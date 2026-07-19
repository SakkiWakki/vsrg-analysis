"""Closure compiler for the frame interpreter: AST -> nested Python closures.

`frame_eval.Interpreter` tree-walks the AST, re-dispatching on node type every
tick (250M+ `_eval` calls on a real chart - the compiled-body perf wall). This
compiles a body's AST ONCE into nested closures: every node becomes a
`(scope) -> value` (expressions) or `(scope) -> None` (statements) that closes
over its children's compiled closures. The per-tick cost is then a direct call
chain - no type dispatch, no `match`, no per-node attribute pulls.

Semantics are IDENTICAL to the interpreter's - this is a compile of the SAME
behaviour, not a variant. It reuses `frame_eval`'s Scope/LuaTable/builtins and
calls back into the owning `Interpreter` for surface reads/writes, closures,
and traces, so the two paths differ only in dispatch cost. The keyframe-diff
oracle gates that equality; where a node shape is outside the compiled subset
it falls back to `interp._eval`/`_exec` (the tree-walk floor), so the compiler
is a strict speedup, never a coverage regression.
"""
from __future__ import annotations

from analysis.player.render.expr import ast
from analysis.player.render.expr.frame_eval import (
    LuaTable, Scope, UNRESOLVED, _NO_BUILTIN, _binary, _builtin_call, _num,
    _resolved_nil, _truthy, _truthy_raw, _unary)


def compile_body(stmts, interp):
    """Compile a statement sequence to one `run(scope)` closure. `interp` owns
    the surface and runtime helpers the compiled closures call back into."""
    compiled = [compile_stmt(s, interp) for s in stmts]

    def run(scope):
        for c in compiled:
            c(scope)
    return run


# -- expressions: compile to `(scope) -> value` ------------------------------

def compile_expr(node, interp):
    """A `(scope) -> value` closure for an expression node. Falls back to the
    interpreter's tree-walk for a node the compiler does not model (still one
    closure, just slower for that node)."""
    t = node.__class__
    if t is ast.Num or t is ast.Str or t is ast.Bool:
        v = node.value
        return lambda scope: v
    if t is ast.Nil:
        return lambda scope: None
    if t is ast.Sym:
        return _compile_sym(node.name, interp)
    if t is ast.Binary:
        return _compile_binary(node, interp)
    if t is ast.Index:
        return _compile_index(node, interp)
    if t is ast.Field:
        return _compile_field(node, interp)
    if t is ast.Unary:
        operand = compile_expr(node.operand, interp)
        op = node.op
        return lambda scope: _unary(op, operand(scope))
    if t is ast.Call:
        return _compile_call(node, interp)
    if t is ast.Method:
        return _compile_method(node, interp)
    if t is ast.FuncExpr:
        params, body = node.params, node.body
        return lambda scope: interp._make_closure(params, body, scope)
    if t is ast.Table:
        return _compile_table(node, interp)
    # Unmodeled expression node: defer to the tree-walk (rare).
    return lambda scope: interp._eval(node, scope, 0)


def _compile_sym(name, interp):
    surface = interp._surface

    def read(scope):
        found, value = scope.lookup(name)
        return value if found else surface.symbol(name)
    return read


def _compile_binary(node, interp):
    left = compile_expr(node.left, interp)
    right = compile_expr(node.right, interp)
    op = node.op
    if op == 'and':
        def eval_and(scope):
            a = left(scope)
            if a is UNRESOLVED:
                return UNRESOLVED
            return right(scope) if _truthy_raw(a) else a
        return eval_and
    if op == 'or':
        def eval_or(scope):
            a = left(scope)
            if a is UNRESOLVED:
                return UNRESOLVED
            return a if _truthy_raw(a) else right(scope)
        return eval_or
    return lambda scope: _binary(op, left(scope), right(scope))


def _compile_index(node, interp):
    surface = interp._surface
    key = compile_expr(node.key, interp)
    base_node = node.base
    # `_G[k]` is a computed-name global read - route through the scope.
    if base_node.__class__ is ast.Sym and base_node.name == '_G':
        def read_global(scope):
            k = key(scope)
            if k is UNRESOLVED:
                return UNRESOLVED
            name = str(k)
            found, value = scope.lookup(name)
            return value if found else surface.symbol(name)
        return read_global
    base = compile_expr(base_node, interp)

    def read_index(scope):
        b = base(scope)
        k = key(scope)
        if b is UNRESOLVED or k is UNRESOLVED:
            return UNRESOLVED
        if b.__class__ is LuaTable:
            return b.get(k)
        return _resolved_nil(surface.index(b, k))
    return read_index


def _compile_field(node, interp):
    surface = interp._surface
    base = compile_expr(node.base, interp)
    name = node.name

    def read_field(scope):
        b = base(scope)
        if b is UNRESOLVED:
            return UNRESOLVED
        if b.__class__ is LuaTable:
            return b.get(name)
        return _resolved_nil(surface.index(b, name))
    return read_field


def _compile_call(node, interp):
    surface = interp._surface
    fn_node = node.fn
    arg_fns = [compile_expr(a, interp) for a in node.args]

    if fn_node.__class__ is ast.Sym:
        name = fn_node.name

        def call_sym(scope):
            arg_vs = [a(scope) for a in arg_fns]
            builtin = _builtin_call(fn_node, arg_vs, surface)
            if builtin is not _NO_BUILTIN:
                return builtin
            found, bound = scope.lookup(name)
            if found and callable(bound):
                return interp._call_closure(bound, arg_vs, 0)
            if not found:
                global_fn = surface.symbol(name)
                if global_fn is not UNRESOLVED and callable(global_fn):
                    return interp._call_closure(global_fn, arg_vs, 0)
            if UNRESOLVED in arg_vs:
                return UNRESOLVED
            return surface.call(name, arg_vs)
        return call_sym

    fn = compile_expr(fn_node, interp)

    def call_expr(scope):
        arg_vs = [a(scope) for a in arg_fns]
        builtin = _builtin_call(fn_node, arg_vs, surface)
        if builtin is not _NO_BUILTIN:
            return builtin
        target = fn(scope)
        if callable(target):
            return interp._call_closure(target, arg_vs, 0)
        return UNRESOLVED
    return call_expr


def _compile_method(node, interp):
    surface = interp._surface
    recv = compile_expr(node.recv, interp)
    name = node.name
    arg_fns = [compile_expr(a, interp) for a in node.args]

    def read_method(scope):
        recv_v = recv(scope)
        if recv_v is UNRESOLVED:
            return UNRESOLVED
        return surface.method(recv_v, name, [a(scope) for a in arg_fns])
    return read_method


def _compile_table(node, interp):
    array_fns = [compile_expr(v, interp) for v in node.array]
    field_fns = [(k, compile_expr(v, interp)) for k, v in node.fields]

    def build(scope):
        table = LuaTable()
        for fn in array_fns:
            table.append(fn(scope))
        for k, fn in field_fns:
            table.set(k, fn(scope))
        return table
    return build


# -- statements: compile to `(scope) -> None` --------------------------------

def compile_stmt(node, interp):
    t = node.__class__
    if t is ast.ExprStmt:
        return _compile_expr_stmt(node, interp)
    if t is ast.If:
        return _compile_if(node, interp)
    if t is ast.Local:
        return _compile_local(node, interp)
    if t is ast.Assign:
        return _compile_assign(node, interp)
    if t is ast.NumericFor:
        return _compile_numeric_for(node, interp)
    if t is ast.GenericFor:
        return _compile_generic_for(node, interp)
    if t is ast.While:
        return _compile_while(node, interp)
    if t is ast.FuncDef:
        params, body, name = node.params, node.body, node.name
        return lambda scope: scope.assign(
            name, interp._make_closure(params, body, scope))
    # Return / Unparsed / unmodeled: defer to the tree-walk (Return needs its
    # _Return unwind, which the interpreter owns; these are rare in a body).
    return lambda scope: interp._exec(node, scope, 0)


def _compile_expr_stmt(node, interp):
    expr = node.expr
    if expr.__class__ is ast.Method:
        surface = interp._surface
        recv = compile_expr(expr.recv, interp)
        name = expr.name
        arg_fns = [compile_expr(a, interp) for a in expr.args]

        def poke(scope):
            recv_v = recv(scope)
            surface.poke(recv_v, name, [a(scope) for a in arg_fns])
        return poke
    compiled = compile_expr(expr, interp)
    return lambda scope: compiled(scope)


def _compile_if(node, interp):
    cond = compile_expr(node.cond, interp)
    then_body = compile_body(node.body, interp)
    elifs = [(compile_expr(c, interp), compile_body(b, interp))
             for c, b in node.elifs]
    orelse = compile_body(node.orelse, interp) if node.orelse else None

    def run_if(scope):
        if _truthy(cond(scope)):
            then_body(_child(node.body, scope))
            return
        for econd, ebody_run, ebody in elifs_meta:
            if _truthy(econd(scope)):
                ebody_run(_child(ebody, scope))
                return
        if orelse is not None:
            orelse(_child(node.orelse, scope))
    # Precompute (cond, run, body) so the elif loop needs no zip per tick.
    elifs_meta = [(c, r, b) for (c, r), (_oc, b) in
                  zip(elifs, node.elifs)]
    return run_if


def _compile_local(node, interp):
    names = node.names
    val_fns = [compile_expr(v, interp) for v in node.values]

    def run_local(scope):
        vals = [f(scope) for f in val_fns]
        for i, name in enumerate(names):
            scope.set_local(name, vals[i] if i < len(vals) else UNRESOLVED)
    return run_local


def _compile_assign(node, interp):
    val_fns = [compile_expr(v, interp) for v in node.values]
    target_setters = [_compile_target(t, interp) for t in node.targets]

    def run_assign(scope):
        vals = [f(scope) for f in val_fns]
        for i, setter in enumerate(target_setters):
            setter(scope, vals[i] if i < len(vals) else UNRESOLVED)
    return run_assign


def _compile_target(target, interp):
    """A `(scope, value) -> None` setter for one assignment target."""
    t = target.__class__
    if t is ast.Sym:
        name = target.name
        return lambda scope, value: scope.assign(name, value)
    if t is ast.Index and target.base.__class__ is ast.Sym \
            and target.base.name == '_G':
        key = compile_expr(target.key, interp)

        def set_global(scope, value):
            k = key(scope)
            if k is not UNRESOLVED:
                scope.assign(str(k), value)
        return set_global
    if t is ast.Index:
        base = compile_expr(target.base, interp)
        key = compile_expr(target.key, interp)
        return _element_setter(base, key, interp)
    if t is ast.Field:
        base = compile_expr(target.base, interp)
        name = target.name
        return _element_setter(base, lambda scope: name, interp)
    return lambda scope, value: interp._emit_trace('skip-target', target)


def _element_setter(base, key, interp):
    surface = interp._surface

    def set_element(scope, value):
        table = base(scope)
        k = key(scope)
        if table.__class__ is LuaTable and k is not UNRESOLVED:
            table.set(k, value)
        else:
            surface.set_index(table, k, value)
    return set_element


def _compile_numeric_for(node, interp):
    start = compile_expr(node.start, interp)
    stop = compile_expr(node.stop, interp)
    step_fn = compile_expr(node.step, interp) if node.step else None
    body = compile_body(node.body, interp)
    var = node.var
    from analysis.player.render.expr.frame_eval import _MAX_LOOP

    def run_for(scope):
        s = _num(start(scope))
        e = _num(stop(scope))
        st = _num(step_fn(scope)) if step_fn else 1.0
        if s is None or e is None or st is None or st == 0.0:
            return
        i = s
        count = 0
        while (st > 0 and i <= e) or (st < 0 and i >= e):
            if count >= _MAX_LOOP:
                break
            child = Scope(scope)
            child.bindings[var] = i
            body(child)
            i += st
            count += 1
    return run_for


def _compile_generic_for(node, interp):
    body = compile_body(node.body, interp)
    names = node.names
    exprs = node.exprs

    def run_generic(scope):
        pairs = interp._iter_pairs(exprs, scope, 0)
        if pairs is None:
            interp._emit_trace('generic-for-skip', node)
            return
        for kv in pairs:
            child = Scope(scope)
            b = child.bindings
            for i, name in enumerate(names):
                b[name] = kv[i] if i < len(kv) else UNRESOLVED
            body(child)
    return run_generic


def _compile_while(node, interp):
    cond = compile_expr(node.cond, interp)
    body = compile_body(node.body, interp)
    from analysis.player.render.expr.frame_eval import _MAX_LOOP

    def run_while(scope):
        count = 0
        while _truthy(cond(scope)):
            if count >= _MAX_LOOP:
                break
            body(_child(node.body, scope))
            count += 1
    return run_while


def _child(body, scope):
    """A child scope for a block. A block that declares no local/function reads
    and writes identically in the parent scope (nothing new binds), so it reuses
    the parent - avoiding a per-tick allocation for the vast majority of
    if-bodies. Cached per body tuple."""
    return Scope(scope) if _declares_local(body) else scope


_DECLARES_LOCAL: dict = {}


def _declares_local(body) -> bool:
    key = id(body)
    cached = _DECLARES_LOCAL.get(key)
    if cached is None:
        cached = any(s.__class__ is ast.Local or s.__class__ is ast.FuncDef
                     for s in body)
        _DECLARES_LOCAL[key] = cached
    return cached
