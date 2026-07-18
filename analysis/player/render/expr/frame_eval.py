"""Statement-level tree-walk interpreter: run a chart body against a Surface.

`tree_eval` evaluates one EXPRESSION to a value (read-only, stateless). This is
its statement-level sibling: it EXECUTES a body's statements - assignments,
control flow, and effect calls - stepping the same AST against the same
`Surface`. It is the map-general replacement for running the body as Lua: the
interpreter covers the LANGUAGE (the node grammar), not any one chart's idioms,
so a body from any map runs here without lupa, and a construct outside the
modeled subset (`Unparsed`) is skipped per-node - never a hard fault.

It is NOT a NotITG VM. The engine surface (actor pokes, getter reads, clock
symbols) is entirely the `Surface`'s job; this interpreter only decides
control flow and threads values. So the same interpreter drives any game whose
`Surface` implements the effect/read verbs - it is an API layer feeding the
host engine, not a reimplementation of one.

Statefulness over `tree_eval`:
- a `Scope` chain holds locals (`local x = ...`) and assigned globals, so an
  accumulator (`x = a*x + b`) reads its previous value and stores the next -
  the recurrence a pure expression evaluator cannot carry;
- an unbound name falls through the scope chain to `surface.symbol`, so
  `beat`/`mod_time`/actor globals resolve exactly as a guard's would.

Effects vs values (the Phase 1 Surface split):
- `recv:setter(args)` as a STATEMENT -> `surface.poke` (mutate the actor);
- `recv:getter(args)` in an EXPRESSION -> `surface.method` (read a value).

Closures (`FuncExpr`/`FuncDef`) evaluate to a Python callable capturing the
defining scope; calling one re-enters the interpreter. A body that schedules
`mm(beat, function() ... end)` therefore defers that closure as a value the
host's scheduler runs later - the interpreter does not run it eagerly.

`trace` (optional) is called at each effect and branch with its source span:
the "why does this look wrong" hook, designed in - a wrong actor value traces
back to the exact statement that produced it.
"""
from __future__ import annotations

import operator
from typing import Callable

from analysis.player.render.expr import ast
from analysis.player.render.expr.surface import UNRESOLVED, Surface

_CMP = {
    '<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge,
    '==': operator.eq, '~=': operator.ne,
}
_ARITH = {
    '+': operator.add, '-': operator.sub, '*': operator.mul,
    '/': operator.truediv, '%': operator.mod, '^': operator.pow,
}

# Cap on interpreter recursion (closure calls, nested loops) so a runaway
# body cannot blow the Python stack - the engine has its own dispatch caps and
# this mirrors that ceiling for the interpreter path.
_MAX_DEPTH = 64

# Cap on a single numeric-for's iteration count: a chart writing
# `for i = 1, huge do` (or an unresolved bound coerced wrong) must not spin.
_MAX_LOOP = 100000


class _Return(Exception):
    """Non-local unwind carrying a `return`'s values up to the call frame."""
    def __init__(self, values):
        self.values = values


class LuaTable:
    """The interpreter's own table type - Lua semantics without a host
    dependency, so a `{...}` constructor round-trips through `table.insert`,
    `ipairs`, and index reads in pure Python (no lupa). One store for both
    parts: 1-based integer keys are the array part, everything else the hash
    part. This is what keeps table-building charts running with no Lua table."""

    __slots__ = ('_data',)

    def __init__(self):
        self._data: dict = {}

    def append(self, value) -> None:
        self._data[self.length() + 1] = value

    def set(self, key, value) -> None:
        # Lua: `t[k] = nil` removes the key (so length() stays contiguous).
        k = _norm_key(key)
        if value is None or value is UNRESOLVED:
            self._data.pop(k, None)
        else:
            self._data[k] = value

    def get(self, key):
        return self._data.get(_norm_key(key), UNRESOLVED)

    def length(self) -> int:
        # Lua `#t` / table.getn: the border of the 1..n contiguous array run.
        n = 0
        while (n + 1) in self._data:
            n += 1
        return n

    def ipairs(self):
        i = 1
        while i in self._data:
            yield i, self._data[i]
            i += 1

    def pairs(self):
        return list(self._data.items())


def _norm_key(key):
    # Lua has no int/float distinction for keys: 1 and 1.0 are the same key, so
    # normalise a whole float to int to keep array keys contiguous.
    if isinstance(key, float) and key.is_integer():
        return int(key)
    return key


class GlobalStore:
    """The interpreter's global namespace, so a body's globals live in ONE
    place a guard/other body can also read. The default is a private dict (the
    pure-logic path); a game backs it with its own shared store (e.g. bound to
    a live host env, so a load-populated global and a per-frame accumulator
    share a namespace, and a guard reading a body-written global sees the value
    the body just wrote). `get` returns UNRESOLVED for an absent name."""

    __slots__ = ('_d',)

    def __init__(self):
        self._d: dict = {}

    def has(self, name: str) -> bool:
        return name in self._d

    def get(self, name: str):
        return self._d.get(name, UNRESOLVED)

    def set(self, name: str, value) -> None:
        self._d[name] = value


class Scope:
    """A lexical scope: local bindings plus a parent link. Name resolution
    walks up the chain; a bare assignment to an unbound name is a GLOBAL (Lua
    default), stored in the root's `store` so sibling scopes and guards see it.
    Only the ROOT scope carries a `store`; child scopes reach it up the chain."""

    __slots__ = ('bindings', 'parent', 'store')

    def __init__(self, parent: 'Scope | None' = None,
                 store: GlobalStore | None = None):
        self.bindings: dict = {}
        self.parent = parent
        # The root owns the global store; a child inherits None and defers to
        # the root's via the chain walk.
        self.store = store if parent is None else None

    def _root(self) -> 'Scope':
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def get(self, name: str):
        scope = self
        while scope is not None:
            b = scope.bindings
            if name in b:
                return b[name]
            store = scope.store       # only the root carries a store
            scope = scope.parent
        return store.get(name) if store is not None else UNRESOLVED

    def has(self, name: str) -> bool:
        scope = self
        while scope is not None:
            if name in scope.bindings:
                return True
            store = scope.store
            scope = scope.parent
        return store is not None and store.has(name)

    def lookup(self, name: str):
        """One chain walk returning `(found, value)` - collapses the
        `has()`+`get()` double walk in the symbol hot path. `found` False means
        neither a local nor the global store had the name (the caller then
        asks the surface). The store IS consulted here (a store-backed global
        is 'found')."""
        scope = self
        store = None
        while scope is not None:
            b = scope.bindings
            if name in b:
                return True, b[name]
            store = scope.store
            scope = scope.parent
        if store is not None and store.has(name):
            return True, store.get(name)
        return False, UNRESOLVED

    def set_local(self, name: str, value) -> None:
        self.bindings[name] = value

    def assign(self, name: str, value) -> None:
        """`name = value`: rebind the nearest enclosing local; else it is a
        GLOBAL (Lua's implicit-global rule) - written to the root's store when
        one backs it, otherwise the root's own bindings."""
        scope = self
        while scope is not None:
            if name in scope.bindings:
                scope.bindings[name] = value
                return
            scope = scope.parent
        root = self._root()
        if root.store is not None:
            root.store.set(name, value)
        else:
            root.bindings[name] = value


class Interpreter:
    """Runs statement bodies against a `Surface`. One interpreter per run;
    the top scope persists across `run` calls so a per-frame body's globals
    (a frame counter, an accumulator) carry between ticks exactly as the
    engine's persistent Lua globals do."""

    def __init__(self, surface: Surface,
                 trace: Callable | None = None,
                 store: GlobalStore | None = None):
        self._surface = surface
        self._trace = trace
        self.root = Scope(store=store)

    def run(self, stmts, scope: Scope | None = None) -> None:
        """Execute a statement sequence in `scope` (the root when omitted)."""
        self._exec_block(stmts, scope or self.root, 0)

    # -- statements ----------------------------------------------------------

    def _exec_block(self, stmts, scope: Scope, depth: int) -> None:
        for stmt in stmts:
            self._exec(stmt, scope, depth)

    def _exec(self, node: ast.Node, scope: Scope, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        match node:
            case ast.Local(names=names, values=values):
                self._exec_local(names, values, scope, depth)
            case ast.Assign(targets=targets, values=values):
                self._exec_assign(targets, values, scope, depth)
            case ast.ExprStmt(expr=expr):
                self._exec_expr_stmt(expr, scope, depth)
            case ast.If():
                self._exec_if(node, scope, depth)
            case ast.NumericFor():
                self._exec_numeric_for(node, scope, depth)
            case ast.GenericFor():
                self._exec_generic_for(node, scope, depth)
            case ast.While():
                self._exec_while(node, scope, depth)
            case ast.FuncDef(name=name, params=params, body=body):
                scope.assign(name, self._make_closure(params, body, scope))
            case ast.Return(values=values):
                raise _Return([self._eval(v, scope, depth) for v in values])
            case ast.Unparsed():
                self._emit_trace('unparsed', node)
            case _:
                # An unmodeled statement node is skipped, not fatal (the
                # map-general floor); a value node used as a statement is a
                # no-op effect.
                self._emit_trace('skip', node)

    def _exec_local(self, names, values, scope, depth) -> None:
        vals = [self._eval(v, scope, depth) for v in values]
        for i, name in enumerate(names):
            scope.set_local(name, vals[i] if i < len(vals) else UNRESOLVED)

    def _exec_assign(self, targets, values, scope, depth) -> None:
        vals = [self._eval(v, scope, depth) for v in values]
        for i, target in enumerate(targets):
            value = vals[i] if i < len(vals) else UNRESOLVED
            self._assign_target(target, value, scope, depth)

    def _assign_target(self, target, value, scope, depth) -> None:
        # A bare name binds a frame variable/global; `t[k] = v` / `t.f = v`
        # write into a LuaTable, and `_G[k] = v` is a GLOBAL write (the Lua
        # global-table idiom for a computed name, `_G['P'..pn] = ...`).
        match target:
            case ast.Sym(name=name):
                scope.assign(name, value)
            case ast.Index(base=ast.Sym(name='_G'), key=key):
                k = self._eval(key, scope, depth)
                if k is not UNRESOLVED:
                    scope.assign(str(k), value)
            case ast.Index(base=base, key=key):
                self._assign_element(base, key, value, scope, depth)
            case ast.Field(base=base, name=name):
                self._assign_element(base, ast.Str(name), value, scope, depth)
            case _:
                self._emit_trace('skip-target', target)

    def _assign_element(self, base, key, value, scope, depth) -> None:
        table = self._eval(base, scope, depth)
        k = key.value if isinstance(key, ast.Str) \
            else self._eval(key, scope, depth)
        if isinstance(table, LuaTable) and k is not UNRESOLVED:
            table.set(k, value)
        else:
            self._emit_trace('skip-target', base)

    def _exec_expr_stmt(self, expr, scope, depth) -> None:
        # A method statement is an EFFECT (`self:zoom(x)`); a plain call is a
        # free-function effect (`update_proxies()`); anything else is an
        # expression evaluated for side effects only (rare).
        match expr:
            case ast.Method(recv=recv, name=name, args=args):
                self._poke(recv, name, args, scope, depth)
            case _:
                self._eval(expr, scope, depth)

    def _poke(self, recv, name, args, scope, depth) -> None:
        recv_v = self._eval(recv, scope, depth)
        arg_vs = [self._eval(a, scope, depth) for a in args]
        self._surface.poke(recv_v, name, arg_vs)
        self._emit_trace('poke', recv, name=name, args=arg_vs)

    def _exec_if(self, node: ast.If, scope, depth) -> None:
        if _truthy(self._eval(node.cond, scope, depth)):
            self._emit_trace('branch-then', node)
            self._exec_block(node.body, Scope(scope), depth)
            return
        for econd, ebody in node.elifs:
            if _truthy(self._eval(econd, scope, depth)):
                self._emit_trace('branch-elif', node)
                self._exec_block(ebody, Scope(scope), depth)
                return
        if node.orelse:
            self._emit_trace('branch-else', node)
            self._exec_block(node.orelse, Scope(scope), depth)

    def _exec_numeric_for(self, node: ast.NumericFor, scope, depth) -> None:
        start = _num(self._eval(node.start, scope, depth))
        stop = _num(self._eval(node.stop, scope, depth))
        step = _num(self._eval(node.step, scope, depth)) if node.step else 1.0
        if start is None or stop is None or step is None or step == 0.0:
            return
        i = start
        count = 0
        while (step > 0 and i <= stop) or (step < 0 and i >= stop):
            if count >= _MAX_LOOP:
                break
            body_scope = Scope(scope)
            body_scope.set_local(node.var, i)
            self._exec_block(node.body, body_scope, depth + 1)
            i += step
            count += 1

    def _exec_generic_for(self, node: ast.GenericFor, scope, depth) -> None:
        # `for k, v in ipairs(t) do` / `for k, v in pairs(t) do` over the
        # interpreter's own LuaTable. The iterator expression names ipairs or
        # pairs on a table; each (k, v) binds the loop names in a fresh scope.
        # Anything else (a custom iterator, a live host collection) is skipped
        # and traced - never a fault.
        pairs = self._iter_pairs(node.exprs, scope, depth)
        if pairs is None:
            self._emit_trace('generic-for-skip', node)
            return
        for kv in pairs:
            body_scope = Scope(scope)
            for i, name in enumerate(node.names):
                body_scope.set_local(name, kv[i] if i < len(kv) else UNRESOLVED)
            self._exec_block(node.body, body_scope, depth + 1)

    def _iter_pairs(self, exprs, scope, depth):
        """The (key, value) sequence a generic-for iterates, or None when the
        iterator is not a recognised ipairs/pairs over a table. Iterates the
        interpreter's own LuaTable directly, and falls back to the surface for
        a HOST table (a lupa table a load pass created), so a chart mixing
        both - `local t = {}` at load, `table.insert(t, ...)` in Update - still
        loops."""
        if not exprs:
            return None
        match exprs[0]:
            case ast.Call(fn=ast.Sym(name='ipairs' | 'pairs' as kind),
                          args=(arg, *_)):
                table = self._eval(arg, scope, depth)
                if isinstance(table, LuaTable):
                    return (list(table.ipairs()) if kind == 'ipairs'
                            else table.pairs())
                if table is not UNRESOLVED:
                    return self._surface.iter_table(table)
        return None

    def _exec_while(self, node: ast.While, scope, depth) -> None:
        count = 0
        while _truthy(self._eval(node.cond, scope, depth)):
            if count >= _MAX_LOOP:
                break
            self._exec_block(node.body, Scope(scope), depth + 1)
            count += 1

    # -- expressions ---------------------------------------------------------

    def _eval(self, node: ast.Node, scope: Scope, depth: int):
        # Dispatch by exact type (faster than structural `match` in the hot
        # per-tick path: 250M+ calls on a real chart). Ordered hottest-first -
        # Sym / Index / Binary / Num dominate a driver body's expressions.
        t = node.__class__
        if t is ast.Sym:
            return self._eval_symbol(node.name, scope)
        if t is ast.Num or t is ast.Str or t is ast.Bool:
            return node.value
        if t is ast.Index:
            return self._eval_index(node.base, node.key, scope, depth)
        if t is ast.Binary:
            op = node.op
            if op == 'and':
                return self._eval_and(node.left, node.right, scope, depth)
            if op == 'or':
                return self._eval_or(node.left, node.right, scope, depth)
            return _binary(op, self._eval(node.left, scope, depth),
                           self._eval(node.right, scope, depth))
        if t is ast.Call:
            return self._eval_call(node.fn, node.args, scope, depth)
        if t is ast.Method:
            return self._eval_method(node.recv, node.name, node.args,
                                     scope, depth)
        if t is ast.Field:
            b = self._eval(node.base, scope, depth)
            if b is UNRESOLVED:
                return UNRESOLVED
            if b.__class__ is LuaTable:
                return b.get(node.name)
            return self._surface.index(b, node.name)
        if t is ast.Unary:
            return _unary(node.op, self._eval(node.operand, scope, depth))
        if t is ast.Nil:
            return None
        if t is ast.FuncExpr:
            return self._make_closure(node.params, node.body, scope)
        if t is ast.Table:
            return self._eval_table(node.array, node.fields, scope, depth)
        return UNRESOLVED

    def _eval_table(self, array, fields, scope, depth):
        """A `{a, b, k = v}` constructor -> a `LuaTable` (1-indexed array part +
        named part), the interpreter's own table type so it round-trips through
        `table.insert`/`ipairs`/index without a host dependency. An UNRESOLVED
        entry is stored as-is (a hole), mirroring Lua's sparse tolerance."""
        table = LuaTable()
        for item in array:
            table.append(self._eval(item, scope, depth))
        for key, value in fields:
            table.set(key, self._eval(value, scope, depth))
        return table

    def _eval_symbol(self, name: str, scope: Scope):
        # A bound local/global shadows the surface; else the surface resolves
        # it (a clock symbol, an actor global, a state var). ONE chain walk.
        found, value = scope.lookup(name)
        return value if found else self._surface.symbol(name)

    def _eval_index(self, base, key, scope, depth):
        # `_G[k]` is a global read by computed name - route through the scope so
        # it sees globals the interpreter wrote (`_G['P'..pn] = ...`), matching
        # the assign side.
        if base.__class__ is ast.Sym and base.name == '_G':
            k = self._eval(key, scope, depth)
            if k is UNRESOLVED:
                return UNRESOLVED
            name = str(k)
            found, value = scope.lookup(name)
            return value if found else self._surface.symbol(name)
        b = self._eval(base, scope, depth)
        k = self._eval(key, scope, depth)
        if b is UNRESOLVED or k is UNRESOLVED:
            return UNRESOLVED
        if b.__class__ is LuaTable:
            return b.get(k)
        return self._surface.index(b, k)

    def _eval_method(self, recv, name, args, scope, depth):
        # A method in VALUE position is a getter read through the surface.
        recv_v = self._eval(recv, scope, depth)
        arg_vs = [self._eval(a, scope, depth) for a in args]
        if recv_v is UNRESOLVED:
            return UNRESOLVED
        return self._surface.method(recv_v, name, arg_vs)

    def _eval_call(self, fn, args, scope, depth):
        arg_vs = [self._eval(a, scope, depth) for a in args]
        # Lua stdlib over the interpreter's own LuaTable (table.insert, ipairs,
        # ...): handled here so a table-building chart never needs a host table.
        builtin = _builtin_call(fn, arg_vs)
        if builtin is not _NO_BUILTIN:
            return builtin
        # A call to an in-scope closure re-enters the interpreter; otherwise
        # the surface may know the free function (`perframe`), else UNRESOLVED.
        match fn:
            case ast.Sym(name=name):
                found, bound = scope.lookup(name)
                if found and callable(bound):
                    return self._call_closure(bound, arg_vs, depth)
                if UNRESOLVED in arg_vs:
                    return UNRESOLVED
                return self._surface.call(name, arg_vs)
            case _:
                target = self._eval(fn, scope, depth)
                if callable(target):
                    return self._call_closure(target, arg_vs, depth)
                return UNRESOLVED

    def _eval_and(self, left, right, scope, depth):
        # Lua `and` returns the OPERAND: `a and b` is `a` when `a` is falsy,
        # else `b`. An UNRESOLVED left makes the choice unknowable (the answer
        # is `a` on one branch, `b` on the other), so the whole expression is
        # UNRESOLVED - the right operand's value cannot rescue it.
        a = self._eval(left, scope, depth)
        if a is UNRESOLVED:
            return UNRESOLVED
        if not _truthy_raw(a):
            return a
        return self._eval(right, scope, depth)

    def _eval_or(self, left, right, scope, depth):
        # Lua `or` returns the OPERAND: `a or b` is `a` when `a` is truthy,
        # else `b`. An UNRESOLVED left is unknowable (the answer is `a` if
        # truthy, else `b`), so the expression is UNRESOLVED - "skip, do not
        # guess" (never fabricate a seed for an unknown accumulator).
        a = self._eval(left, scope, depth)
        if a is UNRESOLVED:
            return UNRESOLVED
        if _truthy_raw(a):
            return a
        return self._eval(right, scope, depth)

    # -- closures ------------------------------------------------------------

    def _make_closure(self, params, body, defining_scope: Scope):
        """A Python callable capturing `defining_scope`; calling it binds the
        params in a child scope and runs the body, returning the `return`
        values (or None)."""
        def closure(*call_args):
            call_scope = Scope(defining_scope)
            for i, param in enumerate(params):
                call_scope.set_local(
                    param, call_args[i] if i < len(call_args) else UNRESOLVED)
            try:
                self._exec_block(body, call_scope, _closure_depth_guard(self))
            except _Return as ret:
                return ret.values[0] if ret.values else None
            return None
        closure._frame_closure = True   # UNRESOLVED-aware (no nil marshalling)
        return closure

    def _call_closure(self, fn, args, depth: int):
        if depth > _MAX_DEPTH:
            return UNRESOLVED
        # A closure the interpreter made is UNRESOLVED-aware; any OTHER callable
        # is a host (lupa) function that must receive Lua values - the sentinel
        # marshals to nil, so a Lua helper never does arithmetic on userdata
        # (the `compare userdata with number` fault). Interpreter closures carry
        # the marker set in `_make_closure`.
        if not getattr(fn, '_frame_closure', False):
            args = [None if a is UNRESOLVED else a for a in args]
        try:
            return fn(*args)
        except _Return as ret:
            return ret.values[0] if ret.values else None

    # -- trace ---------------------------------------------------------------

    def _emit_trace(self, kind: str, node, **extra) -> None:
        if self._trace is not None:
            self._trace(kind, getattr(node, 'span', None), extra)


def _closure_depth_guard(interp) -> int:
    # Closures start a fresh depth budget rooted below the cap so a deeply
    # self-calling closure still terminates; the per-call _MAX_DEPTH check in
    # `_call_closure` is the real ceiling.
    return 1


# Sentinel: this call is not a stdlib builtin (distinct from a builtin that
# legitimately returns None/UNRESOLVED).
_NO_BUILTIN = object()


def _builtin_call(fn, args):
    """The Lua stdlib functions the interpreter services over its own
    `LuaTable` - `table.insert/remove/getn`, `#`-style length. Returns
    `_NO_BUILTIN` when `fn` is not one of them (so the caller falls through to
    closures/the surface). A call whose table arg is not a LuaTable is not a
    builtin here either - it flows on (a host table handles its own)."""
    if not isinstance(fn, ast.Field) or not isinstance(fn.base, ast.Sym):
        return _NO_BUILTIN
    if fn.base.name != 'table' or not args or not isinstance(args[0], LuaTable):
        return _NO_BUILTIN
    table = args[0]
    match fn.name:
        case 'insert':
            _table_insert(table, args[1:])
            return None
        case 'remove':
            return _table_remove(table, args[1:])
        case 'getn':
            return float(table.length())
    return _NO_BUILTIN


def _table_insert(table: LuaTable, rest) -> None:
    # table.insert(t, v) appends; table.insert(t, pos, v) inserts at pos.
    if len(rest) >= 2 and isinstance(rest[0], (int, float)):
        pos = int(rest[0])
        for i in range(table.length(), pos - 1, -1):
            table.set(i + 1, table.get(i))
        table.set(pos, rest[1])
    elif rest:
        table.append(rest[0])


def _table_remove(table: LuaTable, rest):
    n = table.length()
    if n == 0:
        return None
    pos = int(rest[0]) if rest and isinstance(rest[0], (int, float)) else n
    value = table.get(pos)
    for i in range(pos, n):
        table.set(i, table.get(i + 1))
    table.set(n, None)
    return value


def _truthy(value) -> bool:
    # Lua truthiness with the UNRESOLVED discipline: an unprovable condition is
    # FALSE for control flow (skip, do not guess) - the branch runs only when
    # positively true, mirroring window extraction's "cannot prove -> skip".
    if value is UNRESOLVED:
        return False
    return _truthy_raw(value)


def _truthy_raw(value) -> bool:
    # Lua: only nil and false are falsy; 0 and '' are TRUE.
    return value is not None and value is not False


def _unary(op: str, x):
    if x is UNRESOLVED:
        return UNRESOLVED
    if op == '-':
        return -x if isinstance(x, (int, float)) else UNRESOLVED
    if op == 'not':
        return not _truthy_raw(x)
    if op == '#':
        try:
            return len(x)
        except TypeError:
            return UNRESOLVED
    return UNRESOLVED


def _binary(op: str, a, b):
    if a is UNRESOLVED or b is UNRESOLVED:
        return UNRESOLVED
    if op == '..':
        return f'{_concat(a)}{_concat(b)}'
    fn = _CMP.get(op) or _ARITH.get(op)
    if fn is None:
        return UNRESOLVED
    try:
        return fn(a, b)
    except (TypeError, ZeroDivisionError):
        return UNRESOLVED


def _concat(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _num(value):
    return float(value) if isinstance(value, (int, float)) \
        and not isinstance(value, bool) else None
