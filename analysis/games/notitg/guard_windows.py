"""NotITG bridge: an Update body's live per-frame windows, via the Lua AST.

Replaces the regex window extraction in `update_integrator._live_windows`.
Parses the body once, walks the AST for the two window sources - `if`
conditions that are `beat`/`mod_time` ranges, and `perframe(a, b)` calls
anywhere - and reduces each to a `(start, end)` beat span through the
game-agnostic `expr.windows` extractor. Bounds resolve against a constant
surface (literals + any compiled `v`/`e` tables); a guard over live locals
or nil tables yields no window (skipped, not guessed).

The window set is merged and sorted, matching the consumed shape at
`update_integrator._live_windows`.
"""
from __future__ import annotations

from analysis.games.notitg.xml_actors import _lua50_compat, _strip_lua_wrapper
from analysis.player.render.expr import ast
from analysis.player.render.expr.eval_tree import tree_eval
from analysis.player.render.expr.parser import parse_body
from analysis.player.render.expr.surface import ConstSurface, Surface
from analysis.player.render.expr.windows import guard_windows as _guard_windows


def _prepare(body: str) -> str:
    """A command body ready for the Lua parser: apply the same Lua-5.0
    compat rewrite the engine sees (nested long-comments blanked, number/
    keyword spacing, escape fixups) THEN unwrap the `%function(self)...end`
    command wrapper to its statement body. Running compat first is what lets
    a guard survive a `--[[ .. [[ .. ]] .. ]]` nested comment that a naive
    strip would mis-close."""
    return _strip_lua_wrapper(_lua50_compat(body))


def windows_from_body(body: str,
                      const_surface: Surface | None = None) -> list:
    """Sorted, merged (start, end) beat windows for `body`. `const_surface`
    resolves compiled constants (v/e tables); default is literals only. A
    `%function(self)...end` command wrapper is unwrapped to its statement
    body first (the AST parses Lua statements, not the `%`-expression form)."""
    surface = const_surface or ConstSurface()
    stmts, _sink = parse_body(_prepare(body))
    spans = []
    for node in _guard_nodes(stmts):
        spans.extend(_guard_windows(node, surface))
    return _merge_spans(sorted(spans))


def rearm_period(body: str, update_command: str = 'Update',
                 const_surface: Surface | None = None) -> float | None:
    """Seconds between self-scheduled invocations, from a
    `self:sleep(X)` paired with a `self:queuecommand('<update_command>')`
    in the body - or None when the body never re-arms. Replaces the
    `_REARM_RE` scrape: walk for a `sleep` method call whose arg is a
    positive number when a matching queuecommand is present.

    `const_surface` resolves a period AUTHORED AS AN EXPRESSION - gat 2
    re-arms with `self:sleep(1 / gf2_fps)` where `gf2_fps` is a chart
    global set at load. Missing it is not a cosmetic default: the sim
    then ticks the body at ITS rate instead of the chart's, and every
    per-frame integrator (`ry = ry + rotspd` per tick) accumulates
    proportionally too much - a spin phase that drifts off the engine's.
    """
    stmts, _sink = parse_body(_prepare(body))
    has_rearm = False
    period: float | None = None
    for node in _walk(stmts):
        match node:
            case ast.Method(name='queuecommand', args=(ast.Str(value=v), *_)) \
                    if v == update_command:
                has_rearm = True
            case ast.Method(name='sleep', args=(arg, *_)):
                value = _positive_seconds(arg, const_surface)
                if value is not None:
                    period = value
    return period if has_rearm else None


def _positive_seconds(arg, const_surface: Surface | None) -> float | None:
    """`arg` as a positive number of seconds: a literal directly, any other
    expression through the constant evaluator (None without a surface, or
    when the expression reads something the surface cannot answer)."""
    match arg:
        case ast.Num(value=n):
            return n if n > 0.0 else None
    if const_surface is None:
        return None
    try:
        value = tree_eval(arg, const_surface)
    except Exception:
        return None
    if isinstance(value, (int, float)) and value > 0.0:
        return float(value)
    return None


def bound_global_name(body: str) -> str | None:
    """The Lua global a body self-assigns (`some_actor = self`), or None.
    Walks for an `Assign` whose value is the bare `self` and whose target is
    a plain name - the name the scheduled closures poke. Replaces the
    `_BIND_RE` scrape (which matched a bare `NAME = self` anywhere, even in
    `x = self:GetShader()`, a false bind the AST rejects)."""
    stmts, _sink = parse_body(_prepare(body))
    for node in _walk(stmts):
        match node:
            case ast.Assign(targets=targets, values=values):
                for target, value in zip(targets, values):
                    if (isinstance(target, ast.Sym)
                            and isinstance(value, ast.Sym)
                            and value.name == 'self'
                            and target.name != 'self'):
                        return target.name
    return None


def _guard_nodes(stmts):
    """Every window-bearing guard node in the tree: each `if`/`elseif`
    condition, and every `perframe(...)` call anywhere (a chart may write
    `if perframe(a,b) then` or bare `perframe(a,b)`)."""
    for node in _walk(stmts):
        match node:
            case ast.If(cond=cond, elifs=elifs):
                yield cond
                for econd, _body in elifs:
                    yield econd
            case ast.Call(fn=ast.Sym(name=name), args=args) \
                    if name.endswith('perframe'):
                # `perframe(a,b)` and chart wrappers around it
                # (`floral_perframe`, `smf_perframe`) - all range-membership
                # helpers. Normalize the wrapper name so the game-agnostic
                # extractor reads them as a plain perframe window.
                yield ast.Call(ast.Sym('perframe'), args, span=node.span)


def _walk(nodes):
    """Depth-first over every Node in a statement/expression tree."""
    for node in nodes:
        yield from _walk_one(node)


def _walk_one(node):
    yield node
    match node:
        case ast.If(cond=cond, body=body, elifs=elifs, orelse=orelse):
            yield from _walk_one(cond)
            yield from _walk(body)
            for econd, ebody in elifs:
                yield from _walk_one(econd)
                yield from _walk(ebody)
            yield from _walk(orelse)
        case ast.NumericFor(start=start, stop=stop, step=step, body=body):
            yield from _walk_one(start)
            yield from _walk_one(stop)
            if step is not None:
                yield from _walk_one(step)
            yield from _walk(body)
        case ast.GenericFor(exprs=exprs, body=body):
            yield from _walk(exprs)
            yield from _walk(body)
        case ast.While(cond=cond, body=body):
            yield from _walk_one(cond)
            yield from _walk(body)
        case ast.FuncDef(body=body):
            yield from _walk(body)
        case ast.FuncExpr(body=body):
            yield from _walk(body)
        case ast.Assign(values=values):
            yield from _walk(values)
        case ast.Local(values=values):
            yield from _walk(values)
        case ast.Return(values=values):
            yield from _walk(values)
        case ast.ExprStmt(expr=expr):
            yield from _walk_one(expr)
        case ast.Binary(left=left, right=right):
            yield from _walk_one(left)
            yield from _walk_one(right)
        case ast.Unary(operand=operand):
            yield from _walk_one(operand)
        case ast.Call(fn=fn, args=args):
            yield from _walk_one(fn)
            yield from _walk(args)
        case ast.Method(recv=recv, args=args):
            yield from _walk_one(recv)
            yield from _walk(args)
        case ast.Index(base=base, key=key):
            yield from _walk_one(base)
            yield from _walk_one(key)
        case ast.Field(base=base):
            yield from _walk_one(base)
        case ast.Table(array=array, fields=fields):
            yield from _walk(array)
            for _key, value in fields:
                yield from _walk_one(value)


def _merge_spans(spans):
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
