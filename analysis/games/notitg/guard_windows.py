"""NotITG bridge: an Update body's live per-frame windows, via the Lua AST.

Replaces the regex window extraction in `update_integrator._live_windows`.
Parses the body once, walks the AST for the two window sources - `if`
conditions that are `beat`/`mod_time` ranges, and `perframe(a, b)` calls
anywhere - and reduces each to a `(start, end)` beat span through the
game-agnostic `expr.windows` extractor. Bounds resolve against a constant
surface (literals + any compiled `v`/`e` tables); a guard over live locals
or nil tables yields no window (skipped, not guessed).

The window set is merged and sorted, matching the consumed shape at
`mod_stubs._run_update_ticks`.
"""
from __future__ import annotations

from analysis.games.notitg.xml_actors import _strip_lua_wrapper
from analysis.player.render.expr import ast
from analysis.player.render.expr.parser import parse_body
from analysis.player.render.expr.surface import ConstSurface, Surface
from analysis.player.render.expr.windows import guard_window


def windows_from_body(body: str,
                      const_surface: Surface | None = None) -> list:
    """Sorted, merged (start, end) beat windows for `body`. `const_surface`
    resolves compiled constants (v/e tables); default is literals only. A
    `%function(self)...end` command wrapper is unwrapped to its statement
    body first (the AST parses Lua statements, not the `%`-expression form)."""
    surface = const_surface or ConstSurface()
    stmts, _sink = parse_body(_strip_lua_wrapper(body))
    spans = []
    for node in _guard_nodes(stmts):
        window = guard_window(node, surface)
        if window is not None:
            spans.append(window)
    return _merge_spans(sorted(spans))


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
            case ast.Call(fn=ast.Sym(name='perframe')):
                yield node


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
        case ast.While(cond=cond, body=body):
            yield from _walk_one(cond)
            yield from _walk(body)
        case ast.FuncDef(body=body):
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


def _merge_spans(spans):
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
