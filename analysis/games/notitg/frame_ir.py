"""Frame IR: compile a NotITG Update body into a scope+window tree.

The per-frame `%function(self) ... end` body is a nest of lexical blocks.
Each block (`if..then..end`, `for..do..end`) is a FRAME: a scope with
variable bindings and a temporal window. Names resolve up the parent chain
exactly like an interpreter's `[frame, parent]` walk; a write inside a frame
is live over the INTERSECTION of its enclosing frames' windows, under their
bindings.

`build_frames(body)` parses the body once (the same `_prepare` +
`parse_body` the guard-window path uses) and walks the statements into a
`Frame` tree:
- an `if` with a beat/perframe guard opens a child frame WINDOWED by that
  guard; an `if` with a non-window guard opens a child frame that inherits
  the parent window and carries the guard as a `gate`;
- a numeric `for` with literal bounds UNROLLS into one child frame per loop
  value, each binding the loop var; non-literal bounds open a single frame
  with the loop var UNBOUND (its writes route to attention downstream);
- a `local` adds to the current frame's bindings;
- a setter poke (`recv:prop(arg)` where `prop` is a verb-surface setter) or
  an assignment becomes a `VarUpdate` leaf; any other statement becomes a
  `VarUpdate` carrying the raw node (attention fallback).

`resolve`, `effective_window`, and `iter_updates` read the tree: name
lookup up the parent chain, the intersected live window of a frame, and a
flat enumeration of every `(VarUpdate, owning Frame)` pair for the router
and sinks to consume.

The router (a downstream lane) fills each `VarUpdate.closed_form`; `route`
is DERIVED from it - None means attention, a set form means SSM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from analysis.games.notitg import guard_windows
from analysis.games.notitg.sim import verb_surface
from analysis.player.render.expr import ast
from analysis.player.render.expr.parser import parse_body
from analysis.player.render.expr.surface import ConstSurface, Surface
from analysis.player.render.expr.windows import guard_windows as _guard_windows

# Method names that set a per-frame property. Built from the verb surface's
# setter tables, so the builder keys on the engine mechanism, never on
# chart-specific actor names.
_SETTER_NAMES = frozenset(verb_surface.SCALAR_SETTERS) \
    | frozenset(verb_surface.BULK_SETTERS) \
    | frozenset(verb_surface.ADD_SETTERS) \
    | frozenset(verb_surface.BULK_ADD_SETTERS)

# Cap so a pathological literal loop cannot explode the frame tree; real
# per-player loops are 1..8.
_MAX_UNROLL = 64

# Sentinel returned by `resolve` when a name is bound nowhere up the chain.
UNBOUND = object()


@dataclass(frozen=True)
class ClosedForm:
    """What a flattener emits for an SSM-routed variable: a kernel plus the
    coefficient channels it reads and the initial value. `kernel(t_or_n, h0,
    coeffs) -> value` is evaluable at any point with no stepping, so the SSM
    sink stays flattener-agnostic."""
    kernel: object          # callable (t_or_n, h0, coeffs) -> value
    coeffs: dict            # {name: Channel}
    h0: object              # initial value or Channel


@dataclass
class VarUpdate:
    """One write inside a frame - a property poke or a variable assignment.
    `name` is `actor.prop` (a poked setter) or the bare assigned name; `node`
    is the raw write AST, always kept for the attention fallback.
    `closed_form` is set by a flattener when the write is a recognized
    recurrence; None routes it to attention."""
    name: str
    node: ast.Node
    closed_form: ClosedForm | None = None


@dataclass
class Frame:
    """A lexical block: a scope (`bindings`) with a temporal `window` and
    nested `children`. `window` None inherits the parent's window (see
    `effective_window`). `gate` is a non-window guard condition the frame is
    conditioned on (an `if <flag>` that is not a beat range); None when the
    frame opens unconditionally. `parent` links up the chain for name
    resolution."""
    window: tuple | None
    bindings: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    parent: object = None
    gate: ast.Node | None = None


def build_frames(body: str,
                 const_surface: Surface | None = None) -> Frame:
    """Parse an Update `body` into a Frame tree. `const_surface` resolves
    compiled constants for guard bounds (v/e tables); default is literals
    only. Returns the root Frame (window None = the whole body)."""
    surface = const_surface or ConstSurface()
    stmts, _sink = parse_body(guard_windows._prepare(body))
    root = Frame(window=None)
    _build_stmts(stmts, root, surface)
    return root


def _build_stmts(stmts: Iterable[ast.Node], frame: Frame,
                 surface: Surface) -> None:
    for stmt in stmts:
        _build_one(stmt, frame, surface)


def _build_one(stmt: ast.Node, frame: Frame, surface: Surface) -> None:
    match stmt:
        case ast.If():
            _build_if(stmt, frame, surface)
        case ast.NumericFor():
            _build_for(stmt, frame, surface)
        case ast.Local(names=names, values=values):
            _bind_local(names, values, frame)
        case ast.ExprStmt(expr=ast.Method(name=name)) if name in _SETTER_NAMES:
            frame.children.append(_poke_update(stmt.expr))
        case ast.Assign(targets=targets):
            frame.children.append(VarUpdate(_target_name(targets), stmt))
        case _:
            frame.children.append(VarUpdate(_attention_name(stmt), stmt))


def _build_if(node: ast.If, frame: Frame, surface: Surface) -> None:
    """An `if` opens a child frame. A beat/perframe guard windows the child;
    a non-window guard leaves the child window None (inherit parent) and
    records the guard as its `gate`. `elseif` branches open sibling child
    frames the same way; a bare `else` body opens an ungated inheriting
    frame."""
    _add_branch(node.cond, node.body, frame, surface)
    for econd, ebody in node.elifs:
        _add_branch(econd, ebody, frame, surface)
    if node.orelse:
        child = Frame(window=None, parent=frame)
        frame.children.append(child)
        _build_stmts(node.orelse, child, surface)


def _add_branch(cond: ast.Node, body: Iterable[ast.Node], frame: Frame,
                surface: Surface) -> None:
    window = _cond_window(cond, surface)
    gate = None if window is not None else cond
    child = Frame(window=window, parent=frame, gate=gate)
    frame.children.append(child)
    _build_stmts(body, child, surface)


def _cond_window(cond: ast.Node, surface: Surface) -> tuple | None:
    """The single beat window a guard is live over, or None for a non-window
    guard. `perframe(...)` wrappers are normalized to a plain perframe call
    (as the guard-window path does) before extraction."""
    windows = _guard_windows(_normalize_perframe(cond), surface)
    return windows[0] if windows else None


def _normalize_perframe(node: ast.Node) -> ast.Node:
    """Rewrite a `<name>perframe(a,b)` wrapper call to a bare `perframe(a,b)`
    so the game-agnostic extractor reads chart wrappers as plain windows."""
    match node:
        case ast.Call(fn=ast.Sym(name=name), args=args) \
                if name.endswith('perframe') and name != 'perframe':
            return ast.Call(ast.Sym('perframe'), args, span=node.span)
    return node


def _build_for(node: ast.NumericFor, frame: Frame, surface: Surface) -> None:
    """A numeric `for` with LITERAL bounds unrolls to one child frame per
    loop value, each binding the loop var. Non-literal bounds open a single
    child frame with the loop var UNBOUND, so its writes route to attention
    downstream."""
    values = _unroll_values(node)
    if values is None:
        child = Frame(window=None, parent=frame)
        frame.children.append(child)
        _build_stmts(node.body, child, surface)
        return
    for value in values:
        child = Frame(window=None, parent=frame, bindings={node.var: value})
        frame.children.append(child)
        _build_stmts(node.body, child, surface)


def _unroll_values(node: ast.NumericFor) -> list[float] | None:
    """The loop-variable values a literal-bounds numeric `for` visits, or
    None when a bound is non-literal, the step is zero, or the count exceeds
    the unroll cap (all route to a single unbound frame)."""
    lo = _literal(node.start)
    hi = _literal(node.stop)
    stride = _literal(node.step) if node.step is not None else 1.0
    if lo is None or hi is None or stride is None or stride == 0:
        return None
    if abs(hi - lo) / abs(stride) > _MAX_UNROLL:
        return None
    values = []
    value = lo
    while (stride > 0 and value <= hi) or (stride < 0 and value >= hi):
        values.append(value)
        value += stride
    return values


def _bind_local(names: Iterable[str], values: Iterable[ast.Node],
                frame: Frame) -> None:
    """Add `local names = values` to the frame's bindings. Fewer values than
    names leaves the trailing names bound to nil (the raw None sentinel);
    each name binds to its value AST node (a downstream reader evaluates or
    routes it)."""
    values = tuple(values)
    for i, name in enumerate(names):
        frame.bindings[name] = values[i] if i < len(values) else None


def _poke_update(method: ast.Method) -> VarUpdate:
    """A setter poke `recv:prop(arg)` -> a VarUpdate named `<recv>.<prop>`."""
    return VarUpdate(f'{_render(method.recv)}.{method.name}', method)


def _target_name(targets: tuple[ast.Node, ...]) -> str:
    return _render(targets[0]) if targets else '?'


def _attention_name(stmt: ast.Node) -> str:
    """A stable name for an unmodeled statement's VarUpdate (attention). A
    bare call names its callee; anything else names its node type."""
    match stmt:
        case ast.ExprStmt(expr=ast.Call(fn=fn)):
            return _render(fn)
        case ast.ExprStmt(expr=ast.Method(recv=recv, name=name)):
            return f'{_render(recv)}:{name}'
    return type(stmt).__name__


def _render(node: ast.Node) -> str:
    """A canonical string for a receiver/target expression - enough to name a
    property channel, not a full unparse. Keys on structure, never on any
    chart-specific spelling."""
    match node:
        case ast.Sym(name=name):
            return name
        case ast.Field(base=base, name=name):
            return f'{_render(base)}.{name}'
        case ast.Index(base=base, key=key):
            return f'{_render(base)}[{_render(key)}]'
        case ast.Call(fn=fn, args=args):
            return f'{_render(fn)}({",".join(_render(a) for a in args)})'
        case ast.Method(recv=recv, name=name, args=args):
            return f'{_render(recv)}:{name}({",".join(_render(a) for a in args)})'
        case ast.Num(value=value):
            return _num_str(value)
        case ast.Str(value=value):
            return repr(value)
        case _:
            return type(node).__name__


def _num_str(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _literal(node: ast.Node) -> float | None:
    match node:
        case ast.Num(value=value):
            return value
        case ast.Unary(op='-', operand=ast.Num(value=value)):
            return -value
    return None


def resolve(name: str, frame: Frame):
    """Walk `frame` and its `.parent` chain for `name`'s binding, returning
    the bound value (a loop-var value, a local's value AST) or `UNBOUND` when
    it is bound nowhere. Mirrors an interpreter's `[frame, parent]` resolve:
    the innermost binding wins."""
    current = frame
    while current is not None:
        if name in current.bindings:
            return current.bindings[name]
        current = current.parent
    return UNBOUND


def effective_window(frame: Frame) -> tuple | None:
    """The window a frame is actually live over: the intersection of every
    window up its parent chain. A None window inherits (contributes nothing),
    so the result is None only when no frame up the chain windows it (live
    over the whole body). When the windows do not overlap the intersection is
    an EMPTY span `(start, end)` with `end <= start` - a caller reads that as
    never-live, distinct from the unbounded None."""
    start: float | None = None
    end: float | None = None
    current = frame
    while current is not None:
        if current.window is not None:
            wstart, wend = current.window
            start = wstart if start is None else max(start, wstart)
            end = wend if end is None else min(end, wend)
        current = current.parent
    return None if start is None else (start, end)


def iter_updates(root: Frame) -> Iterator[tuple[VarUpdate, Frame]]:
    """Every `(VarUpdate, owning Frame)` pair in the tree, depth-first. The
    router (to fill `closed_form`) and the sinks (to emit per-property
    channels) iterate over this."""
    for child in root.children:
        match child:
            case VarUpdate():
                yield (child, root)
            case Frame():
                yield from iter_updates(child)
