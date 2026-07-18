"""Guard AST -> (start, end) beat window, for the Update integrator.

The integrator ticks over the union of a body's live per-frame windows. A
window comes from either form:

- `perframe(a)` / `perframe(a, b)` - a Call node; the window is `(a, b)` (or
  `(a, a+1)` for the one-arg form, the helper's `endBeat = beat+1` default).
- a driver range guard `beat > A and beat < B` (or `>=`/`<=`, or `mod_time`/
  `measure`) - reduce the guard to a conjunction of `driver OP bound`
  clauses where every `bound` evaluates to a constant, a proven-true
  non-driver conjunct drops out, and any unresolved conjunct means the
  window cannot be bounded (skip it, never guess).

`guard_window(node, const_surface)` returns `(start, end)` or None. The
`const_surface` resolves literal arithmetic and compiled constant tables
(`v[]`/`e[]`) but leaves driver symbols (`beat`, `mod_time`, `measure`)
UNRESOLVED - so a bound like `e[1] + e[2]` folds to a number when `e` is
compiled, while `beat` stays the structural driver.
"""
from __future__ import annotations

from analysis.player.render.expr import ast
from analysis.player.render.expr.eval_tree import tree_eval
from analysis.player.render.expr.surface import UNRESOLVED, Surface

# Driver symbols a window ranges over. A guard clause `driver OP bound`
# contributes a start (lower bound) or end (upper bound) to the window.
_DRIVERS = frozenset({'beat', 'mod_time', 'measure', 'time', 'curtime'})

_LOWER_OPS = frozenset({'>', '>='})     # driver > bound  -> start = bound
_UPPER_OPS = frozenset({'<', '<='})     # driver < bound  -> end   = bound


def guard_window(node: ast.Node,
                 const_surface: Surface) -> tuple[float, float] | None:
    """The beat window a guard is live over, or None when it is not a
    resolvable driver range."""
    if isinstance(node, ast.Call) and _call_name(node) == 'perframe':
        return _perframe_window(node, const_surface)
    start, end = _range_bounds(node, const_surface)
    if start is None or end is None or end <= start:
        return None
    return (start, end)


def _perframe_window(node: ast.Call,
                     const_surface: Surface) -> tuple[float, float] | None:
    args = node.args
    if not args:
        return None
    a = tree_eval(args[0], const_surface)
    if a is UNRESOLVED:
        return None
    if len(args) >= 2:
        b = tree_eval(args[1], const_surface)
        if b is UNRESOLVED:
            return None
    else:
        b = a + 1.0                      # perframe(a) -> [a, a+1]
    return (float(a), float(b)) if b > a else None


def _range_bounds(node: ast.Node,
                  const_surface: Surface) -> tuple[float | None, float | None]:
    """Reduce a guard to (start, end) driver bounds. Recurses through `and`
    (both sides contribute), drops a proven-true non-driver conjunct, and
    returns (None, None) when a conjunct is unresolved or not a driver
    range."""
    match node:
        case ast.Binary(op='and', left=left, right=right):
            ls, le = _range_bounds(left, const_surface)
            rs, re_ = _range_bounds(right, const_surface)
            if (ls, le) == (None, None) and _is_true(left, const_surface):
                return rs, re_
            if (rs, re_) == (None, None) and _is_true(right, const_surface):
                return ls, le
            return (_pick(ls, rs, max), _pick(le, re_, min))
        case ast.Binary(op=op, left=left, right=right) if op in _LOWER_OPS | _UPPER_OPS:
            return _clause_bound(op, left, right, const_surface)
    return (None, None)


def _clause_bound(op: str, left: ast.Node, right: ast.Node,
                  const_surface: Surface) -> tuple[float | None, float | None]:
    """One `driver OP bound` (or `bound OP driver`) clause -> a (start, end)
    with one side filled."""
    driver_left = _is_driver(left)
    driver_right = _is_driver(right)
    if driver_left and not driver_right:
        bound = tree_eval(right, const_surface)
        lower = op in _LOWER_OPS
    elif driver_right and not driver_left:
        bound = tree_eval(left, const_surface)
        # `bound < driver` is a LOWER bound on the driver; flip.
        lower = op in _UPPER_OPS
    else:
        return (None, None)
    if bound is UNRESOLVED or not isinstance(bound, (int, float)):
        return (None, None)
    return (float(bound), None) if lower else (None, float(bound))


def _is_driver(node: ast.Node) -> bool:
    return isinstance(node, ast.Sym) and node.name in _DRIVERS


def _is_true(node: ast.Node, const_surface: Surface) -> bool:
    """A non-driver conjunct that provably holds (a resolved-true state
    guard like `fgcurcommand == 2` when fgcurcommand is compiled)."""
    value = tree_eval(node, const_surface)
    return value is not UNRESOLVED and bool(value)


def _pick(a: float | None, b: float | None, combine):
    if a is None:
        return b
    if b is None:
        return a
    return combine(a, b)


def _call_name(node: ast.Call) -> str | None:
    return node.fn.name if isinstance(node.fn, ast.Sym) else None
