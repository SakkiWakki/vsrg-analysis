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
    """The single beat window a guard is live over, or None. A guard that is
    a DISJUNCTION of ranges has more than one window - use `guard_windows`
    for those; this returns the first (kept for single-range callers)."""
    windows = guard_windows(node, const_surface)
    return windows[0] if windows else None


def guard_windows(node: ast.Node,
                  const_surface: Surface) -> list[tuple[float, float]]:
    """Every beat window a guard is live over. The guard is put in
    disjunctive normal form (a disjunction of conjunct-clause sets), so a
    section live in either of two ranges - even nested, like
    `gate and ((beat>A and beat<B) or (beat>C and beat<D))` - yields one
    window per range. Each DNF term reduces to a driver range or a
    `perframe` call; a term that resolves to neither is skipped."""
    windows = []
    for clauses in _to_dnf(node):
        window = _clauses_window(clauses, const_surface)
        if window is not None:
            windows.append(window)
    return windows


def _to_dnf(node: ast.Node) -> list[list[ast.Node]]:
    """Disjunctive normal form as a list of clause-lists (each inner list is
    an AND of clauses; the outer list is the OR). Distributes `and` over
    `or`: `and(x, or(a,b))` -> `[[x,a],[x,b]]`."""
    if isinstance(node, ast.Binary) and node.op == 'or':
        return _cap(_to_dnf(node.left) + _to_dnf(node.right))
    if isinstance(node, ast.Binary) and node.op == 'and':
        left = _to_dnf(node.left)
        right = _to_dnf(node.right)
        return _cap([lc + rc for lc in left for rc in right])
    return [[node]]


# DNF distribution can blow up on a pathological guard (deeply nested
# and/or); cap the term count so window extraction stays bounded. A guard
# past the cap keeps its first terms - the integrator over-ticks a hair
# rather than spins, and real chart guards are far under it.
_MAX_DNF_TERMS = 256


def _cap(terms: list) -> list:
    return terms[:_MAX_DNF_TERMS]


def _clauses_window(clauses: list[ast.Node],
                    const_surface: Surface) -> tuple[float, float] | None:
    """A conjunction of clauses -> its window: a `perframe(a,b)` clause, or
    the driver range formed by the resolvable `driver OP bound` clauses
    (others ignored). None when no window resolves."""
    for clause in clauses:
        if isinstance(clause, ast.Call) and _call_name(clause) == 'perframe':
            return _perframe_window(clause, const_surface)
    start: float | None = None
    end: float | None = None
    for clause in clauses:
        cs, ce = _resolve_clause(clause, const_surface)
        start = _pick(start, cs, max)
        end = _pick(end, ce, min)
    if start is not None and end is not None and end > start:
        return (start, end)
    return None


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


def _resolve_clause(node: ast.Node,
                    const_surface: Surface) -> tuple[float | None, float | None]:
    """A single conjunct -> a (start, end) contribution. A resolvable
    `driver OP bound` gives one side; anything else (a state guard, a driver
    clause over a live bound, a nested form) contributes nothing."""
    if (isinstance(node, ast.Binary)
            and node.op in _LOWER_OPS | _UPPER_OPS):
        return _clause_bound(node.op, node.left, node.right, const_surface)
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


def _pick(a: float | None, b: float | None, combine):
    if a is None:
        return b
    if b is None:
        return a
    return combine(a, b)


def _call_name(node: ast.Call) -> str | None:
    return node.fn.name if isinstance(node.fn, ast.Sym) else None
