"""Classify per-frame Update sections: schedulable vs integrated.

The Update integrator polls - it re-runs the whole Lua body every tick. But
a `if perframe(a,b) then <poke driven by a curve> end` section is really a
SCHEDULING directive: over `[a,b)` bind the poked property to a curve. This
module reads a section's AST and decides whether it is such a pure-curve
section (schedulable, compiles to `PropBinding`s over its window) or has
stateful / unmodeled logic (integrated, falls back to the lupa tick loop).

Conservative by construction: a section leaves the lupa path ONLY when every
leaf statement provably compiles to the same pokes - any accumulator, any
unmodeled call, any non-literal loop bound, any nested condition that is not
a plain target gate forces the whole section to 'integrated'. So the ticked
output is unchanged; scheduling is a pure speedup on the sections that qualify.

Section body shapes handled (corpus: pokes/loops/nested-ifs dominate):
- actor-poke method `recv:prop(argexpr)` where `prop` is a known setter and
  `argexpr` compiles to a curve (a data-holder read, a closed-form beat expr,
  or a literal);
- `for v = lo, hi do <body> end` with LITERAL lo/hi -> unrolled per v;
- `if <targetexpr> then <body> end` -> a per-target gate (the body's pokes
  apply only when that target is present); the condition itself must be a
  bare target reference, not a state predicate;
- `local name = <compilable>` -> inlined into later arg expressions.
"""
from __future__ import annotations

from dataclasses import dataclass

from analysis.games.notitg.sim import verb_surface
from analysis.player.render.expr import ast

# Method names that set a per-frame property we can bind to a curve. Built
# from the verb surface's setter tables (scalar / bulk / add), so the
# classifier keys on the engine mechanism, never on chart-specific names.
_SETTER_NAMES = frozenset(verb_surface.SCALAR_SETTERS) \
    | frozenset(verb_surface.BULK_SETTERS) \
    | frozenset(verb_surface.ADD_SETTERS)


@dataclass(frozen=True)
class PropBinding:
    """A scheduled poke: over the section window, `target`'s `prop` follows
    the value of `arg` (an AST expression compiled to a curve downstream).
    `loop_vars` records the unrolled `(name, value)` bindings in scope so the
    arg expression resolves a `Proxy(pn)`-style target and any loop-indexed
    read. `gates` are the target expressions that must be present (the
    `if P1 then` folds)."""
    target: ast.Node        # the receiver expression (a Sym, or Call like Proxy(pn))
    prop: str               # the setter name (rotationz, diffusealpha, ...)
    arg: ast.Node           # the poked value expression
    loop_vars: tuple[tuple[str, float], ...] = ()
    gates: tuple[ast.Node, ...] = ()


@dataclass(frozen=True)
class SectionPlan:
    """One guarded section, classified. `kind` is 'scheduled' (compiles to
    `bindings` over `window`) or 'integrated' (stays on the lupa tick loop).
    `reason` names why an integrated section did not schedule (diagnostics)."""
    window: tuple[float, float] | None
    kind: str
    bindings: tuple[PropBinding, ...] = ()
    reason: str = ''


def classify_section(body: tuple[ast.Node, ...],
                     window: tuple[float, float] | None) -> SectionPlan:
    """Classify a section's statement body. A window-bounded section whose
    every leaf compiles to a PropBinding is 'scheduled'; anything else is
    'integrated' with a reason."""
    if window is None:
        return SectionPlan(None, 'integrated', reason='no window (flag guard)')
    bindings: list[PropBinding] = []
    reason = _collect_bindings(body, bindings, loop_vars=(), gates=())
    if reason:
        return SectionPlan(window, 'integrated', reason=reason)
    if not bindings:
        return SectionPlan(window, 'integrated', reason='no poke bindings')
    return SectionPlan(window, 'scheduled', bindings=tuple(bindings))


def _collect_bindings(stmts, out: list, loop_vars, gates) -> str:
    """Append a PropBinding per leaf poke; return a non-empty reason string
    the moment a statement is not schedulable (the section then integrates)."""
    locals_map = dict(loop_vars)
    for stmt in stmts:
        reason = _collect_one(stmt, out, tuple(locals_map.items()), gates)
        if reason:
            return reason
    return ''


def _collect_one(stmt, out: list, loop_vars, gates) -> str:
    match stmt:
        case ast.ExprStmt(expr=ast.Method(recv=recv, name=name, args=args)) \
                if name in _SETTER_NAMES and len(args) == 1:
            out.append(PropBinding(recv, name, args[0], loop_vars, gates))
            return ''
        case ast.NumericFor(var=var, start=start, stop=stop, step=step,
                            body=body):
            return _unroll_for(var, start, stop, step, body, out, loop_vars,
                               gates)
        case ast.If(cond=cond, body=body, elifs=(), orelse=()) \
                if _is_target_gate(cond):
            return _collect_bindings(body, out, loop_vars, gates + (cond,))
        case ast.Local():
            # A local binding we do not track yet forces integration only if a
            # later poke reads it; conservatively, allow a local that is never
            # a poke target (it is inlined by value at compile downstream).
            return ''
        case ast.ExprStmt(expr=ast.Method()):
            return 'poke to a non-setter or multi-arg method'
        case ast.ExprStmt(expr=ast.Call()):
            return 'bare call in section (possible side effect)'
        case ast.Assign():
            return 'assignment in section (state mutation)'
        case _:
            return f'unschedulable statement: {type(stmt).__name__}'


def _unroll_for(var, start, stop, step, body, out, loop_vars, gates) -> str:
    lo = _literal(start)
    hi = _literal(stop)
    stride = _literal(step) if step is not None else 1.0
    if lo is None or hi is None or stride is None or stride == 0:
        return 'loop with non-literal bounds'
    if (hi - lo) / stride > _MAX_UNROLL:
        return 'loop too large to unroll'
    value = lo
    while (stride > 0 and value <= hi) or (stride < 0 and value >= hi):
        reason = _collect_bindings(body, out, loop_vars + ((var, value),),
                                   gates)
        if reason:
            return reason
        value += stride
    return ''


def _is_target_gate(cond) -> bool:
    """True when a condition is a bare target reference (`if P1 then`,
    `if a then`) - a presence gate, not a state predicate. A comparison or
    logical op is a predicate and forces integration."""
    return isinstance(cond, (ast.Sym, ast.Index, ast.Call))


def _literal(node) -> float | None:
    match node:
        case ast.Num(value=v):
            return v
        case ast.Unary(op='-', operand=ast.Num(value=v)):
            return -v
    return None


# Cap so a pathological literal loop cannot explode the binding list; real
# per-player loops are 1..8.
_MAX_UNROLL = 64
