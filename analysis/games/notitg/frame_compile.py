"""Integrator seam sketch: split an Update body into closed-form-baked keyframes and
evaluated frames, so the lupa tick loop runs ONLY over what cannot be
flattened.

This is the compile pre-pass the frame-IR design calls for, kept as a standalone
entry (NOT wired into `update_integrator.integrate_update`) until the router is
correct enough for a byte-parity production cutover. It exists so the split can
be measured and its reconstruction proven against the all-lupa oracle:

    plan = compile_update(body, surface, to_seconds)
    plan.closed_keyframes    # {(actor_global, prop): [Keyframe]} baked from
                          # each closed-form-routed poke's ClosedForm over its window
    plan.evaluated_windows  # merged beat windows the lupa loop must still
                          # sample (every non-flattened VarUpdate's frame)

The contract with the existing integrator (the byte-parity target): for each
recorded (actor, prop),

    {closed-form-baked keyframes} + {keyframes the lupa loop records over the evaluated
    windows} == {today's all-lupa keyframes}

The closed-form half is baked here with no lupa; the evaluated half is the SAME lupa
tick loop the integrator runs today, only bounded to `evaluated_windows` instead
of the whole body's live-window union. When `evaluated_windows` still covers
every live window (nothing flattened - today's poke-router state), this pre-pass
is a no-op and the integrator's output is unchanged: that is the safe floor the
production cutover starts from.

Bake grid: the closed form keyframes are sampled on the integrator's own tick grid
(`_TICK_HZ`) so a baked keyframe and a lupa-recorded keyframe share tick times
exactly - the two halves compose without re-alignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from analysis.games.notitg import update_integrator
from analysis.games.notitg.frame_ir import (
    build_frames, effective_window, iter_updates)
from analysis.games.notitg.flatteners import affine_kernel, route
from analysis.games.notitg.sim import verb_surface
from analysis.player.render.effects.timeline import Keyframe
from analysis.player.render.expr import ast
from analysis.player.render.expr.surface import UNRESOLVED


_TICK_STEP_S = 1.0 / update_integrator._TICK_HZ

# The time-driver symbols an closed-form curve may read: they vary correctly per tick
# through a clock reader. ANY other name (a local like `b = beat-1093.5`, a
# live global) must NOT resolve for flattening - if it did, the flattener
# would fold the curve to a stale snapshot of that name's current value and
# bake a frozen constant. Restricting to drivers routes such pokes to
# evaluation, where they evaluate live (the value-read hazard the parity
# harness caught on a `diffusealpha(f(b))` fade).
_DRIVER_SYMBOLS = frozenset({'beat', 'mod_time', 'time', 'curtime', 'measure'})


class _DriverOnlySurface:
    """Wraps a live surface but resolves ONLY the time drivers (symbol + clock
    reader); every other symbol/index is UNRESOLVED. A pure-curve poke thus
    flattens to a closed form only when its argument is a genuine function of time; a
    poke reading a local or non-driver global fails to compile and stays
    evaluation."""

    def __init__(self, inner):
        self._inner = inner

    def symbol(self, name):
        if name in _DRIVER_SYMBOLS:
            return self._inner.symbol(name)
        return UNRESOLVED

    def index(self, base, key):
        return UNRESOLVED

    def call(self, name, args):
        return self._inner.call(name, args)

    def clock_reader(self, name):
        if name in _DRIVER_SYMBOLS:
            return self._inner.clock_reader(name)
        return None

# setter method -> the single scalar property it records (a bulk setter writing
# a tuple of props is left to evaluation here - the closed form bake covers scalars).
_SCALAR_PROP = {name: prop for name, prop in verb_surface.SCALAR_SETTERS.items()
                if isinstance(prop, str)}

# Instant keyframe defaults: a baked closed-form sample is an untweened point (the
# recording actor emits the same shape for a per-tick poke).
_INSTANT_DURATION = 0.0
_LINEAR_EASE = 0


@dataclass
class UpdatePlan:
    """The closed-form/evaluated split of an Update body. `closed_keyframes` are baked with
    no lupa; `evaluated_windows` are the beat windows the lupa tick loop must
    still cover. `closed` / `evaluated` count the VarUpdates on each side."""
    closed_keyframes: dict = field(default_factory=dict)
    evaluated_windows: list = field(default_factory=list)
    closed: int = 0
    evaluated: int = 0


def compile_update(body: str, surface, to_seconds) -> UpdatePlan:
    """Route every VarUpdate in `body`; bake the closed-form-routed scalar pokes to
    keyframes over their effective window, and collect the windows of everything
    that stays evaluated. `surface` compiles coefficient channels (a live
    `NotitgGuardSurface` for beat-driven curves); `to_seconds` maps beats to the
    song-time tick grid."""
    frame_root = build_frames(body)
    updates = list(iter_updates(frame_root))
    sole = _sole_writer_props(updates)
    driver_surface = _DriverOnlySurface(surface)
    plan = UpdatePlan()
    attention_spans = []
    for update, frame in updates:
        window = effective_window(frame)
        closed_form = route(update, frame, surface=driver_surface)
        # A property flattens ONLY when it is the SOLE writer over its window.
        # Multiple pokes to one property compose last-writer-wins per tick,
        # which a standalone baked curve cannot reproduce (the parity harness
        # caught this as swapped/duplicate values on multi-written props).
        if (closed_form is not None and _bakeable(update, window)
                and update.name in sole):
            _bake_into(plan, update, closed_form, window, to_seconds)
            plan.closed += 1
        else:
            plan.evaluated += 1
            if window is not None:
                attention_spans.append(window)
    plan.evaluated_windows = _merge(attention_spans)
    return plan


def _sole_writer_props(updates) -> set:
    """Property names written by exactly one VarUpdate in the whole body. A
    property touched by two or more writes composes across ticks/windows
    (last-writer-wins) and cannot be a standalone closed-form curve, so it is excluded
    from flattening and stays on the evaluated path."""
    counts: dict = {}
    for update, _frame in updates:
        counts[update.name] = counts.get(update.name, 0) + 1
    return {name for name, n in counts.items() if n == 1}


def _bakeable(update, window) -> bool:
    """True when the update is a scalar actor poke with a bounded window - the
    only shape this sketch bakes. A bare frame-variable assignment (no actor
    prop) and an unbounded window are left to the evaluated/live-channel paths."""
    split = _split_actor_prop(update.name)
    return split is not None and window is not None


def _split_actor_prop(name: str):
    actor, _dot, method = name.rpartition('.')
    prop = _SCALAR_PROP.get(method)
    if not actor or prop is None:
        return None
    return actor, prop


def _bake_into(plan, update, closed_form, window, to_seconds) -> None:
    """Sample `closed_form` over `window` on the tick grid into instant
    keyframes, keyed by (actor global, property)."""
    actor, prop = _split_actor_prop(update.name)
    keyframes = _bake_keyframes(closed_form, window, to_seconds)
    plan.closed_keyframes.setdefault((actor, prop), []).extend(keyframes)


def _bake_keyframes(closed_form, window, to_seconds) -> list:
    """The closed-form value stream over `window` as instant Keyframes on the tick grid.
    A pure curve (a=0) samples its coefficient channel at each tick coordinate
    (the true closed form); an accumulator steps the recurrence per tick index.
    """
    ticks = _window_ticks(window, to_seconds)
    values = _value_stream(closed_form, ticks)
    return [Keyframe(t, (v,), _INSTANT_DURATION, _LINEAR_EASE)
            for t, v in zip(ticks, values)]


def _window_ticks(window, to_seconds) -> list:
    t = to_seconds(window[0])
    t_end = to_seconds(window[1])
    ticks = []
    while t <= t_end:
        ticks.append(t)
        t += _TICK_STEP_S
    return ticks


def _value_stream(closed_form, ticks) -> list:
    """The closed-form value at each tick. A pure curve (a=0) is `b` sampled at the tick
    coordinate (the true closed form, evaluable at any t); an accumulator (a!=0)
    is the frozen affine_kernel at the tick index n, which per the Stage A
    assumption samples its coefficients once at coord 0.0."""
    b_channel = closed_form.coeffs['b']
    if closed_form.coeffs['a'].at(0.0) == 0:
        return [b_channel.at(t) for t in ticks]
    return [affine_kernel(n, closed_form.h0, closed_form.coeffs)
            for n in range(len(ticks))]


def _merge(spans) -> list:
    """Sorted, merged (start, end) windows - the contiguous beat ranges the
    evaluation lupa loop must sample."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
