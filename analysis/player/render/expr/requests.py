"""The compiler->backend request vocabulary (COMPILER_CONTRACT.md).

A script body compiles DOWN into these requests - the language- and game-neutral
seam a future Rust core emits into. Each request is a small typed record; the
backend (the scheduler timeline) consumes them. This module is the CONTRACT made
concrete: `EMIT_CURVE` / `EMIT_EVENT` / `EMIT_SAMPLED` become `CurveRequest` /
`EventRequest` / `SampledRequest`.

A curve/sampled request carries a `Channel` (scheduler.py) - curve + clock
together - so "the value of `prop` on `target` over time" is one object the
renderer evaluates live (`channel.at(t_seconds)`), never pre-sampled. The clock
is where rate-mods / stops / warps live (a curve authored in beat coordinate,
evaluated through the beat-integral clock, gets them for free - see
`analysis/player/sv/DESIGN.tex`); the curve itself does not change.

Two paths produce a `CurveRequest`:
- ANALYTIC (this module's `compile_curves`, measured ~56% of setter pokes): the
  arg is a pure function of clock+math+consts, lowered ONCE to a closed-form
  Channel. No ticking.
- RESIDUE (`EMIT_SAMPLED`, the ~44%): the arg reads live state, so the core
  ticks it and emits the sampled piecewise Channel. Same Channel shape, sampled
  rather than closed-form.
"""
from __future__ import annotations

from dataclasses import dataclass

from analysis.games.notitg.frame_ir import build_frames, iter_updates
from analysis.games.notitg.flatteners import route
from analysis.player.render.effects.timeline import EventTimeline
from analysis.player.render.scheduler import Channel, timeline_channel


@dataclass(frozen=True)
class CurveRequest:
    """`EMIT_CURVE`: a property that evolves continuously as a function of a
    clock. `target` is the actor the poke names (its global/label, `p` in
    `p:x(...)`); `prop` is the timeline property (`x`, `rotationz`); `channel`
    is the curve + its clock. `channel.at(t_seconds)` is the value at song time
    t. `analytic` records whether it came from the closed-form path (True) or
    the residue sampler (False) - diagnostics only; the backend treats both the
    same."""
    target: str
    prop: str
    channel: Channel
    analytic: bool = True


@dataclass(frozen=True)
class EventRequest:
    """`EMIT_EVENT`: a discrete thing at a point on a clock (a mod window
    opening, a message dispatch, a tween start). `clock_value` is the position
    on `clock` (a beat, a song-time); `payload` is the backend-opaque event
    (a mod string, a command name). Feeds `EventSchedule`."""
    clock: object
    clock_value: float
    payload: object


@dataclass(frozen=True)
class SampledRequest:
    """`EMIT_SAMPLED`: the residue path - a property whose per-tick value reads
    live state the compiler cannot characterize analytically. The core ticks it
    and emits the resulting piecewise Channel (still clock-named, just sampled).
    Same fields as `CurveRequest`; a distinct type marks the provenance."""
    target: str
    prop: str
    channel: Channel


def compile_curves(body: str, surface) -> tuple[list[CurveRequest], list]:
    """Lower a per-frame `body`'s ANALYTIC setter pokes to `CurveRequest`s.

    Returns `(curves, residue)`: `curves` are the closed-form pokes (arg is a
    pure function of the driver clock + math + consts), each a live `Channel`;
    `residue` is the list of `(VarUpdate, Frame)` the analytic path could NOT
    characterize (they read live state) - the caller routes those to the sampler
    (`EMIT_SAMPLED`). This is the analytic/residue SPLIT the contract's
    classification names, done structurally: a poke whose arg lowers is
    analytic, else residue - never a guess.

    `surface` resolves the driver clock symbols to their readers (a
    `_DriverOnlySurface` wrapping the live surface: only beat/mod_time/... and
    math/consts resolve, so a poke reading an actor's live `GetX` fails to lower
    and falls to residue). The emitted Channel's curve is authored in song time
    with the driver clock applied at the leaves (the `compile_guard` form), so
    `channel.at(t_seconds)` is the poke's value at song time t."""
    root = build_frames(body)
    updates = list(iter_updates(root))
    sole = _sole_writer_targets(updates)
    curves: list[CurveRequest] = []
    residue: list = []
    for update, frame in updates:
        split = _split_actor_prop(update.name)
        closed = route(update, frame, surface=surface) if split else None
        # A (target, prop) written by more than one poke composes last-writer-
        # wins per tick, which a standalone curve cannot reproduce - so only a
        # SOLE-writer analytic poke becomes a curve; the rest stay residue (the
        # sampler ticks the composition). Mirrors frame_compile's _sole_writer.
        if split is not None and split in sole and _is_pure_curve(closed):
            curves.append(CurveRequest(split[0], split[1], closed.coeffs['b']))
        else:
            residue.append((update, frame))
    return curves, residue


def _sole_writer_targets(updates) -> set:
    """The `(target, prop)` pairs written by exactly ONE poke across the body. A
    pair touched by two+ pokes composes last-writer-wins per tick and cannot be
    a standalone analytic curve, so it is excluded from the curve path."""
    counts: dict = {}
    for update, _frame in updates:
        split = _split_actor_prop(update.name)
        if split is not None:
            counts[split] = counts.get(split, 0) + 1
    return {pair for pair, n in counts.items() if n == 1}


def compile_sampled(recorded: dict, residue_props: set,
                    clock) -> list[SampledRequest]:
    """Package the residue tick loop's RECORDED streams as `SampledRequest`s.

    The residue path (the ~44%: pokes reading live state, accumulators, bulk
    setters) is RUN over the tick grid by the core - here, the sim's interpreter
    tick loop - which records a sparse keyframe stream per (target, prop) (an
    instant only where the value changes, NOT a dense per-tick grid, so this is
    NOT the baked-keyframe explosion of [[project_events_not_keyframes]]). This
    wraps each such stream as one live Channel over `clock` (`EventTimeline`
    sampled live), so a residue poke emits the SAME request shape as an analytic
    one - the timeline holds Channels either way.

    `recorded` is `{target: {prop: [Keyframe]}}` from the tick run;
    `residue_props` is the `{(target, prop)}` set the analytic split routed to
    residue (so a (target, prop) the analytic path already emitted as a
    closed-form curve is NOT re-emitted here - analytic wins, it is exact).
    `clock` is the timeline Clock the sampled Channels are named over."""
    out: list[SampledRequest] = []
    for target, props in recorded.items():
        for prop, keyframes in props.items():
            if (target, prop) not in residue_props:
                continue
            channel = timeline_channel(EventTimeline(keyframes,
                                                     (_rest(prop),)), clock)
            out.append(SampledRequest(target, prop, channel))
    return out


def _rest(prop: str) -> float:
    """The property's resting value, returned by the Channel before its first
    keyframe (the recorder's REST for that prop)."""
    from analysis.games.notitg.lua_api import _REST

    value = _REST.get(prop)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _is_pure_curve(closed) -> bool:
    """A closed form is a PURE CURVE (a=0, no self-reference) exactly when its
    `a` coefficient is the constant 0 - then value_n = b(clock), the analytic
    curve. An accumulator (a != 0) is a recurrence, not a clock-only curve, so
    it stays residue here (the sampler ticks it)."""
    if closed is None:
        return False
    a_channel = closed.coeffs.get('a')
    return a_channel is not None and a_channel.at(0.0) == 0


def _split_actor_prop(name: str) -> tuple[str, str] | None:
    """`actor.prop` (a scalar setter poke's VarUpdate name) -> (actor, prop),
    or None for a bare frame-variable assignment (no timeline target)."""
    from analysis.games.notitg.sim import verb_surface

    actor, _dot, method = name.rpartition('.')
    prop = verb_surface.SCALAR_SETTERS.get(method)
    if not actor or not isinstance(prop, str):
        return None
    return actor, prop
