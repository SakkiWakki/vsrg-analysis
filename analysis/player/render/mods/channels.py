"""Mod-channel compilation: approach chasing resolved to linear segments.

# What a mod channel is

A StepMania / NotITG modfile drives per-note mods (`drunk`, `tornado`,
`movex`, ...) by pushing `ApplyModifiers` strings over time. Each string
carries a target percentage and an approach-speed prefix:

    "*5 40 drunk"   -> approach 40% drunk at speed 5
    "*-1 100 drunk" -> snap to 100% drunk instantly

The engine holds a *current* value per (mod, player) and, every frame,
moves it toward the *target* value. A channel is the resolved history of
one (mod, player): a function of time returning the current percentage.

# Approach semantics (verified against OpenITG source)

OpenITG PlayerState::Update calls PlayerOptions::Approach every frame
(src/PlayerState.cpp:12), which for each option runs (PlayerOptions.cpp:42)

    fapproach( current, target, fDeltaSeconds * speed )

and `fapproach` (RageUtil.cpp:51) is a constant-rate linear chase capped
so it never overshoots:

    delta   = target - current
    to_move = sign(delta) * (dt * speed)
    if |to_move| > |delta|: to_move = delta   # snap on arrival
    current += to_move

So `speed` is in units of *percent-fraction per second*: at speed S the
value closes the gap to its target at S per second (0.4 -> 1.0 at speed
5 takes (1.0 - 0.4) / 5 = 0.12 s). This is a straight line in time, which
is why the whole chase compiles to explicit linear segments here with no
per-frame simulation.

`speed = -1` is the Mirin/NotITG instant-snap convention: the Mirin
template computes the exact eased percent itself every frame and always
emits `*-1` so the engine snaps rather than double-easing
(template.lua:1127). OpenITG's raw `fapproach` would assert on a negative
`to_move`; we treat any `speed <= 0` as an instant snap, which matches
both the Mirin convention and the "*-1 = instant" community docs.

# Compilation model

Events for one (mod, player) are sorted by time. We walk them, tracking
the current value and the time it was last exact. Each event either:

  - snaps (speed <= 0): emit a zero-duration jump to the target.
  - chases (speed > 0): emit a ramp from the value *at this event's time*
    (found by sampling the segments built so far, so a re-target
    mid-approach starts from wherever the chase had reached) toward the
    target, reaching it after |target - from| / speed seconds, then
    holding until the next event.

The result is a list of (t, value) breakpoints per channel: piecewise
LINEAR, sampled by binary search. No frame is ever simulated.

# PORT BOUNDARY

Compilation is Python (runs once per chart). Sampling (`value` /
`values_at`) is the hot path: pure array/scalar math over precompiled
breakpoints, no engine or Qt imports, ready to lift into a native
evaluator. Times are seconds; callers convert beat-keyed events via a
supplied beat->time function so this module never touches timing data.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

DEFAULT_REST = 0.0


@dataclass(frozen=True)
class ModEvent:
    """One `(beat, value_percent, approach_speed, mod_name, player)` row.

    `value` is a fraction (40% -> 0.4). `speed` is the `*S` prefix:
    S > 0 chases at S/sec, S <= 0 snaps instantly (`*-1`)."""
    beat: float
    value: float
    speed: float
    mod: str
    player: int = 0


@dataclass
class _Segments:
    """Breakpoints of one channel: parallel `times`/`values`, sorted by
    time, sampled as a piecewise-linear curve holding the last value."""
    times: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def _sample(self, t: float) -> float:
        times = self.times
        if not times:
            return DEFAULT_REST
        idx = bisect_right(times, t) - 1
        if idx < 0:
            return self.values[0] if times[0] <= t else DEFAULT_REST
        if idx + 1 >= len(times):
            return self.values[idx]
        t0, t1 = times[idx], times[idx + 1]
        v0, v1 = self.values[idx], self.values[idx + 1]
        if t1 <= t0:
            return v1
        f = (t - t0) / (t1 - t0)
        return v0 + (v1 - v0) * f

    def _append(self, t: float, v: float) -> None:
        if self.times and self.times[-1] == t:
            self.values[-1] = v
            return
        self.times.append(t)
        self.values.append(v)

    def _append_step(self, t: float, v: float) -> None:
        """A vertical jump to `v` at `t`: a second point sharing the last
        point's time, which `_sample`'s `t1 <= t0` branch reads as the
        value at and after `t` while the prior point holds up to `t`."""
        self.times.append(t)
        self.values.append(v)


def _add_snap(seg: _Segments, t: float, target: float) -> None:
    """Jump to `target` at `t`, holding the current value up to `t` so a
    snap that follows an earlier breakpoint steps vertically instead of
    interpolating across the gap (the engine holds a snapped value until
    the next change)."""
    frm = seg._sample(t)
    seg._append(t, frm)
    if target != frm:
        seg._append_step(t, target)


def _add_chase(seg: _Segments, t: float, target: float, speed: float,
               until: float) -> None:
    """Emit the fapproach chase toward `target` starting at `t`, clamped
    to end at `until` (the next event's time).

    fapproach (PlayerOptions::Approach) moves the live value toward the
    current target by dt*speed every frame - a constant-rate ramp, then
    flat once it arrives. Its continuous form is a single line from the
    value at `t` to the value REACHED by `until`: the full `target` when
    the chase completes first (`arrival <= until`), else the partial
    value at `until` when a re-target interrupts it. Clamping to `until`
    is what keeps the breakpoints time-ordered - an unclamped ramp can
    arrive past the next event and make `times` non-monotonic, which
    breaks the bisect in `_sample`."""
    frm = seg._sample(t)
    seg._append(t, frm)
    gap = abs(target - frm)
    if gap == 0.0:
        return
    arrival = t + gap / speed
    if arrival <= until:
        seg._append(arrival, target)
    else:
        reached = frm + (target - frm) * (until - t) / (arrival - t)
        seg._append(until, reached)


def _compile_channel(events: list) -> _Segments:
    ordered = sorted(events, key=lambda e: e.beat)
    # The clamp bound for each event is the next distinct event time; the
    # last event chases unbounded (nothing re-targets it).
    next_times = [ordered[j].beat for j in range(1, len(ordered))]
    next_times.append(float('inf'))
    seg = _Segments()
    for ev, until in zip(ordered, next_times):
        if ev.speed <= 0.0:
            _add_snap(seg, ev.beat, ev.value)
        else:
            _add_chase(seg, ev.beat, ev.value, ev.speed, until)
    return seg


class ModChannels:
    """Compiled (mod, player) -> piecewise-linear value curve.

    Build with `ModChannels.compile(events, beat_to_time=...)`; the
    caller-supplied `beat_to_time` maps event beats to seconds (identity
    if events are already time-keyed). Query with `value(mod, t)` for one
    channel or `values_at(t)` for every active mod at once."""

    def __init__(self, channels: dict, players: tuple):
        self._channels = channels
        self._players = players

    @classmethod
    def compile(cls, events, beat_to_time: Callable[[float], float] | None = None):
        to_time = beat_to_time if beat_to_time is not None else (lambda b: b)
        grouped = defaultdict(list)
        players = set()
        for ev in events:
            players.add(ev.player)
            timed = ModEvent(to_time(ev.beat), ev.value, ev.speed, ev.mod, ev.player)
            grouped[(ev.mod, ev.player)].append(timed)
        channels = {key: _compile_channel(evs) for key, evs in grouped.items()}
        return cls(channels, tuple(sorted(players)))

    @property
    def players(self) -> tuple:
        return self._players

    def mods(self, player: int = 0) -> tuple:
        return tuple(sorted(mod for (mod, pn) in self._channels if pn == player))

    def value(self, mod: str, t: float, player: int = 0) -> float:
        """Current percentage of one (mod, player) at time `t` (seconds).
        Returns the rest value (0) for a mod that has no events."""
        seg = self._channels.get((mod, player))
        return DEFAULT_REST if seg is None else seg._sample(float(t))

    def values_at(self, t: float, player: int = 0) -> dict:
        """Every mod's current percentage for `player` at time `t`, as
        `{mod: value}`. Only mods with events appear; a consumer treats a
        missing mod as rest (0)."""
        t = float(t)
        return {mod: seg._sample(t)
                for (mod, pn), seg in self._channels.items() if pn == player}

    def values_over(self, ts, player: int = 0) -> dict:
        """Vectorized `values_at` over a time array: `{mod: ndarray}` with
        one sampled value per t in `ts`. Convenience for batch prepasses;
        each channel is sampled with numpy interpolation."""
        ts = np.asarray(ts, dtype=np.float64)
        out = {}
        for (mod, pn), seg in self._channels.items():
            if pn != player:
                continue
            out[mod] = _sample_array(seg, ts)
        return out


def _sample_array(seg: _Segments, ts: np.ndarray) -> np.ndarray:
    """Piecewise-linear sample of one channel over a time array. `np.interp`
    holds the endpoints (flat before the first / after the last point),
    matching `_Segments._sample`; the pre-first-point rest value is 0,
    which `np.interp`'s left-hold gives only when the first value is 0, so
    a nonzero opening value is masked back to rest explicitly."""
    if not seg.times:
        return np.zeros_like(ts)
    times = np.asarray(seg.times, dtype=np.float64)
    values = np.asarray(seg.values, dtype=np.float64)
    out = np.interp(ts, times, values)
    if values[0] != DEFAULT_REST:
        out = np.where(ts < times[0], DEFAULT_REST, out)
    return out
