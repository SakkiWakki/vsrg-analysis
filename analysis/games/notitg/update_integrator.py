"""Compile-time per-frame Update integrator for NotITG modfiles.

The classic template drives its most stateful effects from a single
self-scheduling per-frame command:

    UpdateCommand="%function(self)
        ...
        if perframe(128,252) then gat_updateproxies() end   -- 3x3 grid
        if perframe(1175,1236) then ... Proxy(pn):rotationz(..) end
        ...
        self:sleep(0.02); self:queuecommand('Update')       -- ~50Hz loop
    end"

Everything inside runs EVERY frame while the song plays. Some blocks are
pure functions of already-compiled data-holder quads (`gat_scroller`,
`gat_aftzoom`), but many ACCUMULATE state across frames - the flagship is
`gat_updateproxies()`, which does `gat_allproxies:addx(...)` with a
wrap-around each frame and reads that running total back to place a 3x3
grid of notefield proxies. The one-shot mod_actions replay never runs
these, so their outputs are missing or frozen (scoping items 25, 30, 35,
36, 45).

This module runs the RECORDED Update body itself on a fixed tick grid
against the same stubbed host and RecordingActor mirrors the load pass
built, harvesting each tick's actor pokes into dense keyframe streams
(the shape aft_drivers.py produced by hand, generalized to the whole
Update body):

- TICK GRID: `_TICK_HZ` samples per second in SONG TIME, deterministic,
  over the union of the body's live `perframe(a,b)` windows (so a chart
  with no per-frame drivers costs nothing and we never integrate dead
  time). Ticks are song-time driven; `GetSongBeat()` returns the tick
  beat so `perframe`/`beat` branches evaluate correctly.

- STATE MIRRORING: each RecordingActor enters `begin_sampling` so a
  driver reading a source quad (`gat_scroller:GetX()`) gets its value AT
  THE TICK'S TIME, while an actor the pass itself pokes
  (`gat_allproxies`) becomes a live accumulator reading its running
  total. Lua globals (`gat_frame`, `gat_screen_zoom`, the `gat_proxies`
  tables) persist across ticks in the host env, so frame-count parity and
  accumulators carry exactly as the engine's per-frame loop would.

- ISOLATION: the body also holds the `mods`/`mods2` window reader and the
  `mod_actions` curaction loop; those are compiled by other passes, so we
  empty the three tables for the duration of the integration and restore
  them after. Only the per-frame drivers' actor pokes and their direct
  `ApplyGameCommand` mods (the walking `movey` family) are harvested.

- TERMINATION: bounded by the perframe-window union and a hard tick cap;
  the body never blocks (its `sleep`/`queuecommand` self-schedule is a
  no-op under the recording stub).

The integrator MUTATES the recorders in place - a proxy grid frame that
had no keyframes gains a dense per-tick stream - so the existing
`field_copies` / element / mod-event producers pick the new streams up
with no change to their shape.
"""
from __future__ import annotations

import re

# Song-time tick rate. ~60Hz tracks the source quad eases finely and, per
# the template's own `self:sleep(0.02)` (50Hz) loop, over-samples the real
# frame cadence so wrap-around accumulators land smoothly.
_TICK_HZ = 60.0
_TICK_STEP_S = 1.0 / _TICK_HZ

# Hard cap so a pathological window union cannot spin the compile. gat's
# live windows span ~beats 0-1500 (~7 min); 60Hz over that is ~26k ticks.
_MAX_TICKS = 60000

# The named command the per-frame loop lives under (`UpdateCommand`), as
# exposed by Actor.named_commands (the `Command` suffix stripped).
_UPDATE_COMMAND = 'Update'

# Live `perframe(a, b)` and `perframe(a)` windows in the body bound the
# tick range. Comment-stripped first so `--[[ perframe(..) ]]` blocks and
# `-- if perframe(..)` lines do not widen the range with dead drivers.
_PERFRAME_RE = re.compile(
    r'perframe\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*(?:,\s*([0-9]+(?:\.[0-9]+)?))?')

# Tables the body reads that other passes own; emptied during integration
# so the window reader and action loop inside Update no-op.
_ISOLATED_TABLES = ('mods', 'mods2', 'mod_actions')


def integrate_update(env, root, to_seconds) -> dict:
    """Run the chart's per-frame Update body on a tick grid, mutating the
    recorders with each tick's pokes. Returns a summary dict
    {'ran', 'ticks', 'windows', 'applied', 'applied_events'}; 'ran' is
    False when the chart has no live Update drivers (nothing to
    integrate). Never raises - a chart whose Update body faults on an
    unmodeled call degrades to whatever it poked before the fault, per
    tick."""
    body = _update_body(root)
    if body is None:
        return _NO_RUN

    windows = _live_windows(body)
    if not windows:
        return _NO_RUN

    to_beats = _beat_inverter(windows, to_seconds)
    return env.run_update_integration(body, windows, to_seconds, to_beats,
                                      _TICK_STEP_S, _MAX_TICKS)


_NO_RUN = {'ran': False, 'ticks': 0, 'windows': 0, 'applied': 0,
           'applied_events': [], 'faults': 0}


def _beat_inverter(windows, to_seconds):
    """A `seconds -> beat` inverter over the window span, built by bisect
    on a dense (beat, seconds) table. `to_seconds` is monotonic in beat
    (chart time only moves forward, warps aside), so a bisect on the
    sampled times inverts it; linear interpolation between samples keeps
    the mapping smooth for tick times between grid beats."""
    from bisect import bisect_right

    lo = min(w[0] for w in windows)
    hi = max(w[1] for w in windows)
    steps = max(2, int((hi - lo) * _INVERT_SAMPLES_PER_BEAT) + 1)
    beats = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    times = [to_seconds(b) for b in beats]

    def to_beats(t):
        idx = bisect_right(times, t) - 1
        if idx < 0:
            return beats[0]
        if idx >= steps - 1:
            return beats[-1]
        span = times[idx + 1] - times[idx]
        frac = (t - times[idx]) / span if span > 0 else 0.0
        return beats[idx] + (beats[idx + 1] - beats[idx]) * frac
    return to_beats


# Inversion table density: samples per beat. gat is ~205 bpm, so a beat is
# ~0.29s; 8 samples/beat puts inversion grid points ~35ms apart, finer
# than the 1/60s tick, so the interpolated beat is accurate.
_INVERT_SAMPLES_PER_BEAT = 8.0


def _update_body(root):
    """The runnable Lua of the first actor's `UpdateCommand`, or None. The
    classic template has exactly one such per-frame loop; the first found
    is authoritative."""
    for actor in _iter_actors(root):
        body = actor.named_commands().get(_UPDATE_COMMAND)
        if isinstance(body, str) and body.startswith('%'):
            return body
    return None


_REARM_RE = re.compile(
    r"sleep\s*\(\s*([0-9.]+)\s*\)\s*;?\s*self\s*:\s*queuecommand\s*\(\s*"
    r"['\"]" + _UPDATE_COMMAND + r"['\"]", re.IGNORECASE)


def _body_rearm_period(body: str) -> float | None:
    """Seconds between update-body invocations, from the rig's own
    re-arm tail (`self:sleep(X); self:queuecommand('Update')`), or None
    when the body never self-schedules. The engine runs the body at
    THIS cadence, and per-call integrators in it (a toss rig's
    `addx(xspd); yspd = yspd + fall` Euler steps, a walker's per-call
    scroll add) carry no dt - running them at the sweep's tick rate
    instead integrates visibly fast (60/50 = 20% at the template's
    0.02s re-arm)."""
    match = _REARM_RE.search(_strip_comments(body))
    if not match:
        return None
    period = float(match.group(1))
    return period if period > 0.0 else None


def _live_windows(body: str):
    """Sorted, merged (start_beat, end_beat) windows for every live
    `perframe(a, b)` in the body. `perframe(a)` (no end) is a one-beat
    window [a, a+1], matching the helper's `endBeat = beat+1` default.
    Overlapping/adjacent windows merge so the tick grid is contiguous
    across a run of drivers."""
    spans = []
    for match in _PERFRAME_RE.finditer(_strip_comments(body)):
        start = float(match.group(1))
        end = float(match.group(2)) if match.group(2) else start + 1.0
        if end > start:
            spans.append((start, end))
    return _merge_spans(sorted(spans))


def _merge_spans(spans):
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _strip_comments(body: str) -> str:
    """Body with Lua comments removed so a commented-out perframe driver
    (`--[[ if perframe(..) ]]`, `-- if perframe(..)`) does not widen the
    tick range. Block comments first (they may span lines), then line
    comments."""
    without_blocks = re.sub(r'--\[\[.*?\]\]', '', body, flags=re.DOTALL)
    return re.sub(r'--[^\n]*', '', without_blocks)


def _iter_actors(actor):
    yield actor
    for child in actor.children:
        yield from _iter_actors(child)
