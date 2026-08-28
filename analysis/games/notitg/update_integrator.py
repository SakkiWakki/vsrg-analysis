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

from analysis.games.notitg import guard_windows
from analysis.player.render.expr.surface import UNRESOLVED, ConstSurface

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
    body, _label, _actor = _update_source(root)
    return body


def _update_source(root):
    """(raw `%`-Lua UpdateCommand body, fault/chunk label, actor) of the
    first actor bearing one, or (None, None, None). The classic template
    has exactly one such per-frame loop; the first found is
    authoritative. The label carries the actor's source XML file and
    Name so a Lua error in the body names its origin."""
    for actor in _iter_actors(root):
        body = actor.named_commands().get(_UPDATE_COMMAND)
        if isinstance(body, str) and body.startswith('%'):
            name = actor.attrs.get('Name') or actor.kind
            src = getattr(actor, '_src_xml', '')
            prefix = f'{src}:{name}' if src else name
            return body, f'{prefix}.{_UPDATE_COMMAND}', actor
    return None, None, None


def _body_rearm_period(body: str, env=None) -> float | None:
    """Seconds between update-body invocations, from the rig's own
    re-arm tail (`self:sleep(X); self:queuecommand('Update')`), or None
    when the body never self-schedules. The engine runs the body at
    THIS cadence, and per-call integrators in it (a toss rig's
    `addx(xspd); yspd = yspd + fall` Euler steps, a walker's per-call
    scroll add) carry no dt - running them at the sweep's tick rate
    instead integrates visibly fast (60/50 = 20% at the template's
    0.02s re-arm).

    `env` (a loaded SimEnvironment) resolves a period authored as an
    expression over chart globals - gat 2's `self:sleep(1 / gf2_fps)`
    with `gf2_fps = 50` set at load. Without it only literal periods
    resolve."""
    surface = _EnvGlobalsSurface(env) if env is not None else None
    return guard_windows.rearm_period(body, _UPDATE_COMMAND, surface)


class _EnvGlobalsSurface(ConstSurface):
    """Constant resolution against a LOADED env's Lua globals, for the
    re-arm walk: the chart computes its own cadence at load time, so the
    number exists only in the env, not in the body text. Non-numeric
    globals stay UNRESOLVED - this answers arithmetic, not tables."""

    def __init__(self, env):
        super().__init__()
        self._read = env._host.global_value

    def symbol(self, name: str):
        value = self._read(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        return UNRESOLVED


def _live_windows(body: str):
    """Sorted, merged (start_beat, end_beat) windows for every live per-frame
    driver in the body - a `perframe(a, b)` call or a `beat`/`mod_time` range
    guard - via the Lua AST front-end (`guard_windows.windows_from_body`),
    which puts each guard in DNF so nested/disjoint ranges each get a window
    and resolves table/arithmetic bounds. Overlapping/adjacent windows merge
    so the tick grid is contiguous across a run of drivers."""
    return guard_windows.windows_from_body(body)


def _iter_actors(actor):
    yield actor
    for child in actor.children:
        yield from _iter_actors(child)
