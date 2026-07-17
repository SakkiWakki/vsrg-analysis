"""The engine loop: tick time, let the chart run itself.

`run_sim` loads an actor tree into a SimEnvironment and advances sim
time on a fixed tick grid to the chart's end. That is the WHOLE
per-frame story: the classic template's Update rig is a queuecommand
chain (`sleep(0.02); queuecommand('Update')`, gat default.xml:4695)
that re-arms itself on the real tween queue, so per-frame drivers,
`mod_actions` scheduling, and message dispatch all execute through the
chart's own Lua against engine time - no window extraction, no harvest
passes.

The chart is loaded slightly before beat 0 (the same anchor rule as the
harvest path): actors exist and their Init/On commands run before any
beat-0 action fires, and `GAMESTATE:GetSongBeat()` is already live and
negative pre-song, which the templates' own `>= 0` gates rely on.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from analysis.games.notitg.sim.env import SimEnvironment

_TICK_HZ = 60.0

# Anchor load a hair before beat 0 when the FGCHANGES start beat maps
# later, so load-time keyframes strictly precede beat-0 actions (same
# rule and rationale as the harvest path's _LOAD_LEAD_S).
_LOAD_LEAD_S = 0.01

# seconds -> beat inversion sampling density.
_INVERT_SAMPLES_PER_BEAT = 8.0


@dataclass
class SimResult:
    """Everything one sim run recorded, for the producers."""
    env: SimEnvironment
    ticks: int = 0
    end_seconds: float = 0.0
    load_seconds: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def actors(self) -> dict:
        return self.env.actors

    @property
    def applied_mods(self) -> list:
        return self.env.applied_mods

    @property
    def shader_flags(self) -> list:
        return self.env.shader_flags

    @property
    def faults(self) -> int:
        return self.env.faults


def load_anchor_seconds(start_beat, to_seconds) -> float:
    start_s = to_seconds(start_beat)
    beat0_s = to_seconds(0.0)
    return min(start_s, beat0_s - _LOAD_LEAD_S) if start_s > beat0_s \
        else start_s


def beat_inverter(to_seconds, end_seconds):
    """seconds -> beat over [load, end], by bisect over a densely sampled
    monotonic (seconds, beat) table - same approach the harvest
    integrator used, but spanning the whole chart."""
    beats = [0.0]
    seconds = [to_seconds(0.0)]
    step = 1.0 / _INVERT_SAMPLES_PER_BEAT
    beat = 0.0
    while seconds[-1] < end_seconds:
        beat += step
        beats.append(beat)
        seconds.append(to_seconds(beat))
        if beat > 1e6:
            break

    def to_beats(t: float) -> float:
        idx = bisect_right(seconds, float(t)) - 1
        if idx < 0:
            first = seconds[0]
            return beats[0] + (t - first) * _INVERT_SAMPLES_PER_BEAT * step
        if idx >= len(beats) - 1:
            return beats[-1]
        span = seconds[idx + 1] - seconds[idx]
        frac = (t - seconds[idx]) / span if span > 0 else 0.0
        return beats[idx] + frac * step

    return to_beats


def run_chart_sim(sm_path, end_seconds: float,
                  tick_hz: float = _TICK_HZ) -> SimResult | None:
    """Load a chart's modfile document (the modfile module's generic
    FGCHANGES/XML/timing layers) and run the sim. None when the chart
    has no resolvable lua modfile."""
    from analysis.games.notitg import modfile

    entries = modfile.parse_fgchanges(sm_path)
    lua_dir = modfile._resolve_lua_dir(sm_path, entries)
    if lua_dir is None:
        return None
    sm_data = modfile.sm_chart.parse_sm(sm_path)
    bg_stem = modfile.Path(modfile._sm_background_name(sm_path)).stem.casefold()
    root, _chunks, _classic = modfile._load_document(lua_dir, bg_stem)
    _bpms, _offset, chart = modfile._timing(sm_data)
    to_seconds = modfile._beat_to_seconds(sm_data, chart)
    start_beat = min((b for b, _n, k in entries if k == 'FGCHANGES'),
                     default=0.0)
    return run_sim(root, to_seconds, start_beat, end_seconds,
                   rng_seed=modfile._chart_rng_seed(lua_dir),
                   tick_hz=tick_hz)


def run_sim(root, to_seconds, start_beat, end_seconds,
            rng_seed: int = 0, tick_hz: float = _TICK_HZ) -> SimResult:
    """Load `root` and run the sim to `end_seconds`. Returns the recorded
    streams; the environment's actors carry the keyframes."""
    load_s = load_anchor_seconds(start_beat, to_seconds)
    to_beats = beat_inverter(to_seconds, end_seconds)
    env = SimEnvironment(load_s, rng_seed)
    env.set_time(load_s, to_beats(load_s))
    warnings = env.load_actors(root)

    step = 1.0 / float(tick_hz)
    t = load_s
    ticks = 0
    while t < end_seconds:
        t = min(t + step, end_seconds)
        env.set_time(t, to_beats(t))
        env.drain(t)
        ticks += 1
    return SimResult(env=env, ticks=ticks, end_seconds=end_seconds,
                     load_seconds=load_s, warnings=warnings)
