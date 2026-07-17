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

# Sweep rate OUTSIDE the perframe driver windows: the body's
# unconditional sections still run (positioning, housekeeping), just
# coarsely - the simplification pass absorbs the density difference.
_COARSE_HZ = 10.0

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


# Sim tail past the chart's last measure: end-of-song actions (score
# pokes, outros) fire just past the notes.
_END_TAIL_S = 4.0


@dataclass
class ChartDocument:
    """A chart's modfile document plus timing, loaded once and shared by
    the loop and the producers (all through the modfile module's generic
    FGCHANGES/XML/timing layers)."""
    root: object
    classic_commands: list
    to_seconds: object
    start_beat: float
    lua_dir: object
    rng_seed: int
    end_seconds: float


def load_chart(sm_path) -> ChartDocument | None:
    """None when the chart has no resolvable lua modfile."""
    from analysis.games.notitg import modfile

    entries = modfile.parse_fgchanges(sm_path)
    lua_dir = modfile._resolve_lua_dir(sm_path, entries)
    if lua_dir is None:
        return None
    sm_data = modfile.sm_chart.parse_sm(sm_path)
    bg_stem = modfile.Path(modfile._sm_background_name(sm_path)).stem.casefold()
    root, _chunks, classic = modfile._load_document(lua_dir, bg_stem)
    _bpms, _offset, chart = modfile._timing(sm_data)
    to_seconds = modfile._beat_to_seconds(sm_data, chart)
    start_beat = min((b for b, _n, k in entries if k == 'FGCHANGES'),
                     default=0.0)
    # SM notedata is comma-separated measures of 4 beats; the last
    # measure bounds everything the chart schedules in song time.
    measures = str(chart.get('notedata', '')).count(',') + 1
    end_seconds = to_seconds(4.0 * measures) + _END_TAIL_S
    return ChartDocument(root, classic, to_seconds, start_beat, lua_dir,
                         modfile._chart_rng_seed(lua_dir), end_seconds)


def run_chart_sim(sm_path, end_seconds: float,
                  tick_hz: float = _TICK_HZ) -> SimResult | None:
    doc = load_chart(sm_path)
    if doc is None:
        return None
    return run_sim(doc.root, doc.to_seconds, doc.start_beat, end_seconds,
                   rng_seed=doc.rng_seed, tick_hz=tick_hz)


def run_sim(root, to_seconds, start_beat, end_seconds,
            rng_seed: int = 0, tick_hz: float = _TICK_HZ) -> SimResult:
    """Load `root` and run the sim to `end_seconds`. Returns the recorded
    streams; the environment's actors carry the keyframes."""
    load_s = load_anchor_seconds(start_beat, to_seconds)
    to_beats = beat_inverter(to_seconds, end_seconds)
    env = SimEnvironment(load_s, rng_seed, to_seconds=to_seconds)
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


def run_declarative(root, to_seconds, start_beat, end_seconds,
                    rng_seed: int = 0, tick_hz: float = _TICK_HZ) -> SimResult:
    """The fast compile: load the actors, fire the scheduled
    `mod_actions`, then tick ONLY the bounded `perframe(a, b)` driver
    windows - not the whole song.

    The declarative bulk (mods/mods2 tables, tween commands) needs no
    simulation: `producers` reads those tables straight into events. Only
    the `UpdateCommand` per-frame drivers (walker/rotator reading other
    actors' curves) need time-stepping, and they run in bounded sections
    the template's `perframe(a, b)` gates declare. Ticking those windows
    only - a small fraction of a whole-song 60Hz sweep - recovers the
    driven-actor curves at a fraction of the cost."""
    from analysis.games.notitg import update_integrator

    load_s = load_anchor_seconds(start_beat, to_seconds)
    to_beats = beat_inverter(to_seconds, end_seconds)
    env = SimEnvironment(load_s, rng_seed, to_seconds=to_seconds)
    env.set_time(load_s, to_beats(load_s))
    warnings = env.load_actors(root)
    # ONE CLOCK: mod-actions are staged and fired at their true times
    # INSIDE the sweep below, so tween-queue state evolves
    # contemporaneously with the update body's reads (a driver sampling
    # a quad's GetX at a tick sees the value in force at that moment).
    # The rig's queue-carried Update re-arm is suppressed - the sweep
    # owns the update body; the queue-borne copy double-ran the drivers
    # at frozen drain clocks.
    env.prepare_mod_actions()
    env.suppressed_queued_commands = frozenset(
        {update_integrator._UPDATE_COMMAND})

    from analysis.games.notitg.xml_actors import _strip_lua_wrapper

    body = update_integrator._update_body(root)
    windows = update_integrator._live_windows(body) if body else ()
    if body:
        # The raw attr is `%function(self) ... end`; the runnable chunk
        # is the inner statements (`self` falls to the permissive stub,
        # so the rig's own re-arm tail no-ops - the window sweep drives
        # time instead).
        body = _strip_lua_wrapper(body)
    # Hybrid-rate sweep: the body runs full-rate inside the declared
    # `perframe(a, b)` driver windows, and at a coarse rate everywhere
    # else - the body's UNCONDITIONAL sections (player positioning, the
    # intro block, always-on pokes) run every engine frame in-game, and
    # skipping them outside windows loses their state entirely. Coarse
    # ticks suffice there: the unconditional pokes are smooth
    # positioning/housekeeping, and the collinear keyframe
    # simplification absorbs the sampling density either way.
    # Queue drains run at FULL rate over the whole song - the engine
    # drains every frame, and the intro's chained zero-tweens
    # (sleep(0) links) smear if drains lag. Cheap: the queued set holds
    # only actors with live queues. The BODY runs hybrid: full rate
    # inside perframe driver windows, coarse outside (its unconditional
    # positioning/housekeeping is smooth; the simplification absorbs
    # the density).
    window_spans = [(to_seconds(a), to_seconds(b)) for a, b in windows]
    step = 1.0 / float(tick_hz)
    coarse = 1.0 / _COARSE_HZ
    # The body runs at the RIG'S OWN cadence, not the sweep's: its
    # re-arm tail (`sleep(0.02); queuecommand('Update')`) is how often
    # the engine invokes it, and its per-call integrators (dt-less
    # Euler physics, per-call scroll adds) count invocations.
    body_step = (update_integrator._body_rearm_period(body) or step) \
        if body else step
    body_step = max(body_step, step)
    ticks = 0
    t = load_s
    next_body_t = load_s
    while t < end_seconds:
        t = min(t + step, end_seconds)
        env.fire_mod_actions_until(t)
        env.set_time(t, to_beats(t))
        env.drain(t)
        if body and t >= next_body_t:
            env.run_update_body(body)
            in_window = any(a <= t < b for a, b in window_spans)
            next_body_t = t + (body_step if in_window
                               else max(coarse, body_step))
        ticks += 1

    # Self-scheduling chains (a chara Idle loop re-queueing itself) are
    # event lines already: the tween queue is deterministic, so ONE final
    # drain to the end expands every remaining chain at queue-exact times
    # - no tick grid. Runs after the sweep so all timestamps stay
    # monotonic (sync only advances forward).
    env.fire_mod_actions_until(end_seconds)
    env.set_time(end_seconds, to_beats(end_seconds))
    env.drain(end_seconds, defer_queued=False)
    return SimResult(env=env, ticks=ticks, end_seconds=end_seconds,
                     load_seconds=load_s, warnings=warnings)
