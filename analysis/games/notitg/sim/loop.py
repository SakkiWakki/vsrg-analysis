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
                    rng_seed: int = 0, tick_hz: float = _TICK_HZ,
                    song_dir=None, use_compiled_body: bool = False,
                    use_native_body: bool = False) -> SimResult:
    """The fast compile: load the actors, fire the scheduled
    `mod_actions`, and tick the `UpdateCommand` per-frame body at its own
    re-arm cadence across the song.

    The declarative bulk (mods/mods2 tables, tween commands) needs no
    simulation: `producers` reads those tables straight into events. Only
    the `UpdateCommand` per-frame drivers (walker/rotator reading other
    actors' curves, mpf/mod_perframe mod painters) need time-stepping.
    The body runs EVERY re-arm and gates ITSELF - its `perframe(a,b)` /
    `mod_perframes` reader decides what runs each tick - so no external
    window scan is needed; a driver a name-based scan would miss still
    ticks at full rate. Cheap: the body is ~6k invocations over a 2-min
    chart at its 50Hz re-arm, and drains only touch actors with live
    queues."""
    from analysis.games.notitg import update_integrator

    load_s = load_anchor_seconds(start_beat, to_seconds)
    to_beats = beat_inverter(to_seconds, end_seconds)
    env = SimEnvironment(load_s, rng_seed, to_seconds=to_seconds,
                         song_dir=song_dir)
    env.set_time(load_s, to_beats(load_s))
    warnings = env.load_actors(root)
    # Opt-in: drive the per-frame Update body through the AST interpreter
    # (frame_eval) instead of Lua. Set AFTER load (the load pass still runs as
    # Lua); only the per-tick body sweep below is affected.
    env.use_compiled_body = use_compiled_body
    env.use_native_body = use_native_body
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

    body, body_name, update_actor = update_integrator._update_source(root)
    update_rec = getattr(update_actor, '_sim_id', None)
    if body:
        # The raw attr is `%function(self) ... end`; the runnable chunk
        # is the inner statements (`self` falls to the permissive stub,
        # so the rig's own re-arm tail no-ops - the sweep drives time
        # instead).
        body = _strip_lua_wrapper(body)
    # Queue drains run at FULL rate over the whole song - the engine
    # drains every frame, and the intro's chained zero-tweens (sleep(0)
    # links) smear if drains lag. Cheap: the queued set holds only actors
    # with live queues. The BODY runs every re-arm (its own gates decide
    # what fires each tick - see below).
    step = 1.0 / float(tick_hz)
    # The body runs at the RIG'S OWN cadence, the whole song through - its
    # re-arm tail (`sleep(0.02); queuecommand('Update')`) is how often the
    # engine invokes it, and its per-call integrators (dt-less Euler
    # physics, per-call scroll adds) count invocations. It runs EVERY
    # re-arm, not only inside regex-detected `perframe(a, b)` windows: the
    # body's OWN reader (`mod_perframes` / `perframe(a,b)` gates) decides
    # what fires each tick, so an mpf/mod_perframe driver a name-based
    # window scan misses still ticks at full rate and its per-frame math
    # (a sin(beat) alternate/invert) stays smooth instead of degrading to
    # the old ~10Hz coarse rate. Cost is negligible - ~6k body runs over a
    # 2-min chart at 50Hz.
    body_step = (update_integrator._body_rearm_period(body) or step) \
        if body else step
    body_step = max(body_step, step)
    ticks = 0
    t = load_s
    next_body_t = load_s
    while t < end_seconds:
        # The body fires at its EXACT re-arm times, merged into the
        # drain grid, never rounded up to the next tick: sleep(0.02) on
        # a 60Hz grid would quantize to 33ms (30Hz), and every
        # instant-approach driver mod would visibly step below the
        # rig's real 50Hz.
        target = min(t + step, end_seconds)
        run_body = body is not None and next_body_t <= target
        if run_body:
            target = max(next_body_t, t)
        t = target
        env.fire_mod_actions_until(t)
        env.set_time(t, to_beats(t))
        env.drain(t)
        if run_body:
            env.run_update_body(body, name=body_name, rec_id=update_rec)
            next_body_t = t + body_step
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


class LiveSim:
    """A `SimEnvironment` advanced INCREMENTALLY, for lazy replay: `compile`
    loads the chart and stores this (near-instant, no baking); playback calls
    `advance_to(t)` each frame to tick the sim forward to the playhead, then
    reads actor current values.

    INVARIANT (critical): the tick GRID is identical to `run_declarative`'s -
    ticks land only on `load_s + k*step` and exact body re-arm times, NEVER on
    the caller's arbitrary `t`. `advance_to(t)` runs whole grid ticks until the
    next one would pass `t`, so stepping in tiny per-frame increments produces
    the EXACT same tick sequence as one continuous run (a fractional drain would
    leave a queued command half-fired / spawn a dynamic actor out of order).
    A backward jump rebuilds and re-sims forward (sim is ~90x realtime).
    """

    def __init__(self, root, to_seconds, start_beat, end_seconds,
                 rng_seed=0, tick_hz=_TICK_HZ, song_dir=None,
                 use_compiled_body=False):
        # `load_actors` MUTATES the tree it loads (tags nodes with _sim_id /
        # _recorder_id, expands includes, removes Condition-falsy children). The
        # element-tree compiler keys its LiveCurves off those SAME tags on this
        # exact `root`, so the FIRST build must load `root` in place (shared ids).
        # A reset (backward seek) rebuilds a fresh env; re-loading an already-
        # tagged node would skip table creation (_id_for returns the stale id
        # without _new_actor) -> KeyError on _tables[rec_id]. So reset loads a
        # fresh deep copy instead (~4ms), leaving the compiler's `root` intact.
        import copy

        self._root_source = root
        # A PRISTINE snapshot taken before the first build tags `root` in place.
        # Every rebuild loads a fresh copy of this so no stale _sim_id survives.
        self._pristine_root = copy.deepcopy(root)
        self._built_once = False
        self._to_seconds = to_seconds
        self._start_beat = start_beat
        self._end_seconds = end_seconds
        self._rng_seed = rng_seed
        self._tick_hz = float(tick_hz)
        self._song_dir = song_dir
        self._use_compiled_body = use_compiled_body
        self.warnings = []
        self._build()

    def _build(self):
        import copy

        from analysis.games.notitg import update_integrator
        from analysis.games.notitg.xml_actors import _strip_lua_wrapper

        # First build tags `_root_source` in place (the compiler keys its
        # LiveCurves off those tags); every rebuild loads a fresh copy of the
        # pristine snapshot so load's mutation never leaks across a reset.
        self._root = self._root_source if not self._built_once \
            else copy.deepcopy(self._pristine_root)
        self._built_once = True
        self._load_s = load_anchor_seconds(self._start_beat, self._to_seconds)
        self._to_beats = beat_inverter(self._to_seconds, self._end_seconds)
        env = SimEnvironment(self._load_s, self._rng_seed,
                             to_seconds=self._to_seconds,
                             song_dir=self._song_dir)
        env.set_time(self._load_s, self._to_beats(self._load_s))
        self.warnings = env.load_actors(self._root)
        env.use_compiled_body = self._use_compiled_body
        env.prepare_mod_actions()
        env.suppressed_queued_commands = frozenset(
            {update_integrator._UPDATE_COMMAND})

        body, body_name, update_actor = update_integrator._update_source(
            self._root)
        self._update_rec = getattr(update_actor, '_sim_id', None)
        self._body = _strip_lua_wrapper(body) if body else None
        self._body_name = body_name
        self._step = 1.0 / self._tick_hz
        body_step = (update_integrator._body_rearm_period(self._body)
                     or self._step) if self._body else self._step
        self._body_step = max(body_step, self._step)

        self.env = env
        self._t = self._load_s
        self._next_body_t = self._load_s
        # Cleared on every (re)build so a post-reset advance to the same `t`
        # re-simulates (the fast path must never short-circuit a fresh env).
        self._last_advance_t = None

    def advance_to(self, target_t):
        """Tick the sim forward to the largest grid point <= `target_t`. Grid
        ticks are never truncated to `target_t` (the INVARIANT above), so `_t`
        may sit up to one step behind the playhead - the engine's own frame
        quantization, and what keeps stepping identical to a continuous run."""
        # Same-playhead fast path: the renderer samples ~1400 LiveCurves per
        # frame, each calling advance_to with the IDENTICAL frame `t`. Only the
        # first does work; the rest are no-ops. A cached last-target check skips
        # the end-clamp + backward-seek test + loop guard on the other ~1399,
        # the single biggest cheap per-frame win (advance_to went from the hot
        # path to a dict-compare). Any different `t` falls through to the real
        # advance below.
        if target_t == self._last_advance_t:
            return
        self._last_advance_t = target_t
        # The sim loads at `load_s` (gat's is 0.25s - the chart's first FGCHANGE),
        # while the playback clock starts at 0. A playhead BELOW load_s just means
        # "chart not started yet" - the sim sits at load_s, not a backward seek.
        # Clamp the target up to load_s so early playback never trips the reset.
        end = min(max(target_t, self._load_s), self._end_seconds)
        # A genuine BACKWARD jump (scrub to an already-past point) rebuilds and
        # re-sims forward. Compared against load_s-clamped `end`, so pre-start
        # frames do not reset.
        if end < self._t - self._step:
            self.reset()
            end = min(max(target_t, self._load_s), self._end_seconds)
        # Advance whole ticks. `run_declarative`'s loop is: target = min(t+step,
        # END); run_body if next_body_t <= target; snap target to next_body_t.
        # Here END is the far chart end (grid never depends on the call), and we
        # stop once the next tick's target would exceed the playhead.
        while self._t < end:
            target = min(self._t + self._step, self._end_seconds)
            run_body = self._body is not None and self._next_body_t <= target
            if run_body:
                target = max(self._next_body_t, self._t)
            # Stop BEFORE any tick (body or plain) that would pass the playhead -
            # `_t` stays at the last grid point <= playhead. Without this a
            # body-firing tick ran past `end`, overshooting the whole chart on
            # the first call (and tripping the backward-seek reset next call).
            if target > end:
                break
            self._t = target
            self.env.fire_mod_actions_until(self._t)
            self.env.set_time(self._t, self._to_beats(self._t))
            self.env.drain(self._t)
            if run_body:
                self.env.run_update_body(self._body, name=self._body_name,
                                         rec_id=self._update_rec)
                self._next_body_t = self._t + self._body_step

    def reset(self):
        self._build()

    @property
    def now(self):
        return self._t
