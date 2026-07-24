"""SimActor: one SM actor, simulated and recorded at once.

The engine-loop compiler (DESIGN_engine_loop.md) runs chart Lua against
actors that behave like the engine's while emitting the compact
storyboard shapes the element compiler already consumes. SimActor is
that actor. It models Actor's REAL tween queue (openitg Actor.cpp), so
scheduling, reads, and command timing come out engine-exact instead of
approximated:

- `BeginTweening(time, type)` appends a full state snapshot copied from
  the queue tail (Actor.cpp:609); setters write the tail's state
  (`SetX` -> `DestTweenState().pos.x`, Actor.h:113). With an empty
  queue the write is immediate.
- `queuecommand`/`queuemessage` are zero-length tweens carrying a
  command name (Actor.cpp:1074-1086); `sleep(t)` is a t-tween plus a
  zero-tween (Actor.cpp:1068). The tween queue IS the scheduler: a
  queued command fires when the drain reaches it, not "next frame".
- Update drains the queue (Actor::UpdateTweening, Actor.cpp:469): when
  the head tween begins, the current state is snapshotted as the ease
  start and the head's command (if any) fires; the head then eases
  current toward its state over its duration.
- Reads split exactly like the engine: `GetX` returns the in-flight
  interpolated current value (m_current, Actor.h:107) while `addx` adds
  onto the DESTINATION (`AddX(x) = SetX(GetDestX()+x)`, Actor.h:117).
- `stoptweening` clears the queue leaving current mid-flight
  (Actor.cpp:652); `finishtweening` assigns the FINAL queued state and
  clears, never firing queued commands (Actor.cpp:657).
- `hidden` is the immediate visibility bit (`SetHidden`, Actor.h:311),
  bypassing the queue; it records onto its own `hidden` channel exactly
  as before (an alpha crossfade can ride the queue underneath it).

Recording stays compact and storyboard-shaped: when a tween BEGINS, each
property whose destination differs from the ease start emits one
`Keyframe(begin_t, dest, duration, easing)` - precisely EventTimeline's
"ease from previous target over [t, t+duration]" contract - and
immediate writes emit instant keyframes. `stoptweening` pins the frozen
mid-flight value with a zero-duration keyframe at the current time,
which wins the sampler's bisect from that instant on, so the recorded
timeline replays the abandonment exactly.

SM tween types map to easing ids verbatim (Actor.cpp:522-526):
accelerate/decelerate are quad in/out; bouncebegin/bounceend/spring get
their own SM curves (effects.easing negative ids). The old recorder
silently dropped those three verbs.

Effect oscillators (vibrate/wag/bob/bounce/spin + params) stay recorded
as analytic `OscSpan`s, as in recording_actor - continuous
self-animation compiles to a sampled curve, not keyframes. The engine
applies them to the draw-time temp state, NOT to m_current
(Actor.cpp:248-365), so `GetX` correctly excludes them here too.
"""
from __future__ import annotations

import re
from bisect import bisect_right

from analysis.games.notitg.lua_api import (
    _ADD_SETTERS, _FALLBACK_TWEEN_EASING, _REST, _SCALAR_GETTERS,
    _SCALAR_SETTERS, _SIZE_AXIS_SETTERS, _SIZE_PAIR_SETTERS, _TWEEN_EASING,
    AftTexture, _as_float, _as_int)
from analysis.games.notitg.recording_actor import KIND_DEFAULTS
from analysis.player.render.segment_timeline import SegmentTimeline
from analysis.games.notitg.sim import verb_surface
from analysis.player.render import transform3d
from analysis.player.render.expr.eval_tree import eval_number
from analysis.player.render.effects.easing import (
    EASE_SM_BOUNCE_BEGIN, EASE_SM_BOUNCE_END, EASE_SM_SPRING, ease)
from analysis.player.render.effects.timeline import (Keyframe,
                                                     simplify_instants)

# The full SM tween-verb surface. lua_api's table carries the shared
# five; the SM-only curves live here until cutover folds them back.
_SIM_TWEEN_EASING = {
    **_TWEEN_EASING,
    'bouncebegin': EASE_SM_BOUNCE_BEGIN,
    'bounceend': EASE_SM_BOUNCE_END,
    'spring': EASE_SM_SPRING,
}

# Verbs whose lack of a poke dispatch is a DOCUMENTED decision (no
# visual effect / capability not yet built, each with a reason in
# verb_surface). A verb outside this set that falls through every
# dispatch is a silent drop and gets reported via `dropped_notify`.
_UNMODELED_OK = frozenset(verb_surface.IGNORED) | frozenset(
    verb_surface.DEFERRED)

# Actor.cpp:616 guards a runaway queue ("infinitely recursing
# ActorCommand?") by finishing all tweens once the queue passes 50.
_TWEEN_OVERFLOW = 50

# Safety cap on one update's drain iterations: a command that queues
# another zero-length command each time it fires would otherwise spin
# forever inside a single update (the engine's overflow guard above only
# fires on queue DEPTH, not on drain length).
_MAX_DRAIN_STEPS = 10000

_DEFAULT_EFFECT_CLOCK = 'bgm'

# Per-corner diffuse (SetDiffuseUpperLeft etc., openitg Actor.h:190-197).
# The four corner channels the storyboard element carries; each verb
# writes the RageColor(r,g,b,a) to one or two of them (an edge verb sets
# the two corners on that edge). Corner order is UL/UR/LL/LR.
_DIFFUSE_CORNER_VERBS = {
    'diffuseupperleft': ('color_ul',),
    'diffuseupperright': ('color_ur',),
    'diffuselowerleft': ('color_ll',),
    'diffuselowerright': ('color_lr',),
    'diffuseleftedge': ('color_ul', 'color_ll'),
    'diffuserightedge': ('color_ur', 'color_lr'),
    'diffusetopedge': ('color_ul', 'color_ur'),
    'diffusebottomedge': ('color_ll', 'color_lr'),
}

# Edge-fade setters (SetFadeLeft etc., Actor.h:178-181): one 0..1 scalar
# per edge onto its fade channel.
_FADE_VERBS = {
    'fadeleft': ('fade_left',), 'faderight': ('fade_right',),
    'fadetop': ('fade_top',), 'fadebottom': ('fade_bottom',),
    'fadeh': ('fade_left', 'fade_right'),
    'fadev': ('fade_top', 'fade_bottom'),
    'fade': ('fade_left', 'fade_right', 'fade_top', 'fade_bottom'),
}

# Actor natural (unzoomed) pixel size, openitg Actor.cpp:82 - `m_size` is
# born (1, 1) and a Sprite overwrites it with its texture dimensions.
# SetWidth/SetHeight (Actor.h:128-129) override it directly; GetWidth/
# GetHeight (GetUnzoomedWidth/Height, Actor.h:124-125) read it back. The
# sim never loads pixels, so a plain sprite's true natural size is only
# resolvable at render time - this default is the engine's starting m_size
# and the basis SetWidth/SetHeight, GetWidth/GetHeight, and a `(1,1)`-natural
# fit operate against.
_DEFAULT_NATURAL_SIZE = (1.0, 1.0)

# ScaleToCover/ScaleToFitInside modes (Actor.h:226 StretchType). Recorded
# onto the `fit_mode` channel so the renderer - which knows the true natural
# size - picks the uniform zoom (larger ratio covers, smaller fits inside,
# Actor.cpp:690-698). Rest 0 = no fit, so an actor never fit draws through
# the natural*scale path unchanged.
_FIT_COVER = 1.0
_FIT_INSIDE = 2.0

# Custom shader-uniform upload verbs (`GetShader():uniform1f(name, v)`).
# All carry the GLSL uniform name first and its value(s) after; only the
# first scalar component is recorded onto `uniform:<name>` (the
# chart-shader bridge drives scalar strengths). The `*fv` array and
# integer forms are accepted for coverage, reduced to their first value.
_UNIFORM_VERBS = frozenset({
    'uniform1f', 'uniform2f', 'uniform3f', 'uniform4f',
    'uniform1i', 'uniform2i', 'uniform3i', 'uniform4i',
    'uniform1fv', 'uniform2fv', 'uniform3fv', 'uniform4fv'})

# The COMPLETE effect-kind verb surface the NotITG fork registers on
# every actor. Position/rotation kinds synthesize downstream today; the
# color/zoom families (rainbow/diffuse*/glow*/pulse*) record spans for
# a future color-oscillator synthesis.
_EFFECT_KINDS = frozenset({
    'vibrate', 'wag', 'floorwag', 'bob', 'bounce', 'spin',
    'pulse', 'pulseramp', 'rainbow', 'diffuseshift', 'diffuseblink',
    'diffuseramp', 'glowshift', 'glowblink', 'glowramp'})
_EFFECT_PARAM_VERBS = frozenset({
    'effectmagnitude', 'effectperiod', 'effectoffset', 'effectclock',
    'effectdelay', 'effecttiming', 'effectcolor1', 'effectcolor2'})
# Kinds that animate even with no effectmagnitude poke (spin integrates
# a default; the color/zoom families draw from effectcolor/period).
_SELF_EVIDENT_KINDS = frozenset({
    'spin', 'pulse', 'pulseramp', 'rainbow', 'diffuseshift',
    'diffuseblink', 'diffuseramp', 'glowshift', 'glowblink', 'glowramp'})

# SM Actor defaults (Actor.cpp:55,59).
_DEFAULT_EFFECT_PERIOD = 1.0
_DEFAULT_EFFECT_MAGNITUDE = (0.0, 0.0, 10.0)
_DEFAULT_EFFECT_OFFSET = 0.0

# Fork transform-order defaults (Actor::BeginDraw @ 004a4320). The stock
# rotation order is 'xyz' (RageMatrixRotationXYZ), and the dest quaternion
# rests at identity so a never-touched actor composes exactly as before.
_DEFAULT_ROTATION_ORDER = 'xyz'
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
# Spherical single-axis adds -> the quat axis they spin about (RageQuatFromH
# = heading/y, RageQuatFromP = pitch/x, RageQuatFromR = roll/z,
# RageMath.cpp:311-341).
_SPHERICAL_AXIS = {'heading': 'y', 'pitch': 'x', 'roll': 'z'}

# Gap under which two driven-poke times merge into one span (seconds);
# per-frame ticks are 1/60 apart, real gaps between sections are long.
_DRIVEN_SPAN_GAP = 0.5

# The sim dispatches off verb_surface's generated tables (the full actor
# verb surface), which are supersets of lua_api's harvest-path tables: the
# scalar table adds zbias / basezoomz / skewy / the per-axis rotation
# setters, and the bulk / bulk-add / per-axis rotation-add families are
# new. lua_api's tables stay untouched so the harvest path is unchanged
# (parallel-build rule).
_SIM_SCALAR_SETTERS = verb_surface.SCALAR_SETTERS
_SIM_ADD_SETTERS = verb_surface.ADD_SETTERS
_SIM_BULK_SETTERS = verb_surface.BULK_SETTERS
_SIM_BULK_ADD_SETTERS = verb_surface.BULK_ADD_SETTERS
_SIM_CROP_COMPOSITES = verb_surface.CROP_COMPOSITES


# Tuple-valued rests for the color-gradient / glow channels, which
# lua_api._REST (scalar-only) does not carry. A corner rests at the UNSET
# sentinel (any component < 0 = "use the flat diffuse"); glow rests at
# alpha 0 = no glow pass (Actor.cpp:1008). Kept in sync with the storyboard
# model's _COLOR_RESTS - the recording and the render must agree on rest.
_COLOR_UNSET = (-1.0, -1.0, -1.0, -1.0)
_COLOR_RESTS = {
    'color_ul': _COLOR_UNSET, 'color_ur': _COLOR_UNSET,
    'color_ll': _COLOR_UNSET, 'color_lr': _COLOR_UNSET,
    'glow': (1.0, 1.0, 1.0, 0.0),
}


def _rest(prop):
    if prop in _COLOR_RESTS:
        return _COLOR_RESTS[prop]
    return _REST.get(prop, 0.0)


# NotITG evaluates arithmetic in classic command args: charts write beat

def _arg_float(value, default=None):
    """`_as_float` plus classic-arg arithmetic (`60/205`,
    `128*(60/205)`). The harvest path's coercion stays untouched."""
    result = _as_float(value)
    if result is None and isinstance(value, str):
        result = eval_number(value)
    return result if result is not None else default


class OscSpan:
    """One effect-oscillator interval: kind, period/offset/clock, and the
    magnitude as (t, x, y, z) samples (many when a per-frame driver ramps
    it). `end` is None while open. Same shape recording_actor records, so
    the oscillator keyframe synthesis consumes either."""

    __slots__ = ('kind', 'start', 'end', 'period', 'offset', 'clock',
                 'magnitude_samples', 'last_clock', '_clock_index', 'extra',
                 'explicit_end')

    def __init__(self, kind, start, period, offset, clock):
        self.kind = kind
        self.start = float(start)
        self.end = None
        self.period = float(period)
        self.offset = float(offset)
        self.clock = clock
        self.magnitude_samples: list = []
        self._clock_index: list = []
        self.last_clock = float(start)
        # Fork params with no dedicated slot (effecttiming, the color
        # families' effectcolor1/2) - recorded for future synthesis.
        self.extra: dict = {}
        # True when the chart itself stopped the effect (stopeffect or a
        # replacing kind verb); False = still running when recording
        # ended, so the synthesis extends it to the compile end.
        self.explicit_end = False

    def touch(self, clock) -> None:
        self.last_clock = max(self.last_clock, float(clock))

    def set_magnitude(self, clock, vec) -> None:
        self.magnitude_samples.append((float(clock), vec[0], vec[1], vec[2]))
        self.touch(clock)

    def magnitude_at(self, clock):
        samples = self.magnitude_samples
        if not samples:
            return _DEFAULT_EFFECT_MAGNITUDE
        idx = bisect_right(self._sample_clocks(), float(clock)) - 1
        chosen = samples[max(0, idx)]
        return chosen[1], chosen[2], chosen[3]

    def _sample_clocks(self) -> list:
        if len(self._clock_index) != len(self.magnitude_samples):
            self._clock_index = [s[0] for s in self.magnitude_samples]
        return self._clock_index

    def copy(self):
        out = OscSpan(self.kind, self.start, self.period, self.offset,
                      self.clock)
        out.last_clock = self.last_clock
        out.magnitude_samples = list(self.magnitude_samples)
        out.extra = dict(self.extra)
        out.explicit_end = self.explicit_end
        return out


class _Tween:
    """One queue entry: duration/easing, the full destination state
    snapshot, and the optional command name a zero-tween carries
    ('!'-prefixed = broadcast, Actor.cpp:1082)."""

    __slots__ = ('dur', 'ease', 'left', 'state', 'command', 'started')

    def __init__(self, dur, ease_id, state):
        self.dur = max(0.0, float(dur))
        self.ease = ease_id
        self.left = self.dur
        self.state = state
        self.command: str | None = None
        self.started = False


class SimActor:
    """One simulated-and-recorded actor. The loop owns time: it creates
    the actor at its load time and calls `update_to(t, run_command)` each
    tick; command bodies poke the actor in between through `poke`."""

    def __init__(self, now: float = 0.0):
        self._now = float(now)
        self._created = float(now)
        # Actor-level effect clock (SetEffectClockString semantics,
        # Actor.cpp:720): 'music' = song seconds, 'beat'/'bgm' = song
        # beat, 'timer' = per-actor wrapped seconds. Charts set it with
        # no effect running to abuse GetSecsIntoEffect as a clock (gat's
        # mod_time rig). `beat_fn` is wired by the environment.
        self._effect_clock = 'timer'
        self.beat_fn = None
        self._current: dict = {}
        # Props that entered through the SETTER path (_write_dest) - by
        # construction exactly SM's TweenState fields, since immediates
        # (hidden, sprite state, text, vanish) write through
        # _set_immediate instead. Tween snapshots copy ONLY these: the
        # write path is the tweenable/immediate classification, so a new
        # property is immediate-safe by default and never replays a
        # stale value over a later immediate write (Actor.h:56
        # TweenState vs plain actor members).
        self._tweenable: set = set()
        self._tweens: list = []
        self._ease_start: dict = {}
        self._head_begin_t = 0.0
        self._frames: dict = {}
        # The seekable read substrate: every emission mirrors into
        # per-lane SegmentTimelines (numeric) or a step token channel
        # (text / rotation_order), so readers evaluate value-at-any-t
        # without the sim being AT t. prop -> [SegmentTimeline per
        # tuple component] / prop -> ([t], [values]).
        self._seg: dict = {}
        self._seg_tokens: dict = {}
        # Whole-chart declarative lanes from the schedule lowering
        # (schedule_lower.lower_actions); readers fall to these beyond
        # the sweep frontier.
        self._seg_preview: dict = {}
        # Natural (unzoomed) size, m_size (openitg Actor.cpp:82). Only
        # SetWidth/SetHeight move it in the sim (a real sprite's texture
        # size is a render-time fact); GetWidth/GetHeight and the fit verbs
        # read it. Kept off the keyframe channels - it is actor state, not
        # a per-frame draw value, so recording it never perturbs output.
        self._natural = list(_DEFAULT_NATURAL_SIZE)
        self._aft_source: str | None = None
        self._aft_texture_name: str | None = None
        self._osc_spans: list = []
        self._osc_open: OscSpan | None = None
        self._driven = False
        self._driven_spans: list = []
        self._in_update = False
        # Fork transform-order state (Actor::BeginDraw @ 004a4320): the
        # Euler rotation order (SetRotationOrder) and whether each skew axis
        # applies BEFORE the rotation (skewx/y_before_rotation). These are
        # discrete modes, not tween state, so they write as immediate
        # keyframes on their own channels and rest at the engine default
        # ('xyz' order, skew-after). The spherical adds (heading/pitch/roll)
        # accumulate onto a dest quaternion (RageQuatMultiply, Actor.cpp:894)
        # recorded as an immediate 4-tuple resting at identity.
        self._rotation_order = _DEFAULT_ROTATION_ORDER
        self._quat = _IDENTITY_QUAT
        # keyframes() memo, invalidated on emit: the harvest surfaces
        # (named/actor/player keyframes, copies) each re-read every
        # actor, and re-simplifying per read is quadratic in practice.
        self._kf_cache: dict | None = None
        # Set by the environment: called when the queue goes non-empty,
        # so the drain loop can track ONLY actors with live queues
        # instead of scanning everyone per tick.
        self.queue_notify = None
        # Set by the environment: called with a verb that fell through
        # every poke dispatch without a documented IGNORED/DEFERRED
        # entry - either unmapped, or mapped but never routed (the
        # Actor:cmd class of bug). Silence here cost whole sessions.
        self.dropped_notify = None
        # Like dropped_notify, for DEFERRED verbs: documented gaps the
        # chart actually exercises (the coverage report's raw feed).
        self.deferred_notify = None

    @property
    def now(self) -> float:
        return self._now

    # -- time ------------------------------------------------------------

    def update_to(self, t: float, run_command=None,
                  defer_queued: bool = True) -> None:
        """Advance the tween queue to sim time `t` (Actor::UpdateTweening,
        Actor.cpp:469). `run_command(name)` plays a queue-carried command
        (or broadcasts, for '!name') at the exact moment its zero-tween
        begins; commands may poke this actor, appending to the live
        queue. With `defer_queued` (the engine's one-queue-pass-per-frame
        shape) a tween appended DURING this update waits for the next
        one, bounding every self-requeue chain to the tick rate; pass
        False to expand chains to quiescence in one call (the loop's
        final drain). A zero-dt call returns without beginning anything,
        matching the engine's early-out."""
        remaining = float(t) - self._now
        if remaining <= 0.0 or self._in_update:
            return
        self._in_update = True
        entry_tweens = len(self._tweens)
        try:
            for step_count in range(_MAX_DRAIN_STEPS):
                if not self._tweens or remaining <= 0.0:
                    break
                if defer_queued and step_count >= entry_tweens:
                    break
                head = self._tweens[0]
                if not head.started:
                    self._begin_head(head, run_command)
                    if not self._tweens or self._tweens[0] is not head:
                        # The carried command rebuilt the queue
                        # (stoptweening from inside a queued command);
                        # the stale head is gone - work on the new one.
                        continue
                step = min(head.left, remaining)
                head.left -= step
                remaining -= step
                self._now += step
                if head.left <= 0.0:
                    self._complete_head(head)
            self._now += max(0.0, remaining)
        finally:
            self._in_update = False

    def _begin_head(self, head, run_command) -> None:
        """Head tween begins: snapshot the ease start (m_start =
        m_current, Actor.cpp:490), emit the keyframes this tween will
        realize, then fire its carried command (Actor.cpp:493-499)."""
        head.started = True
        self._head_begin_t = self._now
        self._ease_start = dict(self._current)
        for prop, dest in head.state.items():
            start = self._ease_start.get(prop, _rest(prop))
            if dest != start:
                self._emit_at(self._now, prop, dest, head.dur, head.ease,
                              start=start)
        if head.command and run_command is not None:
            run_command(head.command)

    def _complete_head(self, head) -> None:
        # Merge, not replace: tween states snowball every tween-managed
        # property (each copies its predecessor), but the immediate
        # fields (hidden, vanish, frame) live outside SM's TweenState
        # and must survive a completion.
        self._current.update(head.state)
        if self._tweens and self._tweens[0] is head:
            self._tweens.pop(0)

    # -- reads -----------------------------------------------------------

    def _tween_progress(self, head, at_t):
        """The head tween's eased progress in [0, 1]. `at_t` is the CONTINUOUS
        query time: the scheduler principle - `advance_to` establishes WHICH
        tween is live (discrete), then the value is a closed-form ease at the
        real `at_t`, exactly like SV's cumulative_at(raw_t). `head.left` only
        tracks the last GRID tick, so reading progress from it quantizes the
        motion to 60Hz (the frame-lag residual); `(at_t - _head_begin_t)/dur`
        is continuous. `at_t=None` keeps the tick-quantized read for engine-
        internal callers that have no sub-frame query time."""
        if at_t is None:
            frac = 1.0 - head.left / head.dur
        else:
            frac = (float(at_t) - self._head_begin_t) / head.dur
            frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
        return ease(head.ease, frac)

    def get(self, prop: str, at_t=None):
        """The engine-current value: mid-flight interpolation when the
        head tween is running (GetX -> m_current, Actor.h:107), else the
        settled value, else rest. `at_t` (continuous query time) evaluates a
        running tween at the exact sub-frame time (see `_tween_progress`)."""
        head = self._tweens[0] if self._tweens else None
        if head is not None and head.started and head.dur > 0.0:
            dest = head.state.get(prop, _rest(prop))
            start = self._ease_start.get(prop, _rest(prop))
            if dest != start:
                return self._lerp(start, dest,
                                  self._tween_progress(head, at_t))
        value = self._current.get(prop)
        return value if value is not None else _rest(prop)

    def current(self, prop: str, at_t=None):
        """The current value of ANY recorded channel, for a live reader (lazy
        replay's LiveCurve). Same as `get` for tweened/immediate props, but also
        exposes the transform-ORDER state (`rotation_order` token, `quat` tuple)
        that the eager recorder emits as keyframes but which live outside
        `_current` (`_rotation_order`/`_quat`). Returns None when the channel has
        no value yet, so the caller falls to that channel's rest. `at_t` is the
        continuous query time - a running tween is evaluated at the exact sub-
        frame `at_t`, not the last grid tick (the frame-lag residual)."""
        if prop == 'rotation_order':
            return self._rotation_order
        if prop == 'quat':
            return self._quat
        head = self._tweens[0] if self._tweens else None
        if head is not None and head.started and head.dur > 0.0:
            dest = head.state.get(prop)
            start = self._ease_start.get(prop, _rest(prop))
            if dest is not None and dest != start:
                return self._lerp(start, dest,
                                  self._tween_progress(head, at_t))
        return self._current.get(prop)

    def get_dest(self, prop: str):
        """The destination value: the queue tail's state (GetDestX ->
        DestTweenState, Actor.h:110), which `add*` verbs stack onto."""
        if self._tweens:
            return self._tweens[-1].state.get(prop, _rest(prop))
        value = self._current.get(prop)
        return value if value is not None else _rest(prop)

    @staticmethod
    def _lerp(a, b, f):
        if isinstance(a, tuple) or isinstance(b, tuple):
            a = a if isinstance(a, tuple) else (a,) * len(b)
            b = b if isinstance(b, tuple) else (b,) * len(a)
            return tuple(x + (y - x) * f for x, y in zip(a, b))
        return a + (b - a) * f

    def read(self, verb: str):
        """Value for a getter call, or None when `verb` is not a getter
        we model (the Lua bridge then falls back to the poke path)."""
        match verb:
            case v if v in _SCALAR_GETTERS:
                return self.get(_SCALAR_GETTERS[v])
            case 'GetTexture' if self._aft_texture_name is not None:
                return AftTexture(f'aft:{self._aft_texture_name}')
            case 'GetSecsIntoEffect':
                return self._secs_into_effect()
            case 'GetEffectMagnitude' if self._osc_open is not None:
                return self._osc_open.magnitude_at(self._now)
            case 'GetText':
                return str(self._current.get('text', ''))
            case 'getaux':
                return self._current.get('aux', 0.0)
            case 'GetWidth':
                return self._natural[0]
            case 'GetHeight':
                return self._natural[1]
            case 'GetTweenTimeLeft':
                return sum(t.left for t in self._tweens)
            case 'GetRotationOrder':
                return self._rotation_order
            case 'GetSkewXBeforeRotation':
                return self._current.get('skew_x_before', 0.0)
            case 'GetSkewYBeforeRotation':
                return self._current.get('skew_y_before', 0.0)
            case 'getdiffuse':
                return self._get_diffuse()
            case 'getrotation':
                return self.getrotation()
            case 'getcurrentrotation':
                return (self.get('rotation_x'), self.get('rotation_y'),
                        self.get('rotation'))
            case _:
                return None

    def _get_diffuse(self) -> tuple:
        """GetDiffuse -> DestTweenState().diffuse[0] (Actor.h:198): the
        upper-left corner when set individually, else the flat diffuse
        color with its alpha, as an (r, g, b, a) tuple."""
        ul = self.get_dest('color_ul')
        if isinstance(ul, tuple) and ul[0] >= 0.0:
            return ul
        color = self.get_dest('color')
        rgb = color if isinstance(color, tuple) else (1.0, 1.0, 1.0)
        return (rgb[0], rgb[1], rgb[2], self.get_dest('alpha'))

    def _secs_into_effect(self) -> float:
        """m_fSecsIntoEffect by effect clock (Actor.cpp:559-590): the
        BGM clocks TRACK song time/beat outright, so charts read this
        with no effect running as a clock; the timer clock accumulates
        per-actor and wraps at period + delay."""
        match self._effect_clock:
            case 'music':
                return self._now
            case 'beat' | 'bgm':
                return self.beat_fn() if self.beat_fn is not None else self._now
            case _:
                period = (self._osc_open.period if self._osc_open is not None
                          else _DEFAULT_EFFECT_PERIOD)
                elapsed = self._now - self._created
                return elapsed % period if period > 0 else elapsed

    def getrotation(self):
        """(rx, ry, rz) from the DEST state (Actor.h:523 GetRotationX/Y/Z
        read m_baseRotation dest; getcurrentrotation reads current)."""
        return (self.get_dest('rotation_x'), self.get_dest('rotation_y'),
                self.get_dest('rotation'))

    # -- recording surface ----------------------------------------------

    def keyframes(self) -> dict:
        """property -> list[Keyframe], only for properties actually
        poked, with runs of collinear instant points collapsed by
        `timeline.simplify_instants` (which lives NEXT TO EventTimeline
        so the collapse can never drift from the playback semantics it
        must reproduce). A per-frame driver pokes an instant setter
        every tick, so a property that holds constant or ramps linearly
        records hundreds of redundant points; the collapse is
        behavior-preserving and cuts the compiled size by orders of
        magnitude (the events-not-keyframes model). Tweened points (their
        own duration/easing) are structural and never dropped."""
        if self._kf_cache is None:
            self._kf_cache = {prop: simplify_instants(kfs)
                              for prop, kfs in self._frames.items() if kfs}
        return self._kf_cache

    def oscillator_spans(self) -> tuple:
        spans = list(self._osc_spans)
        if self._osc_open is not None:
            span = self._osc_open.copy()
            span.end = span.last_clock
            if span.magnitude_samples or span.kind in _SELF_EVIDENT_KINDS:
                spans.append(span)
        return tuple(spans)

    def set_driven(self, driven: bool) -> None:
        """Mark subsequent pokes as per-frame-driven (the loop sets this
        around UpdateCommand tick dispatch). Driven pokes accumulate the
        visibility spans the producers gate driven visuals to."""
        self._driven = bool(driven)

    def driven_spans(self) -> tuple:
        return tuple((s, e) for s, e in self._driven_spans)

    @property
    def aft_source(self) -> str | None:
        return self._aft_source

    @property
    def is_aft(self) -> bool:
        return self._aft_texture_name is not None

    @property
    def aft_texture_name(self) -> str | None:
        return self._aft_texture_name

    # ActorProxy target bind (SetTarget): the recorder id of the actor
    # this proxy re-renders (ActorProxy.cpp:18 - a proxy is the target
    # re-drawn under the proxy's transform). The environment resolves
    # the Lua table argument and stores the id here; the copy producer
    # keys field copies off it instead of a name list.
    proxy_target: int | None = None

    # -- poke dispatch ---------------------------------------------------

    def poke(self, verb: str, args: list) -> None:
        arg0 = args[0] if args else None
        if self._poke_multi_arg(verb, args) or self._poke_bulk(verb, args):
            return
        if self._poke_effect(verb, args):
            return
        if self._poke_color(verb, args):
            return
        if self._poke_uniform(verb, args):
            return
        if self._poke_channel(verb, arg0) or self._poke_tween(verb, arg0):
            return
        match verb:
            case 'diffuse':
                self._diffuse(args)
            case 'diffusecolor':
                # SetDiffuseColor writes r/g/b onto every corner and
                # leaves alpha alone (Actor.cpp:986-994).
                self._diffuse(args[:3])
            case 'hidden':
                self._visibility(_arg_float(arg0, 1.0) != 0.0)
            case 'visible':
                self._visibility(_arg_float(arg0, 1.0) == 0.0)
            case 'SetTextureName' | 'SetTexture':
                self._texture(verb, arg0)
            case 'blend':
                self._set_immediate(
                    'blend_add',
                    1.0 if str(arg0).strip() == 'add' else 0.0)
            case 'additiveblend':
                self._set_immediate(
                    'blend_add', 1.0 if _arg_float(arg0, 1.0) != 0.0
                    else 0.0)
            case 'setstate':
                self._set_state(_as_int(arg0))
            case 'settext':
                self._set_immediate('text', '' if arg0 is None else str(arg0))
            case 'aux':
                self._set_immediate('aux', _arg_float(arg0))
            case 'addaux':
                delta = _arg_float(arg0)
                if delta is not None:
                    self._set_immediate(
                        'aux', self._current.get('aux', 0.0) + delta)
            case 'animate':
                self._animate(_arg_float(arg0, 1.0) != 0.0)
            case 'play':
                self._animate(True)
            case 'pause':
                self._animate(False)
            case 'heading' | 'pitch' | 'roll':
                self._add_spherical(_SPHERICAL_AXIS[verb], _arg_float(arg0))
            case 'SetRotationOrder':
                self._set_rotation_order(arg0)
            case 'skewx_before_rotation':
                self._set_skew_before('skew_x_before', arg0)
            case 'skewy_before_rotation':
                self._set_skew_before('skew_y_before', arg0)
            case 'skewto':
                self._skewto(args)
            case 'SetWidth':
                self._set_natural(0, _arg_float(arg0))
            case 'SetHeight':
                self._set_natural(1, _arg_float(arg0))
            case _ if verb in verb_surface.DEFERRED:
                # A real capability we have not built yet: swallowed for
                # rendering, but COUNTED so the compile-done report can
                # say what this chart needs (silent gaps cost sessions).
                if self.deferred_notify is not None:
                    self.deferred_notify(verb)
            case _ if verb not in _UNMODELED_OK:
                # Every deliberately-unmodeled verb is in IGNORED or
                # DEFERRED with a reason; anything else reaching the
                # tail is a silent drop - report it.
                if self.dropped_notify is not None:
                    self.dropped_notify(verb)

    def queue_command(self, name: str) -> None:
        """`queuecommand(name)`: a zero-length tween carrying the command
        (Actor.cpp:1074). It fires when the drain reaches it."""
        self._begin_tweening(0.0, 0)
        self._tweens[-1].command = str(name)

    def queue_message(self, name: str) -> None:
        """`queuemessage(name)`: same mechanism, '!'-marked so the runner
        broadcasts instead of playing (Actor.cpp:1080)."""
        self._begin_tweening(0.0, 0)
        self._tweens[-1].command = '!' + str(name)

    def _poke_multi_arg(self, verb, args) -> bool:
        if verb == 'stretchto':
            # StretchTo(x1, y1, x2, y2): fill that rect (SM
            # Actor::StretchTo). Our compiled actors have a CENTER origin
            # (0.5, 0.5), so position at the rect CENTER and set the
            # absolute size to the rect extent - the actor then covers the
            # rect. The chart's full-screen backdrops use
            # stretchto,0,0,SCREEN_WIDTH,SCREEN_HEIGHT.
            if len(args) >= 4:
                x1 = _arg_float(args[0]); y1 = _arg_float(args[1])
                x2 = _arg_float(args[2]); y2 = _arg_float(args[3])
                if None not in (x1, y1, x2, y2):
                    self._set_scalar('x', (x1 + x2) / 2.0)
                    self._set_scalar('y', (y1 + y2) / 2.0)
                    self._set_scalar('size_x', abs(x2 - x1))
                    self._set_scalar('size_y', abs(y2 - y1))
            return True
        if verb == 'scaletocover':
            self._scale_to(args, _FIT_COVER)
            return True
        if verb == 'scaletofit':
            self._scale_to(args, _FIT_INSIDE)
            return True
        if verb in _SIZE_PAIR_SETTERS:
            self._set_scalar('size_x', _arg_float(args[0] if args else None))
            self._set_scalar('size_y',
                             _arg_float(args[1] if len(args) > 1 else None))
            return True
        crop_props = _SIM_CROP_COMPOSITES.get(verb)
        if crop_props is not None:
            # crop / croph / cropv fan one call across the scalar crop
            # edges, one positional arg per edge (same zip the bulk setters
            # use); the storyboard renderer already insets by those edges.
            for prop, arg in zip(crop_props, args):
                self._set_scalar(prop, _arg_float(arg))
            return True
        if verb in _SIZE_AXIS_SETTERS:
            self._set_scalar(_SIZE_AXIS_SETTERS[verb],
                             _arg_float(args[0] if args else None))
            return True
        if verb == 'SetVanishPoint':
            self._set_immediate('vanish_x',
                                _arg_float(args[0] if args else None))
            self._set_immediate(
                'vanish_y', _arg_float(args[1] if len(args) > 1 else None))
            return True
        if verb == 'fov':
            # A frame's perspective camera fov (deg); projects its whole
            # subtree. Immediate, like vanish - it is not tween state.
            self._set_immediate('fov', _arg_float(args[0] if args else None))
            return True
        if verb == 'SetDrawByZPosition':
            # ActorFrame.cpp:194-205: the frame draws its children
            # stable-sorted by GetZ ascending (ActorUtil.cpp:408-416)
            # instead of tree order. Immediate flag, not tween state.
            arg0 = args[0] if args else True
            self._set_immediate('draw_by_z',
                                0.0 if arg0 in (False, 0, '0') else 1.0)
            return True
        return False

    def _poke_channel(self, verb, arg0) -> bool:
        if verb in _SIM_TWEEN_EASING:
            self._begin_tweening(_arg_float(arg0, 0.0),
                                 _SIM_TWEEN_EASING[verb])
        elif verb in _SIM_SCALAR_SETTERS:
            self._set_scalar(_SIM_SCALAR_SETTERS[verb], _arg_float(arg0))
        elif verb in _SIM_ADD_SETTERS:
            self._add_dest(_SIM_ADD_SETTERS[verb], _arg_float(arg0))
        else:
            return False
        return True

    def _poke_uniform(self, verb, args) -> bool:
        """A `self:GetShader():uniform1f(name, value)` custom-uniform
        upload, recorded onto a per-uniform channel `uniform:<name>`,
        returning whether `verb` belonged to the shader-uniform surface.

        `GetShader()` returns the recorder itself (an unmodeled verb
        chains `self` through the sandbox), so the uniform call lands as
        a poke on the frag-owning actor. arg0 is the GLSL uniform name,
        the rest its value; only the first scalar component is kept - the
        chart-shader bridge drives scalar strengths, and a vec's extra
        lanes have no fullscreen-pass consumer yet. Writing through the
        normal dest path means a `linear(t)` before the poke eases the
        uniform exactly as the chart authored it (SM's shader uniforms
        are Actor tween state, openitg RageDisplay::SetUniform*)."""
        if verb == 'GetShader':
            return True
        if verb not in _UNIFORM_VERBS:
            return False
        name = args[0] if args else None
        value = _arg_float(args[1]) if len(args) > 1 else None
        if isinstance(name, str) and value is not None:
            self._set_scalar(f'uniform:{name}', value)
        return True

    def _poke_bulk(self, verb, args) -> bool:
        """Bulk setter (xy/xyz/xyza/xywh/rotationxyz): one positional write
        per property (ACTOR_LUA_API.md 03/04). The bulk-add form
        (addrotationxyz) stacks each component onto its destination."""
        props = _SIM_BULK_SETTERS.get(verb)
        if props is not None:
            for prop, arg in zip(props, args):
                self._set_scalar(prop, _arg_float(arg))
            return True
        props = _SIM_BULK_ADD_SETTERS.get(verb)
        if props is not None:
            for prop, arg in zip(props, args):
                self._add_dest(prop, _arg_float(arg))
            return True
        return False

    def _add_dest(self, prop, delta) -> None:
        """AddX(v) = SetX(GetDestX()+v) - stack onto the destination
        (Actor.h:117)."""
        if delta is not None:
            self._write_dest(prop, self.get_dest(prop) + delta)

    def _set_natural(self, axis, value) -> None:
        """SetWidth/SetHeight: override the natural (unzoomed) size on one
        axis (m_size, Actor.h:128-129). Actor state, not a keyframe channel
        - GetWidth/GetHeight and the fit verbs read it, nothing draws it."""
        if value is not None:
            self._natural[axis] = float(value)

    def _scale_to(self, args, mode) -> None:
        """ScaleToCover/ScaleToFitInside(rect): center on the rect and pick
        a UNIFORM zoom from the actor's natural size (Actor.cpp:672-702).
        The center is natural-independent, so the sim writes x/y now; the
        zoom depends on the true natural size (a render-time fact for a
        sprite), so the sim records the rect + mode onto the `fit_*`
        channels and the renderer resolves the zoom. A negative rect
        dimension flips the actor 180deg about that axis, exactly as the
        engine does before taking |ratio|."""
        if len(args) < 4:
            return
        left, top, right, bottom = (_arg_float(a) for a in args[:4])
        if None in (left, top, right, bottom):
            return
        width, height = right - left, bottom - top
        if width < 0:
            self._set_scalar('rotation_y', 180.0)
        if height < 0:
            self._set_scalar('rotation_x', 180.0)
        self._set_scalar('x', (left + right) / 2.0)
        self._set_scalar('y', (top + bottom) / 2.0)
        self._set_scalar('fit_left', left)
        self._set_scalar('fit_top', top)
        self._set_scalar('fit_right', right)
        self._set_scalar('fit_bottom', bottom)
        self._set_scalar('fit_mode', mode)

    def _poke_tween(self, verb, arg0) -> bool:
        match verb:
            case 'sleep':
                self._sleep(_arg_float(arg0, 0.0))
            case 'queuecommand':
                self.queue_command(arg0)
            case 'queuemessage':
                self.queue_message(arg0)
            case 'finishtweening':
                self._finish_tweening()
            case 'stoptweening':
                self._stop_tweening()
            case 'hurrytweening':
                self._hurry_tweening(_arg_float(arg0, 1.0))
            case _:
                return False
        return True

    # -- tween queue -----------------------------------------------------

    def _begin_tweening(self, duration, ease_id) -> None:
        """Append a queue entry whose state copies the tail (or current
        when the queue is empty), per Actor.cpp:609. Commands never
        inherit. Overflow finishes everything first (Actor.cpp:616).

        Only setter-written props (`_tweenable`, SM's TweenState) enter
        the snapshot: a queued tween beginning later must never replay
        the hidden bit that was current when it was QUEUED over a
        SetHidden made since (the show-then-queued-hide idiom)."""
        if len(self._tweens) > _TWEEN_OVERFLOW:
            self._finish_tweening()
        if self._tweens:
            base = dict(self._tweens[-1].state)
        else:
            base = {prop: value for prop, value in self._current.items()
                    if prop in self._tweenable}
        if not self._tweens and self.queue_notify is not None:
            self.queue_notify()
        self._tweens.append(_Tween(duration, ease_id, base))

    def _sleep(self, duration) -> None:
        """Sleep(t) = a t-tween plus a zero-tween (Actor.cpp:1068)."""
        self._begin_tweening(max(0.0, duration or 0.0), 0)
        self._begin_tweening(0.0, 0)

    def _stop_tweening(self) -> None:
        """Clear the queue, leaving current wherever the head's
        interpolation put it (Actor.cpp:652). Mid-flight properties pin
        their frozen value with an instant keyframe so the recorded
        timeline abandons exactly where the sim did."""
        head = self._tweens[0] if self._tweens else None
        if head is not None and head.started and head.dur > 0.0 and head.left > 0.0:
            for prop, dest in head.state.items():
                start = self._ease_start.get(prop, _rest(prop))
                if dest != start:
                    frozen = self.get(prop)
                    self._current[prop] = frozen
                    self._emit_at(self._now, prop, frozen, 0.0, 0)
        self._tweens = []

    def _hurry_tweening(self, factor) -> None:
        """Scale every queued tween's remaining and total time by
        `factor` (Actor::HurryTweening, openitg Actor.cpp:663-669)."""
        if factor is None or factor < 0.0:
            return
        for tween in self._tweens:
            tween.left *= factor
            tween.dur *= factor

    def _finish_tweening(self) -> None:
        """Jump current to the FINAL queued state and clear
        (Actor.cpp:657). Queued commands are dropped, never fired - the
        engine clears the queue without playing them. Reached-now values
        pin with instant keyframes, overriding any in-flight easing the
        begin-time emission promised."""
        if not self._tweens:
            return
        final = self._tweens[-1].state
        for prop, dest in final.items():
            if dest != self._current.get(prop, _rest(prop)):
                self._emit_at(self._now, prop, dest, 0.0, 0)
        self._current.update(final)
        self._tweens = []

    def _write_dest(self, prop, value) -> None:
        """A setter write: onto the queue tail's state (SetX ->
        DestTweenState, Actor.h:113), or immediately when no tween is
        queued. Retargeting an already-STARTED head re-emits its keyframe
        at the head's begin time; the sampler's bisect makes the newer
        emission win."""
        if value is None:
            return
        self._tweenable.add(prop)
        if self._tweens:
            tail = self._tweens[-1]
            tail.state[prop] = value
            start = self._ease_start.get(prop, _rest(prop))
            if tail.started and value != start:
                self._emit_at(self._head_begin_t, prop, value, tail.dur,
                              tail.ease, start=start)
        else:
            self._current[prop] = value
            self._emit_at(self._now, prop, value, 0.0, 0)

    def _set_scalar(self, prop, value) -> None:
        if value is None:
            return
        targets = prop if isinstance(prop, tuple) else (prop,)
        for target in targets:
            self._write_dest(target, value)

    def _set_immediate(self, prop, value) -> None:
        """A non-tweened engine field (vanish point, hidden, sprite
        state): writes current directly, bypassing the queue."""
        if value is None:
            return
        self._current[prop] = value
        self._emit_at(self._now, prop, value, 0.0, 0)

    # -- non-queue channels ---------------------------------------------

    def _diffuse(self, args) -> None:
        channels = [_arg_float(a) for a in args[:3]]
        if len(channels) == 3 and all(c is not None for c in channels):
            self._write_dest('color', tuple(channels))
        alpha = _arg_float(args[3]) if len(args) > 3 else None
        if alpha is not None:
            self._write_dest('alpha', alpha)

    def _poke_color(self, verb, args) -> bool:
        """Per-corner/edge diffuse gradient, additive glow, and edge fades
        (Sprite.cpp draws the gradient quad, glow pass, and fade ramps).
        The corner/edge/glow verbs carry a RageColor(r,g,b,a); the fade
        verbs carry one 0..1 distance per edge. Each writes the queue-tail
        dest so a `linear(t)` before it eases the channel exactly as the
        chart authored, and rests at the identity sentinel so an actor
        never poked with one draws its flat quad unchanged."""
        corners = _DIFFUSE_CORNER_VERBS.get(verb)
        if corners is not None:
            color = self._rgba(args)
            if color is not None:
                for prop in corners:
                    self._write_dest(prop, color)
            return True
        if verb == 'glow':
            color = self._rgba(args)
            if color is not None:
                self._write_dest('glow', color)
            return True
        edges = _FADE_VERBS.get(verb)
        if edges is not None:
            dist = _arg_float(args[0] if args else None)
            if dist is not None:
                for prop in edges:
                    self._write_dest(prop, dist)
            return True
        return False

    @staticmethod
    def _rgba(args) -> tuple | None:
        """A RageColor(r,g,b,a) from the verb args, alpha defaulting to 1
        (openitg FArg reads a missing 4th as 0, but every gat color verb
        passes alpha; default 1 keeps a 3-arg call opaque)."""
        vals = [_arg_float(a) for a in args[:4]]
        if len(vals) < 3 or any(v is None for v in vals[:3]):
            return None
        alpha = vals[3] if len(vals) == 4 and vals[3] is not None else 1.0
        return (vals[0], vals[1], vals[2], alpha)

    def _visibility(self, hidden: bool) -> None:
        self._set_immediate('hidden', 1.0 if hidden else 0.0)

    def _set_state(self, index) -> None:
        if index is None:
            return
        self._set_immediate('frame', float(index))

    def _animate(self, enabled: bool) -> None:
        self._set_immediate('frame_paused', 0.0 if enabled else 1.0)
        if enabled:
            return
        current = self._current.get('frame', 0.0)
        self._set_immediate('frame', current)

    def _add_spherical(self, axis, deg) -> None:
        """A spherical rotation add (heading/pitch/roll): accumulate the
        axis quaternion onto the dest quat (RageQuatMultiply, Actor.cpp:894).
        Recorded as an immediate 4-tuple on the `quat` channel - the common
        usage sets it with no tween in flight; a slerped quat tween (rare)
        would need a dedicated channel and is not synthesized here."""
        if deg is None:
            return
        self._quat = transform3d.quat_multiply(
            self._quat, transform3d.quat_from_axis(axis, deg))
        self._set_immediate('quat', self._quat)

    def _set_rotation_order(self, token) -> None:
        """SetRotationOrder(token): pick the Euler compose order (fork
        SetRotationOrder @ 004abd70). An unknown token leaves the order be,
        matching the engine's 'Invalid Rotation mode' log-and-ignore. The
        4-char 'xyza' alias collapses to the stock 'xyz'."""
        if not isinstance(token, str):
            return
        order = token.strip().lower()
        if order == 'xyza':
            order = 'xyz'
        if order in transform3d._ROTATION_ORDERS:
            self._rotation_order = order
            self._set_immediate('rotation_order', order)

    def _set_skew_before(self, prop, arg) -> None:
        """skewx/y_before_rotation(flag): whether the skew axis applies
        before the Euler rotation in the compose (fork BeginDraw skew-order
        gate). Immediate mode state resting at 0 (skew-after)."""
        flag = _arg_float(arg)
        if flag is not None:
            self._set_immediate(prop, 1.0 if flag != 0.0 else 0.0)

    def _skewto(self, args) -> None:
        """skewto(x, y): both-axis skew convenience -> skew_x + skew_y
        (fork skewto, no openitg analogue - the fold gives no rect/duration
        arg, so it maps to the two dest skew writes)."""
        sx = _arg_float(args[0]) if args else None
        sy = _arg_float(args[1]) if len(args) > 1 else None
        self._set_scalar('skew_x', sx)
        self._set_scalar('skew_y', sy)

    def _texture(self, verb, arg) -> None:
        arg = getattr(arg, 'marker', arg)
        if not isinstance(arg, str):
            return
        if verb == 'SetTextureName':
            self._aft_texture_name = arg
        elif arg.startswith('aft:'):
            self._aft_source = arg[len('aft:'):]

    def mark_aft(self, name: str) -> None:
        """Declare this actor an ActorFrameTexture render target under
        `name`, so `GetTexture()` returns its `aft:<name>` marker for a
        copy/post-process sprite to pick up. The engine makes an actor
        an AFT by its `Type="ActorFrameTexture"` + `Create()`; a chart
        that also calls `SetTextureName` overrides this default name."""
        if self._aft_texture_name is None:
            self._aft_texture_name = name

    # -- effect oscillators ---------------------------------------------

    def _poke_effect(self, verb, args) -> bool:
        if verb in _EFFECT_KINDS:
            self._open_effect(verb)
            return True
        if verb == 'stopeffect':
            self._close_effect()
            return True
        if verb in _EFFECT_PARAM_VERBS:
            self._effect_param(verb, args)
            return True
        return False

    def _open_effect(self, kind) -> None:
        """Open (or re-affirm) an effect span. The 2D kind setters
        overwrite period/magnitude with their engine defaults (a bare
        `vibrate()` shakes +-10px), recorded as a magnitude sample; the
        color/zoom families have no magnitude defaults and record their
        params via `extra` (see KIND_DEFAULTS in recording_actor)."""
        defaults = KIND_DEFAULTS.get(kind)
        if self._osc_open is None or self._osc_open.kind != kind:
            self._close_effect()
            period = defaults[0] if defaults else _DEFAULT_EFFECT_PERIOD
            self._osc_open = OscSpan(kind, self._now, period,
                                     _DEFAULT_EFFECT_OFFSET,
                                     _DEFAULT_EFFECT_CLOCK)
        elif defaults:
            self._osc_open.period = defaults[0]
        if defaults:
            self._osc_open.set_magnitude(self._now, defaults[1])
        else:
            self._osc_open.touch(self._now)

    def _close_effect(self) -> None:
        span = self._osc_open
        if span is not None:
            span.explicit_end = True
            span.end = max(self._now, span.start)
            if span.magnitude_samples or span.kind in _SELF_EVIDENT_KINDS:
                self._osc_spans.append(span)
            self._osc_open = None

    def _effect_param(self, verb, args) -> None:
        span = self._osc_open
        if span is None:
            # The clock is ACTOR state, not span state: charts set
            # `effectclock,music` with no effect running purely to turn
            # GetSecsIntoEffect into a song clock (gat's mod_time rig).
            if verb == 'effectclock' and args and isinstance(args[0], str):
                self._effect_clock = str(args[0]).strip().lower()
            return
        span.touch(self._now)
        match verb:
            case 'effectperiod':
                span.period = _arg_float(args[0] if args else None,
                                        span.period)
            case 'effectoffset':
                span.offset = _arg_float(args[0] if args else None,
                                        span.offset)
            case 'effectclock':
                clock = args[0] if args else None
                if isinstance(clock, str):
                    span.clock = str(clock).strip().lower()
                    self._effect_clock = span.clock
            case 'effectmagnitude':
                span.set_magnitude(self._now, tuple(
                    _arg_float(args[i] if i < len(args) else None, 0.0)
                    for i in range(3)))
            case 'effectdelay' | 'effecttiming' | 'effectcolor1' \
                    | 'effectcolor2':
                span.extra[verb] = tuple(
                    _arg_float(a, 0.0) for a in args)

    # -- emission --------------------------------------------------------

    def _emit_at(self, t, prop, values, dur, ease_id, start=None) -> None:
        """Record one keyframe. Eased emissions pin their ease-from via
        the Keyframe.start override: the sampler otherwise eases from the
        PREVIOUS keyframe's target, which goes wrong when a retarget
        re-emits at the same t (the stale emission would become the
        ease-from) or after a stop pin."""
        if not isinstance(values, tuple):
            values = (values,)
        if start is not None and not isinstance(start, tuple):
            start = (start,)
        self._frames.setdefault(prop, []).append(
            Keyframe(t, values, dur, ease_id, start=start))
        self._kf_cache = None
        self._seg_emit(t, prop, values, dur, ease_id, start)
        if self._driven:
            self._track_driven(t)

    def _seg_emit(self, t, prop, values, dur, ease_id, start) -> None:
        if not all(isinstance(v, (int, float)) for v in values):
            ts, vals = self._seg_tokens.setdefault(prop, ([], []))
            ts.append(t)
            vals.append(values)
            return

        lanes = self._seg.setdefault(prop, [])
        while len(lanes) < len(values):
            lanes.append(self._new_lane(prop, len(lanes)))
        # Collapse-eligibility mirrors simplify_instants._plain_instant
        # exactly: only single-value no-ease-from instants may join a
        # corridor run; everything else is structural and recorded
        # verbatim, so the lanes reproduce the batch pipeline.
        if dur > 0.0 and start is not None:
            for lane, v0, v1 in zip(lanes, start, values):
                lane.add_ramp(t, t + dur, v0, v1, ease_id)
        elif len(values) == 1 and start is None:
            lanes[0].poke(t, values[0])
        else:
            for lane, v in zip(lanes, values):
                lane.add_hold(t, v)

    def _new_lane(self, prop, i) -> SegmentTimeline:
        rest = _rest(prop)
        if isinstance(rest, tuple):
            rest = rest[i] if i < len(rest) else 0.0
        lane = SegmentTimeline(rest=float(rest))
        # Frontier gating belongs to the reader (it clamps queries to
        # the recording sim's clock); the lane itself is writer-truth.
        lane.frontier = float('inf')
        return lane

    def _track_driven(self, t: float) -> None:
        spans = self._driven_spans
        if spans and t - spans[-1][1] <= _DRIVEN_SPAN_GAP:
            spans[-1][1] = max(spans[-1][1], t)
        elif not spans or t > spans[-1][1]:
            spans.append([t, t])
