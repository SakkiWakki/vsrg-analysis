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

from bisect import bisect_right

from analysis.games.notitg.lua_api import (
    _ADD_SETTERS, _FALLBACK_TWEEN_EASING, _REST, _SCALAR_GETTERS,
    _SCALAR_SETTERS, _SIZE_AXIS_SETTERS, _SIZE_PAIR_SETTERS, _TWEEN_EASING,
    _as_float, _as_int)
from analysis.player.render.effects.easing import (
    EASE_SM_BOUNCE_BEGIN, EASE_SM_BOUNCE_END, EASE_SM_SPRING, ease)
from analysis.player.render.effects.timeline import Keyframe

# The full SM tween-verb surface. lua_api's table carries the shared
# five; the SM-only curves live here until cutover folds them back.
_SIM_TWEEN_EASING = {
    **_TWEEN_EASING,
    'bouncebegin': EASE_SM_BOUNCE_BEGIN,
    'bounceend': EASE_SM_BOUNCE_END,
    'spring': EASE_SM_SPRING,
}

# Actor.cpp:616 guards a runaway queue ("infinitely recursing
# ActorCommand?") by finishing all tweens once the queue passes 50.
_TWEEN_OVERFLOW = 50

# Safety cap on one update's drain iterations: a command that queues
# another zero-length command each time it fires would otherwise spin
# forever inside a single update (the engine's overflow guard above only
# fires on queue DEPTH, not on drain length).
_MAX_DRAIN_STEPS = 10000

_DEFAULT_EFFECT_CLOCK = 'bgm'
# The COMPLETE effect-kind verb surface, from the NotITG decompile
# (Actor::PushSelf registrations, refs/notitg/decompile/c/actors/
# Actor.clean.c:3445-3480; SetEffect* bodies @0x4ab230-0x4ab630).
# Position/rotation kinds synthesize downstream today; the color/zoom
# families (rainbow/diffuse*/glow*/pulse*) record spans for a future
# color-oscillator synthesis.
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

# Gap under which two driven-poke times merge into one span (seconds);
# per-frame ticks are 1/60 apart, real gaps between sections are long.
_DRIVEN_SPAN_GAP = 0.5


def _rest(prop):
    return _REST.get(prop, 0.0)


class OscSpan:
    """One effect-oscillator interval: kind, period/offset/clock, and the
    magnitude as (t, x, y, z) samples (many when a per-frame driver ramps
    it). `end` is None while open. Same shape recording_actor records, so
    the oscillator keyframe synthesis consumes either."""

    __slots__ = ('kind', 'start', 'end', 'period', 'offset', 'clock',
                 'magnitude_samples', 'last_clock', '_clock_index', 'extra')

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
        self._tweens: list = []
        self._ease_start: dict = {}
        self._head_begin_t = 0.0
        self._frames: dict = {}
        self._aft_source: str | None = None
        self._aft_texture_name: str | None = None
        self._osc_spans: list = []
        self._osc_open: OscSpan | None = None
        self._driven = False
        self._driven_spans: list = []
        self._in_update = False

    @property
    def now(self) -> float:
        return self._now

    # -- time ------------------------------------------------------------

    def update_to(self, t: float, run_command=None) -> None:
        """Advance the tween queue to sim time `t` (Actor::UpdateTweening,
        Actor.cpp:469). `run_command(name)` plays a queue-carried command
        (or broadcasts, for '!name') at the exact moment its zero-tween
        begins; commands may poke this actor, appending to the live
        queue, and the drain continues. A zero-dt call returns without
        beginning anything, matching the engine's early-out."""
        remaining = float(t) - self._now
        if remaining <= 0.0 or self._in_update:
            return
        self._in_update = True
        try:
            for _ in range(_MAX_DRAIN_STEPS):
                if not self._tweens or remaining <= 0.0:
                    break
                head = self._tweens[0]
                if not head.started:
                    self._begin_head(head, run_command)
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
        self._tweens.pop(0)

    # -- reads -----------------------------------------------------------

    def get(self, prop: str):
        """The engine-current value: mid-flight interpolation when the
        head tween is running (GetX -> m_current, Actor.h:107), else the
        settled value, else rest."""
        head = self._tweens[0] if self._tweens else None
        if head is not None and head.started and head.dur > 0.0:
            dest = head.state.get(prop, _rest(prop))
            start = self._ease_start.get(prop, _rest(prop))
            if dest != start:
                progress = 1.0 - head.left / head.dur
                return self._lerp(start, dest, ease(head.ease, progress))
        value = self._current.get(prop)
        return value if value is not None else _rest(prop)

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
                return f'aft:{self._aft_texture_name}'
            case 'GetSecsIntoEffect':
                return self._secs_into_effect()
            case 'GetEffectMagnitude' if self._osc_open is not None:
                return self._osc_open.magnitude_at(self._now)
            case 'GetText':
                return str(self._current.get('text', ''))
            case 'getaux':
                return self._current.get('aux', 0.0)
            case 'GetTweenTimeLeft':
                return sum(t.left for t in self._tweens)
            case _:
                return None

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
        return (self.get('rotation_x'), self.get('rotation_y'),
                self.get('rotation'))

    # -- recording surface ----------------------------------------------

    def keyframes(self) -> dict:
        """property -> list[Keyframe], only for properties actually
        poked. The recorded timeline replays exactly what the sim
        displayed (begin-time emission + stop pins)."""
        return {prop: kfs for prop, kfs in self._frames.items() if kfs}

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

    # -- poke dispatch ---------------------------------------------------

    def poke(self, verb: str, args: list) -> None:
        arg0 = args[0] if args else None
        if self._poke_multi_arg(verb, args) or self._poke_effect(verb, args):
            return
        if self._poke_channel(verb, arg0) or self._poke_tween(verb, arg0):
            return
        match verb:
            case 'diffuse':
                self._diffuse(args)
            case 'hidden':
                self._visibility(_as_float(arg0, 1.0) != 0.0)
            case 'visible':
                self._visibility(_as_float(arg0, 1.0) == 0.0)
            case 'SetTextureName' | 'SetTexture':
                self._texture(verb, arg0)
            case 'setstate':
                self._set_state(_as_int(arg0))
            case 'settext':
                self._set_immediate('text', '' if arg0 is None else str(arg0))
            case 'aux':
                self._set_immediate('aux', _as_float(arg0))
            case 'addaux':
                delta = _as_float(arg0)
                if delta is not None:
                    self._set_immediate(
                        'aux', self._current.get('aux', 0.0) + delta)
            case 'animate':
                self._animate(_as_float(arg0, 1.0) != 0.0)
            # Any other verb pokes actor state we do not model; ignore it.

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
        if verb in _SIZE_PAIR_SETTERS:
            self._set_scalar('size_x', _as_float(args[0] if args else None))
            self._set_scalar('size_y',
                             _as_float(args[1] if len(args) > 1 else None))
            return True
        if verb in _SIZE_AXIS_SETTERS:
            self._set_scalar(_SIZE_AXIS_SETTERS[verb],
                             _as_float(args[0] if args else None))
            return True
        if verb == 'SetVanishPoint':
            self._set_immediate('vanish_x',
                                _as_float(args[0] if args else None))
            self._set_immediate(
                'vanish_y', _as_float(args[1] if len(args) > 1 else None))
            return True
        return False

    def _poke_channel(self, verb, arg0) -> bool:
        if verb in _SIM_TWEEN_EASING:
            self._begin_tweening(_as_float(arg0, 0.0),
                                 _SIM_TWEEN_EASING[verb])
        elif verb in _SCALAR_SETTERS:
            self._set_scalar(_SCALAR_SETTERS[verb], _as_float(arg0))
        elif verb in _ADD_SETTERS:
            prop = _ADD_SETTERS[verb]
            delta = _as_float(arg0)
            if delta is not None:
                self._write_dest(prop, self.get_dest(prop) + delta)
        else:
            return False
        return True

    def _poke_tween(self, verb, arg0) -> bool:
        match verb:
            case 'sleep':
                self._sleep(_as_float(arg0, 0.0))
            case 'queuecommand':
                self.queue_command(arg0)
            case 'queuemessage':
                self.queue_message(arg0)
            case 'finishtweening':
                self._finish_tweening()
            case 'stoptweening':
                self._stop_tweening()
            case _:
                return False
        return True

    # -- tween queue -----------------------------------------------------

    def _begin_tweening(self, duration, ease_id) -> None:
        """Append a queue entry whose state copies the tail (or current
        when the queue is empty), per Actor.cpp:609. Commands never
        inherit. Overflow finishes everything first (Actor.cpp:616)."""
        if len(self._tweens) > _TWEEN_OVERFLOW:
            self._finish_tweening()
        base = (dict(self._tweens[-1].state) if self._tweens
                else dict(self._current))
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
        channels = [_as_float(a) for a in args[:3]]
        if len(channels) == 3 and all(c is not None for c in channels):
            self._write_dest('color', tuple(channels))
        alpha = _as_float(args[3]) if len(args) > 3 else None
        if alpha is not None:
            self._write_dest('alpha', alpha)

    def _visibility(self, hidden: bool) -> None:
        self._set_immediate('hidden', 1.0 if hidden else 0.0)

    def _set_state(self, index) -> None:
        if index is None:
            return
        self._set_immediate('frame', float(index))

    def _animate(self, enabled: bool) -> None:
        if enabled:
            return
        current = self._current.get('frame', 0.0)
        self._set_immediate('frame', current)

    def _texture(self, verb, arg) -> None:
        if not isinstance(arg, str):
            return
        if verb == 'SetTextureName':
            self._aft_texture_name = arg
        elif arg.startswith('aft:'):
            self._aft_source = arg[len('aft:'):]

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
        if self._osc_open is not None and self._osc_open.kind == kind:
            self._osc_open.touch(self._now)
            return
        self._close_effect()
        self._osc_open = OscSpan(kind, self._now, _DEFAULT_EFFECT_PERIOD,
                                 _DEFAULT_EFFECT_OFFSET,
                                 _DEFAULT_EFFECT_CLOCK)

    def _close_effect(self) -> None:
        span = self._osc_open
        if span is not None:
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
                span.period = _as_float(args[0] if args else None,
                                        span.period)
            case 'effectoffset':
                span.offset = _as_float(args[0] if args else None,
                                        span.offset)
            case 'effectclock':
                clock = args[0] if args else None
                if isinstance(clock, str):
                    span.clock = str(clock).strip().lower()
                    self._effect_clock = span.clock
            case 'effectmagnitude':
                span.set_magnitude(self._now, tuple(
                    _as_float(args[i] if i < len(args) else None, 0.0)
                    for i in range(3)))
            case 'effectdelay' | 'effecttiming' | 'effectcolor1' \
                    | 'effectcolor2':
                span.extra[verb] = tuple(
                    _as_float(a, 0.0) for a in args)

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
        if self._driven:
            self._track_driven(t)

    def _track_driven(self, t: float) -> None:
        spans = self._driven_spans
        if spans and t - spans[-1][1] <= _DRIVEN_SPAN_GAP:
            spans[-1][1] = max(spans[-1][1], t)
        elif not spans or t > spans[-1][1]:
            spans.append([t, t])
