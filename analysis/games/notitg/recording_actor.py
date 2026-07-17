"""Recording actor surface: turns StepMania actor pokes into keyframes.

A NotITG modfile drives named actors two ways, and both must land as
keyframes on ONE timeline per actor:

- at LOAD, an actor's InitCommand/OnCommand Lua runs `self:x(100)`,
  `self:linear(1)`, ... on itself;
- at fire time, a scheduled `mod_actions` closure pokes the same actor
  through the Lua global it self-assigned (`gat_g_rot_intro:y(60)`).

`RecordingActor` is the object those pokes hit. It reproduces enough of
the SM tween model to place keyframes at the right times:

- Property setters (x/y/z, zoom/zoomx/zoomy, rotationx/y/z, diffuse,
  diffusealpha, ...) emit a `Keyframe` at the actor's CURRENT local
  clock, using whatever tween interval is open (instant when none is).
- Visibility setters (`hidden`, `visible`) are SEPARATE from alpha: SM's
  `hidden` flag is a hard visibility bit, independent of diffusealpha, so
  it records onto its own `hidden` channel (1 hidden, 0 shown). An actor
  can hold a diffusealpha crossfade while `hidden,1` gates it off; the
  renderer draws only when NOT hidden AND alpha > 0.
- Tween verbs (linear/accelerate/decelerate/smooth/tween/sleep) open a
  new interval: they first CLOSE the previous one (advancing the local
  clock by its duration, so chained tweens accumulate) then set the new
  pending duration + easing. `sleep` opens an empty interval and closes
  it at once (a pure time gap).
- `finishtweening` closes the open interval and clears the pending
  tween (subsequent setters are instant again). `stoptweening` clears
  the pending tween WITHOUT advancing (SM abandons the in-flight tween
  in place); we treat the remaining setters as instant from the clock.

The actor's local clock starts at the moment the poke stream begins
(the closure's fire time in seconds, or the actor-creation time for
InitCommand strings). `add*` verbs read the last target so relative
moves accumulate. Unknown verbs are ignored (they poke actor state we
do not model, e.g. effect params), never faulting the recording.
"""
from __future__ import annotations

from analysis.games.notitg.lua_api import (
    _ADD_SETTERS, _DRIVEN_SPAN_GAP, _FALLBACK_TWEEN_EASING, _LIVE_RESET,
    _REST, _SCALAR_GETTERS, _SCALAR_SETTERS, _SIZE_AXIS_SETTERS,
    _SIZE_PAIR_SETTERS, _TWEEN_EASING, _as_float, _as_int,
    _resolve_screen_expr)  # noqa: F401 - screen-constant resolution, re-exported
from analysis.player.render.effects.timeline import EventTimeline, Keyframe


class RecordingActor:
    """One recorder per named actor. Pokes append Keyframes onto
    per-property lists; `keyframes()` hands them to the compiler."""

    def __init__(self, clock: float = 0.0):
        self._base_clock = float(clock)
        self._clock = float(clock)
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_TWEEN_EASING
        self._frames: dict = {}
        self._current: dict = {}
        self._aft_source: str | None = None
        self._aft_texture_name: str | None = None
        # Sampling-mirror state, set only while a per-frame update
        # integrator drives this actor (begin_sampling..end_sampling).
        self._baseline: dict | None = None
        self._sample_clock: list | None = None
        self._live_props: set = set()
        self._driven_spans: list = []

    def reset_clock(self, clock: float) -> None:
        """Point the local clock at a new poke stream's start time (the
        closure fire time). The pending tween is cleared: a fresh stream
        starts untweened, matching a freshly scheduled command."""
        self._base_clock = float(clock)
        self._clock = float(clock)
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_TWEEN_EASING

    def advance_clock_by_pending(self) -> None:
        """Advance the local clock past the open tween, as a
        `queuecommand` does: SM runs the queued command on the next frame,
        after the in-flight tween finishes, so its keyframes start where
        the current tween ends. Approximation (SM's real delay is one
        frame, not the exact tween length); documented as such."""
        self._clock += self._pending_dur
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_TWEEN_EASING

    def keyframes(self) -> dict:
        """property -> list[Keyframe], only for properties actually
        poked. Empty when the actor was never touched."""
        return {prop: kfs for prop, kfs in self._frames.items() if kfs}

    # -- sampling mirror (per-frame update integrator) --------------------

    def begin_sampling(self, sample_clock: list) -> None:
        """Enter sampling-mirror mode for a per-frame update pass.

        The update integrator runs the chart's real `UpdateCommand` on a
        tick grid; a driver in it reads other actors (`gat_scroller:GetX()`)
        expecting their value AT THE TICK'S TIME, not a frozen load-time
        snapshot. `sample_clock` is a one-element list the integrator
        rewrites each tick (a shared cell, so every mirror tracks the same
        time); `get()` samples this actor's PRE-PASS timeline at that time
        for any property the pass has not yet poked. Once the pass pokes a
        property (`gat_allproxies:addx(..)`), it becomes a live accumulator
        and `get()` returns the accumulated value instead - so stateful
        per-frame accumulation reads its own running total, while inert
        data-holder quads read their compiled curve."""
        self._baseline = {prop: EventTimeline(frames, rest=(_REST.get(prop,
                          0.0),)) for prop, frames in self._frames.items()
                          if frames}
        self._sample_clock = sample_clock
        self._live_props = set()

    def end_sampling(self) -> None:
        self._baseline = None
        self._sample_clock = None
        self._live_props = set()

    def live_poke(self, verb: str, args: list) -> None:
        """Reset an ALREADY-LIVE accumulator property, without emitting a
        keyframe, during a sampling pass.

        The update integrator re-fires the scheduled `mod_actions` with
        keyframe recording frozen (their persistent timelines were already
        captured by the one-shot replay), but some closures RESET an
        accumulator a per-frame driver reads back - gat's `Toss` message
        re-anchors each toss quad (`a.actor:y(SCREEN_HEIGHT+100)`) so the
        next tick's `gat_update_toss` reads the reset position, not the
        runaway fall total. Dropping the poke leaves the accumulator
        running away; recording it would double the compiled timeline.

        The reset applies ONLY to a property the pass is already
        accumulating (`prop in _live_props`): those are the quads the
        per-frame body drives via `add*`, whose running value `get()` reads
        off `_current`. A property the pass never poked stays on its
        baseline timeline (the message's own tween curve, already
        captured), so a message that merely tweens a data-holder quad the
        driver READS - gat's slam quads (`SlamLeft1`) that the split loop
        samples but never pokes - is untouched here and keeps its curve.
        Only scalar/add setters carry an accumulator value; tween opens and
        `hidden` have nothing to reset."""
        props, relative = _LIVE_RESET.get(verb, (None, False))
        amount = _as_float(args[0]) if args else None
        if self._baseline is not None and props is not None and amount is not None:
            self._reset_live(props, amount, relative)

    def _reset_live(self, props, amount, relative) -> None:
        """Re-anchor each ALREADY-LIVE property in `props` (see
        `live_poke`): absolute setters replace the running value with
        `amount`, `add*` setters offset it. A property the pass never poked
        is skipped so it keeps its baseline curve."""
        for target in props:
            if target in self._live_props:
                base = self.get(target) if relative else 0.0
                self._current[target] = (base + amount,)

    def _track_driven(self, t: float) -> None:
        spans = self._driven_spans
        if spans and t - spans[-1][1] <= _DRIVEN_SPAN_GAP:
            spans[-1][1] = max(spans[-1][1], t)
        elif not spans or t > spans[-1][1]:
            spans.append([t, t])

    def driven_spans(self) -> tuple:
        """(start, end) second-spans in which a per-frame integration pass
        actually poked this actor. A per-frame-driven visual only exists
        while its driver runs, so consumers gate visibility to these spans
        instead of letting the last recorded state hold forever."""
        return tuple((s, e) for s, e in self._driven_spans)

    @property
    def aft_source(self) -> str | None:
        """The ActorFrameTexture name this actor draws, when it is an
        AFT-screen-copy sprite (it called `SetTexture(AFT:GetTexture())`
        so its texture is the captured field), else None. The field
        producer uses this to tell field copies from ordinary sprites."""
        return self._aft_source

    @property
    def is_aft(self) -> bool:
        """True when this actor is itself an ActorFrameTexture render
        target (it called `SetTextureName(...)`), the source a copy
        sprite draws - not a copy."""
        return self._aft_texture_name is not None

    def get(self, prop: str) -> float:
        """Current value of a scalar property (rest when never set).
        Reads the last SET target, ignoring any in-flight tween - a
        load-time snapshot, matching what a driver closure sees when it
        reads a sibling actor mid-command.

        Under a sampling-mirror pass (begin_sampling), a property the pass
        has not yet poked is read from the actor's PRE-PASS timeline at the
        tick clock instead, so an update driver reads a source quad's value
        at the current tick time. A property the pass HAS poked is a live
        accumulator and keeps reading its running value."""
        if self._baseline is not None and prop not in self._live_props:
            timeline = self._baseline.get(prop)
            if timeline is not None:
                return timeline.sample(self._sample_clock[0])[0]
        return self._current.get(prop, (_REST.get(prop, 0.0),))[0]

    # -- poke dispatch ----------------------------------------------------

    def poke(self, verb: str, args: list) -> None:
        arg0 = args[0] if args else None
        if self._poke_size(verb, args):
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
            case 'animate':
                self._animate(_as_float(arg0, 1.0) != 0.0)
            # Any other verb pokes actor state we do not model; ignore it.

    def _poke_size(self, verb, args) -> bool:
        """Absolute-size setters (`zoomto`/`setsize` take w AND h;
        `zoomtowidth`/`zoomtoheight` one axis), returning whether `verb`
        was one. Kept off `_poke_channel` because they read two args."""
        if verb in _SIZE_PAIR_SETTERS:
            self._set_scalar('size_x', _as_float(args[0] if args else None))
            self._set_scalar('size_y',
                             _as_float(args[1] if len(args) > 1 else None))
            return True
        if verb in _SIZE_AXIS_SETTERS:
            self._set_scalar(_SIZE_AXIS_SETTERS[verb],
                             _as_float(args[0] if args else None))
            return True
        return False

    def _poke_channel(self, verb, arg0) -> bool:
        """Handle the value-carrying verbs - tween opens and scalar/add
        setters, all keyed by lookup table - returning whether `verb` was
        one. Split out of `poke` so its match stays flat."""
        if verb in _TWEEN_EASING:
            self._open_tween(_as_float(arg0, 0.0), _TWEEN_EASING[verb])
        elif verb in _SCALAR_SETTERS:
            self._set_scalar(_SCALAR_SETTERS[verb], _as_float(arg0))
        elif verb in _ADD_SETTERS:
            self._add_scalar(_ADD_SETTERS[verb], _as_float(arg0))
        else:
            return False
        return True

    def _poke_tween(self, verb, arg0) -> bool:
        """The tween-interval control verbs (no channel value), returning
        whether `verb` was one."""
        match verb:
            case 'sleep':
                self._sleep(_as_float(arg0, 0.0))
            case 'finishtweening':
                self._finish_tweening()
            case 'stoptweening':
                self._stop_tweening()
            case _:
                return False
        return True

    def read(self, verb: str):
        """Value for a getter call, or None when `verb` is not a getter
        we model (the Lua bridge then falls back to the poke path so the
        chained-table behaviour is preserved for unknown reads).

        `GetTexture` on an AFT returns the 'aft:<name>' marker a copy
        sprite feeds to its own SetTexture, so the copy learns which
        capture it draws."""
        match verb:
            case v if v in _SCALAR_GETTERS:
                return self.get(_SCALAR_GETTERS[v])
            case 'GetTexture' if self._aft_texture_name is not None:
                return f'aft:{self._aft_texture_name}'
            case _:
                return None

    def getrotation(self):
        """SM's `getrotation()` returns (rx, ry, rz). Copy sprites read
        the z component (`local x,y,z = a:getrotation()`) to mirror a
        source's spin, so all three come back as numbers."""
        return (self.get('rotation_x'), self.get('rotation_y'),
                self.get('rotation'))

    def _texture(self, verb, arg) -> None:
        """`SetTextureName('gat_aft')` marks this actor AS an AFT render
        target; `SetTexture(AFT:GetTexture())` marks it a COPY of one
        (GetTexture bridges to the 'aft:<name>' marker). Any other
        texture (a name/path) is an ordinary sprite and leaves both
        unset."""
        if not isinstance(arg, str):
            return
        if verb == 'SetTextureName':
            self._aft_texture_name = arg
        elif arg.startswith('aft:'):
            self._aft_source = arg[len('aft:'):]

    # -- tween interval bookkeeping ---------------------------------------

    def _open_tween(self, duration: float, easing: int) -> None:
        self._clock += self._pending_dur
        self._pending_dur = max(0.0, duration)
        self._pending_ease = easing

    def _sleep(self, duration: float) -> None:
        self._clock += self._pending_dur + max(0.0, duration)
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_TWEEN_EASING

    def _finish_tweening(self) -> None:
        self._clock += self._pending_dur
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_TWEEN_EASING

    def _stop_tweening(self) -> None:
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_TWEEN_EASING

    # -- property emission ------------------------------------------------

    def _emit(self, prop: str, values: tuple) -> None:
        self._frames.setdefault(prop, []).append(
            Keyframe(self._clock, values, self._pending_dur,
                     self._pending_ease))
        self._current[prop] = values
        if self._baseline is not None:
            self._live_props.add(prop)
            self._track_driven(self._clock)

    def _set_scalar(self, prop, value) -> None:
        if value is None:
            return
        targets = prop if isinstance(prop, tuple) else (prop,)
        for target in targets:
            self._emit(target, (value,))

    def _add_scalar(self, prop, delta) -> None:
        if delta is None:
            return
        self._emit(prop, (self.get(prop) + delta,))

    def _diffuse(self, args) -> None:
        channels = [_as_float(a) for a in args[:3]]
        if len(channels) == 3 and all(c is not None for c in channels):
            self._emit('color', tuple(channels))
        alpha = _as_float(args[3]) if len(args) > 3 else None
        if alpha is not None:
            self._emit('alpha', (alpha,))

    def _visibility(self, hidden: bool) -> None:
        self._emit('hidden', (1.0 if hidden else 0.0,))

    def _set_state(self, index) -> None:
        """`setstate(i)` pins the sprite to grid frame `i` at the current
        clock (SM `SetState`). Recorded as a step keyframe on the `frame`
        channel, which the compiler turns into a pin timeline that
        overrides the sheet's auto-animation."""
        if index is None:
            return
        self._emit('frame', (float(index),))

    def _animate(self, enabled: bool) -> None:
        """`animate(false)` freezes the sprite on its current frame (SM
        `EnableAnimation(false)`); we pin that frame from the clock on.
        `animate(true)` with no prior `setstate` leaves the sheet on its
        default auto-animation (no pin emitted)."""
        if enabled:
            return
        current = self._current.get('frame', (0.0,))[0]
        self._emit('frame', (current,))
