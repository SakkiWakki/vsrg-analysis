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

import re

from analysis.player.render.effects.timeline import EventTimeline, Keyframe

# Tween verb -> easing id (osu.Framework Easing enum shared by the
# storyboard timelines). SM's accelerate = ease-in quad, decelerate =
# ease-out quad, smooth = in-out cubic; linear/tween are linear.
_TWEEN_EASING = {
    'linear': 0, 'tween': 0, 'accelerate': 3, 'decelerate': 4,
    'smooth': 8,
}
_FALLBACK_TWEEN_EASING = 0

# Setter verb -> storyboard property (or a tuple of properties it feeds,
# for uniform zoom). 'z'/'rotationx'/'rotationy' have no 2D-storyboard
# analogue; they still record so an actor's full poke stream is legible,
# just onto their own synthetic property names. `basezoom` is SM's
# separate pre-multiplier (the AFT copies set basezoomy(-1) to flip the
# upside-down capture); it records onto its own property so `zoom` can
# animate on top without clobbering it (the field producer folds the
# two).
_SCALAR_SETTERS = {
    'x': 'x', 'y': 'y', 'z': 'z',
    'zoom': ('scale_x', 'scale_y'), 'zoomx': 'scale_x', 'zoomy': 'scale_y',
    'zoomz': 'scale_z',
    'basezoom': ('base_scale_x', 'base_scale_y'),
    'basezoomx': 'base_scale_x', 'basezoomy': 'base_scale_y',
    'rotationz': 'rotation', 'rotationx': 'rotation_x',
    'rotationy': 'rotation_y',
    'diffusealpha': 'alpha', 'skewx': 'skew_x', 'skewy': 'skew_y',
}
_ADD_SETTERS = {'addx': 'x', 'addy': 'y', 'addz': 'z'}

_REST = {
    'x': 0.0, 'y': 0.0, 'z': 0.0,
    'scale_x': 1.0, 'scale_y': 1.0, 'scale_z': 1.0,
    'base_scale_x': 1.0, 'base_scale_y': 1.0,
    'rotation': 0.0, 'rotation_x': 0.0, 'rotation_y': 0.0,
    'alpha': 1.0, 'skew_x': 0.0, 'skew_y': 0.0,
    'color': (1.0, 1.0, 1.0),
    'frame': 0.0,
    'hidden': 0.0,
}

# Actor getter verb -> the property whose CURRENT value it returns.
# Per-frame driver closures read these (`gat_afty2:GetY()`,
# `gat_aft_target:GetZoom()`) to feed sibling actors, so a getter must
# hand back a real number for that arithmetic to land as a keyframe
# instead of faulting on a table. Reads use the last set value (rest
# when never poked), ignoring in-flight tweens - a load-time snapshot.
_SCALAR_GETTERS = {
    'GetX': 'x', 'GetY': 'y', 'GetZ': 'z',
    'GetZoom': 'scale_x', 'GetZoomX': 'scale_x', 'GetZoomY': 'scale_y',
    'GetRotationX': 'rotation_x', 'GetRotationY': 'rotation_y',
    'GetRotationZ': 'rotation',
}


# SM screen constants that appear as classic-command args (`x,
# SCREEN_CENTER_X`). The Lua stub env resolves these for %function
# bodies; classic strings are parsed to raw tokens, so the recorder
# resolves them here. 640x480 design space, matching mod_stubs.
_SCREEN_CONSTANTS = {
    'SCREEN_WIDTH': 640.0, 'SCREEN_HEIGHT': 480.0,
    'SCREEN_CENTER_X': 320.0, 'SCREEN_CENTER_Y': 240.0,
    'SCREEN_LEFT': 0.0, 'SCREEN_RIGHT': 640.0,
    'SCREEN_TOP': 0.0, 'SCREEN_BOTTOM': 480.0,
    'sw': 640.0, 'sh': 480.0,
}


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        resolved = _resolve_screen_expr(value.strip())
        if resolved is not None:
            return resolved
    return default


# A classic-command arg that is a screen constant, optionally negated and
# scaled/offset by a literal (`-SCREEN_WIDTH/2`, `SCREEN_HEIGHT*0.55`,
# `112*(SCREEN_HEIGHT/480)`). The proxy-grid frames center themselves with
# these, so resolving them places the grid where the engine does.
_SCREEN_EXPR_RE = re.compile(
    r'^(?P<lead>-?)\s*(?:(?P<pre>[0-9.]+)\s*\*\s*)?'
    r'\(?\s*(?P<name>[A-Za-z_]+)\s*(?:(?P<op>[*/])\s*(?P<num>[0-9.]+))?\s*\)?$')


def _resolve_screen_expr(text: str):
    """A screen constant with an optional literal multiply/divide and a
    leading minus (`SCREEN_CENTER_X`, `-SCREEN_WIDTH/2`,
    `112*(SCREEN_HEIGHT/480)`), or None when `text` is not that shape."""
    constant = _SCREEN_CONSTANTS.get(text)
    if constant is not None:
        return constant
    match = _SCREEN_EXPR_RE.match(text)
    if match is None:
        return None
    base = _SCREEN_CONSTANTS.get(match['name'])
    if base is None:
        return None
    value = base
    if match['op'] == '/':
        value /= float(match['num'])
    elif match['op'] == '*':
        value *= float(match['num'])
    if match['pre']:
        value *= float(match['pre'])
    if match['lead'] == '-':
        value = -value
    return value


def _as_int(value, default=None):
    f = _as_float(value)
    return int(f) if f is not None else default


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
