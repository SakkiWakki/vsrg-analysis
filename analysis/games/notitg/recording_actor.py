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
  diffusealpha, hidden, visible, ...) emit a `Keyframe` at the actor's
  CURRENT local clock, using whatever tween interval is open (instant
  when none is).
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

from analysis.player.render.effects.timeline import Keyframe

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
# just onto their own synthetic property names.
_SCALAR_SETTERS = {
    'x': 'x', 'y': 'y', 'z': 'z',
    'zoom': ('scale_x', 'scale_y'), 'zoomx': 'scale_x', 'zoomy': 'scale_y',
    'zoomz': 'scale_z',
    'basezoomx': 'scale_x', 'basezoomy': 'scale_y',
    'rotationz': 'rotation', 'rotationx': 'rotation_x',
    'rotationy': 'rotation_y',
    'diffusealpha': 'alpha', 'skewx': 'skew_x', 'skewy': 'skew_y',
}
_ADD_SETTERS = {'addx': 'x', 'addy': 'y', 'addz': 'z'}

_REST = {
    'x': 0.0, 'y': 0.0, 'z': 0.0,
    'scale_x': 1.0, 'scale_y': 1.0, 'scale_z': 1.0,
    'rotation': 0.0, 'rotation_x': 0.0, 'rotation_y': 0.0,
    'alpha': 1.0, 'skew_x': 0.0, 'skew_y': 0.0,
    'color': (1.0, 1.0, 1.0),
}


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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

    def reset_clock(self, clock: float) -> None:
        """Point the local clock at a new poke stream's start time (the
        closure fire time). The pending tween is cleared: a fresh stream
        starts untweened, matching a freshly scheduled command."""
        self._base_clock = float(clock)
        self._clock = float(clock)
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_TWEEN_EASING

    def keyframes(self) -> dict:
        """property -> list[Keyframe], only for properties actually
        poked. Empty when the actor was never touched."""
        return {prop: kfs for prop, kfs in self._frames.items() if kfs}

    # -- poke dispatch ----------------------------------------------------

    def poke(self, verb: str, args: list) -> None:
        arg0 = args[0] if args else None
        match verb:
            case v if v in _TWEEN_EASING:
                self._open_tween(_as_float(arg0, 0.0), _TWEEN_EASING[v])
            case v if v in _SCALAR_SETTERS:
                self._set_scalar(_SCALAR_SETTERS[v], _as_float(arg0))
            case v if v in _ADD_SETTERS:
                self._add_scalar(_ADD_SETTERS[v], _as_float(arg0))
            case 'sleep':
                self._sleep(_as_float(arg0, 0.0))
            case 'finishtweening':
                self._finish_tweening()
            case 'stoptweening':
                self._stop_tweening()
            case 'diffuse':
                self._diffuse(args)
            case 'hidden':
                self._visibility(_as_float(arg0, 1.0) != 0.0)
            case 'visible':
                self._visibility(_as_float(arg0, 1.0) == 0.0)
            # Any other verb pokes actor state we do not model; ignore it.

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

    def _set_scalar(self, prop, value) -> None:
        if value is None:
            return
        targets = prop if isinstance(prop, tuple) else (prop,)
        for target in targets:
            self._emit(target, (value,))

    def _add_scalar(self, prop, delta) -> None:
        if delta is None:
            return
        base = self._current.get(prop, (_REST[prop],))[0]
        self._emit(prop, (base + delta,))

    def _diffuse(self, args) -> None:
        channels = [_as_float(a) for a in args[:3]]
        if len(channels) == 3 and all(c is not None for c in channels):
            self._emit('color', tuple(channels))
        alpha = _as_float(args[3]) if len(args) > 3 else None
        if alpha is not None:
            self._emit('alpha', (alpha,))

    def _visibility(self, hidden: bool) -> None:
        self._emit('alpha', (0.0 if hidden else 1.0,))
