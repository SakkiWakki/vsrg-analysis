"""Recording actor for Etterna SM5 modfile command functions.

An Etterna modfile actor carries InitCommand/OnCommand FUNCTIONS (not
strings): the engine calls `command(self)` at load, and the body pokes
`self` with the SM5 actor verbs (`self:xy(x,y)`, `self:zoomto(w,h)`,
`self:diffuse(r,g,b,a)`, `self:rotationz(deg)`, `self:linear(1)`, ...).
`RecordingActor` is the object those pokes hit; it reproduces enough of
the SM tween model to place storyboard `Keyframe`s at the right times.

The command clock is SHARED (`ActorClock`): the same clock the modfile
environment reads when a `po:Drunk(v, s)` PlayerOptions call fires, so a
script that interleaves `self:sleep(1)` with poptions calls lays down
BOTH a visual keyframe timeline and a mod-event timeline on one clock.
The actor's InitCommand advances the clock as its tweens/sleeps run;
poptions events fired mid-command land at the matching time.

Tween model (as in SM):
- Setters (`xy`/`x`/`y`/`zoom`/`zoomto`/`diffuse`/`rotationz`/...) emit a
  Keyframe at the clock's CURRENT time using the open tween interval
  (instant when none is open).
- Tween verbs (`linear`/`accelerate`/`decelerate`/`smooth`/`sleep`) open
  a new interval: they first CLOSE the previous one (advancing the clock
  by its duration, so chained tweens accumulate) then set the pending
  duration + easing. `sleep` is a pure time gap.
- `finishtweening` closes the open interval; `hurrytweening`/
  `stoptweening` clear the pending tween without advancing.
Unknown verbs poke actor state we do not model and are ignored, never
faulting the recording.
"""
from __future__ import annotations

from analysis.player.render.effects.timeline import Keyframe

# Tween verb -> osu.Framework easing id (shared by the storyboard
# timelines). accelerate = ease-in quad, decelerate = ease-out quad,
# smooth = in-out cubic; linear/tween are linear.
_TWEEN_EASING = {
    'linear': 0, 'tween': 0, 'accelerate': 3, 'decelerate': 4, 'smooth': 8,
}
_FALLBACK_EASING = 0

# Setter verb -> the storyboard properties it feeds, one per positional
# arg (so `xy(x, y)` and `zoomto(w, h)` fill two, `zoom(s)` fans one
# value to both scale axes). SM5 uses `zoom` for uniform scale and
# `zoomx`/`zoomy` for axes; `rotationz` is the in-plane 2D rotation
# (rotationx/y have no 2D analogue and are dropped by the compiler's
# drawable-prop filter). A single-element tuple whose value repeats
# across several props (uniform zoom) is handled by `_set_scalars`.
_SCALAR_SETTERS = {
    'x': ('x',), 'y': ('y',),
    'xy': ('x', 'y'), 'zoomto': ('scale_x', 'scale_y'),
    'zoomx': ('scale_x',), 'zoomy': ('scale_y',),
    'zoomtowidth': ('scale_x',), 'zoomtoheight': ('scale_y',),
    'rotationz': ('rotation',), 'rotationx': ('rotation_x',),
    'rotationy': ('rotation_y',), 'diffusealpha': ('alpha',),
}
# Verbs whose single arg fans out to several props (uniform zoom).
_FANOUT_SETTERS = {'zoom': ('scale_x', 'scale_y')}
_ADD_SETTERS = {'addx': 'x', 'addy': 'y'}

_REST = {
    'x': 0.0, 'y': 0.0, 'scale_x': 1.0, 'scale_y': 1.0,
    'rotation': 0.0, 'rotation_x': 0.0, 'rotation_y': 0.0,
    'alpha': 1.0, 'color': (1.0, 1.0, 1.0),
}

# Getter verb -> the property whose current value it returns. A driver
# command that reads one actor to size another (`b:zoomx(a:GetX())`)
# needs a real number back or the arithmetic faults on a table.
_SCALAR_GETTERS = {
    'GetX': 'x', 'GetY': 'y', 'GetZoom': 'scale_x',
    'GetZoomX': 'scale_x', 'GetZoomY': 'scale_y',
    'GetRotationZ': 'rotation',
}


class ActorClock:
    """A command clock shared by an actor's RecordingActor and the mod
    recorder, both keyed to the same load-time timeline in seconds."""

    def __init__(self, seconds: float = 0.0):
        self._t = float(seconds)

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += max(0.0, float(seconds))

    def reset(self, seconds: float) -> None:
        self._t = float(seconds)

    def beat(self, to_seconds) -> float:
        """Best-effort inverse: modfiles rarely read the beat back at
        load, so an identity in seconds is enough for the permissive
        getter; kept for GAMESTATE:GetSongBeat callers."""
        return self._t


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class RecordingActor:
    """One recorder per drawable actor; pokes append Keyframes onto
    per-property lists keyed to the shared clock."""

    def __init__(self, clock: ActorClock):
        self._clock = clock
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_EASING
        self._frames: dict = {}
        self._current: dict = {}

    def keyframes(self) -> dict:
        return {prop: kfs for prop, kfs in self._frames.items() if kfs}

    def read(self, verb: str):
        prop = _SCALAR_GETTERS.get(verb)
        return None if prop is None else self.get(prop)

    def get(self, prop: str) -> float:
        return self._current.get(prop, (_REST.get(prop, 0.0),))[0]

    def poke(self, verb: str, args: list) -> None:
        arg0 = args[0] if args else None
        match verb:
            case v if v in _TWEEN_EASING:
                self._open_tween(_as_float(arg0, 0.0), _TWEEN_EASING[v])
            case v if v in _SCALAR_SETTERS:
                self._set_positional(_SCALAR_SETTERS[v], args)
            case v if v in _FANOUT_SETTERS:
                value = _as_float(arg0)
                for prop in _FANOUT_SETTERS[v]:
                    self._emit_value(prop, value)
            case v if v in _ADD_SETTERS:
                self._add_scalar(_ADD_SETTERS[v], _as_float(arg0))
            case 'sleep':
                self._sleep(_as_float(arg0, 0.0))
            case 'diffuse':
                self._diffuse(args)
            case 'finishtweening':
                self._finish_tweening()
            case 'stoptweening' | 'hurrytweening':
                self._stop_tweening()
            case 'visible':
                self._visibility(_as_float(arg0, 1.0) == 0.0)
            # Any other verb pokes state we do not model; ignore it.

    # -- tween interval bookkeeping ---------------------------------------

    def _open_tween(self, duration: float, easing: int) -> None:
        self._clock.advance(self._pending_dur)
        self._pending_dur = max(0.0, duration)
        self._pending_ease = easing

    def _sleep(self, duration: float) -> None:
        self._clock.advance(self._pending_dur + max(0.0, duration))
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_EASING

    def _finish_tweening(self) -> None:
        self._clock.advance(self._pending_dur)
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_EASING

    def _stop_tweening(self) -> None:
        self._pending_dur = 0.0
        self._pending_ease = _FALLBACK_EASING

    # -- property emission ------------------------------------------------

    def _emit(self, prop: str, values: tuple) -> None:
        self._frames.setdefault(prop, []).append(
            Keyframe(self._clock.now(), values, self._pending_dur,
                     self._pending_ease))
        self._current[prop] = values

    def _set_positional(self, props, args) -> None:
        """Emit one keyframe per (prop, positional arg) pair, so `xy(x, y)`
        fills x then y; a missing or non-numeric arg leaves that prop
        untouched."""
        for prop, arg in zip(props, args):
            self._emit_value(prop, _as_float(arg))

    def _emit_value(self, prop, value) -> None:
        if value is not None:
            self._emit(prop, (value,))

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
