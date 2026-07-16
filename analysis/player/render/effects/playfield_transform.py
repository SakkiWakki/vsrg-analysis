"""Playfield move / scale / rotate as one affine effect.

fluXis applies these about the playfield center. Translations are in
fluXis's virtual playfield pixels (a 512-wide reference field); we
scale x/y by `field_w / 512` so the motion tracks our lane geometry
regardless of window size.

Prby should be generalized to other games
"""
from __future__ import annotations

from PySide6.QtGui import QTransform

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import EventTimeline, Keyframe

_FLUXIS_FIELD_REF = 512.0


def _keyframes(events, value_keys, rest, *, scale=1.0):
    out = []
    for e in events or []:
        if not isinstance(e, dict):
            continue
        values = tuple(float(e.get(k, r)) * scale
                       for k, r in zip(value_keys, rest))
        out.append(Keyframe(
            t=float(e.get('time', 0.0)) / 1000.0,
            values=values,
            duration=max(0.0, float(e.get('duration', 0.0))) / 1000.0,
            easing=int(e.get('ease', 0)),
        ))
    return out


class PlayfieldTransformEffect:
    """Composes move + scale + rotate streams into one QTransform about
    the field center. Any stream may be empty."""

    def __init__(self, *, move=None, scale=None, rotate=None):
        self._move = EventTimeline(_keyframes(move, ('x', 'y'), (0.0, 0.0)),
                                   rest=(0.0, 0.0))
        self._scale = EventTimeline(
            _keyframes(scale, ('x', 'y'), (1.0, 1.0)), rest=(1.0, 1.0))
        self._rotate = EventTimeline(
            _keyframes(rotate, ('roll',), (0.0,)), rest=(0.0,))

    def __bool__(self):
        return bool(self._move or self._scale or self._rotate)

    def at(self, ctx) -> EffectFrame | None:
        t = ctx.t_now
        mx, my = self._move.sample(t)
        sx, sy = self._scale.sample(t)
        (roll,) = self._rotate.sample(t)
        if mx == 0 and my == 0 and sx == 1 and sy == 1 and roll == 0:
            return None

        cx = ctx.x0 + ctx.player.keycount * ctx.lane_w / 2.0
        cy = ctx.judge_y
        field_scale = ctx.player.keycount * ctx.lane_w / _FLUXIS_FIELD_REF

        transform = QTransform()
        transform.translate(cx + mx * field_scale, cy + my * field_scale)
        transform.rotate(roll)
        transform.scale(sx, sy)
        transform.translate(-cx, -cy)
        return EffectFrame(transform=transform)
