"""Playfield move / scale / rotate as one affine effect.

Ports fluXis's `Playfield.updatePositionScale` + the rotate/scale
transforms. The move event carries `(x, y, z)`; fluXis places a camera
at z=-100 and projects:

    scale_z = 100 / max(1, z + 100)
    pos     = (xy - cam_xy) * scale_z + cam_xy          # cam_xy = (0, 0)
    Scale   = scale_z * AnimationScale

so `z` pushes the field into / out of the screen (a perspective
depth-zoom about screen center) on top of the planar `(x, y)` slide.
`x`/`y` are screen pixels centered at 0 in fluXis's reference draw
resolution; we scale by `window / ref` so the motion is
screen-proportional at any size.

Not fluXis-specific: any game emitting move/scale/rotate keyframe
streams (the `.ffx`-style shape) reuses this effect.
"""
from __future__ import annotations

from PySide6.QtGui import QTransform

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import EventTimeline, Keyframe

# osu.Framework DrawSizePreservingFillContainer reference (fluXis's
# gameplay draw space); event x/y are pixels in this space.
_REF_W = 1366.0
_REF_H = 768.0
# Camera distance for the z-depth projection (fluXis: camera at z=-100).
_CAM_Z = 100.0
# Safety clamp: never translate the field center past the window edge.
_MAX_OFFSET_FRAC = 0.5


def _keyframes(events, value_keys, rest):
    out = []
    for e in events or []:
        if not isinstance(e, dict):
            continue
        values = tuple(float(e.get(k, r))
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
        self._move = EventTimeline(
            _keyframes(move, ('x', 'y', 'z'), (0.0, 0.0, 0.0)),
            rest=(0.0, 0.0, 0.0))
        self._scale = EventTimeline(
            _keyframes(scale, ('x', 'y'), (1.0, 1.0)), rest=(1.0, 1.0))
        self._rotate = EventTimeline(
            _keyframes(rotate, ('roll',), (0.0,)), rest=(0.0,))

    def __bool__(self):
        return bool(self._move or self._scale or self._rotate)

    def at(self, ctx) -> EffectFrame | None:
        t = ctx.t_now
        mx, my, mz = self._move.sample(t)
        sx, sy = self._scale.sample(t)
        (roll,) = self._rotate.sample(t)
        z_scale = _CAM_Z / max(1.0, mz + _CAM_Z)   # perspective depth-zoom
        if (mx == 0 and my == 0 and z_scale == 1.0
                and sx == 1 and sy == 1 and roll == 0):
            return None

        cx = ctx.x0 + ctx.player.keycount * ctx.lane_w / 2.0
        cy = ctx.judge_y
        rx, ry, W, H = ctx.chart_rect
        scx, scy = rx + W / 2.0, ry + H / 2.0

        # Screen-proportional translation, clamped so the field center
        # can't leave the window.
        dx = mx * W / _REF_W
        dy = my * H / _REF_H
        dx = max(-W * _MAX_OFFSET_FRAC, min(W * _MAX_OFFSET_FRAC, dx))
        dy = max(-H * _MAX_OFFSET_FRAC, min(H * _MAX_OFFSET_FRAC, dy))

        transform = QTransform()
        # z-depth: perspective zoom about screen center (fluXis camera).
        transform.translate(scx, scy)
        transform.scale(z_scale, z_scale)
        transform.translate(-scx, -scy)
        # planar move + rotate + authored scale, about the receptor.
        transform.translate(cx + dx, cy + dy)
        transform.rotate(roll)
        transform.scale(sx, sy)
        transform.translate(-cx, -cy)
        return EffectFrame(transform=transform)
