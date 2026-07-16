"""Scene-wide camera from `.ffx`-style camera events.

Ports fluXis's CameraContainer: a centre-anchored container wrapping
the entire gameplay scene (background, playfields, storyboard layers;
only the pulse ring and foreground flash sit outside). The proxy's
position is applied NEGATED (`Position = -proxy.Position`), so moving
the camera right slides the scene left, and scale/rotate pivot on the
screen centre.

Camera move x/y are pixels in the 1366x768 reference space, scaled to
the chart region; unlike the playfield move there is no offset clamp,
because cameras are authored to leave and return while the chart-rect
clip keeps everything contained. Emitted on the `scene_transform`
channel, which the renderer applies around every draw below the
scene-top z cutoff.

Not fluXis-specific: any game emitting scene transform keyframes
(NotITG whole-field mods) lands on the same channel.
"""
from __future__ import annotations

from PySide6.QtGui import QTransform

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.ref_space import REF_H, REF_W
from analysis.player.render.effects.timeline import (
    EventTimeline, keyframes_from_events)


class CameraEffect:
    def __init__(self, *, move=None, scale=None, rotate=None):
        self._move = EventTimeline(
            keyframes_from_events(move, ('x', 'y'), (0.0, 0.0)),
            rest=(0.0, 0.0))
        self._scale = EventTimeline(
            keyframes_from_events(scale, ('scale',), (1.0,)), rest=(1.0,))
        self._rotate = EventTimeline(
            keyframes_from_events(rotate, ('roll',), (0.0,)), rest=(0.0,))

    def __bool__(self):
        return bool(self._move or self._scale or self._rotate)

    def at(self, ctx) -> EffectFrame | None:
        t = ctx.t_now
        mx, my = self._move.sample(t)
        (s,) = self._scale.sample(t)
        (roll,) = self._rotate.sample(t)
        if mx == 0 and my == 0 and s == 1 and roll == 0:
            return None

        rx, ry, w, h = ctx.chart_rect
        cx, cy = rx + w / 2.0, ry + h / 2.0
        dx = -mx * w / REF_W
        dy = -my * h / REF_H

        transform = QTransform()
        transform.translate(cx + dx, cy + dy)
        transform.rotate(roll)
        transform.scale(s, s)
        transform.translate(-cx, -cy)
        return EffectFrame(scene_transform=transform)
