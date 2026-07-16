"""Fullscreen color flashes from `.ffx`-style flash events.

Ports fluXis's FlashOverlay: at each event the overlay snaps to
`start-color`/`start-alpha`, then fades both to `end-color`/`end-alpha`
over the duration with the event's easing; later events override
earlier ones mid-fade (fluXis absolute transform sequences do the
same, which is exactly `EventTimeline`'s last-keyframe-wins sampling).
Effective opacity is the color's own alpha times the event alpha,
matching a Box's Colour.A x Drawable.Alpha.

fluXis splits flashes into two layers by the event's `background`
flag: in front of the playfield, or behind it but over the map
background. Those map to one draw above the chart layers and one
below (between MapBackgroundEffect's z=-1000 and the field).
"""
from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import EventTimeline, Keyframe

_Z_FRONT = 900
_Z_BACK = -500
# White, opaque color channel, zero event alpha: invisible until the
# first event, like FlashOverlay's initial Alpha = 0 box.
_REST = (1.0, 1.0, 1.0, 1.0, 0.0)
_MIN_VISIBLE_ALPHA = 1.0 / 255.0


def _rgba(raw) -> tuple:
    raw = raw if isinstance(raw, dict) else {}
    return tuple(float(raw.get(k, 1.0)) for k in 'RGBA')


def _keyframe(event) -> Keyframe:
    return Keyframe(
        t=float(event.get('time', 0.0)) / 1000.0,
        values=_rgba(event.get('end-color'))
               + (float(event.get('end-alpha', 0.0) or 0.0),),
        duration=max(0.0, float(event.get('duration', 0.0) or 0.0)) / 1000.0,
        easing=int(event.get('ease', 0) or 0),
        start=_rgba(event.get('start-color'))
              + (float(event.get('start-alpha', 1.0)),),
    )


class FlashEffect:
    def __init__(self, events):
        front = []
        back = []
        for event in events or []:
            if not isinstance(event, dict):
                continue
            side = back if event.get('background') else front
            side.append(_keyframe(event))
        self._layers = tuple(
            (z, EventTimeline(keyframes, rest=_REST))
            for z, keyframes in ((_Z_FRONT, front), (_Z_BACK, back))
            if keyframes)

    def __bool__(self):
        return bool(self._layers)

    def at(self, ctx) -> EffectFrame | None:
        draws = []
        for z, timeline in self._layers:
            r, g, b, color_a, alpha = timeline.sample(ctx.t_now)
            fill = self._fill_draw(r, g, b, color_a * alpha)
            if fill is not None:
                draws.append((z, fill))
        if not draws:
            return None
        return EffectFrame(draws=tuple(draws))

    @staticmethod
    def _fill_draw(r, g, b, alpha):
        if alpha < _MIN_VISIBLE_ALPHA:
            return None
        color = QColor.fromRgbF(max(0.0, min(1.0, r)),
                                max(0.0, min(1.0, g)),
                                max(0.0, min(1.0, b)),
                                min(1.0, alpha))
        return lambda ctx, painter: painter.fillRect(
            QRectF(*ctx.chart_rect), color)
