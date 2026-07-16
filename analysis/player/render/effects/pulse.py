"""Playfield-edge pulse from `.ffx` pulse events.

Ports fluXis's PulseEffect: a masked, white-bordered container over the
playfield whose border thickness animates per event -- growing 0 ->
`width` over `in-percent` of the duration, then shrinking back to 0 over
the rest, both with the event's easing (default Out). Masking clips the
border inward from the region edges, so it reads as a bright frame that
blooms and recedes around the field.

fluXis drives this with absolute transform sequences and
RemoveCompletedTransforms = false, so the most recent event wins where
they overlap and the border holds at 0 between events. We reproduce that
by picking the last event started at or before `t_now` and evaluating its
two-phase width; outside every event's span the border is 0 and the
effect contributes nothing.
"""
from __future__ import annotations

from bisect import bisect_right

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPen

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import bloom

_Z = 800
_EASE_OUT = 1
_BORDER_COLOR = QColor(255, 255, 255)
_MIN_VISIBLE_PX = 0.5


def _pulse(event) -> tuple | None:
    duration = max(0.0, float(event.get('duration', 0.0) or 0.0)) / 1000.0
    width = float(event.get('width', 32.0) or 0.0)
    if duration <= 0.0 or width <= 0.0:
        return None
    return (float(event.get('time', 0.0)) / 1000.0, duration, width,
            max(0.0, min(1.0, float(event.get('in-percent', 0.0) or 0.0))),
            int(event.get('easing', _EASE_OUT)))


def _border_width(pulse, t_now) -> float:
    start, duration, width, in_pct, easing = pulse
    return bloom(t_now - start, duration, in_pct, easing, rest=0.0, peak=width)


class PulseEffect:
    def __init__(self, events):
        pulses = (_pulse(e) for e in events or [] if isinstance(e, dict))
        self._pulses = sorted(p for p in pulses if p is not None)
        self._starts = [p[0] for p in self._pulses]

    def __bool__(self):
        return bool(self._pulses)

    def at(self, ctx) -> EffectFrame | None:
        idx = bisect_right(self._starts, float(ctx.t_now)) - 1
        if idx < 0:
            return None
        border = _border_width(self._pulses[idx], ctx.t_now)
        if border < _MIN_VISIBLE_PX:
            return None
        return EffectFrame(draws=((_Z, self._draw(border)),))

    @staticmethod
    def _draw(border):
        def draw(ctx, painter):
            x, y, w, h = ctx.chart_rect
            painter.save()
            painter.setPen(QPen(_BORDER_COLOR, border))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            inset = border / 2.0
            painter.drawRect(QRectF(x + inset, y + inset,
                                    w - border, h - border))
            painter.restore()
        return draw
