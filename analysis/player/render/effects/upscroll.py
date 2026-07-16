"""Upscroll presentation: mirror the field vertically.

Games whose native orientation is receptors-on-top (NotITG/ITG) emit
this as a standing effect; it is exactly ITG's `reverse` mod as a
transform. Flipping the chart-layer group about the chart region's
vertical center puts the judgment line at the top and notes rising
from the bottom, with zero changes to the y math anywhere else. The
adapter appends it LAST so the flip wraps the other field transforms.
"""
from __future__ import annotations

from PySide6.QtGui import QTransform

from analysis.player.render.effects.base import EffectFrame


class UpscrollEffect:
    def __bool__(self):
        return True

    def at(self, ctx) -> EffectFrame:
        _rx, ry, _w, h = ctx.chart_rect
        cy = ry + h / 2.0
        flip = QTransform()
        flip.translate(0.0, cy)
        flip.scale(1.0, -1.0)
        flip.translate(0.0, -cy)
        return EffectFrame(transform=flip)
