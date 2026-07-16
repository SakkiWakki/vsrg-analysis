"""Map-background image drawn behind the playfield.

A `below`-draw effect (z < 0): the image is loaded once, cover-fit to
the window, and dimmed so the chart stays readable. This is the first
concrete instance of the "background storyboard" layer -- it lives in
the effects pipeline so playfield transforms above it compose cleanly
and future storyboard draws share the same slot.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QPixmap

from analysis.player.render.effects.base import EffectFrame

# Background z; well below any storyboard/field content.
_BG_Z = -1000
# Dim so notes and the receptor line stay legible over busy art.
_DIM = 0.35


class MapBackgroundEffect:
    def __init__(self, path):
        self._path = path
        self._pixmap = None      # lazy-loaded on first paint (needs a QApp)
        self._loaded = False

    def __bool__(self):
        return bool(self._path)

    def _pixmap_or_none(self):
        if not self._loaded:
            self._loaded = True
            pm = QPixmap(self._path)
            self._pixmap = pm if not pm.isNull() else None
        return self._pixmap

    def at(self, ctx) -> EffectFrame | None:
        if self._pixmap_or_none() is None:
            return None
        return EffectFrame(draws=((_BG_Z, self._draw),))

    def _draw(self, ctx, painter):
        pm = self._pixmap
        x, y, w, h = ctx.chart_rect
        src = _cover_src_rect(pm.width(), pm.height(), w, h)
        painter.save()
        painter.setOpacity(_DIM)
        painter.drawPixmap(QRectF(x, y, w, h), pm, src)
        painter.restore()


def _cover_src_rect(pw, ph, w, h) -> QRectF:
    """Source rect that cover-fits a `pw x ph` image into `w x h`:
    scale to fill, center-crop the overflowing axis."""
    if pw <= 0 or ph <= 0:
        return QRectF(0, 0, pw, ph)
    scale = max(w / pw, h / ph)
    crop_w = w / scale
    crop_h = h / scale
    return QRectF((pw - crop_w) / 2, (ph - crop_h) / 2, crop_w, crop_h)
