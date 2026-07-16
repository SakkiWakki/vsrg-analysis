"""StoryboardEffect: draws a compiled Storyboard through the effects
pipeline.

Each storyboard layer becomes one z-slot draw (below or above the
chart depending on the layer's z), so playfield transforms, flashes,
and shaders compose around storyboard content exactly like any other
effect; draws are clipped to the chart region by the existing effect
machinery.

Design-space mapping: the design rect scales uniformly into the chart
region ('min' fit letterboxes, 'height' fit matches osu's 480-tall
convention where wide regions extend sideways) and everything else is
painted in design units under that transform. Element state is
sampled per frame from the IR timelines, so scrubbing is stateless.

Assets load lazily into a per-effect pixmap cache; a missing file
skips its element with one warning. Sprite tinting (color != white)
uses a cache of pre-tinted pixmaps keyed by quantized color so common
strobe tints don't re-rasterize every frame.
"""
from __future__ import annotations

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QPainter, QPen,
                           QPixmap)

from analysis.player.render.effects.base import EffectFrame

_MIN_VISIBLE_ALPHA = 1.0 / 255.0
_TINT_QUANT_LEVELS = 32


def _design_transform(storyboard, chart_rect):
    """(scale, offset_x, offset_y) mapping design space to screen."""
    x, y, w, h = chart_rect
    if storyboard.fit == 'height':
        k = h / storyboard.design_h
    else:
        k = min(w / storyboard.design_w, h / storyboard.design_h)
    ox = x + (w - storyboard.design_w * k) / 2.0
    oy = y + (h - storyboard.design_h * k) / 2.0
    return k, ox, oy


def _quantize_color(r, g, b) -> tuple:
    q = _TINT_QUANT_LEVELS - 1
    return (round(r * q), round(g * q), round(b * q))


def _is_white(quantized) -> bool:
    q = _TINT_QUANT_LEVELS - 1
    return quantized == (q, q, q)


class StoryboardEffect:
    def __init__(self, storyboard):
        elements = sorted(storyboard.elements,
                          key=lambda e: (e.z, e.z_index, e.t_start))
        self._sb = storyboard
        self._elements = tuple(elements)
        self._starts = np.array([e.t_start for e in elements],
                                dtype=np.float64)
        self._ends = np.array([e.t_end for e in elements], dtype=np.float64)
        zs = sorted({e.z for e in elements})
        self._layer_indices = tuple(
            (z, tuple(i for i, e in enumerate(elements) if e.z == z))
            for z in zs)
        self._pixmaps = {}
        self._tinted = {}
        self._text_metrics = {}

    def __bool__(self):
        return bool(self._elements)

    def at(self, ctx) -> EffectFrame | None:
        t = float(ctx.t_now)
        active = (self._starts <= t) & (t < self._ends)
        if not active.any():
            return None

        draws = []
        for z, indices in self._layer_indices:
            live = tuple(i for i in indices if active[i])
            if live:
                draws.append((z, self._layer_draw(live, t)))
        if not draws:
            return None
        return EffectFrame(draws=tuple(draws))

    def _layer_draw(self, indices, t):
        def draw(ctx, painter):
            k, ox, oy = _design_transform(self._sb, ctx.chart_rect)
            for i in indices:
                self._paint_element(painter, self._elements[i], t, k, ox, oy)
        return draw

    # -- element painting -------------------------------------------------

    def _paint_element(self, painter, el, t, k, ox, oy) -> None:
        alpha = el.sample('alpha', t)[0]
        if alpha < _MIN_VISIBLE_ALPHA:
            return
        size = self._element_size(el, t)
        if size is None:
            return
        w, h = size

        (x,) = el.sample('x', t)
        (y,) = el.sample('y', t)
        (rotation,) = el.sample('rotation', t)
        (sx,) = el.sample('scale_x', t)
        (sy,) = el.sample('scale_y', t)
        if el.flip_h:
            sx = -sx
        if el.flip_v:
            sy = -sy
        if sx == 0.0 or sy == 0.0:
            return

        ax, ay = el.anchor
        painter.save()
        painter.translate(ox + (ax * self._sb.design_w + x) * k,
                          oy + (ay * self._sb.design_h + y) * k)
        painter.scale(k, k)
        if rotation:
            painter.rotate(rotation)
        painter.scale(sx, sy)
        painter.translate(-el.origin[0] * w, -el.origin[1] * h)
        painter.setOpacity(min(1.0, alpha))
        if el.additive:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Plus)
        self._paint_kind(painter, el, t, w, h)
        painter.restore()

    def _paint_kind(self, painter, el, t, w, h) -> None:
        color = self._qcolor(el.sample('color', t))
        rect = QRectF(0.0, 0.0, w, h)
        match el.kind:
            case 'sprite' | 'frames':
                pm = self._tinted_pixmap(self._asset_at(el, t), color)
                painter.drawPixmap(rect, pm, QRectF(pm.rect()))
            case 'rect':
                painter.fillRect(rect, color)
            case 'ellipse':
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(rect)
            case 'outline_rect' | 'outline_ellipse':
                (border,) = el.sample('border', t)
                painter.setPen(QPen(color, max(0.1, border)))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                if el.kind == 'outline_rect':
                    painter.drawRect(rect)
                else:
                    painter.drawEllipse(rect)
            case 'text':
                font, metrics = self._font_for(el)
                painter.setFont(font)
                painter.setPen(color)
                painter.drawText(QPointF(0.0, metrics.ascent()), el.text)

    def _element_size(self, el, t):
        """Natural (w, h) in design units, or None when undrawable."""
        match el.kind:
            case 'sprite' | 'frames':
                pm = self._pixmap(self._asset_at(el, t))
                return (pm.width(), pm.height()) if pm is not None else None
            case 'text':
                _font, metrics = self._font_for(el)
                bounds = metrics.boundingRect(el.text)
                return (bounds.width(), metrics.height())
            case _:
                (w,) = el.sample('w', t)
                (h,) = el.sample('h', t)
                return (w, h) if w > 0 and h > 0 else None

    # -- asset caches -------------------------------------------------------

    def _asset_at(self, el, t) -> str | None:
        if el.kind != 'frames':
            return el.asset
        if not el.frames or el.frame_delay <= 0:
            return el.frames[0] if el.frames else None
        i = int((t - el.t_start) / el.frame_delay)
        if el.loop_forever:
            i %= len(el.frames)
        return el.frames[min(i, len(el.frames) - 1)]

    def _pixmap(self, path) -> QPixmap | None:
        if path is None:
            return None
        if path not in self._pixmaps:
            pm = QPixmap(path)
            if pm.isNull():
                print(f'storyboard asset missing/unreadable: {path}')
            self._pixmaps[path] = None if pm.isNull() else pm
        return self._pixmaps[path]

    def _tinted_pixmap(self, path, color: QColor) -> QPixmap:
        pm = self._pixmap(path)
        quantized = _quantize_color(color.redF(), color.greenF(),
                                    color.blueF())
        if _is_white(quantized):
            return pm

        key = (path, quantized)
        if key not in self._tinted:
            tinted = QPixmap(pm.size())
            tinted.fill(Qt.GlobalColor.transparent)
            p = QPainter(tinted)
            p.drawPixmap(0, 0, pm)
            p.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Multiply)
            p.fillRect(tinted.rect(), color)
            p.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_DestinationIn)
            p.drawPixmap(0, 0, pm)
            p.end()
            self._tinted[key] = tinted
        return self._tinted[key]

    def _font_for(self, el):
        key = el.font_px
        if key not in self._text_metrics:
            font = QFont()
            font.setPixelSize(max(1, int(el.font_px or 32)))
            self._text_metrics[key] = (font, QFontMetricsF(font))
        return self._text_metrics[key]

    def _qcolor(self, rgb) -> QColor:
        r, g, b = rgb
        return QColor.fromRgbF(max(0.0, min(1.0, r)),
                               max(0.0, min(1.0, g)),
                               max(0.0, min(1.0, b)))
