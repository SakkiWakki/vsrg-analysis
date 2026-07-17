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
from analysis.player.render.storyboard.sprite_sheet import (frame_at_time,
                                                            frame_source_rect)

_MIN_VISIBLE_ALPHA = 1.0 / 255.0
_TINT_QUANT_LEVELS = 32


def _is_sheet(el) -> bool:
    """A sprite whose asset is an SM NxM grid (more than one frame)."""
    return el.sheet_cols * el.sheet_rows > 1


def _sheet_frame(el, t: float) -> int:
    """The frame index a sheet sprite shows at time `t`: the pinned frame
    from a recorded setstate/animate timeline when present, otherwise the
    auto-animated frame stepped through `sheet_states` on the effect
    clock (relative to the element's own start)."""
    if el.state_pin is not None:
        return int(round(el.state_pin.sample(t)[0]))
    return frame_at_time(el.sheet_states, t - el.t_start)


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


def _bitmaptext_width(font, text: str) -> float:
    """Total pen advance for `text` in an SM bitmap font (design px)."""
    return sum(font.advance(ord(char)) for char in text)


def _is_white_texture(path: str) -> bool:
    """SM's built-in solid-white texture. Quads/Sprites that fill a
    flat color reference the virtual asset name 'white' (no real file on
    disk), so it is synthesized rather than warned as missing."""
    return isinstance(path, str) and path.strip().lower() == 'white'


def _white_pixmap() -> QPixmap:
    """A 1x1 opaque-white pixmap; drawPixmap stretches it to the
    element's size, and the sprite tint path recolors it like any
    texture (a white base multiplies cleanly to the target color)."""
    pm = QPixmap(1, 1)
    pm.fill(Qt.GlobalColor.white)
    return pm


def _quantize_color(r, g, b) -> tuple:
    q = _TINT_QUANT_LEVELS - 1
    return (round(r * q), round(g * q), round(b * q))


def _is_white(quantized) -> bool:
    q = _TINT_QUANT_LEVELS - 1
    return quantized == (q, q, q)


class StoryboardEffect:
    def __init__(self, storyboard):
        # Only TOP-LEVEL elements own z-slots; a group draws its whole
        # subtree inside its own slot bracket, so nested children never
        # appear in the layer table.
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
                self._paint_element(painter, self._elements[i], t, k, ox, oy,
                                    self._sb.design_w, self._sb.design_h)
        return draw

    # -- element painting -------------------------------------------------

    def _paint_element(self, painter, el, t, k, ox, oy,
                       ref_w, ref_h, inherited_alpha=1.0) -> None:
        alpha = el.sample('alpha', t)[0] * inherited_alpha
        if alpha < _MIN_VISIBLE_ALPHA:
            return
        # A group has no natural size; it rotates/scales about its own
        # anchor + position point (zero-size origin), then draws children.
        size = (0.0, 0.0) if el.kind == 'group' else self._element_size(el, t)
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
        painter.translate(ox + (ax * ref_w + x) * k,
                          oy + (ay * ref_h + y) * k)
        painter.scale(k, k)
        if rotation:
            painter.rotate(rotation)
        painter.scale(sx, sy)
        painter.translate(-el.origin[0] * w, -el.origin[1] * h)
        painter.setOpacity(min(1.0, alpha))
        if el.additive:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Plus)
        if el.kind == 'group':
            self._paint_children(painter, el, t, alpha)
        else:
            self._paint_kind(painter, el, t, w, h)
        painter.restore()

    def _paint_children(self, painter, el, t, group_alpha) -> None:
        """Draw a group's subtree in the group's own transformed space.
        The group bracket already applied its translate/rotate/scale, so
        the painter origin is now the frame's local (0, 0): children
        position by raw (x, y) relative to it (SM ActorFrame semantics,
        a zero-size anchor box) at k=1. Each child re-checks its own
        window, so one outside [t_start, t_end) is skipped while siblings
        still draw; if the whole group is outside its window the caller
        never reaches here."""
        for child in el.children:
            if child.t_start <= t < child.t_end:
                self._paint_element(painter, child, t, 1.0, 0.0, 0.0,
                                    0.0, 0.0, group_alpha)

    def _paint_kind(self, painter, el, t, w, h) -> None:
        color = self._qcolor(el.sample('color', t))
        rect = QRectF(0.0, 0.0, w, h)
        match el.kind:
            case 'sprite' | 'frames':
                pm = self._tinted_pixmap(self._asset_at(el, t), color)
                src = self._source_rect(el, t, pm)
                painter.drawPixmap(rect, pm, src)
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
            case 'bitmaptext':
                self._paint_bitmaptext(painter, el, color)

    def _paint_bitmaptext(self, painter, el, color) -> None:
        """Composite `el.text` from its SM bitmap-font atlas: one glyph
        cell per character, each drawn centred on the advancing pen so
        ink lines up like StepMania's own drawing, tinted by `color`.
        The whole string is centred on the element origin (SM BitmapText
        default), matching the (0.5, 0.5) origin the compiler assigns."""
        atlas = self._pixmap(el.font.texture_path)
        if atlas is None:
            return
        glyphs = self._tinted_pixmap(el.font.texture_path, color)
        pen = -_bitmaptext_width(el.font, el.text) / 2.0
        cell_h = atlas.height() / el.font.rows
        top = -cell_h / 2.0
        for char in el.text:
            codepoint = ord(char)
            advance = el.font.advance(codepoint)
            cell = el.font.cell(codepoint, atlas.width(), atlas.height())
            if cell is not None:
                cx, cy, cw, ch = cell
                dest = QRectF(pen + (advance - cw) / 2.0, top, cw, ch)
                painter.drawPixmap(dest, glyphs, QRectF(cx, cy, cw, ch))
            pen += advance

    def _source_rect(self, el, t, pm) -> QRectF:
        """The region of `pm` this sprite draws: one grid cell for an SM
        NxM sheet (the current frame), else the whole pixmap."""
        if not _is_sheet(el):
            return QRectF(pm.rect())
        frame = _sheet_frame(el, t)
        x, y, w, h = frame_source_rect(frame, pm.width(), pm.height(),
                                       el.sheet_cols, el.sheet_rows)
        return QRectF(x, y, w, h)

    def _element_size(self, el, t):
        """Natural (w, h) in design units, or None when undrawable. A
        sheet sprite's natural size is ONE frame, not the whole sheet."""
        match el.kind:
            case 'sprite' | 'frames':
                pm = self._pixmap(self._asset_at(el, t))
                if pm is None:
                    return None
                if _is_sheet(el):
                    return (pm.width() / el.sheet_cols,
                            pm.height() / el.sheet_rows)
                return (pm.width(), pm.height())
            case 'text':
                _font, metrics = self._font_for(el)
                bounds = metrics.boundingRect(el.text)
                return (bounds.width(), metrics.height())
            case 'bitmaptext':
                atlas = self._pixmap(el.font.texture_path)
                if atlas is None:
                    return None
                return (_bitmaptext_width(el.font, el.text),
                        atlas.height() / el.font.rows)
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
            pm = (_white_pixmap() if _is_white_texture(path)
                  else QPixmap(path))
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
