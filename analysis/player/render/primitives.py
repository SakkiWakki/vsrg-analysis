"""Cached Qt drawing primitives used by every renderer layer.

Splitting the primitive helpers out of qt_renderer.py keeps the layer
files (notes, chart_extras, sidebar, ...) from importing their own
cache state — every caller goes through the same `_qcolor`/`_qpen`/
`_qbrush` interners so the same (r,g,b) tuple produces one QColor
instance across the whole process. The cache is module-global because
Qt objects are interchangeable across paint operations and we never
want to be rebuilding them.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QBrush, QColor, QPen




# Qt primitive cache. Every frame used to spin up a fresh QColor/QPen/QBrush
# per draw call — on heavy Etterna charts that's ~thousands of transient Qt
# objects per second, enough to visibly jitter frame timing. Intern by tuple
# so the same (r,g,b[,a]) / (color, width) / color keys reuse one instance
# forever. Color palette is finite (lane palette, judge colors, theme
# constants) so the cache stabilizes within the first second of play.
_QCOLOR_CACHE: dict[tuple, QColor] = {}
_QPEN_CACHE: dict[tuple, QPen] = {}
_QBRUSH_CACHE: dict[tuple, QBrush] = {}
_NO_PEN = QPen(QColor(0, 0, 0, 0))
_NO_BRUSH = QBrush(QColor(0, 0, 0, 0))


def _qcolor(color):
    if isinstance(color, QColor):
        return color
    key = tuple(color)
    cached = _QCOLOR_CACHE.get(key)
    if cached is not None:
        return cached
    if len(key) == 4:
        qc = QColor(int(key[0]), int(key[1]), int(key[2]), int(key[3]))
    else:
        qc = QColor(int(key[0]), int(key[1]), int(key[2]))
    _QCOLOR_CACHE[key] = qc
    return qc


def _qpen(color, width=1):
    key = (tuple(color) if not isinstance(color, QColor) else id(color),
           int(width))
    cached = _QPEN_CACHE.get(key)
    if cached is not None:
        return cached
    pen = QPen(_qcolor(color), int(width))
    _QPEN_CACHE[key] = pen
    return pen


def _qbrush(color):
    key = tuple(color) if not isinstance(color, QColor) else id(color)
    cached = _QBRUSH_CACHE.get(key)
    if cached is not None:
        return cached
    brush = QBrush(_qcolor(color))
    _QBRUSH_CACHE[key] = brush
    return brush



def _rect_tuple(rect):
    if hasattr(rect, 'getRect'):
        return rect.getRect()
    return tuple(rect)


def _fmt_num(x, decimals=2):
    """Render a scroll-speed scalar as an int when it's near-integer,
    else fixed-width decimal. Cross-mode translations produce values
    like 35.0212 that collapse to '35' but remain distinguishable from
    an actual 35 after the user nudges."""
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f'{x:.{decimals}f}'


def _line(painter, color, start, end, width=1):
    painter.setPen(_qpen(color, width))
    painter.drawLine(QPointF(start[0], start[1]),
                     QPointF(end[0], end[1]))


def _rect(painter, color, rect):
    x, y, w, h = _rect_tuple(rect)
    painter.setPen(_NO_PEN)
    painter.setBrush(_qbrush(color))
    painter.drawRect(QRectF(x, y, w, h))


def _rect_outline(painter, color, rect, width=1):
    x, y, w, h = _rect_tuple(rect)
    painter.setPen(_qpen(color, width))
    painter.setBrush(_NO_BRUSH)
    painter.drawRect(QRectF(x, y, w, h))


def _ellipse(painter, color, cx, cy, rx, ry):
    painter.setPen(_NO_PEN)
    painter.setBrush(_qbrush(color))
    painter.drawEllipse(QPointF(cx, cy), rx, ry)


def _ellipse_outline(painter, color, cx, cy, rx, ry, width=1):
    painter.setPen(_qpen(color, width))
    painter.setBrush(_NO_BRUSH)
    painter.drawEllipse(QPointF(cx, cy), rx, ry)


def _text(painter, text, color, x, baseline):
    painter.setPen(_qpen(color, 1))
    painter.drawText(QPointF(x, baseline), str(text))
