"""Chart-field background painters: bg, lanes, judgment windows.

All QColor / QPen / QBrush objects for constant palette entries are
built once at module load. Per-frame work is reduced to the minimum
number of QPainter state changes and draw calls.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPen

import numpy as np


# ── palette: built once at import, never again ──────────────────────

_BG_BASE     = QColor(14, 14, 16)
_LANE_BG     = QColor(22, 22, 24)
_LANE_LINE_C = QColor(40, 40, 44)
_WHITE       = QColor(255, 255, 255)
_DEATH_RED   = QColor(220, 50, 50)

_BG_BRUSH       = QBrush(_BG_BASE)
_ENGINE_BLACK_BRUSH = QBrush(QColor(0, 0, 0))
_LANE_BG_BRUSH  = QBrush(_LANE_BG)
_LANE_LINE_PEN  = QPen(_LANE_LINE_C, 1)
_JUDGE_LINE_PEN = QPen(_WHITE, 2)
_DEATH_LINE_PEN = QPen(_DEATH_RED, 2)
_NO_PEN         = QPen(QColor(0, 0, 0, 0))

# Per-column receptor notch: a thin white bar filling most of the lane
# width, centered on the lane center at the hit line. Height and the
# fraction of the lane it spans are the only geometry knobs; the mark
# picks up its position/rotation/zoom/alpha from ctx.receptor_offsets.
_RECEPTOR_H          = 4.0
_RECEPTOR_LANE_FRAC  = 0.82
_RECEPTOR_BRUSH      = QBrush(_WHITE)

# Judgment-window overlay brushes are keyed by (r, g, b) and built on
# first use ; the color set depends on the active judge scheme which
# can change at runtime, but within a session the same ~5 colors repeat
# every frame.
_JUDGE_BRUSHES: dict[tuple, QBrush] = {}


def _judge_brush(color: tuple) -> QBrush:
    cached = _JUDGE_BRUSHES.get(color)
    if cached is not None:
        return cached
    brush = QBrush(QColor(color[0], color[1], color[2], 24))
    _JUDGE_BRUSHES[color] = brush
    return brush


# ── draw functions ──────────────────────────────────────────────────

def draw_background(ctx, painter):
    """Single fullscreen fill ; one draw call, zero allocations.

    Transparent-field games clear TRUE BLACK, matching the engine's
    framebuffer clear: the app's near-black canvas tint (14,14,16) is
    invisible live but ADDITIVE capture copies sum it - a dozen stacked
    AFT generations turned the tint into visible gray phantom quads
    (gat 2's cyriak wash)."""
    adapter = getattr(ctx.player, '_adapter', None)
    brush = (_ENGINE_BLACK_BRUSH
             if adapter is not None and adapter.transparent_field()
             else _BG_BRUSH)
    painter.fillRect(QRectF(0, 0, ctx.player.W, ctx.player.H), brush)


def draw_lanes(ctx, painter):
    """Fill all lane backgrounds in one composite rect, then draw the
    column dividers in a single pen pass.

    Before: 2 * keycount draw calls (fillRect + drawLine per lane)
    After:  1 fillRect + (keycount + 1) drawLines, one setPen.

    Games with a transparent field (NotITG-style floating notes over
    the scene) skip the fills and dividers entirely; notes, judgment
    line, and press marks still draw."""
    p = ctx.player
    if p._adapter.transparent_field():
        return
    x0 = ctx.x0
    lane_w = ctx.lane_w
    kc = p.keycount
    H = p.H

    if ctx.lane_xs is None:
        # Static layout: one rect covers every lane, uniform dividers.
        painter.fillRect(QRectF(x0, 0, lane_w * kc, H), _LANE_BG_BRUSH)
        painter.setPen(_LANE_LINE_PEN)
        x = x0
        for _ in range(kc + 1):
            painter.drawLine(QPointF(x, 0.0), QPointF(x, H))
            x += lane_w
        return

    # Lane-switch layout: per-column animated x/width (collapsed lanes
    # have width ~0, so their fill vanishes and their divider merges
    # with the neighbor's).
    xs, ws = ctx.lane_xs, ctx.lane_ws
    for x, w in zip(xs, ws):
        if w > 0.5:
            painter.fillRect(QRectF(x, 0, w, H), _LANE_BG_BRUSH)
    painter.setPen(_LANE_LINE_PEN)
    for x in xs:
        painter.drawLine(QPointF(x, 0.0), QPointF(x, H))
    right = xs[-1] + ws[-1]
    painter.drawLine(QPointF(right, 0.0), QPointF(right, H))


def _field_span(ctx):
    """`(left_x, width)` of the visible playfield: the active-lane
    extent under a lane switch, else the full uniform field."""
    if ctx.lane_xs is None:
        return ctx.x0, ctx.player.keycount * ctx.lane_w
    left = min((x for x, w in zip(ctx.lane_xs, ctx.lane_ws) if w > 0.5),
               default=ctx.x0)
    right = max((x + w for x, w in zip(ctx.lane_xs, ctx.lane_ws) if w > 0.5),
                default=ctx.x0)
    return left, right - left


def _receptor_offsets(ctx, keycount):
    """Per-column receptor mod arrays in OUR pixel space, defaulting to
    the identity when the producer hasn't stashed anything on the ctx.

    Returns `(dx, dy, rotation_deg, zoom, alpha)`, each length keycount.
    The producer (games/*/note_mods) sets `ctx.receptor_offsets` to a
    dict of numpy arrays; we consume via getattr so an unmodded chart --
    every game by default -- pays only the zeros allocation."""
    offs = getattr(ctx, 'receptor_offsets', None)
    zeros = np.zeros(keycount, dtype=np.float64)
    if offs is None:
        return zeros, zeros, zeros, None, None
    return (offs.get('dx', zeros), offs.get('dy', zeros),
            offs.get('rotation_deg', zeros),
            offs.get('zoom', None), offs.get('alpha', None))


def _draw_receptor_mark(painter, cx, cy, lane_w, rotation_deg, zoom, alpha):
    """One per-column receptor notch centered at `(cx, cy)`.

    rotation/zoom are applied about the mark's own center through the
    painter transform, and only when non-identity, so unmodded marks
    draw with a single fillRect and no save/restore."""
    bar_w = lane_w * _RECEPTOR_LANE_FRAC
    rect = QRectF(-bar_w / 2.0, -_RECEPTOR_H / 2.0, bar_w, _RECEPTOR_H)
    transformed = rotation_deg or (zoom is not None and zoom != 1.0)
    faded = alpha is not None and alpha < 1.0

    if not transformed and not faded:
        painter.fillRect(rect.translated(cx, cy), _RECEPTOR_BRUSH)
        return

    painter.save()
    if faded:
        painter.setOpacity(painter.opacity() * max(0.0, alpha))
    painter.translate(cx, cy)
    if rotation_deg:
        painter.rotate(rotation_deg)
    if zoom is not None and zoom != 1.0:
        painter.scale(zoom, zoom)
    painter.fillRect(rect, _RECEPTOR_BRUSH)
    painter.restore()


def draw_judgment(ctx, painter):
    p = ctx.player
    t_now = ctx.t_now
    sps = ctx.scroll_speed
    judge_y = ctx.judge_y
    x0, field_w = _field_span(ctx)

    # Window shading. Anchor both edges at judge_y and ask the SV engine
    # for the render-space distance across each half independently --
    # going through ctx.time_to_y instead would route through the
    # smoothed cull-space predictor (visual_cum_now), which drifts
    # relative to t_now frame-to-frame and makes the band jitter.
    sv = p.sv_render if (p.sv_enabled and p._sv_engine.enabled) else None
    painter.setPen(_NO_PEN)
    bands = []
    for name, w in reversed(p.windows):
        if sv is not None:
            half_top = sv.sv_distance(t_now - w, t_now) * sps
            half_bot = sv.sv_distance(t_now, t_now + w) * sps
        else:
            half_top = half_bot = w * sps
        bands.append((_judge_brush(p.judge_colors[name]), half_top, half_bot))

    bars = p._adapter.receptor_style() != 'line'
    per_column = bars and getattr(ctx, 'receptor_offsets', None) is not None
    if not per_column:
        for brush, half_top, half_bot in bands:
            painter.setBrush(brush)
            painter.drawRect(QRectF(x0, judge_y - half_top, field_w,
                                    half_top + half_bot))

    x_end = x0 + field_w
    if not bars:
        # Legacy single full-width line across the field.
        painter.setPen(_JUDGE_LINE_PEN)
        painter.drawLine(QPointF(x0, judge_y), QPointF(x_end, judge_y))
    elif per_column:
        # Receptors are displaced per column, so the window coloring
        # adheres to each receptor's own frame instead of staying behind
        # as a full-width band at the untransformed judge line.
        _draw_column_judgments(ctx, painter, judge_y, bands)
    else:
        _draw_receptors(ctx, painter, judge_y)

    # Death line
    death_t = p.replay.get('death_time')
    if death_t is not None:
        y = ctx.time_to_y(float(death_t))
        painter.setPen(_DEATH_LINE_PEN)
        painter.drawLine(QPointF(x0, y), QPointF(x_end, y))


def _draw_column_judgments(ctx, painter, judge_y, bands):
    """Window coloring + receptor notch per column, drawn together in the
    receptor's local frame (its lane center + mod displacement, rotation
    and zoom about its own center) so the judgment coloring rides every
    receptor transform. Band widths span the lane; each column's bands
    draw under its own notch."""
    kc = ctx.keycount
    dx, dy, rot, zoom, alpha = _receptor_offsets(ctx, kc)
    painter.setPen(_NO_PEN)
    for col in range(kc):
        lane_w = ctx.lane_width(col)
        if lane_w <= 0.5:
            continue
        cx = ctx.lane_center(col) + float(dx[col])
        cy = judge_y + float(dy[col])
        col_alpha = None if alpha is None else float(alpha[col])
        col_zoom = None if zoom is None else float(zoom[col])

        painter.save()
        if col_alpha is not None and col_alpha < 1.0:
            painter.setOpacity(painter.opacity() * max(0.0, col_alpha))
        painter.translate(cx, cy)
        if rot[col]:
            painter.rotate(float(rot[col]))
        if col_zoom is not None and col_zoom != 1.0:
            painter.scale(col_zoom, col_zoom)
        for brush, half_top, half_bot in bands:
            painter.setBrush(brush)
            painter.drawRect(QRectF(-lane_w / 2.0, -half_top, lane_w,
                                    half_top + half_bot))
        bar_w = lane_w * _RECEPTOR_LANE_FRAC
        painter.fillRect(QRectF(-bar_w / 2.0, -_RECEPTOR_H / 2.0, bar_w,
                                _RECEPTOR_H), _RECEPTOR_BRUSH)
        painter.restore()


def _draw_receptors(ctx, painter, judge_y):
    """Per-column receptor notches. Each rides its lane center
    (`ctx.lane_x` + width, so lane switches and animated widths track
    for free) plus the per-column receptor mod displacement, and lives
    inside the effect transform bracket so every field transform carries
    it too. Fill-only, no pen state."""
    kc = ctx.keycount
    dx, dy, rot, zoom, alpha = _receptor_offsets(ctx, kc)
    painter.setPen(_NO_PEN)
    for col in range(kc):
        lane_w = ctx.lane_width(col)
        if lane_w <= 0.5:
            continue
        cx = ctx.lane_center(col) + float(dx[col])
        cy = judge_y + float(dy[col])
        _draw_receptor_mark(painter, cx, cy, lane_w, float(rot[col]),
                            None if zoom is None else float(zoom[col]),
                            None if alpha is None else float(alpha[col]))