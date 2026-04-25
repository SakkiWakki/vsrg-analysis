"""Chart-field background painters: bg, lanes, judgment windows.

All QColor / QPen / QBrush objects for constant palette entries are
built once at module load. Per-frame work is reduced to the minimum
number of QPainter state changes and draw calls.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPen


# ── palette: built once at import, never again ──────────────────────

_BG_BASE     = QColor(14, 14, 16)
_LANE_BG     = QColor(22, 22, 24)
_LANE_LINE_C = QColor(40, 40, 44)
_WHITE       = QColor(255, 255, 255)
_DEATH_RED   = QColor(220, 50, 50)

_BG_BRUSH       = QBrush(_BG_BASE)
_LANE_BG_BRUSH  = QBrush(_LANE_BG)
_LANE_LINE_PEN  = QPen(_LANE_LINE_C, 1)
_JUDGE_LINE_PEN = QPen(_WHITE, 2)
_DEATH_LINE_PEN = QPen(_DEATH_RED, 2)
_NO_PEN         = QPen(QColor(0, 0, 0, 0))

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
    """Single fullscreen fill ; one draw call, zero allocations."""
    painter.fillRect(QRectF(0, 0, ctx.player.W, ctx.player.H), _BG_BRUSH)


def draw_lanes(ctx, painter):
    """Fill all lane backgrounds in one composite rect, then draw the
    column dividers in a single pen pass.

    Before: 2 * keycount draw calls (fillRect + drawLine per lane)
    After:  1 fillRect + (keycount + 1) drawLines, one setPen."""
    p = ctx.player
    x0 = ctx.x0
    lane_w = ctx.lane_w
    kc = p.keycount
    H = p.H

    # One rect covers every lane
    # TODO: Custom behaviour
    total_w = lane_w * kc
    painter.fillRect(QRectF(x0, 0, total_w, H), _LANE_BG_BRUSH)

    # Set pen once
    painter.setPen(_LANE_LINE_PEN)
    x = x0
    for _ in range(kc + 1):
        painter.drawLine(QPointF(x, 0.0), QPointF(x, H))
        x += lane_w


def draw_judgment(ctx, painter):
    p = ctx.player
    t_now = ctx.t_now
    sps = ctx.scroll_speed
    x0 = ctx.x0
    judge_y = ctx.judge_y
    field_w = p.keycount * ctx.lane_w

    # Window shading. Anchor both edges at judge_y and ask the SV engine
    # for the render-space distance across each half independently --
    # going through ctx.time_to_y instead would route through the
    # smoothed cull-space predictor (visual_cum_now), which drifts
    # relative to t_now frame-to-frame and makes the band jitter.
    sv = p.sv_render if (p.sv_enabled and p._sv_engine.enabled) else None
    painter.setPen(_NO_PEN)
    for name, w in reversed(p.windows):
        if sv is not None:
            half_top = sv.sv_distance(t_now - w, t_now) * sps
            half_bot = sv.sv_distance(t_now, t_now + w) * sps
        else:
            half_top = half_bot = w * sps
        painter.setBrush(_judge_brush(p.judge_colors[name]))
        painter.drawRect(QRectF(x0, judge_y - half_top, field_w,
                                half_top + half_bot))

    # Judgment line
    painter.setPen(_JUDGE_LINE_PEN)
    x_end = x0 + field_w
    painter.drawLine(QPointF(x0, judge_y), QPointF(x_end, judge_y))

    # Death line
    death_t = p.replay.get('death_time')
    if death_t is not None:
        y = ctx.time_to_y(float(death_t))
        painter.setPen(_DEATH_LINE_PEN)
        painter.drawLine(QPointF(x0, y), QPointF(x_end, y))