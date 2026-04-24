"""Chart-extras layer: mines, lifts, fakes, ghost taps, and miss-holds.

Note-like elements outside the main replay stream. Shares the same cull
window with the notes layer (single SV-space bisect per bucket).

All visible-index computation is vectorized with NumPy — the Python loop
only runs over notes that actually need drawing.
"""
from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

import numpy as np

from PySide6.QtCore import QPointF
from analysis.player.render.primitives import _line, _rect

if TYPE_CHECKING:
    from analysis.player.render.context import RenderContext


# ── public entry points ─────────────────────────────────────────────


def draw_mines(ctx: RenderContext, painter) -> None:
    p = ctx.player
    _draw_chart_sprites(ctx, painter,
                        p._mine_times, p._mine_cols, p.notes.mine_sv,
                        sprite='mine', keyed=False, y_center=True)


def draw_lifts(ctx: RenderContext, painter) -> None:
    p = ctx.player
    _draw_chart_sprites(ctx, painter,
                        p._lift_times, p._lift_cols, p.notes.lift_sv,
                        sprite='lift', keyed=True, y_center=False)


def draw_fakes(ctx: RenderContext, painter) -> None:
    p = ctx.player
    _draw_chart_sprites(ctx, painter,
                        p._fake_times, p._fake_cols, p.notes.fake_sv,
                        sprite='fake', keyed=True, y_center=False)


def draw(ctx: RenderContext, painter) -> None:
    """Backwards-compat alias: draws mines + lifts + fakes in one pass.
    Retained for plugins that registered against the old combined layer
    before the per-type split."""
    draw_mines(ctx, painter)
    draw_lifts(ctx, painter)
    draw_fakes(ctx, painter)


def draw_ghost_taps(ctx: RenderContext, painter) -> None:
    p = ctx.player
    ghost_times = p._ghost_times
    if not ghost_times.size:
        ctx.visible_ghost_taps = []
        return

    search = p._ghost_sv_times if ctx.use_sv_space else ghost_times
    indices = _cull_indices(search, ctx.target_lo, ctx.target_hi)
    indices = indices[p._ghost_cols[indices] < p.keycount]

    pm = ctx.sprite_cache.get('ghost_tap', ctx)

    lane_x = ctx.lane_x
    lane_w = ctx.lane_w
    time_to_y = ctx.time_to_y

    visible = []
    for k in indices:
        col = int(p._ghost_cols[k])
        y = time_to_y(float(ghost_times[k]))

        cx = lane_x(col) + lane_w / 2
        painter.drawPixmap(
            QPointF(float(cx - pm.width() / 2),
                    float(y - pm.height() / 2)),
            pm,
        )
        visible.append(k)

    ctx.visible_ghost_taps = visible


def draw_miss_holds(ctx: RenderContext, painter) -> None:
    p = ctx.player
    if not p._miss_hold_press.size:
        ctx.visible_miss_holds = []
        return

    indices = _visible_miss_hold_indices(ctx)
    indices = indices[p._miss_hold_cols[indices] < p.keycount]

    from PySide6.QtCore import QPointF
    red = p.judge_colors['miss']
    lane_x, lane_w = ctx.lane_x, ctx.lane_w
    time_to_y = ctx.time_to_y
    H = p.H
    tick_pm = ctx.sprite_cache.get('tick', ctx, color=red)

    visible = []
    for k in indices:
        col = int(p._miss_hold_cols[k])
        y_press = time_to_y(float(p._miss_hold_press[k]))
        y_release = time_to_y(float(p._miss_hold_release[k]))

        top, bot = min(y_press, y_release), max(y_press, y_release)
        if bot < 0 or top > H:
            continue
        top = max(0.0, top)
        bot = min(float(H), bot)

        lx = int(lane_x(col))
        # Vertical stroke (vector, per-note endpoints) + two cached
        # ticks at press and release times.
        draw_lane_line(painter, red, lx, lane_w, top, bot, width=2)
        painter.drawPixmap(QPointF(float(lx), float(y_press - 2)), tick_pm)
        painter.drawPixmap(QPointF(float(lx), float(y_release - 2)), tick_pm)
        visible.append(k)
    ctx.visible_miss_holds = visible


# ── drawing helpers ──────────────────────────────────────────────────


def default_miss_hold_stroke(painter, lx, lane_w, y_top, y_bot,
                              y_press, y_release, color):
    draw_lane_line(painter, color, lx, lane_w, y_top, y_bot, width=2)
    draw_tick(painter, color, lx, y_press, lane_w)
    draw_tick(painter, color, lx, y_release, lane_w)


def draw_tick(painter, color, lane_x, y, lane_w):
    """Horizontal tick centered in the lane at `y`."""
    _rect(painter, color, (lane_x + 8, y - 2, lane_w - 16, 4))


def draw_lane_line(painter, color, lane_x, lane_w, y0, y1, width=1):
    """Vertical line down the lane centerline from `y0` to `y1`."""
    cx = lane_x + lane_w / 2
    _line(painter, color, (cx, y0), (cx, y1), width)


# ── internals ────────────────────────────────────────────────────────


def _cull_indices(sorted_keys: np.ndarray,
                  lo: float, hi: float) -> np.ndarray:
    """Return an int array of indices whose keys fall within [lo, hi).
    Uses bisect on the sorted array — O(log n) + O(visible)."""
    i = bisect.bisect_left(sorted_keys, lo)
    j = bisect.bisect_right(sorted_keys, hi)
    return np.arange(i, j, dtype=np.intp)


def _draw_chart_sprites(ctx, painter, times, cols, sv_times, *,
                        sprite, keyed, y_center):
    """Cull + blit a chart-stream sprite bucket (mines/lifts/fakes).

    - `sprite`   — sprite cache key
    - `keyed`    — True when the sprite keys on `col` (lifts, fakes).
      False for palette-independent glyphs like mines.
    - `y_center` — True when the sprite's pixmap is `(lane_w, lane_w)`
      and should blit centered on `y` (mines). False for head-shaped
      pixmaps `(lane_w, note_h)` that blit at `y - note_h / 2`.
    """
    if not times.size:
        return
    search = sv_times if (ctx.use_sv_space and sv_times.size) else times
    indices = _cull_indices(search, ctx.target_lo, ctx.target_hi)
    indices = indices[cols[indices] < ctx.player.keycount]
    if not indices.size:
        return

    from PySide6.QtCore import QPointF
    cache = ctx.sprite_cache
    lane_x = ctx.lane_x
    note_h = ctx.note_h
    lane_w = ctx.lane_w
    time_to_y = ctx.time_to_y

    # Mines' sprite is a square with side == lane_w, centered on y.
    # Other chart sprites use the head shape (lane_w, note_h), blitted
    # top-left at (lx, y - note_h / 2).
    y_offset = lane_w / 2 if y_center else note_h / 2

    if keyed:
        for k in indices:
            c = int(cols[k])
            pm = cache.get(sprite, ctx, col=c)
            y = time_to_y(float(times[k]))
            painter.drawPixmap(QPointF(float(lane_x(c)),
                                        float(y - y_offset)), pm)
    else:
        pm = cache.get(sprite, ctx)
        for k in indices:
            c = int(cols[k])
            y = time_to_y(float(times[k]))
            painter.drawPixmap(QPointF(float(lane_x(c)),
                                        float(y - y_offset)), pm)


def _visible_miss_hold_indices(ctx) -> np.ndarray:
    """Vectorized miss-hold culling: press-release spans that could
    touch the visible window."""
    p = ctx.player
    if ctx.use_sv_space:
        press = p._miss_hold_press_sv
        release = p._miss_hold_release_sv
        max_dur = p._miss_hold_max_sv_dur
    else:
        press = p._miss_hold_press
        release = p._miss_hold_release
        max_dur = p._miss_hold_max_dur

    # coarse cull by press time (accounts for max duration overshoot)
    i = bisect.bisect_left(press, ctx.target_lo - max_dur)
    j = bisect.bisect_right(press, ctx.target_hi)
    candidates = np.arange(i, j, dtype=np.intp)
    if not candidates.size:
        return candidates

    # fine cull: release must reach into the visible window
    return candidates[release[candidates] >= ctx.target_lo]