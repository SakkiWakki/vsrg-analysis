"""Chart-extras layer: mines, lifts, fakes, ghost taps, and miss-holds.

Note-like elements outside the main replay stream. Shares the same cull
window with the notes layer (single SV-space bisect per bucket).

All visible-index computation is vectorized with NumPy ; the Python loop
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


# Hold-mine body stroke (Quaver): connects the armed span so the player
# can see how long the lane stays hot.
_MINE_BODY_COLOR = (170, 60, 60)
_MINE_BODY_WIDTH = 3


def draw_mines(ctx: RenderContext, painter) -> None:
    p = ctx.player
    _draw_chart_sprites(ctx, painter,
                        p.notes.mine_times, p.notes.mine_cols, p.notes.mine_sv,
                        p.notes.mine_until,
                        sprite='mine', keyed=False, rows=p.notes.mine_rows)
    _draw_hold_mine_spans(ctx, painter)
    _draw_mine_detonations(ctx, painter)


def _draw_hold_mine_spans(ctx, painter) -> None:
    """Body stroke + end sprite for Quaver hold mines (finite
    `mine_end_times`; NaN marks a point mine). The head sprite is
    already drawn by the shared mine pass."""
    p = ctx.player
    n = p.notes
    ends = n.mine_end_times
    if not ends.size:
        return

    margin = ctx.screen_margin
    lo, hi = -margin, p.H + margin
    end_pm = ctx.sprite_cache.get('mine', ctx)
    for k in np.nonzero(np.isfinite(ends))[0]:
        c = int(n.mine_cols[k])
        if c >= p.keycount:
            continue
        y_head = _chart_sprite_y(ctx, float(n.mine_times[k]), n.mine_sv, k)
        y_end = ctx.time_to_y(float(ends[k]))
        if (y_head < lo and y_end < lo) or (y_head > hi and y_end > hi):
            continue
        lx = ctx.lane_x(c)
        draw_lane_line(painter, _MINE_BODY_COLOR, lx, ctx.lane_width(c),
                       y_head, y_end, _MINE_BODY_WIDTH)
        painter.drawPixmap(
            QPointF(float(lx), float(y_end - end_pm.height() / 2)), end_pm)


def _draw_mine_detonations(ctx, painter) -> None:
    """Miss-X over every mine the player actually set off (Quaver
    scores each detonation as a Miss). Appears from the press time
    onward so scrubbing back before the press hides it again."""
    p = ctx.player
    n = p.notes
    idx = n.mine_hit_idx
    if not idx.size:
        return

    shown = np.nonzero(n.mine_hit_press <= float(ctx.t_now))[0]
    if not shown.size:
        return
    pm = ctx.sprite_cache.get('miss_x', ctx, jcolor=p.judge_colors['miss'])
    margin = ctx.screen_margin
    for k in shown:
        i = int(idx[k])
        c = int(n.mine_cols[i])
        if c >= p.keycount:
            continue
        y = _chart_sprite_y(ctx, float(n.mine_times[i]), n.mine_sv, i)
        if not (-margin <= y <= p.H + margin):
            continue
        painter.drawPixmap(
            QPointF(float(ctx.lane_x(c)), float(y - pm.height() / 2)), pm)


def draw_lifts(ctx: RenderContext, painter) -> None:
    p = ctx.player
    _draw_chart_sprites(ctx, painter,
                        p.notes.lift_times, p.notes.lift_cols, p.notes.lift_sv,
                        p.notes.lift_until,
                        sprite='lift', keyed=True)


def draw_fakes(ctx: RenderContext, painter) -> None:
    p = ctx.player
    _draw_chart_sprites(ctx, painter,
                        p.notes.fake_times, p.notes.fake_cols, p.notes.fake_sv,
                        p.notes.fake_until,
                        sprite='fake', keyed=True)


def draw(ctx: RenderContext, painter) -> None:
    """Backwards-compat alias: draws mines + lifts + fakes in one pass.
    Retained for plugins that registered against the old combined layer
    before the per-type split."""
    draw_mines(ctx, painter)
    draw_lifts(ctx, painter)
    draw_fakes(ctx, painter)


def draw_ghost_taps(ctx: RenderContext, painter) -> None:
    p = ctx.player
    ghost_times = p.notes.ghost_times
    if not ghost_times.size:
        ctx.visible_ghost_taps = []
        return

    search = p.notes.ghost_sv_times if ctx.use_sv_space else ghost_times
    indices = _cull_indices(search, ctx.target_lo, ctx.target_hi)
    indices = indices[p.notes.ghost_cols[indices] < p.keycount]

    pm = ctx.sprite_cache.get('ghost_tap', ctx)

    lane_x = ctx.lane_x
    lane_w = ctx.lane_w
    time_to_y = ctx.time_to_y

    visible = []
    for k in indices:
        col = int(p.notes.ghost_cols[k])
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
    if not p.notes.miss_hold_press.size:
        ctx.visible_miss_holds = []
        return

    indices = _visible_miss_hold_indices(ctx)
    indices = indices[p.notes.miss_hold_cols[indices] < p.keycount]

    red = p.judge_colors['miss']
    lane_x, lane_w = ctx.lane_x, ctx.lane_w
    time_to_y = ctx.time_to_y
    H = p.H
    tick_pm = ctx.sprite_cache.get('tick', ctx, color=red)

    visible = []
    for k in indices:
        col = int(p.notes.miss_hold_cols[k])
        y_press = time_to_y(float(p.notes.miss_hold_press[k]))
        y_release = time_to_y(float(p.notes.miss_hold_release[k]))

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
    Uses bisect on the sorted array ; O(log n) + O(visible)."""
    i = bisect.bisect_left(sorted_keys, lo)
    j = bisect.bisect_right(sorted_keys, hi)
    return np.arange(i, j, dtype=np.intp)


def _draw_chart_sprites(ctx, painter, times, cols, sv_times, active_until, *,
                        sprite, keyed, rows=None):
    """Cull + blit a chart-stream sprite bucket (mines/lifts/fakes).

    - `sprite` ; sprite cache key
    - `keyed`  ; True when the sprite keys on `col` (lifts, fakes).
      False for palette-independent glyphs like mines.
    - `rows`   ; per-note beat rows (parallel to `times`); when the game
      supplies a per-note mod provider (`ctx.chart_stream_offsets`), these
      feed it so the stream picks up the same NotITG modfield displacement
      the column's taps get. None or no provider = the plain lane rail.

    Every chart-stream pixmap blits centered on its anchor y (square
    mines, head-shaped lifts/fakes), so the y-offset is `pm.height() / 2`
    in both cases.
    """
    if not times.size:
        return
    search = sv_times if (ctx.use_sv_space and sv_times.size) else times
    indices = _cull_indices(search, ctx.target_lo, ctx.target_hi)
    indices = indices[cols[indices] < ctx.player.keycount]
    if active_until.size:
        indices = indices[float(ctx.t_now) < active_until[indices]]
    if not indices.size:
        return

    cache = ctx.sprite_cache
    lane_x = ctx.lane_x
    ys = np.array([_chart_sprite_y(ctx, float(times[k]), sv_times, k)
                   for k in indices], dtype=np.float64)
    mods = _chart_stream_mods(ctx, cols[indices], ys, rows, indices)

    # Both pixmap shapes anchor `y` at the pixmap's vertical center
    # (mines' `(lane_w, lane_w)` square -> `lane_w/2`; head-shaped
    # pixmaps -> `pm.height() / 2`, which adapts to whichever skin is
    # active without the blit site needing to know).
    for n, k in enumerate(indices):
        c = int(cols[k])
        pm = cache.get(sprite, ctx, col=c) if keyed else cache.get(sprite, ctx)
        _blit_chart_sprite(painter, pm, float(lane_x(c)), float(ys[n]), c, ctx,
                           mods.at(n) if mods is not None else None)


def _chart_stream_mods(ctx, cols, ys, rows, indices):
    """The stream's per-note mod offsets, or None when no provider is wired
    or the stream carries no rows. `rows` is parallel to the full stream, so
    it is gathered down to the visible `indices`."""
    provider = getattr(ctx, 'chart_stream_offsets', None)
    if provider is None or rows is None:
        return None
    dx, dy, rot, zoom, alpha = provider(cols, ys, np.asarray(rows)[indices])
    return _StreamMods(dx, dy, rot, zoom, alpha)


class _StreamMods:
    """Column-major mod arrays with a per-index accessor. Small enough to
    stay a thin holder; `at` returns the tuple the blit site consumes."""

    __slots__ = ('dx', 'dy', 'rot', 'zoom', 'alpha')

    def __init__(self, dx, dy, rot, zoom, alpha):
        self.dx, self.dy, self.rot, self.zoom, self.alpha = dx, dy, rot, zoom, alpha

    def at(self, n):
        return (float(self.dx[n]), float(self.dy[n]), float(self.rot[n]),
                float(self.zoom[n]), float(self.alpha[n]))


def _blit_chart_sprite(painter, pm, lx, y, col, ctx, mod) -> None:
    """Blit one chart-stream pixmap centered on `(lx, y)`'s anchor. When
    `mod` is present ((dx, dy, rot, zoom, alpha)) the sprite is displaced and
    drawn inside the same head-center rotate/zoom/alpha bracket the notes
    layer uses (see layers/notes.py `_draw_view`), so a modded mine spins,
    scales, and fades exactly like a tap in its column."""
    y_top = y - pm.height() / 2
    if mod is None:
        painter.drawPixmap(QPointF(lx, float(y_top)), pm)
        return
    dx, dy, rot, zoom, alpha = mod
    lx += dx
    y += dy
    y_top += dy
    faded = alpha < 1.0
    transformed = rot or zoom != 1.0
    if not faded and not transformed:
        painter.drawPixmap(QPointF(float(lx), float(y_top)), pm)
        return
    if faded and alpha < 1.0 / 255.0:
        return
    painter.save()
    if faded:
        painter.setOpacity(painter.opacity() * alpha)
    if transformed:
        cx = lx + pm.width() / 2.0
        painter.translate(cx, float(y))
        if rot:
            painter.rotate(rot)
        if zoom != 1.0:
            painter.scale(zoom, zoom)
        painter.translate(-cx, -float(y))
    painter.drawPixmap(QPointF(float(lx), float(y_top)), pm)
    painter.restore()


def _chart_sprite_y(ctx, t, sv_times, k):
    # Need this otherwise warps don't render correctly
    if ctx.use_sv_space and sv_times.size:
        dist = (
            (float(sv_times[k]) - float(ctx.frame.visual_cum_now))
            * float(ctx.frame.render_multiplier)
        )
        return ctx.judge_y - dist * float(ctx.scroll_speed)
    return ctx.time_to_y(t)


def _visible_miss_hold_indices(ctx) -> np.ndarray:
    """Vectorized miss-hold culling: press-release spans that could
    touch the visible window."""
    p = ctx.player
    if ctx.use_sv_space:
        press = p.notes.miss_hold_press_sv
        release = p.notes.miss_hold_release_sv
        max_dur = p.notes.miss_hold_max_sv_dur
    else:
        press = p.notes.miss_hold_press
        release = p.notes.miss_hold_release
        max_dur = p.notes.miss_hold_max_dur

    # coarse cull by press time (accounts for max duration overshoot)
    i = bisect.bisect_left(press, ctx.target_lo - max_dur)
    j = bisect.bisect_right(press, ctx.target_hi)
    candidates = np.arange(i, j, dtype=np.intp)
    if not candidates.size:
        return candidates

    # fine cull: release must reach into the visible window
    return candidates[release[candidates] >= ctx.target_lo]
