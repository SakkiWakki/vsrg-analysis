"""Chart-extras layer: replay overlays (ghost taps, miss-holds) plus
compat entry points for the chart-stream note layers.

Mines, lifts, and fakes render through the unified stream pipeline in
layers/notes.py (one candidate axis, one mod kernel, one draw bracket
shared with taps); the `draw_mines`/`draw_lifts`/`draw_fakes` functions
here are thin wrappers kept for plugins that registered against this
module. Ghost taps and miss-holds are replay overlays, not note
records; their vectorized culls live here, and the note prepass turns
them into views (`ghost_views` / `miss_hold_views`) so the raster layer
and the drawable feed draw the same list rather than each culling.
"""
from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from PySide6.QtCore import QPointF
from analysis.player.init.notes_model import stream_groups_or_none
from analysis.player.render.primitives import _line, _rect

if TYPE_CHECKING:
    from analysis.player.render.context import RenderContext


# ── chart-stream compat wrappers ─────────────────────────────────────
# The imports are deferred to break the module cycle: layers/notes.py
# imports this module at load. Only plugin-registered callers land
# here; the builtin note-type layers reference the notes.py drawers
# directly.


def draw_mines(ctx: RenderContext, painter) -> None:
    from analysis.player.render.layers import notes
    notes.draw_mines(ctx, painter)


def draw_lifts(ctx: RenderContext, painter) -> None:
    from analysis.player.render.layers import notes
    notes.draw_lifts(ctx, painter)


def draw_fakes(ctx: RenderContext, painter) -> None:
    from analysis.player.render.layers import notes
    notes.draw_fakes(ctx, painter)


def draw_mine_detonations(ctx, painter) -> None:
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
    groups = stream_groups_or_none(n.mine_groups)
    for k in shown:
        i = int(idx[k])
        c = int(n.mine_cols[i])
        if c >= p.keycount:
            continue
        y = _chart_sprite_y(ctx, n.mine_times, n.mine_sv, groups, i)
        if not (-margin <= y <= p.H + margin):
            continue
        painter.drawPixmap(
            QPointF(float(ctx.lane_x(c)), float(y - pm.height() / 2)), pm)


def draw(ctx: RenderContext, painter) -> None:
    """Backwards-compat alias: draws mines + lifts + fakes in one pass.
    Retained for plugins that registered against the old combined layer
    before the per-type split."""
    draw_mines(ctx, painter)
    draw_lifts(ctx, painter)
    draw_fakes(ctx, painter)


def draw_ghost_taps(ctx: RenderContext, painter) -> None:
    pm = ctx.sprite_cache.get('ghost_tap', ctx)
    lane_center = ctx.lane_center
    visible = []
    for view in getattr(ctx, 'ghost_views', ()):
        cx = lane_center(view.col)
        painter.drawPixmap(
            QPointF(float(cx - pm.width() / 2),
                    float(view.y - pm.height() / 2)),
            pm,
        )
        visible.append(view.k)
    ctx.visible_ghost_taps = visible


def draw_miss_holds(ctx: RenderContext, painter) -> None:
    p = ctx.player
    red = p.judge_colors['miss']
    tick_pm = ctx.sprite_cache.get('tick', ctx, color=red)

    visible = []
    for view in getattr(ctx, 'miss_hold_views', ()):
        lx = int(ctx.lane_x(view.col))
        # Vertical stroke (vector, per-note endpoints) + two cached
        # ticks at press and release times.
        draw_lane_line(painter, red, lx, ctx.lane_w, view.top, view.bot,
                       width=2)
        painter.drawPixmap(QPointF(float(lx), float(view.y_press - 2)),
                           tick_pm)
        painter.drawPixmap(QPointF(float(lx), float(view.y_release - 2)),
                           tick_pm)
        visible.append(view.k)
    ctx.visible_miss_holds = visible


class GhostView(NamedTuple):
    """A press with no note under it: a point on a lane, and the record
    it came from."""

    k: int
    col: int
    y: float


class MissHoldView(NamedTuple):
    """A hold the player never held: the stretch between where it should
    have been pressed and released.

    `top`/`bot` are that stretch clamped to the screen. It rides the view
    rather than each drawer reworking it, because a missed hold can run
    many screens and both backends have coordinate limits - a rule two
    drawers must not be able to disagree about."""

    k: int
    col: int
    y_press: float
    y_release: float
    top: float
    bot: float


def ghost_views(ctx) -> list:
    """This frame's visible ghost taps, culled and placed.

    Built by the note prepass so every drawer - the raster layer and the
    drawable feed alike - reads the same list instead of each running the
    cull itself."""
    p = ctx.player
    ghost_times = p.notes.ghost_times
    if not ghost_times.size:
        return []
    search = p.notes.ghost_sv_times if ctx.use_sv_space else ghost_times
    indices = _cull_indices(search, ctx.target_lo, ctx.target_hi)
    indices = indices[p.notes.ghost_cols[indices] < p.keycount]
    return [GhostView(int(k), int(p.notes.ghost_cols[k]),
                      float(ctx.time_to_y(float(ghost_times[k]))))
            for k in indices]


def miss_hold_views(ctx) -> list:
    """This frame's visible miss-hold spans, culled and placed (see
    `ghost_views` for why the prepass owns it)."""
    p = ctx.player
    if not p.notes.miss_hold_press.size:
        return []
    indices = _visible_miss_hold_indices(ctx)
    indices = indices[p.notes.miss_hold_cols[indices] < p.keycount]
    time_to_y = ctx.time_to_y
    height = float(p.H)
    views = []
    for k in indices:
        y_press = float(time_to_y(float(p.notes.miss_hold_press[k])))
        y_release = float(time_to_y(float(p.notes.miss_hold_release[k])))
        top, bot = min(y_press, y_release), max(y_press, y_release)
        if bot < 0 or top > height:
            continue
        views.append(MissHoldView(int(k), int(p.notes.miss_hold_cols[k]),
                                  y_press, y_release,
                                  max(0.0, top), min(height, bot)))
    return views


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


def _chart_stream_ys(ctx, times, sv_times, groups, indices):
    """Screen y for chart-stream sprites at `indices`, routed through
    the same projection primitive taps use (`batch_time_to_y`) so every
    stream inherits the taps' direction/rate/per-group handling.

    The cached `sv_times` projection rides along as `cum` (row-space
    for beat-space engines ; without it, old negative-BPM warp aliases
    render at the wrong position)."""
    cum = sv_times[indices] if sv_times.size else None
    sub_groups = groups[indices] if groups is not None else None
    return ctx.player.batch_time_to_y(times[indices], ctx.frame,
                                       groups=sub_groups, cum=cum)


def _chart_sprite_y(ctx, times, sv_times, groups, k) -> float:
    """Single-entry `_chart_stream_ys` for the low-count draw sites
    (mine detonations)."""
    idx = np.array([k], dtype=np.intp)
    return float(_chart_stream_ys(ctx, times, sv_times, groups, idx)[0])


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
