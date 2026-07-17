"""Notes layer: replay-stream taps and long-notes.

Renders note heads, LN bodies/tails, release guides, press marks, and
miss-X overlays for every visible candidate. Chart-stream notes (mines,
lifts, fakes, ghost taps, miss-holds) live in chart_extras.py.

Public API:
- `prepare(ctx)` builds `ctx.note_views` once per frame from the player
  candidate list (shared by the `taps` + `lns` layers so the per-note
  state is computed once).
- `draw_taps(ctx, painter)` / `draw_lns(ctx, painter)` are the per-layer
  drawers the `NoteType` registrations hand out.
- `NoteType` is the per-adapter note-kind spec; `default_note_types()`
  returns the full set every game defaults to.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QPainterPath, QPainterPathStroker,
                           QTransform)

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np

from analysis.player.notetypes import NT_TICK
from analysis.player.render.layers import chart_extras as _extras
from analysis.player.render.primitives import _NO_PEN

if TYPE_CHECKING:
    from analysis.player.render.render_context import RenderContext


# Shared vector constant for LN release-guide + press-mark strokes.
# Every other sprite color lives inside the sprite-cache rasterize
# callbacks (see `layers/note_sprites.py`).

# ── note state ───────────────────────────────────────────────────

@dataclass
class _NoteView:
    i: int
    col: int
    y: int
    y_end: int
    press_y: float    # cached y at press_t; precomputed batched per frame
    lx: int
    off: float
    press_t: float
    release_t: float | None
    rel_off: float | None
    end_t: float | None
    is_ln: bool
    is_roll: bool
    miss: bool
    state: str       # upcoming | tap | held | released | missed | missed_note
    note_color: tuple
    jcolor: tuple
    # Quaver: tail sprite flips when SV at the LN's end_time is negative.
    # Other games leave this False ; the renderer just skips the flip.
    flip_tail: bool = False
    # fluXis tick notes; drawn bright yellow via the 'tick' sprite state.
    is_tick: bool = False
    # Per-note mod alpha (NotITG stealth/hidden family); 1 = opaque.
    alpha: float = 1.0
    # Per-note mod rotation (deg) / zoom (multiplier), applied about the
    # head center. Defaults are the identity so unmodded notes pay
    # nothing; LN bodies/tails keep their position (only the head sprite
    # spins/scales -- documented in `_draw_view`).
    rotation_deg: float = 0.0
    zoom: float = 1.0
    # Quaver: under SV reversal an LN body covers a wider span than
    # head->tail. `body_min_y` / `body_max_y` are screen-y bounds of the
    # convex hull of cum positions over the LN's chart-time interval.
    # NaN means "use the legacy head/tail span" (every non-LN, non-Quaver
    # note, plus Quaver LNs whose SV doesn't reverse inside the body).
    body_min_y: float = float('nan')
    body_max_y: float = float('nan')
    # NotITG per-note-mod hold-body warp: (xs, ys) polyline (our px)
    # tracing this hold's body bent by drunk/wave/digital etc. None =
    # draw the straight head/tail rect (every mod-free frame + game).
    body_samples: object = None


def _ln_body_y_extent(ctx, i, p):
    """Screen-y bounds of an LN body, reconstructed each frame.

    Mirrors Quaver's `UpdateLongNoteSize`:
        start_cum = playhead_cum if t_now >= t_head else head_cum
        earliest = min(start_cum, tail_cum)
        latest   = max(start_cum, tail_cum)
        for (t_change, c_change) in waypoints:
            if t_change > max(t_now, t_head):
                earliest = min(earliest, c_change)
                latest   = max(latest,   c_change)

    Past sign-change points fall out of the body, so the bar shrinks
    live as the playhead crosses each reversal. Legacy-LN charts have
    empty waypoint arrays so the loop is a no-op and the body just
    spans `[head, tail]`.

    Returns `(NaN, NaN)` when the engine isn't in SV mode or this
    isn't a Quaver-cached LN ; the caller falls back to the simpler
    head/tail rect."""
    head_cum_arr = getattr(p, '_ln_head_cum', None)
    tail_cum_arr = getattr(p, '_ln_tail_cum', None)
    if head_cum_arr is None or tail_cum_arr is None:
        return float('nan'), float('nan')
    if i >= len(head_cum_arr) or not math.isfinite(head_cum_arr[i]):
        return float('nan'), float('nan')
    frame = ctx.frame
    if not getattr(frame, 'use_sv', False):
        return float('nan'), float('nan')

    engine = p._sv_engine
    groups = getattr(p, '_note_sv_groups', None)
    if groups is not None and hasattr(engine, 'cumulative_at_groups'):
        gid = str(groups[i])
        playhead_cum = float(engine.cumulative_at_groups(
            float(frame.raw_t), [gid])[0])
        mult = float(engine.render_multiplier_at_groups(
            float(frame.raw_t), [gid])[0])
    else:
        playhead_cum = float(frame.visual_cum_now)
        mult = float(frame.render_multiplier)

    head_t = float(p.times[i])
    t_now = ctx.t_now
    held = t_now >= head_t
    start_cum = playhead_cum if held else float(head_cum_arr[i])
    tail_cum = float(tail_cum_arr[i])
    earliest = min(start_cum, tail_cum)
    latest = max(start_cum, tail_cum)

    change_times = p._ln_change_times[i]
    change_cums = p._ln_change_cums[i]
    if change_times is not None and change_times.size:
        future_after = max(t_now, head_t)
        future = change_times > future_after
        if future.any():
            future_cums = change_cums[future]
            earliest = min(earliest, float(future_cums.min()))
            latest = max(latest, float(future_cums.max()))

    judge_y = ctx.judge_y
    scroll_speed = float(ctx.player.scroll_speed)
    y_lo = judge_y - (earliest - playhead_cum) * mult * scroll_speed
    y_hi = judge_y - (latest - playhead_cum) * mult * scroll_speed
    return min(y_lo, y_hi), max(y_lo, y_hi)


def _classify(ctx, press_t, release_t, is_ln, miss) -> str:
    if miss:
        return 'missed' if is_ln else 'missed_note'
    if not is_ln:
        return 'tap'
    if ctx.t_now < press_t:
        return 'upcoming'
    if ctx.t_now < release_t:
        return 'held'
    return 'released'


def _lane_width(ctx, col) -> float:
    """`ctx.lane_width(col)` when available, else the uniform `lane_w`.
    The fallback keeps narrowly-mocked test contexts (SimpleNamespace)
    working without a real RenderContext."""
    fn = getattr(ctx, 'lane_width', None)
    return fn(col) if fn is not None else ctx.lane_w


def _blit_lane_pixmap(ctx, painter, pm, lx, y_top, col) -> None:
    """Blit a lane-width sprite. During a lane switch the lane's width
    animates, so the sprite is squeezed horizontally and kept centered
    on the (animated) lane center -- matching fluXis, where notes ride
    the lane's center as it collapses, not its left edge."""
    w = _lane_width(ctx, col)
    if w == ctx.lane_w:
        painter.drawPixmap(QPointF(float(lx), float(y_top)), pm)
        return
    scale = w / ctx.lane_w if ctx.lane_w else 1.0
    draw_w = pm.width() * scale
    # `lx` is the lane's left edge; center the squeezed sprite in the
    # lane so it tracks the collapsing center instead of the edge.
    cx = lx + w / 2.0
    target = QRectF(float(cx - draw_w / 2), float(y_top),
                    draw_w, float(pm.height()))
    painter.drawPixmap(target, pm, QRectF(pm.rect()))


def _is_tick(p, i) -> bool:
    nts = getattr(p, 'notetypes', None)
    return nts is not None and i < len(nts) and int(nts[i]) == NT_TICK


def _build(ctx, i, pos) -> _NoteView | None:
    p = ctx.player
    col = p.notes.columns_list[i]
    if col >= p.keycount:
        return None
    if p.misses[i] and i < len(p.notes.miss_head_suppressed) \
            and p.notes.miss_head_suppressed[i]:
        return None

    note_t = p.times[i]
    end_t = p.notes.ln_tail_times[i]
    is_ln = not math.isnan(end_t)
    off = p.offsets[i]
    press_t = note_t + off

    rel_off = None
    release_t = None
    y_end = 0
    if is_ln:
        rel_off = p.hold_release_offsets.get((p.notes.noterows_list[i], col))
        release_t = end_t + (rel_off or 0.0)
        y_end = float(ctx.candidate_tail_y[pos])
    else:
        end_t = None

    flip_tail = False
    body_min_y = float('nan')
    body_max_y = float('nan')
    if is_ln:
        ln_flip = getattr(p, '_ln_tail_flip', None)
        if ln_flip is not None and i < len(ln_flip):
            flip_tail = bool(ln_flip[i])
        body_min_y, body_max_y = _ln_body_y_extent(ctx, i, p)
        # Quaver anchors the tail sprite at LatestHeldPosition (the
        # body's max-y boundary in down-scroll), not at the note's
        # end_time projection. When SV reverses near the LN end the two
        # diverge ; the body grows past end_t in cum-space and the tail
        # rides that boundary. `y_end` becomes whichever side of the
        # body is "deeper" along the scroll direction.
        if math.isfinite(body_max_y):
            head_y = float(ctx.candidate_head_y[pos])
            # Down-scroll: tail farther from judge_y = larger y. We pick
            # whichever extreme is on the opposite side from the head.
            y_end = body_max_y if head_y <= body_max_y else body_min_y

    mod_dx = getattr(ctx, 'candidate_dx', None)
    mod_alpha = getattr(ctx, 'candidate_alpha', None)
    mod_rot = getattr(ctx, 'candidate_rot_deg', None)
    mod_zoom = getattr(ctx, 'candidate_zoom', None)
    body_samples = None
    if is_ln:
        samples = getattr(ctx, 'hold_body_samples', None)
        if samples is not None:
            body_samples = samples.get(pos)
    return _NoteView(
        i=i, col=col,
        y=float(ctx.candidate_head_y[pos]), y_end=y_end,
        press_y=float(ctx.candidate_press_y[pos]),
        lx=int(ctx.lane_x(col)
               + (mod_dx[pos] if mod_dx is not None else 0.0)),
        alpha=float(mod_alpha[pos]) if mod_alpha is not None else 1.0,
        rotation_deg=float(mod_rot[pos]) if mod_rot is not None else 0.0,
        zoom=float(mod_zoom[pos]) if mod_zoom is not None else 1.0,
        off=off, press_t=press_t,
        release_t=release_t, rel_off=rel_off, end_t=end_t,
        is_ln=is_ln,
        is_roll=bool(is_ln and p.notes.roll_head_keys
                     and (p.notes.noterows_list[i], col) in p.notes.roll_head_keys),
        is_tick=_is_tick(p, i),
        miss=bool(p.misses[i]),
        state=_classify(ctx, press_t, release_t, is_ln, p.misses[i]),
        note_color=p.palette[col],
        jcolor=p.judge_colors[p.note_judges[i]],
        flip_tail=flip_tail,
        body_min_y=body_min_y,
        body_max_y=body_max_y,
        body_samples=body_samples,
    )


# ── public entry points ──────────────────────────────────────────

def prepare(ctx) -> None:
    """Build every per-candidate `_NoteView` once. The `taps` and `lns`
    layer drawers read from `ctx.note_views` ; splitting the previous
    combined loop into two layer passes would otherwise rebuild each
    view twice."""
    views: list[_NoteView | None] = []
    for pos, i in enumerate(ctx.candidates):
        views.append(_build(ctx, i, pos))
    ctx.note_views = views


def _draw_view(ctx, painter, n, draw_fn) -> None:
    """Run one view's drawer, honoring its per-note mod alpha, rotation,
    and zoom. The save/restore pair only exists for modded notes, so
    unmodded charts pay nothing.

    Rotation/zoom apply about the head center via the painter transform.
    For an LN this spins/scales the head sprite only; the body and tail
    are drawn at their own (already-displaced) positions inside the same
    bracket, so under a large rotation the head detaches from the body
    -- an accepted simplification (per-note LN-body deformation would
    need the body to be rebuilt in the rotated frame)."""
    faded = n.alpha < 1.0
    transformed = n.rotation_deg or n.zoom != 1.0
    if not faded and not transformed:
        draw_fn(ctx, painter, n)
        return
    if faded and n.alpha < 1.0 / 255.0:
        return

    painter.save()
    if faded:
        painter.setOpacity(painter.opacity() * n.alpha)
    if transformed:
        cx = n.lx + _lane_width(ctx, n.col) / 2.0
        painter.translate(cx, float(n.y))
        if n.rotation_deg:
            painter.rotate(n.rotation_deg)
        if n.zoom != 1.0:
            painter.scale(n.zoom, n.zoom)
        painter.translate(-cx, -float(n.y))
    draw_fn(ctx, painter, n)
    painter.restore()


def _draw_alpha_only(ctx, painter, n, draw_fn) -> None:
    """Like `_draw_view` but honors the mod alpha without the head-only
    rotation/zoom transform -- used for LN bodies/tails, which keep
    their scroll positions while the head spins/scales."""
    if n.alpha >= 1.0:
        draw_fn(ctx, painter, n)
        return
    if n.alpha < 1.0 / 255.0:
        return
    painter.save()
    painter.setOpacity(painter.opacity() * n.alpha)
    draw_fn(ctx, painter, n)
    painter.restore()


def draw_taps(ctx, painter) -> None:
    """Draw taps (non-LN notes) from the prebuilt views."""
    for n in ctx.note_views:
        if n is None or n.is_ln:
            continue
        _draw_view(ctx, painter, n, _draw_replay_note)


def draw_lns(ctx, painter) -> None:
    """Draw LN bodies, tails, release guides, and heads from the
    prebuilt views.

    Per-note rotation/zoom is a head-only transform (see `_draw_view`):
    the body + tail draw under alpha only so they keep their scroll
    positions, and the head sprite gets the full transform bracket."""
    for n in ctx.note_views:
        if n is None or not n.is_ln:
            continue
        _draw_alpha_only(ctx, painter, n, _draw_ln)
        _draw_view(ctx, painter, n, _draw_replay_note)


def _draw_replay_note(ctx, painter, n) -> None:
    if _draw_head(ctx, painter, n):
        _draw_press_mark(ctx, painter, n)
        if n.miss:
            _draw_miss_x(ctx, painter, n)


# ── LN drawing ───────────────────────────────────────────────────

def _draw_ln(ctx, painter, n):
    p = ctx.player
    hide = p.press_hide

    # ── body fill ──
    span = _ln_body_span(ctx, n, hide)
    if span is not None:
        top, bot, body_state = span
        if bot > top:
            if n.body_samples is not None:
                _draw_ln_body_warped(ctx, painter, n, top, bot, body_state)
            else:
                _draw_ln_body_tile(ctx, painter, n, top, bot, body_state)

    # ── tail sprite ──
    on_screen = -ctx.screen_margin <= n.y_end <= p.H + ctx.screen_margin
    hidden = hide and n.state == 'released'
    if on_screen and not hidden:
        _draw_ln_tail_sprite(ctx, painter, n)

    # ── release guide ──
    if n.rel_off is not None and n.state not in ('released', 'missed'):
        rel_y = ctx.time_to_y(float(n.release_t))
        _draw_stroke_with_tick(ctx, painter, n.jcolor,
                                n.lx, n.y_end, rel_y, n.col)


def _ln_body_span(ctx, n, hide) -> tuple | None:
    """Return `(top_y, bot_y, sprite_state)` for an LN body. None means
    no body draws.

    For Quaver, `n.body_min_y` / `n.body_max_y` already reflect Quaver's
    `EarliestHeldPosition` / `LatestHeldPosition`: when held, the start
    side is the playhead's cum (not the head's), and future sign-change
    points extend the body. So both upcoming and held states just read
    those bounds directly. Other games leave them NaN ; the legacy
    head/tail span path handles them."""
    use_dyn = (math.isfinite(n.body_min_y) and math.isfinite(n.body_max_y)
               and n.state in ('upcoming', 'held'))
    if n.state == 'missed':
        return n.y_end, n.y, 'miss_ln'
    match n.state:
        case 'upcoming':
            if use_dyn:
                return n.body_min_y, n.body_max_y, 'normal'
            return n.y_end, n.y, 'normal'
        case 'held':
            if use_dyn:
                return n.body_min_y, n.body_max_y, 'normal'
            return n.y_end, (ctx.judge_y if hide else n.y), 'normal'
        case 'released' if not hide:
            return n.y_end, ctx.judge_y, 'released'
        case _:
            return None


def _draw_ln_body_tile(ctx, painter, n, top, bot, state):
    pm = ctx.sprite_cache.get('ln_body', ctx,
                              col=n.col, state=state, is_roll=n.is_roll)
    # Tile vertically over the body rect. Using QRectF + drawTiledPixmap
    # so fractional pixel heights (high-DPI) render cleanly.
    painter.drawTiledPixmap(
        QRectF(n.lx, top, ctx.lane_width(n.col), bot - top), pm)


def _clip_body_samples(xs, ys, top, bot):
    """Restrict a hold's (xs, ys) body polyline to the visible [top, bot]
    y-window, inserting interpolated endpoints where it crosses either
    edge. The samples run monotonically in y from head to tail (either
    direction); we walk consecutive segments and keep the in-window part.
    Returns (xs, ys) arrays or None if nothing lies inside."""
    out_x, out_y = [], []

    def add(x, y):
        out_x.append(float(x))
        out_y.append(float(y))

    def at_edge(x0, y0, x1, y1, edge):
        f = (edge - y0) / (y1 - y0)
        return x0 + f * (x1 - x0), edge

    for k in range(len(ys)):
        y = ys[k]
        if top <= y <= bot:
            add(xs[k], y)
        if k + 1 < len(ys):
            y0, y1 = ys[k], ys[k + 1]
            x0, x1 = xs[k], xs[k + 1]
            for edge in (top, bot):
                if (y0 - edge) * (y1 - edge) < 0.0:
                    add(*at_edge(x0, y0, x1, y1, edge))
    if len(out_y) < 2:
        return None
    order = np.argsort(out_y)
    return np.asarray(out_x)[order], np.asarray(out_y)[order]


def _draw_ln_body_warped(ctx, painter, n, top, bot, state):
    """Draw a hold body as a BENT vertical strip through the per-note-mod
    sample polyline (`n.body_samples`), so drunk/wave/digital deform the
    body the way ITGmania's per-strip rendering does, instead of the
    straight head/tail rect. The strip is a filled ribbon of lane width
    centered on the polyline, tiled with the same body sprite as a brush
    (vertical tiling matches the rect path)."""
    xs, ys = n.body_samples
    clipped = _clip_body_samples(xs, ys, top, bot)
    if clipped is None:
        return
    xs, ys = clipped

    pm = ctx.sprite_cache.get('ln_body', ctx,
                              col=n.col, state=state, is_roll=n.is_roll)
    w = ctx.lane_width(n.col)
    if len(ys) < 2:
        return

    # `xs` is the body's LEFT edge per sample (lane_x + dx), matching the
    # rect path's `QRectF(n.lx, ...)` origin. Stroke the CENTER polyline
    # with the lane width so the ribbon stays perpendicular to the path
    # everywhere: tracing axis-aligned left/right edges instead collapses
    # into self-intersecting bowties once a strong bend turns the body
    # near-horizontal.
    center = xs + w / 2.0
    spine = QPainterPath()
    spine.moveTo(float(center[0]), float(ys[0]))
    for i in range(1, len(ys)):
        spine.lineTo(float(center[i]), float(ys[i]))

    stroker = QPainterPathStroker()
    stroker.setWidth(float(w))
    stroker.setCapStyle(Qt.FlatCap)
    stroker.setJoinStyle(Qt.RoundJoin)
    path = stroker.createStroke(spine)

    brush = QBrush(pm)
    brush.setTransform(QTransform().translate(float(xs[0]), float(ys[0])))
    painter.save()
    painter.setPen(_NO_PEN)
    painter.setBrush(brush)
    painter.drawPath(path)
    painter.restore()


def _draw_ln_tail_sprite(ctx, painter, n):
    state = _tail_state(n)
    pm = ctx.sprite_cache.get('ln_tail', ctx, col=n.col, state=state)
    if n.body_samples is not None and _draw_tail_on_curve(ctx, painter, n, pm):
        return
    if n.flip_tail:
        # Quaver tails flip vertically when the SV at end_time is
        # negative (the LN is being drawn pointing the other way).
        # Mirror around the tail's centerline by translating down then
        # scaling Y by -1, draw, restore.
        painter.save()
        cx = float(n.lx)
        cy = float(n.y_end)
        painter.translate(cx, cy)
        painter.scale(1.0, -1.0)
        painter.drawPixmap(QPointF(0.0, -pm.height() / 2), pm)
        painter.restore()
    else:
        _blit_lane_pixmap(ctx, painter, pm, n.lx,
                          n.y_end - pm.height() / 2, n.col)


def _draw_tail_on_curve(ctx, painter, n, pm) -> bool:
    """Seat the tail cap on the END of the bent body path, rotated to the
    local tangent, so it traces the arrow-effect curve instead of sitting
    detached at the straight-lane position. Returns False when the path is
    degenerate (caller falls back to the straight blit)."""
    xs, ys = n.body_samples
    if len(ys) < 2:
        return False

    w = ctx.lane_width(n.col)
    end_x = float(xs[-1]) + w / 2.0
    end_y = float(ys[-1])
    # Tangent from the last segment; the sprite's natural orientation is
    # "pointing down the scroll axis" (angle 90 deg in atan2 terms).
    dx = (float(xs[-1]) - float(xs[-2]))
    dy = (float(ys[-1]) - float(ys[-2]))
    if dx == 0.0 and dy == 0.0:
        return False

    angle_deg = math.degrees(math.atan2(dy, dx)) - 90.0
    painter.save()
    painter.translate(end_x, end_y)
    painter.rotate(angle_deg)
    if n.flip_tail:
        painter.scale(1.0, -1.0)
    painter.drawPixmap(QPointF(-w / 2.0, -pm.height() / 2.0), pm)
    painter.restore()
    return True


def _tail_state(n) -> str:
    if n.miss:
        return 'miss_ln'
    if n.is_roll:
        return 'roll'
    if n.state == 'released':
        return 'released'
    return 'normal'


# ── note head ────────────────────────────────────────────────────

def _draw_head(ctx, painter, n) -> bool:
    """Blit the head sprite from the cache if visible. Returns whether
    it was drawn (press-mark and miss-X both key off this)."""
    visible, state, y = _head_vis(ctx, n)
    if not visible:
        return False
    sprite_name = 'ln_head' if n.is_ln else 'tap_head'
    if n.is_tick and state == 'normal':
        state = 'tick'
    pm = ctx.sprite_cache.get(sprite_name, ctx, col=n.col, state=state)
    _blit_lane_pixmap(ctx, painter, pm, n.lx, y - pm.height() / 2, n.col)
    return True


def _head_vis(ctx, n) -> tuple:
    """`(visible, sprite_state, y)` ; sprite_state drives the cache
    key, not a raw color. `normal` / `miss_tap` / `miss_ln`."""
    if n.miss:
        state = 'miss_ln' if n.is_ln else 'miss_tap'
        return True, state, n.y

    hide = ctx.player.press_hide
    if not hide:
        return n.state in ('upcoming', 'tap', 'held'), 'normal', n.y

    if n.is_ln:
        vis = n.state in ('upcoming', 'held')
        y = ctx.judge_y if n.state == 'held' else n.y
        return vis, 'normal', y
    return ctx.t_now < n.press_t, 'normal', n.y


# ── press mark + miss X ──────────────────────────────────────────

def _draw_press_mark(ctx, painter, n):
    p = ctx.player
    # skip: missed LNs, misses where player never pressed, held LNs in press_hide
    if n.miss and (n.is_ln or not p.miss_pressed[n.i]):
        return
    if n.is_ln and n.state == 'held' and p.press_hide:
        return

    # On-screen test: skip when both line endpoints lie off-screen on
    # the same side. The vast majority of upcoming notes (most of every
    # frame on dense charts) have press_y so close to n.y that both
    # endpoints sit far above the receptor; drawing them is invisible
    # GPU work. At the dense start of Hall of Kings this elides ~half
    # the visible-note draws.
    margin = ctx.screen_margin
    h = p.H
    lo = -margin
    hi = h + margin
    if (n.y < lo and n.press_y < lo) or (n.y > hi and n.press_y > hi):
        return
    # A press mark whose tick lands within this many pixels of the note
    # head is visually indistinguishable from the head itself; the
    # press error was sub-pixel so the visualization conveys nothing.
    if abs(n.press_y - n.y) < 2.0:
        return
    color = p.judge_colors['miss'] if n.miss else n.jcolor
    _draw_stroke_with_tick(ctx, painter, color, n.lx, n.y, n.press_y, n.col)


def _draw_stroke_with_tick(ctx, painter, color, lx, y_from, y_to, col):
    """Vertical line from y_from to y_to + cached tick sprite at y_to.
    The line's endpoints change per note so it stays vector; the tick
    is a fixed-geometry sprite cached per color."""
    _extras.draw_lane_line(painter, color, lx, _lane_width(ctx, col),
                           y_from, y_to)
    pm = ctx.sprite_cache.get('tick', ctx, color=color)
    _blit_lane_pixmap(ctx, painter, pm, lx, y_to - 2, col)


def _draw_miss_x(ctx, painter, n):
    pm = ctx.sprite_cache.get('miss_x', ctx, jcolor=n.jcolor)
    _blit_lane_pixmap(ctx, painter, pm, n.lx,
                      n.y - pm.height() / 2, n.col)



# ── per-adapter note-type registration ───────────────────────────

class NoteType(NamedTuple):
    """One note kind a game declares. Each maps 1:1 to a toggleable
    layer in the render plan.

    - `key`   ; stable layer id (matches `LayerRegistry` key).
    - `name`  ; human label for the HUD visibility tree.
    - `source` ; 'player' (shares the prebuilt candidate list +
      `_NoteView` prepass) or 'chart' (owns its own cull).
    - `draw`  ; `(ctx, painter) -> None`; same signature as any layer.
    - `stage` ; optional plugin stage to fire *after* this layer draws.
      Preserves the `AFTER_NOTES` / `AFTER_GHOSTS` hook points now that
      the layers that used to own them are per-adapter.
    """
    key: str
    name: str
    source: str
    draw: Callable
    stage: object = None


# Keys match the existing `LayerRegistry` builtin names so tests that
# assert on layer ordering, plugin stages, and visibility config keep
# working. New keys ('taps', 'lns') replace the old combined 'notes'.
def default_note_types() -> list[NoteType]:
    # Imported locally so core modules don't depend on the plugin API.
    from analysis.player.plugin.plugin_api import Stage
    return [
        NoteType('taps',       'Taps',        'player', draw_taps),
        NoteType('lns',        'Long notes',  'player', draw_lns),
        NoteType('mines',      'Mines',       'chart',  _extras.draw_mines),
        NoteType('lifts',      'Lifts',       'chart',  _extras.draw_lifts),
        NoteType('fakes',      'Fakes',       'chart',  _extras.draw_fakes,
                 stage=Stage.AFTER_NOTES),
        NoteType('miss_holds', 'Miss holds',  'chart',  _extras.draw_miss_holds),
        NoteType('ghost_taps', 'Ghost taps',  'chart',  _extras.draw_ghost_taps,
                 stage=Stage.AFTER_GHOSTS),
    ]