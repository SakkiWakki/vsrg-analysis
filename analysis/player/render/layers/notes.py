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
from analysis.player.render.layers.note_sprites import ln_body_width
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
    # The hold's body as a path: `(xs, ys)` arrays where `xs` is the
    # per-sample LEFT edge (lane_x + any per-note dx) and `ys` the screen
    # y, running head -> tail. Every LN body is this path; the renderer
    # strokes a constant-width ribbon along it and seats the head/tail
    # caps on its endpoints oriented by the local tangent. Three
    # producers feed it (`_build`): mod-bent samples (drunk/wave), SV
    # folds (negative-SV holds bend back on themselves), or -- for a
    # plain vertical unmodded hold -- `None`, which selects the rect
    # fast-path (byte-identical to the historical straight blit).
    body_path: object = None


def _sv_fold_path(ctx, i, pos, p, head_y, tail_y):
    """Body path `(xs, ys)` for an LN whose SV folds inside its span.

    Under negative / reversing SV the body is not a straight head->tail
    segment: it doubles back on itself. We trace it through the SAME
    time->y pipeline the renderer uses for note positions (`batch_time_to_y`)
    by projecting a handful of sample times -- the head, every SV
    sign-change instant still ahead of the (dynamically shrinking)
    playhead, and the tail. Between two consecutive breakpoints the
    scroll velocity is constant, so linear interpolation of those
    projected ys is exact; the polyline is the body verbatim.

    Reversal breakpoints already fold out live as the playhead crosses
    them (Quaver's `UpdateLongNoteSize`), so the visible body shrinks and
    the tail cap's orientation falls out of the path's end tangent -- the
    former `flip_tail` special case dissolves.

    Returns `None` (caller keeps the straight head->tail path) when the
    engine isn't in SV mode, this isn't a Quaver-cached LN, or the span
    holds no reversal (a monotone body is already the straight path)."""
    change_arr = getattr(p, '_ln_change_times', None)
    if change_arr is None or i >= len(change_arr):
        return None
    frame = ctx.frame
    if not getattr(frame, 'use_sv', False):
        return None
    change_times = change_arr[i]
    if change_times is None or not change_times.size:
        return None

    head_t = float(p.times[i])
    end_t = float(p.notes.ln_tail_times[i])
    # Past reversals have already folded out of the body; start the trace
    # at the deeper of the head and the playhead once held.
    start_t = max(head_t, ctx.t_now) if ctx.t_now >= head_t else head_t
    future = change_times[change_times > start_t]
    if not future.size:
        return None

    sample_t = np.concatenate(([start_t], future, [end_t]))
    groups = getattr(p, '_note_sv_groups', None)
    cand_groups = (np.full(sample_t.shape, groups[i])
                   if groups is not None else None)
    ys = p.batch_time_to_y(sample_t, frame, groups=cand_groups)
    # The head cap already sits at `head_y`; when held the trace begins at
    # the playhead, so pin the first sample there for a seamless join.
    ys = np.asarray(ys, dtype=np.float64)
    if ctx.t_now < head_t:
        ys[0] = head_y
    ys[-1] = tail_y

    lx = float(ctx.lane_x(p.notes.columns_list[i]))
    xs = np.full(ys.shape, lx, dtype=np.float64)
    return xs, ys


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

    head_y = float(ctx.candidate_head_y[pos])
    rel_off = None
    release_t = None
    y_end = 0
    if is_ln:
        rel_off = p.hold_release_offsets.get((p.notes.noterows_list[i], col))
        release_t = end_t + (rel_off or 0.0)
        y_end = float(ctx.candidate_tail_y[pos])
    else:
        end_t = None

    mod_dx = getattr(ctx, 'candidate_dx', None)
    body_path = _ln_body_path(ctx, i, pos, p, head_y, y_end) if is_ln else None
    if body_path is not None:
        # The tail cap seats on the path's LAST sample (deepest point of a
        # fold, the bent end of a mod body); keep `y_end` in sync so the
        # on-screen / release-guide anchoring reads the same point.
        y_end = float(body_path[1][-1])

    mod_alpha = getattr(ctx, 'candidate_alpha', None)
    mod_rot = getattr(ctx, 'candidate_rot_deg', None)
    mod_zoom = getattr(ctx, 'candidate_zoom', None)
    return _NoteView(
        i=i, col=col,
        y=head_y, y_end=y_end,
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
        body_path=body_path,
    )


def _ln_body_path(ctx, i, pos, p, head_y, tail_y):
    """The hold's body path `(xs, ys)`, or `None` for a plain straight
    body (rect fast-path). Three producers, in priority order:

    1. mod-bent samples (`ctx.hold_body_samples`, NotITG drunk/wave/...)
       -- already the full displaced polyline;
    2. SV folds (`_sv_fold_path`) -- negative-SV holds that double back;
    3. otherwise `None`, so a straight vertical unmodded hold renders via
       the byte-identical rect blit.

    Mods win over SV folds: when a per-note mod is displacing the body,
    its samples already carry the final displaced geometry."""
    samples = getattr(ctx, 'hold_body_samples', None)
    if samples is not None:
        bent = samples.get(pos)
        if bent is not None:
            return bent
    return _sv_fold_path(ctx, i, pos, p, head_y, tail_y)


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
        if _body_alphas(n) is not None:
            # The body carries its own per-strip visibility (engine
            # semantics: hidden/sudden evaluate each drawn part at ITS
            # y), so it must not inherit the head's alpha - a blanked
            # head leaves the body up, fading through the window.
            _draw_ln(ctx, painter, n)
        else:
            _draw_alpha_only(ctx, painter, n, _draw_ln)
        _draw_view(ctx, painter, n, _draw_replay_note)


def _body_alphas(n):
    """The hold body's per-sample visibility array, or None when its
    path carries none (SV folds, straight bodies)."""
    path = n.body_path
    return path[2] if path is not None and len(path) > 2 else None


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
        # A hold IS a path: stroke a constant-width ribbon along its
        # samples. `body_path is None` is the plain vertical unmodded
        # body, drawn via the byte-identical rect fast-path. `top`/`bot`
        # are the visible y-window every source clips to.
        if n.body_path is not None and n.state in ('upcoming', 'held'):
            _draw_ln_body_stroke(ctx, painter, n, top, bot, body_state)
        elif bot > top:
            _draw_ln_body_tile(ctx, painter, n, top, bot, body_state)

    # ── tail sprite ──
    # A released hold with no release data (autoplay streams) is fully
    # consumed at the tail: no cap drifts on past the receptor.
    on_screen = -ctx.screen_margin <= n.y_end <= p.H + ctx.screen_margin
    hidden = n.state == 'released' and (hide or n.rel_off is None)
    if on_screen and not hidden:
        alphas = _body_alphas(n)
        if alphas is None:
            _draw_ln_tail_sprite(ctx, painter, n)
        elif alphas[-1] >= 1.0 / 255.0:
            # The tail cap fades at ITS OWN y (the body's last sample).
            painter.save()
            painter.setOpacity(painter.opacity() * min(1.0, alphas[-1]))
            _draw_ln_tail_sprite(ctx, painter, n)
            painter.restore()

    # ── release guide ──
    # Straight-lane analyzer UI: on a curved body the vertical stroke
    # would slash across the noodle (it assumes the lane is the path).
    # Skipped until the guide learns to follow the path; the tail cap
    # still marks the release end.
    if (n.rel_off is not None and n.body_path is None
            and n.state not in ('released', 'missed')):
        rel_y = ctx.time_to_y(float(n.release_t))
        _draw_stroke_with_tick(ctx, painter, n.jcolor,
                                n.lx, n.y_end, rel_y, n.col)


def _display_judge_y(ctx, col) -> float:
    """The judge line where `col`'s notes visually land. `ctx.judge_y` is
    the native downscroll anchor; a mod consumer that reorients columns
    (NotITG's reverse family mirrors the field to upscroll) stashes each
    receptor's shift from that anchor in `ctx.receptor_offsets['dy']`."""
    offs = getattr(ctx, 'receptor_offsets', None)
    if offs is None:
        return float(ctx.judge_y)
    return float(ctx.judge_y) + float(offs['dy'][col])


def _ln_body_span(ctx, n, hide) -> tuple | None:
    """Return `(top_y, bot_y, sprite_state)` for an LN body -- the visible
    y-window the body clips to plus its sprite state. None means no body
    draws.

    When the hold carries a `body_path` (SV fold or mod bend) the window
    is the path's own y-extent, so a folded noodle that reaches past the
    straight head/tail span still clips correctly. Straight windows are
    ordered min/max so both scroll orientations work (NotITG's engine
    field scrolls up); the press-hide clamp cuts the body at the display
    judge line without extending past the tail, so a hold whose tail has
    already crossed it draws nothing.

    Released holds keep the downscroll-literal window: it exists to show
    the body remainder between an early release and the judge line, so
    it only draws while the tail is still above it (the caller's
    `bot > top` guard) and only when release data exists at all --
    autoplay streams carry none, and their holds vanish at the tail."""
    match n.state:
        case 'missed':
            lo, hi = sorted((n.y, n.y_end))
            return lo, hi, 'miss_ln'
        case 'upcoming' | 'held' if n.body_path is not None:
            ys = n.body_path[1]
            return float(ys.min()), float(ys.max()), 'normal'
        case 'upcoming':
            lo, hi = sorted((n.y, n.y_end))
            return lo, hi, 'normal'
        case 'held':
            cap = n.y
            if hide:
                lo, hi = sorted((n.y, n.y_end))
                cap = min(max(_display_judge_y(ctx, n.col), lo), hi)
            lo, hi = sorted((cap, n.y_end))
            return lo, hi, 'normal'
        case 'released' if not hide and n.rel_off is not None:
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


def _clip_body_samples(xs, ys, top, bot, alphas=None):
    """Restrict a hold's (xs, ys) body polyline to the visible [top, bot]
    y-window, inserting interpolated points where it crosses either edge.

    PATH ORDER is preserved (samples are walked head -> tail, not sorted)
    so a folded noodle -- where `ys` runs down then back up -- keeps both
    arms in trace order for the stroker. Samples outside the window are
    dropped; the crossing points are inserted where they occur so a
    segment straddling an edge is cut cleanly. `alphas` (per-sample body
    visibility) rides along, interpolated at the crossings. Returns
    (xs, ys) or (xs, ys, alphas) arrays, or None if fewer than two
    points survive."""
    out_x, out_y, out_a = [], [], []

    def add(x, y, a=None):
        out_x.append(float(x))
        out_y.append(float(y))
        if alphas is not None:
            out_a.append(1.0 if a is None else float(a))

    def at_edge(k, y0, y1, edge):
        f = (edge - y0) / (y1 - y0)
        x = xs[k] + f * (xs[k + 1] - xs[k])
        a = None if alphas is None \
            else alphas[k] + f * (alphas[k + 1] - alphas[k])
        return x, edge, a

    n = len(ys)
    for k in range(n):
        y = ys[k]
        if top <= y <= bot:
            add(xs[k], y, None if alphas is None else alphas[k])
        if k + 1 < n:
            y0, y1 = ys[k], ys[k + 1]
            # Insert edge crossings in the order the segment meets them so
            # path order survives (a segment can cross both edges).
            crossings = [edge for edge in (top, bot)
                         if (y0 - edge) * (y1 - edge) < 0.0]
            crossings.sort(key=lambda e: abs(e - y0))
            for edge in crossings:
                add(*at_edge(k, y0, y1, edge))
    if len(out_y) < 2:
        return None
    if alphas is None:
        return np.asarray(out_x), np.asarray(out_y)
    return np.asarray(out_x), np.asarray(out_y), np.asarray(out_a)


# pm.cacheKey() -> the sprite's central body color; one pixel read per
# distinct sprite pixmap, then cached for the session.
_BODY_FILL_CACHE: dict = {}


def _body_fill_color(pm):
    key = pm.cacheKey()
    color = _BODY_FILL_CACHE.get(key)
    if color is None:
        img = pm.toImage()
        color = img.pixelColor(img.width() // 2, img.height() // 2)
        _BODY_FILL_CACHE[key] = color
    return color


def _draw_ln_body_stroke(ctx, painter, n, top, bot, state):
    """Draw a hold body as a constant-width ribbon stroked along its path
    (`n.body_path`). This is the ONE body renderer for every non-straight
    hold: mod-bent bodies (drunk/wave/digital) and SV folds alike -- the
    producer differs, the stroke is identical. The strip is a filled
    ribbon of lane width centered on the polyline, tiled with the same
    body sprite as a brush (vertical tiling matches the rect path)."""
    xs, ys = n.body_path[0], n.body_path[1]
    alphas = _body_alphas(n)
    clipped = _clip_body_samples(xs, ys, top, bot, alphas)
    if clipped is None:
        return
    xs, ys = clipped[0], clipped[1]
    alphas = clipped[2] if len(clipped) > 2 else None

    pm = ctx.sprite_cache.get('ln_body', ctx,
                              col=n.col, state=state, is_roll=n.is_roll)
    w = ctx.lane_width(n.col)
    if len(ys) < 2:
        return

    # `xs` is the body's LEFT edge per sample (lane_x + dx), matching the
    # rect path's `QRectF(n.lx, ...)` origin. Stroke the CENTER polyline
    # so the ribbon stays perpendicular to the path everywhere: tracing
    # axis-aligned left/right edges instead collapses into
    # self-intersecting bowties once a strong bend turns the body
    # near-horizontal. The stroke width is the sprite strip's visible
    # width, so a body flipping between this path and the rect tile
    # (producers skip constant-dx frames) keeps one thickness.
    center = xs + w / 2.0
    stroker = QPainterPathStroker()
    stroker.setWidth(float(ln_body_width(
        getattr(ctx.player, 'skin', 'bar'), w)))
    stroker.setCapStyle(Qt.FlatCap)
    stroker.setJoinStyle(Qt.RoundJoin)

    # Flat fill: tiling the body sprite under an axis-aligned brush prints
    # its edge/border pixels as seams across a diagonal ribbon (the brush
    # never rotates with the path). Warped bodies carry no noteskin
    # detail, so the sprite's flat body color is the faithful fill.
    # Per-sample visibility (hidden/sudden evaluated per strip, engine
    # ArrowGetPercentVisible) strokes the spine in runs of similar
    # alpha; a fully-visible body is the single-run fast path.
    painter.save()
    painter.setPen(_NO_PEN)
    painter.setBrush(_body_fill_color(pm))
    if alphas is None:
        spine = QPainterPath()
        spine.moveTo(float(center[0]), float(ys[0]))
        for i in range(1, len(ys)):
            spine.lineTo(float(center[i]), float(ys[i]))
        painter.drawPath(stroker.createStroke(spine))
    else:
        base_opacity = painter.opacity()
        for lo, hi, level in _alpha_runs(alphas, len(ys)):
            if level < 1.0 / 255.0:
                continue
            spine = QPainterPath()
            spine.moveTo(float(center[lo]), float(ys[lo]))
            for i in range(lo + 1, hi + 1):
                spine.lineTo(float(center[i]), float(ys[i]))
            painter.setOpacity(base_opacity * min(1.0, level))
            painter.drawPath(stroker.createStroke(spine))
    painter.restore()


# Per-strip body visibility quantization: runs of segments within one
# step stroke as one path (few draw calls, no visible banding at 1/8).
_BODY_ALPHA_STEP = 1.0 / 8.0


def _alpha_runs(alphas, count):
    """(start, end, alpha) index runs over a polyline's segments, merging
    consecutive segments whose (endpoint-averaged, step-quantized)
    visibility matches. No alphas -> one opaque run."""
    if alphas is None:
        return [(0, count - 1, 1.0)]
    seg = (np.asarray(alphas[:-1]) + np.asarray(alphas[1:])) / 2.0
    levels = np.round(seg / _BODY_ALPHA_STEP) * _BODY_ALPHA_STEP
    runs = []
    start = 0
    for i in range(1, len(levels)):
        if levels[i] != levels[start]:
            runs.append((start, i, float(levels[start])))
            start = i
    runs.append((start, len(levels), float(levels[start])))
    return runs


def _draw_ln_tail_sprite(ctx, painter, n):
    state = _tail_state(n)
    pm = ctx.sprite_cache.get('ln_tail', ctx, col=n.col, state=state)
    if n.body_path is not None and _draw_tail_on_curve(ctx, painter, n, pm):
        return
    _blit_lane_pixmap(ctx, painter, pm, n.lx,
                      n.y_end - pm.height() / 2, n.col)


def _draw_tail_on_curve(ctx, painter, n, pm) -> bool:
    """Seat the tail cap on the END of the body path, rotated to the local
    tangent, so it caps whatever the path does there: a mod-bent curve, or
    a fold whose end tangent points back UP the lane (which is how the old
    `flip_tail` vertical mirror emerges naturally -- a folded noodle's end
    segment runs opposite the head, so the tangent already flips the cap).
    Returns False when the path is degenerate (caller falls back to the
    straight blit)."""
    xs, ys = n.body_path[0], n.body_path[1]
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
        y = _display_judge_y(ctx, n.col) if n.state == 'held' else n.y
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