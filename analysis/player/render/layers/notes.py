"""Notes layer: every chart note record, replay-stream and chart-stream.

Renders note heads, LN bodies/tails, release guides, press marks, and
miss-X overlays for every visible replay candidate, plus the unified
chart-stream records (mines/lifts/fakes) whose kind selects only the
sprite -- positioning, per-note mods, per-column reverse, and
stealth/glow visibility ride the same candidate arrays and `_draw_view`
bracket taps use. Ghost taps and miss-holds (replay overlays, not note
records) stay in chart_extras.py.

Public API:
- `prepare(ctx)` builds `ctx.note_views` + `ctx.stream_views` once per
  frame from the player candidate list (shared by the layers so the
  per-note state is computed once).
- `draw_taps` / `draw_lns` / `draw_mines` / `draw_lifts` / `draw_fakes`
  are the per-layer drawers the `NoteType` registrations hand out.
- `NoteType` is the per-adapter note-kind spec; `default_note_types()`
  returns the full set every game defaults to.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QPainter, QPainterPath,
                           QPainterPathStroker, QPixmap, QTransform)

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np

from analysis.player.init import notes_model as _nm
from analysis.player.notetypes import NT_TICK
from analysis.player.render import lane_path
from analysis.player.render.layers import chart_extras as _extras
from analysis.player.render.layers.note_sprites import ln_body_width
from analysis.player.render.mods.arrow_effects import (
    display_alpha, perspective_z_scale)
from analysis.player.render.primitives import _NO_PEN

if TYPE_CHECKING:
    pass


# Per-note 3D projection corners: a note-sized quad centered at the
# origin (the note center is conjugated in around it). Only the planar
# homography + front/behind verdict are read from these, so the exact
# size just needs to bracket the sprite; ARROW_SIZE (64) is one arrow.
_NOTE_HALF = 32.0
_NOTE_CORNERS = ((-_NOTE_HALF, -_NOTE_HALF), (_NOTE_HALF, -_NOTE_HALF),
                 (_NOTE_HALF, _NOTE_HALF), (-_NOTE_HALF, _NOTE_HALF))


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
    # The judgment's NAME, alongside its colour. The raster path only ever
    # needs the colour, but a fed overlay is a rasterised sprite VARIANT
    # (the miss X is drawn in the judgment colour, not tinted to it), and a
    # variant is chosen by name. Empty means no variant is known, which
    # is what a synthetic view carries; the overlay then finds no sprite.
    judgment: str = ''
    # fluXis tick notes; drawn bright yellow via the 'tick' sprite state.
    is_tick: bool = False
    # Per-note mod alpha (NotITG stealth/hidden family); 1 = opaque.
    alpha: float = 1.0
    # Per-note glow (NotITG stealthglow): the fill is hidden (alpha->0) but
    # the note still renders as an additive glow at this strength. 0 = no
    # glow (unmodded notes keep their plain draw).
    glow: float = 0.0
    # Glow tint (r, g, b) from the stealthglow color companions, or None
    # for the untinted glow (noteskin colors, today's exact draw).
    glow_rgb: tuple | None = None
    # Per-note mod rotation (deg) / zoom (multiplier), applied about the
    # head center. Defaults are the identity so unmodded notes pay
    # nothing; LN bodies/tails keep their position (only the head sprite
    # spins/scales -- documented in `_draw_view`).
    rotation_deg: float = 0.0
    zoom: float = 1.0
    # Per-note 3D: depth (engine px, +z toward the camera) and the
    # out-of-plane tilts roll/twirl (deg). When any is non-rest the head
    # sprite projects through the field's perspective camera about its
    # center (real depth scale + tilt) instead of the 2D zoom/rotate
    # bracket; all-rest keeps the flat path (unmodded notes pay nothing).
    z: float = 0.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    # The hold's body as the lane curve's span between its head and tail
    # offsets (`lane_path.LaneSamples`, running head -> tail, x at the
    # lane center). The renderer strokes a ribbon along it and seats the
    # caps on its endpoints oriented by the local tangent. Two producers
    # feed it (`_build`): the lane curve itself, or an SV fold (a
    # negative-SV hold bending back on itself). `None` is the plain
    # vertical unmodded hold, which selects the rect fast-path
    # (byte-identical to the historical straight blit).
    body_path: object = None
    # Per-sample depth foreshortening for the body ribbon (aligned with
    # body_path's samples), or None for an in-plane body. When present, the
    # stroke lays each cross-section's width perpendicular to the spine and
    # scales it by this d/(d-z) factor, so a body pushed into z narrows -
    # the 3D ribbon. Rides alongside body_path (the spine still drives
    # clipping / tail / alpha); only the fill geometry changes.
    body_scale: object = None


def _sv_fold_path(ctx, i, pos, p, head_y, tail_y):
    """The body span of an LN whose SV folds inside it.

    This is the lane curve too, just one a straight-lane game bends
    through TIME rather than through mods: the column stays put and the
    scroll axis doubles back, so the samples come out in the same shape
    and every consumer treats them the same way.

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

    col = p.notes.columns_list[i]
    center = float(ctx.lane_x(col)) + float(ctx.lane_width(col)) / 2.0
    return lane_path.flat_samples(np.full(ys.shape, center), ys)


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
    body_scale = None
    if body_path is not None:
        # The tail cap seats on the path's LAST sample (deepest point of a
        # fold, the bent end of a mod body); keep `y_end` in sync so the
        # on-screen / release-guide anchoring reads the same point.
        y_end = float(body_path.y[-1])
        body_scale = _ln_body_scale(body_path)

    mod_alpha = getattr(ctx, 'candidate_alpha', None)
    mod_rot = getattr(ctx, 'candidate_rot_deg', None)
    mod_zoom = getattr(ctx, 'candidate_zoom', None)
    mod_z = getattr(ctx, 'candidate_z', None)
    mod_rx = getattr(ctx, 'candidate_rot_x', None)
    mod_ry = getattr(ctx, 'candidate_rot_y', None)
    mod_glow = getattr(ctx, 'candidate_glow', None)
    mod_glow_rgb = getattr(ctx, 'candidate_glow_rgb', None)
    return _NoteView(
        i=i, col=col,
        y=head_y, y_end=y_end,
        press_y=float(ctx.candidate_press_y[pos]),
        lx=int(ctx.lane_x(col)
               + (mod_dx[pos] if mod_dx is not None else 0.0)),
        alpha=(float(display_alpha(mod_alpha[pos]))
               if mod_alpha is not None else 1.0),
        glow=(float(mod_glow[pos])
              if mod_glow is not None else 0.0),
        glow_rgb=(tuple(mod_glow_rgb[pos])
                  if mod_glow_rgb is not None else None),
        rotation_deg=float(mod_rot[pos]) if mod_rot is not None else 0.0,
        zoom=float(mod_zoom[pos]) if mod_zoom is not None else 1.0,
        z=float(mod_z[pos]) if mod_z is not None else 0.0,
        rot_x=float(mod_rx[pos]) if mod_rx is not None else 0.0,
        rot_y=float(mod_ry[pos]) if mod_ry is not None else 0.0,
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
        judgment=p.note_judges[i],
        body_path=body_path,
        body_scale=body_scale,
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


# ── chart-stream views ───────────────────────────────────────────

@dataclass
class _StreamView:
    """One visible chart-stream record (mine/lift/fake): the same
    positioning/mods/visibility surface as `_NoteView`, minus the
    replay-only state (judgment, LN body, press marks). `kind` selects
    only the sprite; `_draw_view` consumes the shared mod fields, so
    stealth/glow/rotation/zoom/3D behave exactly as they do for taps."""
    k: int              # index into the unified stream table
    kind: int
    col: int
    y: float
    y_end: float        # hold-mine span end y; NaN for point records
    lx: float
    # False for span records pulled in only for their body stroke (the
    # cull window missed the head, or it expired): the head sprite
    # stays undrawn, exactly as if the record weren't a candidate.
    head_in_window: bool
    alpha: float = 1.0
    glow: float = 0.0
    glow_rgb: tuple | None = None
    rotation_deg: float = 0.0
    zoom: float = 1.0
    z: float = 0.0
    rot_x: float = 0.0
    rot_y: float = 0.0


def _build_stream_views(ctx) -> list:
    """`_StreamView`s for this frame's stream candidates, reading the
    shared candidate arrays at positions `len(ctx.candidates)` onward
    (where the renderer appended the stream records)."""
    s_idx = getattr(ctx, 'stream_candidates', None)
    if s_idx is None or not len(s_idx):
        return []
    p = ctx.player
    n = p.notes
    base = len(ctx.candidates)
    heads = getattr(ctx, 'stream_head_in_window', None)
    mod_dx = getattr(ctx, 'candidate_dx', None)
    # A stream record fades under the mine stealth family, not the tap
    # one, so it prefers the alpha computed for that family when a chart
    # drives either (see arrow_effects.stream_stealth_active).
    mod_alpha = getattr(ctx, 'candidate_stream_alpha', None)
    if mod_alpha is None:
        mod_alpha = getattr(ctx, 'candidate_alpha', None)
    mod_glow = getattr(ctx, 'candidate_glow', None)
    mod_glow_rgb = getattr(ctx, 'candidate_glow_rgb', None)
    mod_rot = getattr(ctx, 'candidate_rot_deg', None)
    mod_zoom = getattr(ctx, 'candidate_zoom', None)
    mod_z = getattr(ctx, 'candidate_z', None)
    mod_rx = getattr(ctx, 'candidate_rot_x', None)
    mod_ry = getattr(ctx, 'candidate_rot_y', None)

    def at(arr, pos, default):
        return float(arr[pos]) if arr is not None else default

    views = []
    for j, k in enumerate(s_idx):
        pos = base + j
        col = int(n.stream_cols[k])
        views.append(_StreamView(
            k=int(k), kind=int(n.stream_kinds[k]), col=col,
            y=float(ctx.candidate_head_y[pos]),
            y_end=float(ctx.candidate_tail_y[pos]),
            lx=float(ctx.lane_x(col)) + at(mod_dx, pos, 0.0),
            head_in_window=bool(heads[j]) if heads is not None else True,
            alpha=(float(display_alpha(mod_alpha[pos]))
                   if mod_alpha is not None else 1.0),
            glow=at(mod_glow, pos, 0.0),
            glow_rgb=(tuple(mod_glow_rgb[pos])
                      if mod_glow_rgb is not None else None),
            rotation_deg=at(mod_rot, pos, 0.0),
            zoom=at(mod_zoom, pos, 1.0),
            z=at(mod_z, pos, 0.0),
            rot_x=at(mod_rx, pos, 0.0),
            rot_y=at(mod_ry, pos, 0.0),
        ))
    return views


# ── public entry points ──────────────────────────────────────────

def prepare(ctx) -> None:
    """Build every per-candidate `_NoteView` (and `_StreamView`) once.
    The `taps` / `lns` / stream layer drawers read from
    `ctx.note_views` / `ctx.stream_views` ; splitting the previous
    combined loop into per-layer passes would otherwise rebuild each
    view several times. `ctx.ghost_views` / `ctx.miss_hold_views` carry
    the replay overlays that belong to no note record."""
    views: list[_NoteView | None] = []
    for pos, i in enumerate(ctx.candidates):
        views.append(_build(ctx, i, pos))
    ctx.note_views = views
    ctx.stream_views = _build_stream_views(ctx)
    # The replay overlays that belong to no note record. Culled here so
    # every drawer reads one list: they are as much a part of the frame
    # as the notes are, and a backend should not have to know they exist
    # to draw them.
    ctx.ghost_views = _extras.ghost_views(ctx)
    ctx.miss_hold_views = _extras.miss_hold_views(ctx)


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
    glowing = n.glow > 0.0
    is_3d = n.z != 0.0 or n.rot_x != 0.0 or n.rot_y != 0.0
    transformed = n.rotation_deg or n.zoom != 1.0 or is_3d
    if not faded and not transformed and not glowing:
        draw_fn(ctx, painter, n)
        return
    if faded and not glowing and n.alpha < 1.0 / 255.0:
        return

    painter.save()
    if transformed:
        cx = n.lx + _lane_width(ctx, n.col) / 2.0
        cy = float(n.y)
        if is_3d:
            painter.setTransform(_note_projection(ctx, n, cx, cy),
                                 combine=True)
        else:
            painter.translate(cx, cy)
            if n.rotation_deg:
                painter.rotate(n.rotation_deg)
            if n.zoom != 1.0:
                painter.scale(n.zoom, n.zoom)
            painter.translate(-cx, -cy)
    base_opacity = painter.opacity()
    if n.alpha >= 1.0 / 255.0:
        painter.setOpacity(base_opacity * n.alpha)
        draw_fn(ctx, painter, n)
    # stealthglow: the fill is hidden (alpha->0) but an additive glow pass
    # keeps the note visible as light. Rest (glow 0) never reaches here, so
    # an unmodded note is unchanged. The rgb companions tint ONLY this
    # pass: `ctx.glow_tint` is live for its duration and the sprite blit
    # sites swap in the tinted pixmap (`_glow_tinted`); the fill pass
    # above never sees it.
    if glowing:
        painter.setOpacity(base_opacity * n.glow)
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        if n.glow_rgb is None:
            draw_fn(ctx, painter, n)
        else:
            ctx.glow_tint = n.glow_rgb
            try:
                draw_fn(ctx, painter, n)
            finally:
                ctx.glow_tint = None
    painter.restore()


def _note_projection(ctx, n, cx, cy):
    """A QTransform projecting the note's head sprite through the field's
    perspective camera about its center `(cx, cy)`: the in-plane
    zoom/rotation, then the out-of-plane tilt (roll/twirl) and z depth.
    General scene projection (any SM-family game's per-note 3D mods),
    built on the shared transform3d authority.

    Works in a frame centered on the note: the model tilts/pushes the
    unit plane, the camera (LoadMenuPerspective at the field's eye
    distance) projects it, and the result is conjugated back to the
    note's screen position. At the design center a pure +z push yields
    exactly the `perspective_z_scale` d/(d-z) the 2D fake used, so this
    is an upgrade (adds off-center parallax + tilt), consistent with the
    field/receptor cameras. A near-flat note yields ~identity."""
    from analysis.player.render import transform3d as t3d

    model = (t3d.scale(n.zoom, n.zoom, 1.0)
             @ t3d.rotate_xyz(n.rot_x, n.rot_y, n.rotation_deg)
             @ t3d.translate(0.0, 0.0, n.z))
    verdict, H, _clip = t3d.project_with_verdict(
        model, _note_camera(), _NOTE_CORNERS)
    if verdict == 'gone':
        return QTransform()
    to_center = QTransform.fromTranslate(-cx, -cy)
    from_center = QTransform.fromTranslate(cx, cy)
    return to_center * t3d.qtransform_from_h(H) * from_center


def _ln_body_scale(body_path):
    """Per-sample depth foreshortening for the hold body, or None when the
    body is in-plane.

    Each cross-section's width scales by d/(d-z) (`perspective_z_scale`)
    at that sample's own depth, so a bumpy body dives toward/away from the
    camera and its ribbon narrows. Returned as a raw array (not projected
    edges) so it rides the SAME clip machinery as the per-sample alphas;
    the stroke builds the perpendicular edges after clipping. None keeps
    the flat constant-width stroke exact for an in-plane body."""
    z = np.asarray(body_path.z, dtype=np.float64)
    if not np.any(z):
        return None
    return np.asarray(perspective_z_scale(z), dtype=np.float64)


@lru_cache(maxsize=1)
def _note_camera():
    """The per-note perspective camera: LoadMenuPerspective at the field's
    fov/eye distance, centered on the origin (the note center is
    conjugated in), so a note's z push scales by the same d/(d-z) the
    field and the old zoom-fake used - consistent depth across the
    scene."""
    from analysis.player.render import transform3d as t3d
    from analysis.games.notitg import field_projection
    # A viewport the design WIDTH wide (the field eye distance depends on
    # fov + width), so the note z-scale matches perspective_z_scale.
    return t3d.projection(field_projection.FOV, field_projection.DESIGN_W,
                          field_projection.DESIGN_W)


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
    """The hold body's per-sample visibility array, or None when the body
    has no fade of its own and every sample draws fully opaque."""
    path = n.body_path
    if path is None or bool(np.all(path.alpha >= 1.0)):
        return None
    return path.alpha


# ── chart-stream drawing (mines / lifts / fakes) ─────────────────

# Hold-mine body stroke (Quaver): connects the armed span so the player
# can see how long the lane stays hot.
_MINE_BODY_COLOR = (170, 60, 60)
_MINE_BODY_WIDTH = 3


def draw_mines(ctx, painter) -> None:
    """Mines from the unified stream views: head sprites through the
    same per-note mod bracket taps use, then hold-mine span bodies
    (Quaver) and detonation overlays."""
    _draw_stream_kind(ctx, painter, _nm.KIND_MINE)
    _draw_hold_mine_spans(ctx, painter)
    _extras.draw_mine_detonations(ctx, painter)


def draw_lifts(ctx, painter) -> None:
    _draw_stream_kind(ctx, painter, _nm.KIND_LIFT)


def draw_fakes(ctx, painter) -> None:
    _draw_stream_kind(ctx, painter, _nm.KIND_FAKE)


def _draw_stream_kind(ctx, painter, kind) -> None:
    for v in getattr(ctx, 'stream_views', ()):
        if v.kind == kind and v.head_in_window:
            _draw_view(ctx, painter, v, _draw_stream_sprite)


def _draw_stream_sprite(ctx, painter, v) -> None:
    """Blit the record's sprite: `kind` selects it and nothing else.
    Mines are palette-independent glyphs; lifts/fakes key on the column
    like note heads. Every stream pixmap anchors `y` at its vertical
    center (square mines, head-shaped lifts/fakes)."""
    match v.kind:
        case _nm.KIND_MINE:
            pm = ctx.sprite_cache.get('mine', ctx)
        case _nm.KIND_LIFT:
            pm = ctx.sprite_cache.get('lift', ctx, col=v.col)
        case _nm.KIND_FAKE:
            pm = ctx.sprite_cache.get('fake', ctx, col=v.col)
    tint = getattr(ctx, 'glow_tint', None)
    if tint is not None:
        pm = _glow_tinted(pm, tint)
    painter.drawPixmap(
        QPointF(float(v.lx), float(v.y - pm.height() / 2)), pm)


def _draw_hold_mine_spans(ctx, painter) -> None:
    """Body stroke + end sprite for hold mines (finite span end; the
    head sprite is drawn by the shared mine pass). Both endpoints come
    from the unified kernel (the head/tail candidate ys), so a modded
    span's ends displace exactly like the column's taps."""
    p = ctx.player
    margin = ctx.screen_margin
    lo, hi = -margin, p.H + margin
    end_pm = None
    for v in getattr(ctx, 'stream_views', ()):
        if v.kind != _nm.KIND_MINE or not math.isfinite(v.y_end):
            continue
        if (v.y < lo and v.y_end < lo) or (v.y > hi and v.y_end > hi):
            continue
        if end_pm is None:
            end_pm = ctx.sprite_cache.get('mine', ctx)
        _extras.draw_lane_line(painter, _MINE_BODY_COLOR, v.lx,
                               _lane_width(ctx, v.col), v.y, v.y_end,
                               _MINE_BODY_WIDTH)
        painter.drawPixmap(
            QPointF(float(v.lx), float(v.y_end - end_pm.height() / 2)),
            end_pm)


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
        else:
            # The tail cap fades at ITS OWN y (the body's last sample).
            tail_alpha = float(display_alpha(alphas[-1]))
            if tail_alpha >= 1.0 / 255.0:
                painter.save()
                painter.setOpacity(painter.opacity() * tail_alpha)
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
    """The judge line where `col`'s notes visually land: the lane curve at
    scroll offset 0, which is where the column's receptor is. A mod
    consumer that reorients columns (NotITG's reverse family mirrors the
    field to upscroll) moves that point, and this follows it."""
    return float(ctx.receptor_marks.y[col])


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
            ys = n.body_path.y
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


# Tinted glow sprites, keyed (pm.cacheKey(), quantized rgb). Quantizing
# to 1/32 steps bounds the key space for animated tints while staying
# finer than the additive pass resolves visually; the size cap sheds
# entries whose sprite pixmap was invalidated (a stale cacheKey never
# matches again).
_GLOW_TINT_STEP = 1.0 / 32.0
_GLOW_TINT_CACHE: dict = {}
_GLOW_TINT_CACHE_MAX = 256


def _glow_tinted(pm, rgb):
    """A copy of `pm` with its color multiplied by `rgb`, alpha kept:
    the stealthglow rgb companions tint the additive glow pass with the
    sprite's own shape. Multiply folds the tint into the sprite pixels;
    DestinationIn restores the original alpha the opaque tint fill
    flattened."""
    quant = tuple(int(round(min(max(c, 0.0), 1.0) / _GLOW_TINT_STEP))
                  for c in rgb)
    key = (pm.cacheKey(), quant)
    cached = _GLOW_TINT_CACHE.get(key)
    if cached is not None:
        return cached

    tinted = QPixmap(pm.size())
    tinted.fill(Qt.transparent)
    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, pm)
    painter.setCompositionMode(QPainter.CompositionMode_Multiply)
    painter.fillRect(tinted.rect(),
                     QColor.fromRgbF(*(q * _GLOW_TINT_STEP for q in quant)))
    painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    painter.drawPixmap(0, 0, pm)
    painter.end()

    if len(_GLOW_TINT_CACHE) >= _GLOW_TINT_CACHE_MAX:
        _GLOW_TINT_CACHE.clear()
    _GLOW_TINT_CACHE[key] = tinted
    return tinted


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


# How far outside the window a clamped body sample may sit. Big enough
# that clamping never changes in-view geometry perceptibly, small enough
# that window + pad stays well under the GL paint engine's +/-32767
# device-coordinate limit even at high device pixel ratios.
_BODY_COORD_PAD = 8000.0


def _draw_ln_body_stroke(ctx, painter, n, top, bot, state):
    """Draw a hold body as a constant-width ribbon stroked along its path
    (`n.body_path`). This is the ONE body renderer for every non-straight
    hold: mod-bent bodies (drunk/wave/digital) and SV folds alike -- the
    producer differs, the stroke is identical. The strip is a filled
    ribbon of lane width centered on the polyline, tiled with the same
    body sprite as a brush (vertical tiling matches the rect path)."""
    if n.body_scale is not None:
        _draw_ln_body_ribbon(ctx, painter, n, top, bot, state)
        return

    xs, ys = n.body_path.x, n.body_path.y
    alphas = _body_alphas(n)
    clipped = _clip_body_samples(xs, ys, top, bot, alphas)
    if clipped is None:
        return
    xs, ys = clipped[0], clipped[1]
    alphas = clipped[2] if len(clipped) > 2 else None

    # Mod slams (`*10000 500 beat`-style pokes) throw samples arbitrarily
    # far offscreen, and the GL paint engine DROPS concave path fills
    # whose device bounds leave +/-32767 px (the raster engine had no
    # such limit) - the whole visible ribbon vanishes for the slam
    # frame, with a "Painter path exceeds +/-32767 pixels" warning.
    # Clamp samples into a generously offscreen box: a clamped point is
    # still thousands of px outside the window in the same direction, so
    # the in-view ribbon keeps its shape while the path stays inside the
    # engine's range at any plausible device pixel ratio.
    p = ctx.player
    xs = np.clip(xs, -_BODY_COORD_PAD,
                 getattr(p, 'W', 0) + _BODY_COORD_PAD)
    ys = np.clip(ys, -_BODY_COORD_PAD,
                 getattr(p, 'H', 0) + _BODY_COORD_PAD)

    pm = ctx.sprite_cache.get('ln_body', ctx,
                              col=n.col, state=state, is_roll=n.is_roll)
    w = ctx.lane_width(n.col)
    if len(ys) < 2:
        return

    # The path IS the body's center line, so stroke it directly: tracing
    # axis-aligned left/right edges instead collapses into
    # self-intersecting bowties once a strong bend turns the body
    # near-horizontal. The stroke width is the sprite strip's visible
    # width, so a body flipping between this path and the rect tile
    # (producers skip constant-dx frames) keeps one thickness.
    center = xs
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


def _draw_ln_body_ribbon(ctx, painter, n, top, bot, state):
    """Draw a hold body whose depth push tilts it out of the receptor
    plane: a filled ribbon between the two perpendicular edges, each
    cross-section's half-width scaled by the per-sample d/(d-z)
    (`n.body_scale`), so the ribbon foreshortens with depth. Unlike the
    constant-width center stroke, the width here is not axis-aligned - it
    follows the spine's screen tangent - so a strong bend does not bowtie.
    Shares the clip / alpha-run / offscreen-clamp machinery with the flat
    stroke; only the fill geometry differs."""
    from analysis.player.render.mods import body_sweep

    w = ctx.lane_width(n.col)
    width = float(ln_body_width(getattr(ctx.player, 'skin', 'bar'), w))
    center = np.stack([np.asarray(n.body_path.x, dtype=np.float64),
                       np.asarray(n.body_path.y, dtype=np.float64)], axis=1)
    left, right = body_sweep.project_screen_ribbon(center, n.body_scale, width)

    alphas = _body_alphas(n)
    lc = _clip_body_samples(left[:, 0], left[:, 1], top, bot, alphas)
    rc = _clip_body_samples(right[:, 0], right[:, 1], top, bot, alphas)
    if lc is None or rc is None or len(lc[1]) != len(rc[1]):
        return

    p = ctx.player
    pad = _BODY_COORD_PAD
    lx = np.clip(lc[0], -pad, getattr(p, 'W', 0) + pad)
    ly = np.clip(lc[1], -pad, getattr(p, 'H', 0) + pad)
    rx = np.clip(rc[0], -pad, getattr(p, 'W', 0) + pad)
    ry = np.clip(rc[1], -pad, getattr(p, 'H', 0) + pad)
    if len(ly) < 2:
        return
    ribbon_alphas = lc[2] if len(lc) > 2 else None

    pm = ctx.sprite_cache.get('ln_body', ctx,
                              col=n.col, state=state, is_roll=n.is_roll)
    painter.save()
    painter.setPen(_NO_PEN)
    painter.setBrush(_body_fill_color(pm))
    base_opacity = painter.opacity()
    for lo, hi, level in _alpha_runs(ribbon_alphas, len(ly)):
        if level < 1.0 / 255.0:
            continue
        poly = QPainterPath()
        poly.moveTo(float(lx[lo]), float(ly[lo]))
        for k in range(lo + 1, hi + 1):
            poly.lineTo(float(lx[k]), float(ly[k]))
        for k in range(hi, lo - 1, -1):
            poly.lineTo(float(rx[k]), float(ry[k]))
        poly.closeSubpath()
        painter.setOpacity(base_opacity * min(1.0, level))
        painter.drawPath(poly)
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
    seg = display_alpha((np.asarray(alphas[:-1]) + np.asarray(alphas[1:]))
                        / 2.0)
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
    path = n.body_path
    if len(path) < 2:
        return False

    end = path.at(-1)
    if (end.x, end.y) == (float(path.x[-2]), float(path.y[-2])):
        return False

    # The sprite's natural orientation is "pointing down the scroll axis"
    # (angle 90 deg in atan2 terms).
    painter.save()
    painter.translate(end.x, end.y)
    painter.rotate(path.tangent_deg(-2, -1) - 90.0)
    painter.drawPixmap(
        QPointF(-ctx.lane_width(n.col) / 2.0, -pm.height() / 2.0), pm)
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
    tint = getattr(ctx, 'glow_tint', None)
    if tint is not None:
        pm = _glow_tinted(pm, tint)
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
        NoteType('mines',      'Mines',       'chart',  draw_mines),
        NoteType('lifts',      'Lifts',       'chart',  draw_lifts),
        NoteType('fakes',      'Fakes',       'chart',  draw_fakes,
                 stage=Stage.AFTER_NOTES),
        NoteType('miss_holds', 'Miss holds',  'chart',  _extras.draw_miss_holds),
        NoteType('ghost_taps', 'Ghost taps',  'chart',  _extras.draw_ghost_taps,
                 stage=Stage.AFTER_GHOSTS),
    ]