"""Notes as drawables: a per-frame emitter of the visible note/receptor
set as frozen feed-v2 SoA items.

This is the FIRST BRICK of "notes are drawables" (drawable-ir.md, Events
and dynamic drawables): the notefield is a `dynamic` drawable whose
command buffer is regenerated each frame. Today's Python note pipeline
produces one item per visible receptor and note head; a Rust note kernel
takes over later with zero change to the evaluator core, because what
crosses the seam is data - the frozen feed v2 SoA layout, never a
closure.

`feed_notes(player, t, image_map)` returns `(u32 SoA, f32 SoA,
item_count, report)` for the notefield drawable at time `t`:

- u32 stride 4 : [source_kind, source_id, frame, flags]
- f32 stride 18: [m00..m22 (row-major mat3), opacity, r, g, b,
                  crop_l, crop_t, crop_r, crop_b, z]

matching `storyboard_native.Evaluator.feed_{u,f}_stride`. The items are
emitted in DRAW ORDER: receptors first, then note heads back-to-front in
the note layer's own `ctx.note_views` order (the renderer's cull +
prepass order).

## The mat3 and the source box

Each item's mat3 places a UNIT source box (0,0)-(1,1) into 640x480
design space, so the scale lane carries the on-screen note size directly
(the "scale from the note size" contract). The evaluator writes a fed
mat3 to the BLIT record verbatim and the executor applies it as
`dest = M . [x, y, 1]` (row-major); a note's screen rect is therefore
`M` applied to the unit square. Callers register each feed image with
natural size 1x1 so the executor's source-pixel box matches this unit
convention.

For a head centered at design `(cx, cy)`, note size `(w, h)`, per-note
zoom `z` and rotation `deg` (about the center, Qt clockwise-positive):

    M = T(cx, cy) . R(deg) . S(z*w, z*h) . T(-1/2, -1/2)

so the unit box's center lands on `(cx, cy)` before the spin/scale.

## image_map vocabulary (documented, kept simple)

`image_map` is `{key: image_id}`; the emitter looks up:

- 'receptor'         : the per-column receptor notch
- 'tap'              : a tap / note head (all columns)
- 'tap_<col>'        : optional per-column tap override (falls back to
                       'tap' when absent)
- 'receptor_<col>'   : optional per-column receptor override

Per-column keys are consulted first so a caller MAY vary the sprite by
column; the plain keys are the required fallback. A missing required key
raises (fail fast at this build boundary); a note whose column has no
sprite is counted in `report['skipped']`.

## Scope (this wave, honest)

Taps + receptors are emitted. LN bodies/tails, mines/lifts/fakes, and
travelpaths are counted in `report['skipped']`, not drawn - they need
the Lines/Path source and per-sample ribbon geometry that a later wave
carries. `report` also records `receptors`, `taps`, and the `seams`
list: the note-pipeline reads that a clean per-note emission still
wants.
"""
from __future__ import annotations

import numpy as np

# Frozen feed v2 strides (drawable-ir.md; storyboard_native.Evaluator
# .feed_u_stride / .feed_f_stride). Named here so the SoA writer and the
# tests share one source of truth.
FEED_U_STRIDE = 4
FEED_F_STRIDE = 18

# u32 source kinds (storyboard_native.SRC_*): note heads and receptors
# are image blits.
SRC_IMAGE = 0

# f32 lane offsets into a fed item's 18-wide row.
_F_MAT = 0        # m00..m22, lanes 0..8 (row-major)
_F_OPACITY = 9
_F_TINT = 10      # r, g, b, lanes 10..12
_F_CROP = 13      # l, t, r, b, lanes 13..16
_F_Z = 17

# Receptor notch geometry, mirrored from layers/field.py so a fed
# receptor covers the same lane rect the raster field draws. The notch
# is a thin white bar spanning most of the lane width at the hit line.
_RECEPTOR_H = 4.0
_RECEPTOR_LANE_FRAC = 0.82

# A note head's design height. The raster head pixmap is note_h + 2*pad
# tall (note_sprites.HEAD_PAD), but the head's VISIBLE extent is one
# note_h band centered on its y; the feed emits that visible band, so
# the unit box scales to (lane_w, note_h). Sourced from the ctx when
# present (note_h is a RenderContext field), else this default.
_DEFAULT_NOTE_H = 14.0

_WHITE = (1.0, 1.0, 1.0)
_NO_CROP = (0.0, 0.0, 0.0, 0.0)


def feed_notes(player, t, image_map):
    """Emit the visible receptors + note heads at time `t` as feed-v2
    SoA buffers for the notefield dynamic drawable.

    Returns `(u32_soa, f32_soa, item_count, report)`:
    - `u32_soa` : np.uint32, shape (item_count * FEED_U_STRIDE,)
    - `f32_soa` : np.float32, shape (item_count * FEED_F_STRIDE,)
    - `report`  : {'receptors', 'taps', 'skipped', 'seams'} counts + notes.

    `image_map` maps sprite keys to image ids (see module docstring for
    the key vocabulary). `player` is a replay Player; its render context
    (note_views + receptor offsets) is built for `t` and read per note."""
    ctx = _resolve_ctx(player, t)
    return feed_from_context(ctx, image_map)


def feed_from_context(ctx, image_map):
    """The pure emitter: build the SoA from an already-prepared render
    context (its `note_views` + receptor state). Split out from
    `feed_notes` so tests drive it with a synthetic ctx and the real
    path shares one code body."""
    rows = _EmitBuffer()
    report = {'receptors': 0, 'taps': 0, 'skipped': 0, 'seams': []}

    _emit_receptors(ctx, image_map, rows, report)
    _emit_note_heads(ctx, image_map, rows, report)
    _note_seams(ctx, report)

    u_soa, f_soa = rows.finish()
    return u_soa, f_soa, rows.count, report


# ── receptors ────────────────────────────────────────────────────────

def _emit_receptors(ctx, image_map, rows, report):
    """One item per visible column receptor, at the hit line plus the
    per-column receptor mod displacement (mirrors field._draw_receptors).
    The notch is a lane-width * _RECEPTOR_LANE_FRAC bar, _RECEPTOR_H tall,
    centered on the lane center."""
    keycount = int(ctx.keycount)
    dx, dy, rot, zoom, alpha = _receptor_offsets(ctx, keycount)
    judge_y = float(ctx.judge_y)

    for col in range(keycount):
        lane_w = float(ctx.lane_width(col))
        if lane_w <= 0.5:
            continue
        image_id = _lookup_image(image_map, 'receptor', col)
        if image_id is None:
            report['skipped'] += 1
            continue
        cx = float(ctx.lane_center(col)) + float(dx[col])
        cy = judge_y + float(dy[col])
        w = lane_w * _RECEPTOR_LANE_FRAC
        col_zoom = 1.0 if zoom is None else float(zoom[col])
        col_alpha = 1.0 if alpha is None else max(0.0, float(alpha[col]))
        mat = _place(cx, cy, w * col_zoom, _RECEPTOR_H * col_zoom,
                     float(rot[col]))
        rows.add(image_id, mat, col_alpha)
        report['receptors'] += 1


def _receptor_offsets(ctx, keycount):
    """`(dx, dy, rotation_deg, zoom, alpha)` per column, identity when the
    ctx carries none - the same read field._receptor_offsets does, kept
    local so this module doesn't import the Qt field layer."""
    offs = getattr(ctx, 'receptor_offsets', None)
    zeros = np.zeros(keycount, dtype=np.float64)
    if offs is None:
        return zeros, zeros, zeros, None, None
    return (offs.get('dx', zeros), offs.get('dy', zeros),
            offs.get('rotation_deg', zeros),
            offs.get('zoom', None), offs.get('alpha', None))


# ── note heads (taps) ────────────────────────────────────────────────

def _emit_note_heads(ctx, image_map, rows, report):
    """One item per visible tap head, in `ctx.note_views` order (the
    renderer's cull + back-to-front prepass order). LN heads/bodies,
    stream records, and 3D-projected heads are counted in
    `report['skipped']` - this wave emits flat taps only."""
    note_h = float(getattr(ctx, 'note_h', _DEFAULT_NOTE_H))
    for n in getattr(ctx, 'note_views', ()):
        if n is None:
            continue
        if n.is_ln:
            report['skipped'] += 1
            continue
        if _is_3d(n):
            report['skipped'] += 1
            continue
        if not _head_visible(n):
            report['skipped'] += 1
            continue
        image_id = _lookup_image(image_map, 'tap', n.col)
        if image_id is None:
            report['skipped'] += 1
            continue
        lane_w = float(ctx.lane_width(n.col))
        cx = float(n.lx) + lane_w / 2.0
        cy = float(n.y)
        mat = _place(cx, cy, lane_w * n.zoom, note_h * n.zoom, n.rotation_deg)
        rows.add(image_id, mat, float(n.alpha))
        report['taps'] += 1


def _is_3d(n):
    """A head projected through the field's perspective camera (z/tilt
    mods) needs the full homography the 2D `_place` mat3 can't carry -
    deferred, so it's a reported skip, not a wrong-looking draw."""
    return n.z != 0.0 or n.rot_x != 0.0 or n.rot_y != 0.0


def _head_visible(n):
    """A tap head draws unless it's blanked to invisibility by a mod
    stealth alpha (the note layer's `< 1/255` early-out) - stealthglow
    (glow > 0) is a later additive-pass concern, out of this wave."""
    return n.alpha >= 1.0 / 255.0


# ── mat3 construction ────────────────────────────────────────────────

def _place(cx, cy, w, h, rotation_deg):
    """Row-major mat3 mapping the unit source box (0,0)-(1,1) to a `w`x`h`
    rect centered at design `(cx, cy)`, rotated `rotation_deg` about the
    center (Qt clockwise-positive). Returned as a flat 9-tuple in the
    feed's [m00, m01, m02, m10, m11, m12, m20, m21, m22] order.

    Composition (applied right-to-left to a unit-box point):
        T(cx, cy) . R(deg) . S(w, h) . T(-1/2, -1/2)
    so the unit box centers on the origin, scales to the note size, spins,
    then translates to the design center."""
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    center = _translate(-0.5, -0.5)
    scale = _scale(w, h)
    rotate = np.array([[cos_t, -sin_t, 0.0],
                       [sin_t, cos_t, 0.0],
                       [0.0, 0.0, 1.0]])
    place = _translate(cx, cy)

    m = place @ rotate @ scale @ center
    return (m[0, 0], m[0, 1], m[0, 2],
            m[1, 0], m[1, 1], m[1, 2],
            m[2, 0], m[2, 1], m[2, 2])


def _translate(tx, ty):
    return np.array([[1.0, 0.0, tx],
                     [0.0, 1.0, ty],
                     [0.0, 0.0, 1.0]])


def _scale(sx, sy):
    return np.array([[sx, 0.0, 0.0],
                     [0.0, sy, 0.0],
                     [0.0, 0.0, 1.0]])


# ── image lookup ─────────────────────────────────────────────────────

def _lookup_image(image_map, kind, col):
    """The image id for `kind` at `col`: a per-column key (`kind_col`)
    wins when present, else the plain `kind` key. Returns None only when
    neither is registered - the caller decides whether that's a required
    kind (raise) or a per-note skip (count)."""
    per_col = image_map.get(f'{kind}_{col}')
    if per_col is not None:
        return per_col
    return image_map.get(kind)


# ── SoA writer ───────────────────────────────────────────────────────

class _EmitBuffer:
    """Accumulates fed items as Python-list rows, then freezes them into
    the two fixed-stride SoA numpy buffers. One image blit per row; every
    fed item is white-tinted, uncropped, source-over, no z (2D taps and
    receptors), so only the mat3 + opacity vary."""

    def __init__(self):
        self._u = []
        self._f = []
        self.count = 0

    def add(self, image_id, mat, opacity):
        self._u.extend((SRC_IMAGE, int(image_id), 0, 0))
        row = [0.0] * FEED_F_STRIDE
        row[_F_MAT:_F_MAT + 9] = [float(v) for v in mat]
        row[_F_OPACITY] = float(opacity)
        row[_F_TINT:_F_TINT + 3] = _WHITE
        row[_F_CROP:_F_CROP + 4] = _NO_CROP
        row[_F_Z] = 0.0
        self._f.extend(row)
        self.count += 1

    def finish(self):
        u_soa = np.asarray(self._u, dtype=np.uint32)
        f_soa = np.asarray(self._f, dtype=np.float32)
        return u_soa, f_soa


# ── context resolution ───────────────────────────────────────────────

def _resolve_ctx(player, t):
    """Build the render context for `player` at `t` (note_views +
    receptor state), reusing the Qt renderer's `build_context`. The
    painter is None: this pass only READS the prepass, it never draws."""
    renderer = _player_renderer(player)
    return renderer.build_context(player, None, float(t))


def _player_renderer(player):
    """The player's Qt renderer, or a bare one. Kept as a seam so a
    headless caller can inject its own context-builder later."""
    renderer = getattr(player, '_renderer', None)
    if renderer is not None:
        return renderer
    from analysis.player.render.qt_renderer import QtRenderer
    return QtRenderer()


# ── seam report ──────────────────────────────────────────────────────

def _note_seams(ctx, report):
    """Record the note-pipeline reads a CLEAN per-note emission still
    needs but this wave doesn't consume. These are the value of the wave
    as much as the code: the next brick removes each seam.

    - LN bodies/tails are a Lines/Path source with per-sample ribbon
      geometry (`_NoteView.body_path` / `body_scale`); not a single mat3.
    - 3D heads (z / rot_x / rot_y) need the field's perspective
      homography (`notes._note_projection`), a full mat3 the 2D `_place`
      doesn't build.
    - stream records (mines/lifts/fakes) live on `ctx.stream_views`, a
      parallel view list this wave doesn't walk.
    - stealthglow (`glow` > 0) is an additive second pass (blend flag +
      tint), not yet emitted.
    - the per-note sprite FRAME (noteskin coloring by column/beat) is
      folded into the raster sprite cache, not exposed as an image frame
      index the feed can carry."""
    seams = report['seams']
    if any(v is not None and v.is_ln for v in getattr(ctx, 'note_views', ())):
        seams.append('ln_body_path_as_lines_source')
    if any(v is not None and _is_3d(v)
           for v in getattr(ctx, 'note_views', ())):
        seams.append('3d_head_perspective_homography')
    if getattr(ctx, 'stream_views', None):
        seams.append('stream_views_second_walk')
    if any(v is not None and v.glow > 0.0
           for v in getattr(ctx, 'note_views', ())):
        seams.append('stealthglow_additive_pass')
