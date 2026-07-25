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
- 'mine' / 'lift' / 'fake' : chart-stream record sprites (with the same
                       optional '<key>_<col>' per-column override)

Per-column keys are consulted first so a caller MAY vary the sprite by
column; the plain keys are the required fallback. A missing required key
raises (fail fast at this build boundary); a note whose column has no
sprite is counted in `report['skipped']`.

## Scope (this wave, honest)

Taps, receptors and stream records (mines/lifts/fakes) are emitted. LN
bodies/tails and travelpaths are counted in `report['skipped']`, not
drawn - they need the Lines/Path source and per-sample ribbon geometry a
later wave carries. `report` records `receptors`, `taps`, `streams` and
the `seams` list: the note-pipeline reads a clean per-note emission
still wants.
"""
from __future__ import annotations

import math

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

# Stream-record kind -> image_map key (notes_model.KIND_*). Mines are
# palette-independent glyphs; lifts/fakes key on the column like note heads,
# so both consult the per-column override first (see _lookup_image).
_STREAM_KEYS = {0: 'mine', 1: 'lift', 2: 'fake'}

# Feed flags (evaluate.rs FEED_FLAG_*): bit 0 = additive blend.
_FEED_FLAG_ADDITIVE = 1

# The note layer's stealth early-out: below this the fill draws nothing.
_MIN_ALPHA = 1.0 / 255.0

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
    report = {'receptors': 0, 'taps': 0, 'streams': 0, 'glows': 0,
              'ln_tails': 0, 'ln_body_segments': 0,
              'skipped': 0, 'seams': []}

    _emit_receptors(ctx, image_map, rows, report)
    _emit_note_heads(ctx, image_map, rows, report)
    _emit_streams(ctx, image_map, rows, report)
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
        if not _head_visible(n):
            report['skipped'] += 1
            continue
        drawable, state, cy = _sprite_state(ctx, n)
        if not drawable:
            report['skipped'] += 1
            continue
        kind = 'ln_head' if n.is_ln else 'tap'
        image_id = _lookup_image(image_map, kind, n.col, state)
        if image_id is None:
            report['skipped'] += 1
            continue
        lane_w = float(ctx.lane_width(n.col))
        cx = float(n.lx) + lane_w / 2.0
        mat = _head_mat(n, cx, cy, lane_w, note_h)
        if mat is None:
            report['skipped'] += 1
            continue
        if float(n.alpha) >= _MIN_ALPHA:
            rows.add(image_id, mat, float(n.alpha))
            report['taps'] += 1
        _emit_glow(n, image_id, mat, rows, report)
        _emit_ln_body(n, image_map, lane_w, rows, report)
        _emit_ln_tail(ctx, n, image_map, lane_w, note_h, rows, report)


def _emit_streams(ctx, image_map, rows, report):
    """One item per visible chart-stream record (mine / lift / fake), in
    `ctx.stream_views` order.

    A `_StreamView` carries the same positioning + mod surface as a note head
    (`lx`/`y`/`zoom`/`rotation_deg`/`alpha`), so placement is the shared
    `_place`; `kind` selects only the sprite. Span records whose head fell
    outside the cull window (`head_in_window` false) are pulled in for their
    body stroke alone and draw no head, exactly as the raster layer does."""
    note_h = float(getattr(ctx, 'note_h', _DEFAULT_NOTE_H))
    for v in getattr(ctx, 'stream_views', ()) or ():
        if v is None:
            continue
        if not getattr(v, 'head_in_window', True):
            report['skipped'] += 1
            continue
        if not _head_visible(v):
            report['skipped'] += 1
            continue
        key = _STREAM_KEYS.get(v.kind)
        image_id = _lookup_image(image_map, key, v.col) if key else None
        if image_id is None:
            report['skipped'] += 1
            continue
        lane_w = float(ctx.lane_width(v.col))
        cx = float(v.lx) + lane_w / 2.0
        mat = _head_mat(v, cx, float(v.y), lane_w, note_h)
        if mat is None:
            report['skipped'] += 1
            continue
        if float(v.alpha) >= _MIN_ALPHA:
            rows.add(image_id, mat, float(v.alpha))
            report['streams'] += 1
        _emit_glow(v, image_id, mat, rows, report)


def _emit_ln_body(n, image_map, lane_w, rows, report):
    """The hold BODY as a strip of quads along its path - one item per path
    segment, each rotated to that segment's angle.

    A ribbon is a quad strip, so it needs no separate Lines tier: every
    segment is an ordinary image item placed by its own mat3, exactly like a
    note head. `body_scale` narrows a cross-section that dives toward the
    camera (per-sample depth foreshortening), so the width is sampled per
    segment rather than held constant."""
    if not n.is_ln:
        return
    path = getattr(n, 'body_path', None)
    if path is None:
        return
    image_id = _lookup_image(image_map, 'ln_body', n.col, _tail_state(n))
    if image_id is None:
        report['skipped'] += 1
        return
    alpha = float(n.alpha)
    if alpha < _MIN_ALPHA:
        return
    xs, ys = path[0], path[1]
    scale = getattr(n, 'body_scale', None)
    half = lane_w / 2.0
    for i in range(len(ys) - 1):
        x0, y0 = float(xs[i]) + half, float(ys[i])
        x1, y1 = float(xs[i + 1]) + half, float(ys[i + 1])
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        width = lane_w * n.zoom
        if scale is not None and i < len(scale):
            width *= float(scale[i])
        mat = _place((x0 + x1) / 2.0, (y0 + y1) / 2.0, width, length,
                     math.degrees(math.atan2(dy, dx)) - 90.0)
        rows.add(image_id, mat, alpha)
        report['ln_body_segments'] += 1


def _emit_ln_tail(ctx, n, image_map, lane_w, note_h, rows, report):
    """The hold's TAIL cap, seated on the end of its body path and rotated to
    the local tangent (mirrors `notes._draw_tail_on_curve`), or at the plain
    `y_end` when the path is absent/degenerate.

    The tangent is what makes a FOLDED noodle look right: the end segment runs
    opposite the head, so the cap flips naturally instead of needing an
    explicit vertical mirror."""
    if not n.is_ln:
        return
    image_id = _lookup_image(image_map, 'ln_tail', n.col, _tail_state(n))
    if image_id is None:
        report['skipped'] += 1
        return
    end = _tail_end(n, lane_w)
    if end is None:
        return
    cx, cy, angle = end
    mat = _place(cx, cy, lane_w * n.zoom, note_h * n.zoom, angle)
    if float(n.alpha) >= _MIN_ALPHA:
        rows.add(image_id, mat, float(n.alpha))
        report['ln_tails'] = report.get('ln_tails', 0) + 1
    _emit_glow(n, image_id, mat, rows, report)


def _tail_state(n) -> str:
    """The tail sprite variant (notes._tail_state)."""
    if getattr(n, 'miss', False):
        return 'miss_ln'
    if getattr(n, 'is_roll', False):
        return 'roll'
    return 'released' if getattr(n, 'state', None) == 'released' else 'normal'


def _tail_end(n, lane_w):
    """`(cx, cy, angle_deg)` for the tail cap: the body path's last segment
    when it has one (angle from its tangent, the sprite's natural orientation
    being 'down the scroll axis' = 90 deg), else the straight `y_end`."""
    path = getattr(n, 'body_path', None)
    y_end = float(getattr(n, 'y_end', n.y))
    straight = (float(n.lx) + lane_w / 2.0, y_end, 0.0)
    if path is None:
        return straight
    xs, ys = path[0], path[1]
    if len(ys) < 2:
        return straight
    dx = float(xs[-1]) - float(xs[-2])
    dy = float(ys[-1]) - float(ys[-2])
    if dx == 0.0 and dy == 0.0:
        return straight
    return (float(xs[-1]) + lane_w / 2.0, float(ys[-1]),
            math.degrees(math.atan2(dy, dx)) - 90.0)


def _head_mat(n, cx, cy, lane_w, note_h):
    """The item mat3 for a head at `(cx, cy)`: the plain 2D placement, or the
    field's perspective homography when the note carries out-of-plane mods.
    None = fully behind the eye, which draws nothing."""
    if _is_3d(n):
        return _place_3d(n, cx, cy, lane_w, note_h)
    return _place(cx, cy, lane_w * n.zoom, note_h * n.zoom, n.rotation_deg)


def _is_3d(n):
    """Whether a head carries out-of-plane mods (z push / roll / twirl), so
    its placement needs the field's perspective homography rather than the
    plain 2D `_place`."""
    return n.z != 0.0 or n.rot_x != 0.0 or n.rot_y != 0.0


# The per-note 3D model plane, mirroring notes._NOTE_CORNERS: the quad the
# perspective verdict is classified against.
_NOTE_HALF = 32.0
_NOTE_CORNERS = ((-_NOTE_HALF, -_NOTE_HALF), (_NOTE_HALF, -_NOTE_HALF),
                 (_NOTE_HALF, _NOTE_HALF), (-_NOTE_HALF, _NOTE_HALF))


def _place_3d(n, cx, cy, w, h):
    """The projective mat3 for an out-of-plane head, or None when the note
    is fully behind the eye (the raster path's 'gone' verdict, which draws
    nothing).

    Mirrors `notes._note_projection`: the model tilts/pushes the unit plane,
    the field camera projects it, and the homography is conjugated back about
    the note centre. Zoom and in-plane rotation live in the MODEL here, so the
    base placement carries neither. `transform3d` returns a ROW-vector
    homography while a fed mat3 is column-vector, hence the transpose."""
    from analysis.player.render import transform3d as t3d

    model = (t3d.scale(n.zoom, n.zoom, 1.0)
             @ t3d.rotate_xyz(n.rot_x, n.rot_y, n.rotation_deg)
             @ t3d.translate(0.0, 0.0, n.z))
    verdict, homography, _clip = t3d.project_with_verdict(
        model, _note_camera(), _NOTE_CORNERS)
    if verdict == 'gone':
        return None
    conjugated = (_translate_row(-cx, -cy) @ np.asarray(homography, np.float64)
                  @ _translate_row(cx, cy))
    return (conjugated.T @ np.asarray(_place(cx, cy, w, h, 0.0),
                                      np.float64).reshape(3, 3)).reshape(9)


def _translate_row(tx, ty):
    """A row-vector translation 3x3 (`[x y 1] @ M`), transform3d's
    convention."""
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [float(tx), float(ty), 1.0]], dtype=np.float64)


def _note_camera():
    """The per-note perspective camera (notes._note_camera): the field's
    LoadMenuPerspective at its fov/eye distance, centred on the origin, so a
    note's z push scales by the same d/(d-z) the field uses."""
    from analysis.player.render import transform3d as t3d
    from analysis.games.notitg import field_projection
    return t3d.projection(field_projection.FOV, field_projection.DESIGN_W,
                          field_projection.DESIGN_W)


def _head_visible(n):
    """Whether a head contributes anything: a mod stealth alpha can blank the
    FILL (the note layer's `< 1/255` early-out), but a stealthglow note is
    still visible as light, so `glow` keeps it alive."""
    return n.alpha >= _MIN_ALPHA or _glow_of(n) > 0.0


def _glow_of(n) -> float:
    return float(getattr(n, 'glow', 0.0) or 0.0)


def _emit_glow(n, image_id, mat, rows, report):
    """The stealthglow pass: a SECOND blit of the same sprite at the same
    placement, composited ADDITIVELY at `glow` strength.

    Mirrors the raster layer - a stealthed note's fill is hidden while the
    glow keeps it visible as light. `glow_rgb` tints only this pass (the fill
    never sees it); absent, the glow keeps the sprite's own colours."""
    glow = _glow_of(n)
    if glow <= 0.0:
        return
    tint = getattr(n, 'glow_rgb', None) or _WHITE
    rows.add(image_id, mat, glow, tint=tint, additive=True)
    report['glows'] = report.get('glows', 0) + 1


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

def _lookup_image(image_map, kind, col, state=None):
    """The image id for `kind` at `col` in sprite `state`, most specific key
    first: `kind_col_state`, `kind_state`, `kind_col`, `kind`.

    The raster cache keys its head sprites on (col, state) - the noteskin
    varies the head by column AND by judgment state (normal / miss / tick) -
    so the feed resolves the same variant through the image map rather than
    flattening every note onto one sprite. Returns None only when no key
    matches; the caller decides whether that's a required kind (raise) or a
    per-note skip (count)."""
    keys = ((f'{kind}_{col}_{state}', f'{kind}_{state}') if state else ())
    for key in (*keys, f'{kind}_{col}', kind):
        found = image_map.get(key)
        if found is not None:
            return found
    return None


def _head_state(ctx, n):
    """`(visible, sprite_state, y)` for a note head - mirrors
    `notes._head_vis` (kept local so this module never imports the Qt note
    layer, exactly as `_receptor_offsets` mirrors the field layer).

    `press_hide` charts hide a head once it is pressed, so visibility is not
    just the stealth alpha; the held-LN head also RIDES the judge line rather
    than its own y."""
    y = float(n.y)
    if getattr(n, 'miss', False):
        return True, ('miss_ln' if n.is_ln else 'miss_tap'), y
    state = getattr(n, 'state', 'tap')
    if not bool(getattr(getattr(ctx, 'player', None), 'press_hide', False)):
        return state in ('upcoming', 'tap', 'held'), 'normal', y
    if n.is_ln:
        held = state == 'held'
        judge_y = float(getattr(ctx, 'judge_y', y)) if held else y
        return state in ('upcoming', 'held'), 'normal', judge_y
    press_t = getattr(n, 'press_t', None)
    if press_t is None:
        return True, 'normal', y
    return float(getattr(ctx, 't_now', 0.0)) < float(press_t), 'normal', y


def _sprite_state(ctx, n):
    """The head's sprite-cache state, with the fluXis tick override the note
    layer applies (`is_tick` + normal -> the bright 'tick' variant)."""
    visible, state, y = _head_state(ctx, n)
    if state == 'normal' and getattr(n, 'is_tick', False):
        state = 'tick'
    return visible, state, y


# ── SoA writer ───────────────────────────────────────────────────────

class _EmitBuffer:
    """Accumulates fed items as Python-list rows, then freezes them into
    the two fixed-stride SoA numpy buffers. One image blit per row,
    uncropped and with no z (2D taps, receptors and stream records), so
    only the mat3, opacity, tint and blend vary."""

    def __init__(self):
        self._u = []
        self._f = []
        self.count = 0

    def add(self, image_id, mat, opacity, tint=_WHITE, additive=False,
            frame=0):
        self._u.extend((SRC_IMAGE, int(image_id), int(frame),
                        _FEED_FLAG_ADDITIVE if additive else 0))
        row = [0.0] * FEED_F_STRIDE
        row[_F_MAT:_F_MAT + 9] = [float(v) for v in mat]
        row[_F_OPACITY] = float(opacity)
        row[_F_TINT:_F_TINT + 3] = [float(c) for c in tint]
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
    """Record the note-pipeline reads a clean per-note emission still needs
    but this module does not yet consume. The remaining one:

    - travelpaths are a Lines/Path source (an arbitrary polyline, not a quad
      strip), and the GL executor has no Lines tier yet.
    """
    seams = report['seams']
    if getattr(ctx, 'travelpaths', None):
        seams.append('travelpath_as_lines_source')
