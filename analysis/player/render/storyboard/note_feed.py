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
- f32 stride 22: [m00..m22 (row-major mat3), opacity, r, g, b,
                  crop_l, crop_t, crop_r, crop_b, z,
                  model_z, rot_x, rot_y, rot_z]

`z` is the local SORT key; lanes 18..22 are the item's 3D MODEL
(field-space depth + engine-ordered rotations in degrees), flagged by
`_FEED_FLAG_HAS_MODEL`. They are different things: a mat3 can carry
neither depth nor the sign of a tilt, so an out-of-plane note used to be
projected about its OWN centre and flattened before the consuming field
chain ever saw it - which meant a chart could not cancel a field's
rotation with a per-note counter-rotation (see
.claude/plans/notitg-note-model-z.md). A modeled item's mat3 carries
scale and field position only; the evaluator composes its rotations with
the chain in 4D and projects once.

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

`image_map` is `{key: (image_id, w, h)}`, where `(w, h)` is the sprite's own
design box - what the raster path blits it at. It travels with the id because
a sprite's box is skin-dependent (a circle head is `lane_w` square; a bar head
is `note_h + 2 * HEAD_PAD` tall) and an adapter may replace the specs outright,
so nothing but the rasterised sprite knows it. The emitter looks up:

- 'receptor'         : the per-column receptor notch
- 'solid'            : a 1x1 white square, stretched and tinted into every
                       stroke and tick (a line is a quad like anything else)
- 'tap'              : a tap / note head (all columns)
- 'tap_<col>'        : optional per-column tap override (falls back to
                       'tap' when absent)
- 'receptor_<col>'   : optional per-column receptor override
- 'mine' / 'lift' / 'fake' : chart-stream record sprites (with the same
                       optional '<key>_<col>' per-column override)
- 'miss_x_<judgment>': the miss overlay, one per judgment - it is drawn IN
                       the judgment colour (over a fixed red outline), so
                       no single tinted sprite serves them all
- 'ghost_tap'        : a press with no note under it

Per-column keys are consulted first so a caller MAY vary the sprite by
column; the plain keys are the required fallback. A missing required key
raises (fail fast at this build boundary); a note whose column has no
sprite is counted in `report['skipped']`.

## One vocabulary, not a list of note kinds

Nothing here decides HOW a thing draws from WHAT it is. Every producer
answers where something sits and which sprite it wears, and `_emit_at` (a
sprite on a rect) or `_add` (a sprite on a mat3 it built itself, which only a
head with an out-of-plane tilt needs) is the only way anything reaches the
buffer. Adding a thing to the field is adding a producer.

That is why strokes need no Lines tier and ribbons need no Mesh tier: a lane
line is the white solid stretched down the lane and tinted, and a hold body is
a strip of quads. The only tier that would genuinely add expressiveness here
is one for a travelpath's own art, and `_note_seams` records that gap as
`travelpath_as_lines_source`.

Emitted: receptors; taps, LN heads, LN bodies, LN tails and their glows; the
stream records (mines / lifts / fakes) and a hold mine's armed span; and every
replay overlay - press marks, release guides, miss overlays, ghost taps and
miss-hold spans. That is the whole field.

The overlays that belong to no note record (ghosts, miss holds) are culled by
the note prepass into `ctx.ghost_views` / `ctx.miss_hold_views`, so this module
and the raster layer draw one list rather than each running the cull.

`report` counts `receptors`, `taps`, `streams`, `glows`, `ln_tails`,
`ln_body_segments`, `strokes`, `miss_marks`, `mine_spans`, `ghosts`,
`miss_holds` and `skipped`, plus the `seams` list.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

# Frozen feed v2 strides (drawable-ir.md; storyboard_native.Evaluator
# .feed_u_stride / .feed_f_stride). Named here so the SoA writer and the
# tests share one source of truth.
FEED_U_STRIDE = 7
FEED_F_STRIDE = 22

# Per-note shader builtins (feed v3, lanes [4..6] of the u32 row: the
# active shader+1 and this item's window into the x32 uniform buffer).
# The engine's NoteDisplay feeds its per-note programs these uniforms
# alongside the chart's own pokes (note-drop.vert reads iCol/isHold/
# beat; note-suzumebachi picks a single target note by fNoteBeat), so a
# stamped item's value block is the chart uniforms (name-sorted, the
# desc's order) followed by these six, in THIS order. The executor
# derives the remaining engine inputs (modelMatrix, noteSize,
# textureSize) from the record itself.
NOTE_SHADER_BUILTIN_NAMES = ('iCol', 'isHold', 'isReceptor', 'fNoteBeat',
                             'beat', 'time')

# u32 source kinds (storyboard_native.SRC_*): note heads and receptors
# are image blits.
SRC_IMAGE = 0

# f32 lane offsets into a fed item's 22-wide row.
_F_MAT = 0        # m00..m22, lanes 0..8 (row-major)
_F_OPACITY = 9
_F_TINT = 10      # r, g, b, lanes 10..12
_F_CROP = 13      # l, t, r, b, lanes 13..16
_F_Z = 17
_F_MODEL = 18     # model_z, rot_x, rot_y, rot_z, lanes 18..21

# Receptor notch geometry, mirrored from layers/field.py so a fed
# receptor covers the same lane rect the raster field draws. The notch
# is a thin white bar spanning most of the lane width at the hit line.
_RECEPTOR_H = 4.0
_RECEPTOR_LANE_FRAC = 0.82

# Stream-record kind -> image_map key (notes_model.KIND_*). Mines are
# palette-independent glyphs; lifts/fakes key on the column like note heads,
# so both consult the per-column override first (see _lookup_sprite).
_STREAM_KEYS = {0: 'mine', 1: 'lift', 2: 'fake'}
_MINE_KIND = 0

# Feed flags (evaluate.rs FEED_FLAG_*): bit 0 = additive blend,
# bit 3 = the item carries a 3D model on the [model_z, rot_x, rot_y,
# rot_z] lanes.
_FEED_FLAG_ADDITIVE = 1
_FEED_FLAG_HAS_MODEL = 1 << 3

# The note layer's stealth early-out: below this the fill draws nothing.
_MIN_ALPHA = 1.0 / 255.0

_WHITE = (1.0, 1.0, 1.0)

# The draw-order SORT KEY, packed into one int: stage, then COLUMN, then the
# layer within that column. Monotone in that order, so ordering the buckets is
# ordering their keys. Stages: receptors sit under every note, and the
# chart-stream records (mine / lift / fake) keep their own pass after the taps
# rather than joining the engine's per-column tap loop. Layers are the engine's
# order within a column - an LN body and tail go down first "so that they
# appear under the tap notes", then the head, then its additive glow.
_STAGE_RECEPTOR, _STAGE_NOTE, _STAGE_STREAM, _STAGE_OVERLAY = 0, 1, 2, 3
(_LAYER_LN_BODY, _LAYER_LN_TAIL, _LAYER_GUIDE, _LAYER_HEAD, _LAYER_PRESS,
 _LAYER_MISS, _LAYER_GLOW) = range(7)

# Tick marker geometry (chart_extras.draw_tick): a horizontal bar inset
# from both lane edges, centred on the marked y.
_TICK_INSET = 8.0
_TICK_H = 4.0
# Vertical stroke widths: chart_extras.draw_lane_line's default for a
# press mark or release guide, and the heavier miss-hold / hold-mine
# spans (notes._MINE_BODY_WIDTH).
_STROKE_W = 1.0
_MISS_HOLD_STROKE_W = 2.0
_MINE_SPAN_STROKE_W = 3.0
_MINE_SPAN_COLOR = (170, 60, 60)
# A press mark whose tick lands this close to its head says nothing the
# head does not (notes._draw_press_mark).
_PRESS_MARK_MIN_ERROR = 2.0


def _sort_key(stage: int, col: int, layer: int) -> int:
    return (stage << 12) | ((col & 0xFF) << 4) | layer
_NO_CROP = (0.0, 0.0, 0.0, 0.0)


def feed_notes(player, t, image_map, design=None):
    """Emit the visible receptors + note heads at time `t` as feed-v2
    SoA buffers for the notefield dynamic drawable.

    Returns `(u32_soa, f32_soa, item_count, report)`:
    - `u32_soa` : np.uint32, shape (item_count * FEED_U_STRIDE,)
    - `f32_soa` : np.float32, shape (item_count * FEED_F_STRIDE,)
    - `report`  : {'receptors', 'taps', 'skipped', 'seams'} counts + notes.

    `image_map` maps sprite keys to image ids (see module docstring for
    the key vocabulary). `player` is a replay Player; its render context
    (note_views + receptor offsets) is built for `t` and read per note.
    `design` is the consuming document's screen size (see
    `feed_from_context`)."""
    ctx = _resolve_ctx(player, t)
    return feed_from_context(ctx, image_map, design=design)


def feed_from_context(ctx, image_map, design=None, note_shader=None):
    """The pure emitter: build the SoA from an already-prepared render
    context (its `note_views` + receptor state). Split out from
    `feed_notes` so tests drive it with a synthetic ctx and the real
    path shares one code body.

    `design` is the consuming document's screen size, e.g. `(640, 480)`; pass
    it whenever that document is NOT the ctx's own screen. See
    `_screen_to_design` for why the fed mat3s must be converted.

    `note_shader` activates the per-note shader tier for this frame:
    {'arrow'|'hold'|'receptor': (shader_plus_one, chart_values), 'beat':
    float, 'time': float, 'to_beat': callable|None}. Engine-drawn items
    (receptors, taps, mines, hold bodies/tails and their glows) stamp
    the category's program; replay overlays (press marks, miss X,
    ghosts) never do - they are ours, not the engine's."""
    feed = _Feed(ctx, image_map, _EmitBuffer(_screen_to_design(ctx, design)),
                 {'receptors': 0, 'taps': 0, 'streams': 0, 'glows': 0,
                  'ln_tails': 0, 'ln_body_segments': 0, 'strokes': 0,
                  'miss_marks': 0, 'mine_spans': 0, 'ghosts': 0,
                  'miss_holds': 0, 'skipped': 0, 'seams': []},
                 note_shader or {})

    _emit_receptors(feed)
    _emit_note_heads(feed)
    _emit_streams(feed)
    _emit_ghosts(feed)
    _emit_miss_holds(feed)
    _note_seams(ctx, feed.report)

    u_soa, f_soa, x_soa = feed.rows.finish()
    return u_soa, f_soa, x_soa, feed.rows.count, feed.report


class _Feed(NamedTuple):
    """What every emitter needs and nothing else: where the field is,
    what art exists, where rows go, and what to count.

    Bundled so a new thing to draw is a new producer of PLACEMENTS, with
    no say in how a placement becomes a row - `_emit_at` and `_add` are
    the only two ways anything reaches the buffer."""

    ctx: object
    image_map: dict
    rows: object
    report: dict
    shader: dict = {}


def _shader_stamp(feed, category, col, note_beat=0.0):
    """(shader_plus_one, value block) for an item of `category`, or
    (0, None) when no program is bound to it this frame. The block is
    the chart uniform values followed by NOTE_SHADER_BUILTIN_NAMES's
    six, in that frozen order."""
    entry = feed.shader.get(category)
    if entry is None:
        return 0, None
    shader_plus_one, chart_values = entry
    return shader_plus_one, [
        *chart_values, float(col),
        1.0 if category == 'hold' else 0.0,
        1.0 if category == 'receptor' else 0.0,
        float(note_beat),
        feed.shader.get('beat', 0.0), feed.shader.get('time', 0.0)]


def _note_beat_of(feed, n):
    """The note's chart beat, snapped to the 1/48 row grid (the engine's
    fNoteBeat is NoteRowToBeat's exact rational - suzumebachi compares
    it with == against a literal). Derived from press_t - off (the
    note's chart time); 0.0 when the ctx offers no beat conversion."""
    to_beat = feed.shader.get('to_beat')
    press_t = getattr(n, 'press_t', None)
    off = getattr(n, 'off', None)
    if to_beat is None or press_t is None or off is None:
        return 0.0
    return round(to_beat(float(press_t) - float(off)) * 48.0) / 48.0


def _sprite_of(feed, key, col, state=None):
    """The sprite `key` resolves to for this column and state, or None
    (counted as skipped) when the caller registered no art for it."""
    sprite = _lookup_sprite(feed.image_map, key, col, state) if key else None
    if sprite is None:
        feed.report['skipped'] += 1
    return sprite


def _add(feed, sprite, col, mat, opacity, counter, stage=_STAGE_NOTE,
         layer=_LAYER_HEAD, tint=None, additive=False,
         model=None, shader_cat=None, note_beat=0.0) -> bool:
    """Put one already-placed sprite in the buffer. Below `_MIN_ALPHA`
    the item draws nothing, so it is dropped rather than fed.

    `model` is the item's `(z, rot_x, rot_y, rot_z)` 3D model, or None
    for a flat item (the hot path - the evaluator folds a flag-free feed
    once instead of projecting per item). `shader_cat` names the note
    element this item draws as ('arrow'/'hold'/'receptor'); when a
    program is bound to that category this frame the item is stamped
    with it (None = a replay overlay, never shaded)."""
    if mat is None or opacity < _MIN_ALPHA:
        return False
    shader, xvals = (_shader_stamp(feed, shader_cat, col, note_beat)
                     if shader_cat is not None else (0, None))
    feed.rows.add(sprite[0], mat, opacity, tint=tint or _WHITE,
                  additive=additive, stage=stage, col=col, layer=layer,
                  model=model, shader=shader, xvals=xvals)
    feed.report[counter] = feed.report.get(counter, 0) + 1
    return True


def _emit_at(feed, key, col, cx, cy, w=None, h=None, rotation_deg=0.0,
             opacity=1.0, state=None, counter='skipped', stage=_STAGE_NOTE,
             layer=_LAYER_HEAD, tint=None, additive=False,
             shader_cat=None, note_beat=0.0) -> bool:
    """Put `key`'s sprite on a `w` x `h` rect centred at `(cx, cy)`.

    The ONE way anything flat reaches the field - a receptor notch, a
    body segment, a press-mark stroke, a tick, a miss overlay. `w`/`h`
    default to the sprite's own design box, which is what a thing drawn
    at its natural size wants."""
    sprite = _sprite_of(feed, key, col, state)
    if sprite is None:
        return False
    if w is None or h is None:
        box = _sprite_box(feed.ctx, sprite, col)
        w = box[0] if w is None else w
        h = box[1] if h is None else h
    return _add(feed, sprite, col, _place(cx, cy, w, h, rotation_deg),
                opacity, counter, stage=stage, layer=layer, tint=tint,
                additive=additive, shader_cat=shader_cat,
                note_beat=note_beat)


# ── receptors ────────────────────────────────────────────────────────

def _emit_receptors(feed):
    """One item per visible column receptor: the lane curve at scroll
    offset 0 (mirrors field._draw_receptors). The notch is a
    lane-width * _RECEPTOR_LANE_FRAC bar, _RECEPTOR_H tall, centered on
    that point, drawn flat - so it takes the curve's `flat_zoom` rather
    than a depth this item cannot carry."""
    ctx = feed.ctx
    marks = ctx.receptor_marks
    alpha = getattr(ctx, 'receptor_alpha', None)

    for col in range(int(ctx.keycount)):
        lane_w = float(ctx.lane_width(col))
        if lane_w <= 0.5:
            continue
        mark = marks.at(col)
        # A receptor's visibility is its own, never the curve's: an arrow
        # at the same point takes the stealth gradients and a receptor
        # never does (see lane_path).
        _emit_at(feed, 'receptor', col, mark.x, mark.y,
                 lane_w * _RECEPTOR_LANE_FRAC * mark.flat_zoom,
                 _RECEPTOR_H * mark.flat_zoom,
                 rotation_deg=mark.rotation_deg,
                 opacity=1.0 if alpha is None else max(0.0, float(alpha[col])),
                 counter='receptors', stage=_STAGE_RECEPTOR,
                 shader_cat='receptor')


# ── note heads (taps) ────────────────────────────────────────────────

def _emit_note_heads(feed):
    """Everything a replay note puts on the field, in `ctx.note_views`
    order (the renderer's cull + back-to-front prepass order): the head
    sprite, its glow, and for a hold its body, tail and release guide,
    plus the press mark and miss overlay that record how it was played.

    A head is the one placement that is not flat - a z push or an
    out-of-plane tilt puts it through the field's perspective homography
    - so it builds its own mat and hands it to `_add`."""
    ctx = feed.ctx
    for n in getattr(ctx, 'note_views', ()):
        if n is None:
            continue
        lane_w = float(ctx.lane_width(n.col))
        _emit_ln_body(feed, n)
        _emit_ln_tail(feed, n, lane_w)
        _emit_release_guide(feed, n, lane_w)
        _emit_press_mark(feed, n, lane_w)
        if not _head_visible(n):
            feed.report['skipped'] += 1
            continue
        drawable, state, cy = _sprite_state(ctx, n)
        if not drawable:
            feed.report['skipped'] += 1
            continue
        sprite = _sprite_of(feed, 'ln_head' if n.is_ln else 'tap', n.col, state)
        if sprite is None:
            continue
        cx = float(n.lx) + lane_w / 2.0
        model = _head_model(n)
        mat = _head_mat(n, cx, cy, *_sprite_box(ctx, sprite, n.col),
                        model=model)
        note_beat = _note_beat_of(feed, n)
        drawn = _add(feed, sprite, n.col, mat, float(n.alpha), 'taps',
                     layer=_LAYER_HEAD, model=model, shader_cat='arrow',
                     note_beat=note_beat)
        _emit_glow(feed, n, sprite, mat, model=model, shader_cat='arrow',
                   note_beat=note_beat)
        if drawn:
            _emit_miss_x(feed, n, cx, cy)


def _emit_streams(feed):
    """One item per visible chart-stream record (mine / lift / fake), in
    `ctx.stream_views` order.

    A `_StreamView` carries the same positioning + mod surface as a note head
    (`lx`/`y`/`zoom`/`rotation_deg`/`alpha`), so placement is the shared
    `_place`; `kind` selects only the sprite. Span records whose head fell
    outside the cull window (`head_in_window` false) are pulled in for their
    body stroke alone and draw no head, exactly as the raster layer does."""
    ctx = feed.ctx
    for v in getattr(ctx, 'stream_views', ()) or ():
        if v is None:
            continue
        _emit_hold_mine_span(feed, v)
        if not getattr(v, 'head_in_window', True):
            feed.report['skipped'] += 1
            continue
        if not _head_visible(v):
            feed.report['skipped'] += 1
            continue
        sprite = _sprite_of(feed, _STREAM_KEYS.get(v.kind), v.col)
        if sprite is None:
            continue
        cx = float(v.lx) + float(ctx.lane_width(v.col)) / 2.0
        model = _head_model(v)
        mat = _head_mat(v, cx, float(v.y), *_sprite_box(ctx, sprite, v.col),
                        model=model)
        note_beat = _note_beat_of(feed, v)
        _add(feed, sprite, v.col, mat, float(v.alpha), 'streams',
             stage=_STAGE_STREAM, model=model, shader_cat='arrow',
             note_beat=note_beat)
        _emit_glow(feed, v, sprite, mat, stage=_STAGE_STREAM, model=model,
                   shader_cat='arrow', note_beat=note_beat)


def _emit_ln_body(feed, n):
    """The hold BODY as a strip of quads along its path - one item per path
    segment, each rotated to that segment's angle.

    A ribbon is a quad strip, so it needs no separate Lines tier: every
    segment is an ordinary image item placed by its own mat3, exactly like a
    note head. `body_scale` narrows a cross-section that dives toward the
    camera (per-sample depth foreshortening), so the width is sampled per
    segment rather than held constant. The body sprite is a one-row tile whose
    length comes from the path, so only its WIDTH is the sprite's."""
    path = getattr(n, 'body_path', None) if n.is_ln else None
    if path is None or float(n.alpha) < _MIN_ALPHA:
        return
    sprite = _sprite_of(feed, 'ln_body', n.col, _tail_state(n))
    if sprite is None:
        return
    xs, ys = path.x, path.y
    scale = getattr(n, 'body_scale', None)
    body_w, _body_h = _sprite_box(feed.ctx, sprite, n.col, n.zoom)
    for i in range(len(ys) - 1):
        x0, y0 = float(xs[i]), float(ys[i])
        x1, y1 = float(xs[i + 1]), float(ys[i + 1])
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        width = body_w
        if scale is not None and i < len(scale):
            width *= float(scale[i])
        _add(feed, sprite, n.col,
             _place((x0 + x1) / 2.0, (y0 + y1) / 2.0, width, length,
                    math.degrees(math.atan2(dy, dx)) - 90.0),
             float(n.alpha), 'ln_body_segments', layer=_LAYER_LN_BODY,
             shader_cat='hold', note_beat=_note_beat_of(feed, n))


def _emit_ln_tail(feed, n, lane_w):
    """The hold's TAIL cap, seated on the end of its body path and rotated to
    the local tangent (mirrors `notes._draw_tail_on_curve`), or at the plain
    `y_end` when the path is absent/degenerate.

    The tangent is what makes a FOLDED noodle look right: the end segment runs
    opposite the head, so the cap flips naturally instead of needing an
    explicit vertical mirror."""
    if not n.is_ln:
        return
    sprite = _sprite_of(feed, 'ln_tail', n.col, _tail_state(n))
    if sprite is None:
        return
    end = _tail_end(n, lane_w)
    if end is None:
        return
    cx, cy, angle = end
    mat = _place(cx, cy, *_sprite_box(feed.ctx, sprite, n.col, n.zoom), angle)
    note_beat = _note_beat_of(feed, n)
    _add(feed, sprite, n.col, mat, float(n.alpha), 'ln_tails',
         layer=_LAYER_LN_TAIL, shader_cat='hold', note_beat=note_beat)
    _emit_glow(feed, n, sprite, mat, shader_cat='hold', note_beat=note_beat)


def _emit_press_mark(feed, n, lane_w):
    """How the note was actually played: a stroke from the head to the
    press position, ticked at the press (mirrors `notes._draw_press_mark`).

    Skipped where it would say nothing - a miss the player never pressed,
    a held hold under press_hide, a stroke entirely off-screen on one
    side, or a press error too small to see."""
    player = getattr(feed.ctx, 'player', None)
    press_y = getattr(n, 'press_y', None)
    if player is None or press_y is None:
        return
    if n.miss and (n.is_ln or not player.miss_pressed[n.i]):
        return
    if n.is_ln and n.state == 'held' and player.press_hide:
        return
    margin = float(getattr(feed.ctx, 'screen_margin', 0.0))
    lo, hi = -margin, float(getattr(player, 'H', 0.0)) + margin
    y = float(n.y)
    press_y = float(press_y)
    if (y < lo and press_y < lo) or (y > hi and press_y > hi):
        return
    if abs(press_y - y) < _PRESS_MARK_MIN_ERROR:
        return
    color = player.judge_colors['miss'] if n.miss else n.jcolor
    _emit_stroke(feed, n.col, lane_w, float(n.lx), y, press_y, color,
                 _LAYER_PRESS)


def _emit_release_guide(feed, n, lane_w):
    """The stroke from a hold's tail to where it was released.

    A bent body would be slashed across by a straight vertical guide, so
    a hold with its own path draws none - the tail cap already marks the
    release end (mirrors `notes._draw_ln`)."""
    if not n.is_ln or getattr(n, 'rel_off', None) is None:
        return
    if getattr(n, 'body_path', None) is not None:
        return
    if n.state in ('released', 'missed'):
        return
    time_to_y = getattr(feed.ctx, 'time_to_y', None)
    if time_to_y is None:
        return
    _emit_stroke(feed, n.col, lane_w, float(n.lx), float(n.y_end),
                 float(time_to_y(float(n.release_t))), n.jcolor,
                 _LAYER_GUIDE)


def _emit_miss_x(feed, n, cx, cy):
    """The miss overlay over a missed head. The sprite is rasterised per
    judgment colour, so the judgment NAME is its variant."""
    if not getattr(n, 'miss', False):
        return
    _emit_at(feed, 'miss_x', n.col, cx, cy,
             state=getattr(n, 'judgment', None), counter='miss_marks',
             layer=_LAYER_MISS)


def _emit_hold_mine_span(feed, v):
    """A hold mine's armed span: the stroke connecting its head to its
    end, plus the mine glyph at that end (mirrors
    `notes._draw_hold_mine_spans`)."""
    y_end = getattr(v, 'y_end', float('nan'))
    if v.kind != _MINE_KIND or not math.isfinite(y_end):
        return
    margin = float(getattr(feed.ctx, 'screen_margin', 0.0))
    player = getattr(feed.ctx, 'player', None)
    lo, hi = -margin, float(getattr(player, 'H', 0.0)) + margin
    y, y_end = float(v.y), float(y_end)
    if (y < lo and y_end < lo) or (y > hi and y_end > hi):
        return
    lane_w = float(feed.ctx.lane_width(v.col))
    _emit_line(feed, v.col, float(v.lx) + lane_w / 2.0, y, y_end,
               _MINE_SPAN_STROKE_W, _MINE_SPAN_COLOR, _STAGE_STREAM,
               _LAYER_LN_BODY, 'mine_spans')
    _emit_at(feed, 'mine', v.col, float(v.lx) + lane_w / 2.0, y_end,
             counter='mine_spans', stage=_STAGE_STREAM,
             layer=_LAYER_LN_TAIL, shader_cat='arrow',
             note_beat=_note_beat_of(feed, v))


def _emit_ghosts(feed):
    """A press with no note under it, one glyph each. The prepass culled
    and placed them (`chart_extras.ghost_views`)."""
    for view in getattr(feed.ctx, 'ghost_views', ()) or ():
        _emit_at(feed, 'ghost_tap', view.col,
                 float(feed.ctx.lane_center(view.col)), float(view.y),
                 counter='ghosts', stage=_STAGE_OVERLAY)


def _emit_miss_holds(feed):
    """A hold the player never held: the stretch it should have been held
    for, ticked at both ends (mirrors `chart_extras.draw_miss_holds`)."""
    ctx = feed.ctx
    views = getattr(ctx, 'miss_hold_views', ()) or ()
    if not views:
        return
    color = ctx.player.judge_colors['miss']
    for view in views:
        lane_w = float(ctx.lane_width(view.col))
        cx = float(ctx.lane_center(view.col))
        _emit_line(feed, view.col, cx, view.top, view.bot,
                   _MISS_HOLD_STROKE_W, color, _STAGE_OVERLAY,
                   _LAYER_LN_BODY, 'miss_holds')
        for y in (view.y_press, view.y_release):
            _emit_at(feed, 'solid', view.col, cx, float(y),
                     max(0.0, lane_w - 2 * _TICK_INSET), _TICK_H,
                     opacity=_opacity_of(color), tint=_tint_of(color),
                     counter='miss_holds', stage=_STAGE_OVERLAY,
                     layer=_LAYER_LN_TAIL)


def _emit_stroke(feed, col, lane_w, lx, y_from, y_to, color, layer,
                 stage=_STAGE_NOTE, width=_STROKE_W):
    """A vertical lane stroke with a tick at its far end - the shape every
    replay overlay is drawn as (`notes._draw_stroke_with_tick`)."""
    cx = lx + lane_w / 2.0
    _emit_line(feed, col, cx, y_from, y_to, width, color, stage, layer,
               'strokes')
    _emit_at(feed, 'solid', col, cx, y_to, max(0.0, lane_w - 2 * _TICK_INSET),
             _TICK_H, opacity=_opacity_of(color), tint=_tint_of(color),
             counter='strokes', stage=stage, layer=layer)


def _emit_line(feed, col, cx, y_from, y_to, width, color, stage, layer,
               counter):
    """One vertical run of colour down a lane. A line is a quad like
    everything else - the solid sprite tinted and stretched - so nothing
    here needs a stroke tier the executors do not have."""
    _emit_at(feed, 'solid', col, cx, (y_from + y_to) / 2.0, width,
             abs(y_to - y_from), opacity=_opacity_of(color),
             tint=_tint_of(color), counter=counter, stage=stage, layer=layer)


def _tint_of(color) -> tuple:
    """An 0-255 draw colour as the feed's 0-1 tint."""
    return (float(color[0]) / 255.0, float(color[1]) / 255.0,
            float(color[2]) / 255.0)


def _opacity_of(color) -> float:
    """A draw colour's own alpha, for the colours that carry one."""
    return float(color[3]) / 255.0 if len(color) > 3 else 1.0


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
    if path is None or len(path) < 2:
        return straight
    end = path.at(-1)
    if (end.x, end.y) == (float(path.x[-2]), float(path.y[-2])):
        return straight
    return (end.x, end.y, path.tangent_deg(-2, -1) - 90.0)


def _head_mat(n, cx, cy, w, h, model=None):
    """The item mat3 placing a `w` x `h` head centred on `(cx, cy)`.

    A modeled head carries ALL its rotations on the model lanes - the
    in-plane `rotation_deg` included, so the engine's X -> Y -> Z order
    survives the fold - leaving the mat pure scale + position. A flat head
    keeps its spin here and never touches the model path."""
    rotation = 0.0 if model is not None else n.rotation_deg
    return _place(cx, cy, w * n.zoom, h * n.zoom, rotation)


def _head_model(n):
    """The head's `(z, rot_x, rot_y, rot_z)` 3D model, or None while it
    sits flat on the field plane.

    Separate from `_head_mat` because the two are consumed at different
    points: the mat3 sizes and positions the note IN the field, the model
    turns and lifts it OFF the plane so the field chain's own rotation
    composes with it (projected once, with the chain, by the evaluator).
    An in-plane spin alone stays in the mat - the flat path is the hot
    one, and a z-rotation needs no 4D fold of its own."""
    z = float(getattr(n, 'z', 0.0) or 0.0)
    if z == 0.0 and n.rot_x == 0.0 and n.rot_y == 0.0:
        return None
    return (z, float(n.rot_x), float(n.rot_y), float(n.rotation_deg))


def _head_visible(n):
    """Whether a head contributes anything: a mod stealth alpha can blank the
    FILL (the note layer's `< 1/255` early-out), but a stealthglow note is
    still visible as light, so `glow` keeps it alive."""
    return n.alpha >= _MIN_ALPHA or _glow_of(n) > 0.0


def _glow_of(n) -> float:
    return float(getattr(n, 'glow', 0.0) or 0.0)


def _emit_glow(feed, n, sprite, mat, stage=_STAGE_NOTE, model=None,
               shader_cat=None, note_beat=0.0):
    """The stealthglow pass: a SECOND blit of the same sprite at the same
    placement, composited ADDITIVELY at `glow` strength.

    Mirrors the raster layer - a stealthed note's fill is hidden while the
    glow keeps it visible as light. `glow_rgb` tints only this pass (the fill
    never sees it); absent, the glow keeps the sprite's own colours. `model`
    is the head's own, so the glow rides the same 3D placement, and
    `shader_cat` the base item's, so a shaded note's glow deforms with it."""
    glow = _glow_of(n)
    if glow <= 0.0:
        return
    _add(feed, sprite, n.col, mat, glow, 'glows', stage=stage,
         layer=_LAYER_GLOW, tint=getattr(n, 'glow_rgb', None) or _WHITE,
         additive=True, model=model, shader_cat=shader_cat,
         note_beat=note_beat)


# ── mat3 construction ────────────────────────────────────────────────

def _place(cx, cy, w, h, rotation_deg):
    """Row-major mat3 mapping the unit source box (0,0)-(1,1) to a `w`x`h`
    rect centered at design `(cx, cy)`, rotated `rotation_deg` about the
    center (Qt clockwise-positive). Returned as a flat 9-tuple in the
    feed's [m00, m01, m02, m10, m11, m12, m20, m21, m22] order.

    Composition (applied right-to-left to a unit-box point):
        T(cx, cy) . R(deg) . S(w, h) . T(-1/2, -1/2)
    so the unit box centers on the origin, scales to the note size, spins,
    then translates to the design center. Multiplied out rather than
    assembled: this runs once per visible note per frame, and four 3x3
    allocations plus two matmuls to produce six non-trivial scalars is the
    emitter's single largest per-frame cost."""
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    m00, m01 = cos_t * w, -sin_t * h
    m10, m11 = sin_t * w, cos_t * h
    return (m00, m01, cx - 0.5 * (m00 + m01),
            m10, m11, cy - 0.5 * (m10 + m11),
            0.0, 0.0, 1.0)


# ── image lookup ─────────────────────────────────────────────────────

def _lookup_sprite(image_map, kind, col, state=None):
    """`(image_id, w, h)` for `kind` at `col` in sprite `state`, most specific
    key first: `kind_col_state`, `kind_state`, `kind_col`, `kind`.

    The raster cache keys its head sprites on (col, state) - the noteskin
    varies the head by column AND by judgment state (normal / miss / tick) -
    so the feed resolves the same variant through the image map rather than
    flattening every note onto one sprite. `(w, h)` is the sprite's own design
    box, which is what places it (see `_sprite_box`). Returns None only when no
    key matches; the caller decides whether that's a required kind (raise) or a
    per-note skip (count)."""
    keys = ((f'{kind}_{col}_{state}', f'{kind}_{state}') if state else ())
    for key in (*keys, f'{kind}_{col}', kind):
        found = image_map.get(key)
        if found is not None:
            return found
    return None


def _sprite_box(ctx, sprite, col, zoom=1.0):
    """The on-screen `(w, h)` a sprite draws at in column `col`, scaled by a
    per-note `zoom`.

    Mirrors `notes._blit_lane_pixmap`: the pixmap draws at its OWN dimensions,
    squeezed horizontally by the lane's animated width (a lane switch collapses
    the lane, and notes ride its centre) and never vertically. The height is
    therefore the sprite's, NOT `note_h` - a circle head is a `lane_w` square
    and a bar head carries `2 * HEAD_PAD` of outline room, so sizing every
    sprite to `note_h` flattens it."""
    _image_id, sw, sh = sprite
    base = float(getattr(ctx, 'lane_w', 0.0) or 0.0)
    squeeze = (float(ctx.lane_width(col)) / base) if base else 1.0
    return sw * squeeze * zoom, sh * zoom


def _head_state(ctx, n):
    """`(visible, sprite_state, y)` for a note head - mirrors
    `notes._head_vis` (kept local so this module never imports the Qt note
    layer, exactly as `_emit_receptors` mirrors the field layer).

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


# ── screen -> design space ───────────────────────────────────────────

def _screen_to_design(ctx, design):
    """`(a, b, c, d)` for the screen -> design affine `x' = a*x + c`,
    `y' = b*y + d`, or None when no conversion is needed.

    The ctx's field geometry is in SCREEN pixels: a game adapter's
    `field_geometry` already stretches its design grid onto `ctx.chart_rect`
    (NotITG: `lane_w = 64 * chart_w / 640`), and `note_views` inherit that
    space. A consuming document with its OWN screen - the drawable doc's fixed
    640x480, which the executor stretches onto the chart rect at present time -
    would then apply that same stretch a SECOND time, so fed notes and
    receptors land scaled and offset by the chart-rect ratio.

    Converting here rather than at each call site keeps the emitter writing one
    space: every mat3 goes through `_EmitBuffer.add`, so one pre-multiply
    covers heads, receptors, stream records, LN ribbons and glow passes alike.
    """
    if design is None:
        return None
    rect = getattr(ctx, 'chart_rect', None)
    if rect is None:
        return None
    x, y, w, h = (float(v) for v in rect)
    design_w, design_h = float(design[0]), float(design[1])
    if w <= 0.0 or h <= 0.0 or design_w <= 0.0 or design_h <= 0.0:
        return None
    a, b = design_w / w, design_h / h
    return a, b, -x * a, -y * b


def _to_design_mat(to_design, mat):
    """`S @ mat`, where S is the screen -> design affine `to_design` names.

    The bottom row rides through untouched, so a PROJECTIVE placement (a
    3D-modded head's homography) converts as correctly as a plain affine
    one."""
    a, b, c, d = to_design
    m0, m1, m2, m3, m4, m5, m6, m7, m8 = mat
    return (a * m0 + c * m6, a * m1 + c * m7, a * m2 + c * m8,
            b * m3 + d * m6, b * m4 + d * m7, b * m5 + d * m8,
            m6, m7, m8)


# ── SoA writer ───────────────────────────────────────────────────────

class _EmitBuffer:
    """Accumulates fed items as Python-list rows, then freezes them into
    the two fixed-stride SoA numpy buffers. One image blit per row,
    uncropped and with no z (2D taps, receptors and stream records), so
    only the mat3, opacity, tint and blend vary.

    `to_design` (see `_screen_to_design`) pre-multiplies every mat3 into the
    consuming document's space. It is applied here, at the one point every
    item passes through, rather than threaded into each placement."""

    def __init__(self, to_design=None):
        # Rows land in the bucket they will be DRAWN from; `finish` just
        # concatenates the buckets in key order.
        self._buckets: dict[int, tuple[list, list, list]] = {}
        self._to_design = to_design
        self.count = 0

    # Lanes 13..17: every fed item is uncropped and never sorts.
    _TAIL = (*_NO_CROP, 0.0)
    # Lanes 18..21: the flat item's rest model (no depth, no rotations).
    _FLAT_MODEL = (0.0, 0.0, 0.0, 0.0)

    def add(self, image_id, mat, opacity, tint=_WHITE, additive=False,
            frame=0, stage=_STAGE_NOTE, col=0, layer=0, model=None,
            shader=0, xvals=None):
        key = _sort_key(stage, col, layer)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = ([], [], [])
        u, f, x = bucket
        flags = _FEED_FLAG_ADDITIVE if additive else 0
        if model is not None:
            # Flagged rather than always-on: a flat item must keep the
            # fold-once path in the evaluator, which is the hot one.
            flags |= _FEED_FLAG_HAS_MODEL
        # Lanes 5/6 (this item's window into the x32 buffer) are patched
        # in `finish`: bucketing reorders items, so offsets are only
        # known once the draw order is.
        u.extend((SRC_IMAGE, image_id, frame, flags, shader, 0, 0))
        f.extend(mat if self._to_design is None else
                 _to_design_mat(self._to_design, mat))
        f.extend((opacity, tint[0], tint[1], tint[2], *self._TAIL,
                  *(model if model is not None else self._FLAT_MODEL)))
        x.append(xvals)
        self.count += 1

    def finish(self):
        """The SoA in DRAW order: stage, then COLUMN, then layer, and within
        one bucket the order the rows were added.

        Column-major is the ENGINE's order, for the engine's own reason.
        `NoteField::DrawPrimitives` loops `for each arrow column` with holds
        before taps: "Draw the arrows in order of column. This minimize
        texture switches and let us draw in big batches." Emitting in
        candidate (time) order interleaves columns instead, and a column IS a
        texture (a head sprite varies by column and state), so every
        consecutive pair differed and the executor batched runs of one.

        BUCKETED ON APPEND, not sorted at the end. The key domain is small
        and known up front, so this needs no comparison sort - and no
        PERMUTATION GATHER, which is the part that would stream 18-float rows
        through cache in random order. Appends are sequential per bucket and
        the concatenation is one sequential pass: O(n) in address order, with
        nothing tuned to a cache size."""
        u_out: list = []
        f_out: list = []
        x_out: list = []
        for key in sorted(self._buckets):
            u, f, x = self._buckets[key]
            base = len(u_out)
            u_out.extend(u)
            f_out.extend(f)
            for index, xvals in enumerate(x):
                if xvals is None:
                    continue
                row = base + index * FEED_U_STRIDE
                u_out[row + 5] = len(x_out)
                u_out[row + 6] = len(xvals)
                x_out.extend(xvals)
        return (np.asarray(u_out, dtype=np.uint32),
                np.asarray(f_out, dtype=np.float32),
                np.asarray(x_out, dtype=np.float32))


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
    # `build_context` reads the note prepass and never consults the plugin
    # manager, so a renderer with none serves this read-only seam.
    from analysis.player.render.qt_renderer import QtPlayerRenderer
    return QtPlayerRenderer(None)


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
