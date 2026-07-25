"""Feed emitter: notes-as-drawables first brick (note_feed.feed_notes).

Synthetic gate: a ctx-like stub with a few taps/receptors at known
columns/times, asserting the emitted item count, draw order (receptors
first, then note-view order), design positions (the mat3 places the unit
box on the note center), opacity, and the frozen feed-v2 SoA layout
(stride + lane correctness against the native Evaluator's feed strides).

A real headless smoke on the gat 1 chart is included but SKIPS unless
the chart + harness resolve cheaply - the synthetic tests are the gate.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.player.render.storyboard import note_feed as nf


# ── synthetic ctx ────────────────────────────────────────────────────

def _note_view(**kw):
    """A minimal object with the `_NoteView` attributes the emitter
    reads. A SimpleNamespace so a test can set exactly the fields under
    test; the emitter reads col/lx/y/is_ln/alpha/zoom/rotation_deg/
    glow/z/rot_x/rot_y."""
    base = dict(col=0, lx=0.0, y=0.0, is_ln=False, alpha=1.0, zoom=1.0,
                rotation_deg=0.0, glow=0.0, z=0.0, rot_x=0.0, rot_y=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _stream_view(**kw):
    """A minimal `_StreamView` stand-in: the same positioning/mod surface as a
    note head plus `kind` (notes_model.KIND_*) and `head_in_window`."""
    base = dict(kind=0, col=0, lx=0.0, y=0.0, head_in_window=True, alpha=1.0,
                zoom=1.0, rotation_deg=0.0, glow=0.0, z=0.0, rot_x=0.0,
                rot_y=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _ctx(note_views, keycount=4, lane_w=64.0, judge_y=400.0, note_h=14.0,
         receptor_offsets=None, stream_views=None, lane_widths=None):
    """A stub with the RenderContext reads the emitter uses: keycount,
    lane geometry (lane_w/lane_x/lane_width/lane_center), judge_y, note_h,
    note_views, and optional receptor_offsets / stream_views.

    `lane_widths` overrides the per-column width while `lane_w` stays the base,
    which is how a lane switch presents (the sprite squeezes by their ratio)."""
    ctx = SimpleNamespace(
        keycount=keycount,
        judge_y=judge_y,
        lane_w=lane_w,
        note_h=note_h,
        note_views=note_views,
        stream_views=stream_views or [],
        receptor_offsets=receptor_offsets,
    )
    ctx.lane_width = (lambda col: lane_w) if lane_widths is None \
        else (lambda col: lane_widths[col])
    ctx.lane_x = lambda col: col * lane_w
    ctx.lane_center = lambda col: col * lane_w + lane_w / 2.0
    return ctx


# A head sprite's design box, deliberately NOT (lane_w, note_h): the raster
# cache sizes heads at `(lane_w, note_h + 2 * HEAD_PAD)` for the bar skin and
# `(lane_w, lane_w)` for the circle skin, so the emitter must place the
# SPRITE's box rather than reconstruct one from note_h.
_SPRITE_W, _SPRITE_H = 64.0, 26.0

_IMAGE_MAP = {'receptor': (100, 1.0, 1.0),
              'tap': (200, _SPRITE_W, _SPRITE_H),
              'mine': (300, _SPRITE_W, _SPRITE_H),
              'lift': (301, _SPRITE_W, _SPRITE_H),
              'fake': (302, _SPRITE_W, _SPRITE_H),
              'ln_head': (220, _SPRITE_W, _SPRITE_H),
              'ln_tail': (230, _SPRITE_W, _SPRITE_H),
              'ln_body': (240, _SPRITE_W, 1.0)}


# ── SoA layout ───────────────────────────────────────────────────────

def test_strides_match_native_evaluator():
    """The module's frozen strides equal the native Evaluator's, so a fed
    row this emitter writes decodes 1:1 in Rust."""
    sn = pytest.importorskip('storyboard_native')
    db = sn.DocBuilder(640, 480)
    db.drawable(640.0, 480.0, False, False)
    ev = db.finish()
    assert nf.FEED_U_STRIDE == ev.feed_u_stride == 4
    assert nf.FEED_F_STRIDE == ev.feed_f_stride == 18


def test_soa_shape_is_stride_multiple():
    ctx = _ctx([_note_view(col=1, lx=64.0, y=200.0)])
    u, f, count, _ = nf.feed_from_context(ctx, _IMAGE_MAP)
    assert count == 4 + 1  # 4 receptors + 1 tap
    assert u.dtype == np.uint32 and f.dtype == np.float32
    assert u.shape == (count * nf.FEED_U_STRIDE,)
    assert f.shape == (count * nf.FEED_F_STRIDE,)


def test_u_lanes_are_image_blits():
    ctx = _ctx([_note_view(col=0, lx=0.0, y=100.0)], keycount=1)
    u, _, count, _ = nf.feed_from_context(ctx, _IMAGE_MAP)
    rows = u.reshape(count, nf.FEED_U_STRIDE)
    # source_kind == SRC_IMAGE, frame == 0, flags == 0 for every item.
    assert (rows[:, 0] == nf.SRC_IMAGE).all()
    assert (rows[:, 2] == 0).all()
    assert (rows[:, 3] == 0).all()
    # receptor image id first, tap id last.
    assert rows[0, 1] == 100
    assert rows[-1, 1] == 200


# ── draw order + counts ──────────────────────────────────────────────

def test_receptors_first_then_note_order():
    views = [_note_view(col=0, lx=0.0, y=50.0),
             _note_view(col=2, lx=128.0, y=250.0)]
    ctx = _ctx(views)
    u, _, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)
    rows = u.reshape(count, nf.FEED_U_STRIDE)
    assert report['receptors'] == 4
    assert report['taps'] == 2
    # first 4 rows are receptors (image 100), then the 2 taps (image 200)
    # in note-view order.
    assert list(rows[:, 1]) == [100, 100, 100, 100, 200, 200]


def test_ln_head_and_tail_draw_with_their_own_sprites():
    image_map = {'receptor': (100, 1.0, 1.0), 'tap': (200, _SPRITE_W, _SPRITE_H),
                 'ln_head': (220, _SPRITE_W, _SPRITE_H),
                 'ln_tail': (230, _SPRITE_W, _SPRITE_H)}
    views = [_note_view(col=1, lx=64.0, y=150.0, is_ln=True, y_end=40.0)]
    u, _f, count, report = nf.feed_from_context(_ctx(views), image_map)
    assert (report['taps'], report['ln_tails']) == (1, 1)
    assert count == 4 + 2
    drawn = [int(u[i * nf.FEED_U_STRIDE + 1]) for i in (4, 5)]
    assert drawn == [220, 230], 'the ln_head sprite, then the ln_tail cap'
    # No body path on this view, so no ribbon segments.
    assert report['ln_body_segments'] == 0


def test_ln_tail_rides_the_body_path_tangent():
    # A folded noodle's end segment runs back UP the lane; the cap follows the
    # tangent, which is how the vertical flip emerges without a special case.
    import numpy as _np
    path = (_np.array([64.0, 64.0]), _np.array([100.0, 140.0]))  # heading DOWN
    views = [_note_view(col=1, lx=64.0, y=150.0, is_ln=True, y_end=40.0,
                        body_path=path)]
    image_map = {'receptor': (100, 1.0, 1.0), 'tap': (200, _SPRITE_W, _SPRITE_H),
                 'ln_head': (220, _SPRITE_W, _SPRITE_H),
                 'ln_tail': (230, _SPRITE_W, _SPRITE_H)}
    _u, f, _count, report = nf.feed_from_context(_ctx(views), image_map)
    assert report['ln_tails'] == 1
    row = f[5 * nf.FEED_F_STRIDE:6 * nf.FEED_F_STRIDE]
    mat = _np.asarray(row[nf._F_MAT:nf._F_MAT + 9], dtype=float).reshape(3, 3)
    # The cap sits at the path END (y=140), not the raw y_end (40).
    centre = mat @ _np.array([0.5, 0.5, 1.0])
    assert centre[1] == pytest.approx(140.0, abs=1e-4)


def test_stream_records_draw_with_their_kind_sprite():
    # mine / lift / fake each emit one item keyed by kind; a span record whose
    # head fell outside the cull window draws no head (body stroke only).
    streams = [_stream_view(kind=0, col=0, lx=0.0, y=50.0),
               _stream_view(kind=1, col=1, lx=64.0, y=60.0),
               _stream_view(kind=2, col=2, lx=128.0, y=70.0),
               _stream_view(kind=0, col=3, lx=192.0, y=80.0,
                            head_in_window=False)]
    ctx = _ctx([], stream_views=streams)
    u, _f, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)
    assert report['streams'] == 3
    assert count == 4 + 3
    assert 'stream_views_second_walk' not in report['seams']
    drawn = [int(u[i * nf.FEED_U_STRIDE + 1]) for i in range(4, count)]
    assert drawn == [300, 301, 302]


def test_3d_head_draws_through_the_perspective_homography():
    # A +z push draws, and its mat3 is PROJECTIVE (a non-zero bottom row) -
    # the 2D placement cannot express that, so its presence proves the note
    # went through the field camera rather than the flat path.
    views = [_note_view(col=0, lx=0.0, y=50.0, z=120.0)]
    ctx = _ctx(views)
    _u, f, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)
    assert report['taps'] == 1
    assert count == 4 + 1
    assert '3d_head_perspective_homography' not in report['seams']

    # A pure z push at the conjugation centre stays AFFINE - it reduces to
    # the field's d/(d-z) depth scale - so the evidence is the scale change.
    def head_mat(ctx_):
        _u, f_, _c, _r = nf.feed_from_context(ctx_, _IMAGE_MAP)
        row = f_[4 * nf.FEED_F_STRIDE:5 * nf.FEED_F_STRIDE]
        return np.asarray(row[nf._F_MAT:nf._F_MAT + 9], dtype=np.float64)

    pushed = head_mat(ctx)
    flat = head_mat(_ctx([_note_view(col=0, lx=0.0, y=50.0)]))
    assert abs(pushed[0] - flat[0]) > 1e-6, 'z push rescales the head'


def test_a_tilted_head_gets_a_projective_mat3():
    # A tilt (roll/twirl) is what the 2D placement genuinely cannot express:
    # its homography carries a non-zero projective bottom row.
    views = [_note_view(col=0, lx=0.0, y=50.0, rot_x=40.0)]
    _u, f, _count, report = nf.feed_from_context(_ctx(views), _IMAGE_MAP)
    assert report['taps'] == 1
    row = f[4 * nf.FEED_F_STRIDE:5 * nf.FEED_F_STRIDE]
    mat = np.asarray(row[nf._F_MAT:nf._F_MAT + 9], dtype=np.float64)
    assert abs(mat[6]) > 1e-9 or abs(mat[7]) > 1e-9, 'projective bottom row'


def test_a_note_behind_the_eye_draws_nothing():
    # 'gone' verdict: pushed through the camera, the head has no valid
    # homography and must be skipped rather than drawn wrong.
    from analysis.games.notitg import field_projection
    behind = field_projection.EYE_DISTANCE * 2.0
    views = [_note_view(col=0, lx=0.0, y=50.0, z=behind)]
    ctx = _ctx(views)
    _, _, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)
    assert report['taps'] == 0
    assert count == 4  # receptors only


def test_stealth_blanked_head_skipped():
    views = [_note_view(col=0, lx=0.0, y=50.0, alpha=0.0)]
    ctx = _ctx(views)
    _, _, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)
    assert report['taps'] == 0
    assert count == 4


# ── design positions ─────────────────────────────────────────────────

def _apply_mat(mat9, x, y):
    """Apply the row-major feed mat3 to a design point, as the executor
    does: dest = M . [x, y, 1]."""
    m = np.asarray(mat9, dtype=np.float64).reshape(3, 3)
    v = m @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def _tap_row(f, count, index):
    return f.reshape(count, nf.FEED_F_STRIDE)[index]


def test_tap_mat3_centers_unit_box_on_note():
    # column 2, left edge 128, lane width 64 -> center x = 160; y = 300.
    ctx = _ctx([_note_view(col=2, lx=128.0, y=300.0)])
    _, f, count, _ = nf.feed_from_context(ctx, _IMAGE_MAP)
    row = _tap_row(f, count, count - 1)  # the tap is last
    mat9 = row[nf._F_MAT:nf._F_MAT + 9]
    cx, cy = _apply_mat(mat9, 0.5, 0.5)  # unit-box CENTER
    assert cx == pytest.approx(160.0)
    assert cy == pytest.approx(300.0)


def test_tap_mat3_scales_to_the_sprite_box_not_note_h():
    # The raster path blits the head pixmap at its OWN size, centred on the
    # note's y (notes._blit_lane_pixmap); sizing the item to note_h instead
    # squashes every sprite - flat ellipses under the circle skin.
    ctx = _ctx([_note_view(col=0, lx=0.0, y=100.0)], lane_w=64.0, note_h=14.0)
    _, f, count, _ = nf.feed_from_context(ctx, _IMAGE_MAP)
    mat9 = _tap_row(f, count, count - 1)[nf._F_MAT:nf._F_MAT + 9]
    x0, y0 = _apply_mat(mat9, 0.0, 0.0)
    x1, y1 = _apply_mat(mat9, 1.0, 1.0)
    assert (x1 - x0) == pytest.approx(_SPRITE_W)
    assert (y1 - y0) == pytest.approx(_SPRITE_H)
    assert y0 == pytest.approx(100.0 - _SPRITE_H / 2.0)
    assert x0 == pytest.approx(0.0)
    assert x1 == pytest.approx(64.0)


def test_tap_squeezes_horizontally_with_the_lane_never_vertically():
    # A lane switch collapses the lane width; the raster blit scales the
    # sprite's WIDTH by lane_width/lane_w and leaves its height alone.
    ctx = _ctx([_note_view(col=0, lx=0.0, y=100.0)], lane_w=64.0,
               lane_widths={0: 32.0, 1: 64.0, 2: 64.0, 3: 64.0})
    _, f, count, _ = nf.feed_from_context(ctx, _IMAGE_MAP)
    mat9 = _tap_row(f, count, count - 1)[nf._F_MAT:nf._F_MAT + 9]
    x0, y0 = _apply_mat(mat9, 0.0, 0.0)
    x1, y1 = _apply_mat(mat9, 1.0, 1.0)
    assert (x1 - x0) == pytest.approx(_SPRITE_W / 2.0)
    assert (y1 - y0) == pytest.approx(_SPRITE_H)


def test_tap_zoom_scales_about_center():
    ctx = _ctx([_note_view(col=0, lx=0.0, y=100.0, zoom=2.0)],
               lane_w=64.0, note_h=14.0)
    _, f, count, _ = nf.feed_from_context(ctx, _IMAGE_MAP)
    mat9 = _tap_row(f, count, count - 1)[nf._F_MAT:nf._F_MAT + 9]
    # doubled size, still centered at (32, 100).
    cx, cy = _apply_mat(mat9, 0.5, 0.5)
    x0, _ = _apply_mat(mat9, 0.0, 0.0)
    x1, _ = _apply_mat(mat9, 1.0, 0.0)
    assert cx == pytest.approx(32.0)
    assert cy == pytest.approx(100.0)
    assert (x1 - x0) == pytest.approx(128.0)


def test_tap_rotation_spins_about_center():
    # 90deg CW about the center swaps the box's width/height extents.
    ctx = _ctx([_note_view(col=0, lx=0.0, y=100.0, rotation_deg=90.0)],
               lane_w=64.0, note_h=14.0)
    _, f, count, _ = nf.feed_from_context(ctx, _IMAGE_MAP)
    mat9 = _tap_row(f, count, count - 1)[nf._F_MAT:nf._F_MAT + 9]
    cx, cy = _apply_mat(mat9, 0.5, 0.5)
    assert cx == pytest.approx(32.0)
    assert cy == pytest.approx(100.0)
    # a horizontal unit edge maps to a vertical screen edge of note WIDTH.
    x0, y0 = _apply_mat(mat9, 0.0, 0.5)
    x1, y1 = _apply_mat(mat9, 1.0, 0.5)
    assert abs(x1 - x0) == pytest.approx(0.0, abs=1e-6)
    assert abs(y1 - y0) == pytest.approx(64.0)


# ── opacity + tint ───────────────────────────────────────────────────

def test_opacity_from_note_alpha_tint_white():
    ctx = _ctx([_note_view(col=0, lx=0.0, y=100.0, alpha=0.5)])
    _, f, count, _ = nf.feed_from_context(ctx, _IMAGE_MAP)
    row = _tap_row(f, count, count - 1)
    assert row[nf._F_OPACITY] == pytest.approx(0.5)
    assert tuple(row[nf._F_TINT:nf._F_TINT + 3]) == (1.0, 1.0, 1.0)
    # no crop, no z.
    assert tuple(row[nf._F_CROP:nf._F_CROP + 4]) == (0.0, 0.0, 0.0, 0.0)
    assert row[nf._F_Z] == 0.0


# ── receptors ────────────────────────────────────────────────────────

def test_receptor_at_hit_line_center():
    ctx = _ctx([], keycount=2, lane_w=64.0, judge_y=400.0)
    _, f, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)
    assert report['receptors'] == 2
    rows = f.reshape(count, nf.FEED_F_STRIDE)
    # column 0 center x = 32, column 1 center x = 96, both at y = judge_y.
    for col, expect_cx in ((0, 32.0), (1, 96.0)):
        mat9 = rows[col][nf._F_MAT:nf._F_MAT + 9]
        cx, cy = _apply_mat(mat9, 0.5, 0.5)
        assert cx == pytest.approx(expect_cx)
        assert cy == pytest.approx(400.0)


def test_receptor_offsets_displace_and_fade():
    offs = {
        'dx': np.array([10.0, 0.0]),
        'dy': np.array([0.0, -5.0]),
        'rotation_deg': np.zeros(2),
        'alpha': np.array([0.3, 1.0]),
    }
    ctx = _ctx([], keycount=2, receptor_offsets=offs)
    _, f, count, _ = nf.feed_from_context(ctx, _IMAGE_MAP)
    rows = f.reshape(count, nf.FEED_F_STRIDE)
    cx0, cy0 = _apply_mat(rows[0][nf._F_MAT:nf._F_MAT + 9], 0.5, 0.5)
    assert cx0 == pytest.approx(32.0 + 10.0)
    assert rows[0][nf._F_OPACITY] == pytest.approx(0.3)
    _, cy1 = _apply_mat(rows[1][nf._F_MAT:nf._F_MAT + 9], 0.5, 0.5)
    assert cy1 == pytest.approx(400.0 - 5.0)


def test_zero_width_lane_receptor_skipped():
    ctx = _ctx([], keycount=2)
    ctx.lane_width = lambda col: 0.0 if col == 1 else 64.0
    _, _, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)
    assert report['receptors'] == 1
    assert count == 1


# ── per-column image override + missing sprite ───────────────────────

def test_per_column_tap_override():
    image_map = {'receptor': (100, 1.0, 1.0), 'tap': (200, _SPRITE_W, _SPRITE_H),
                 'tap_1': (201, _SPRITE_W, _SPRITE_H)}
    ctx = _ctx([_note_view(col=1, lx=64.0, y=100.0),
                _note_view(col=0, lx=0.0, y=120.0)])
    u, _, count, _ = nf.feed_from_context(ctx, image_map)
    rows = u.reshape(count, nf.FEED_U_STRIDE)
    # col 1 tap -> 201 (override), col 0 tap -> 200 (fallback).
    taps = rows[4:]
    assert list(taps[:, 1]) == [201, 200]


def test_missing_tap_sprite_counts_skip():
    ctx = _ctx([_note_view(col=0, lx=0.0, y=100.0)], keycount=1)
    u, _, count, report = nf.feed_from_context(ctx, {'receptor': (100, 1.0, 1.0)})
    assert report['taps'] == 0
    assert report['skipped'] >= 1
    assert count == 1  # only the receptor


# ── optional headless smoke (skips unless cheap) ─────────────────────

def test_gat1_smoke_prints_counts():
    """Emission counts at three times on the gat 1 chart. Skipped unless
    a real Player resolves cheaply - the synthetic tests above are the
    gate; this only confirms the real note pipeline feeds the emitter."""
    resolve = pytest.importorskip(
        'tools.render_frames', reason='render harness unavailable')
    player_fn = getattr(resolve, 'resolve_player', None)
    if player_fn is None:
        pytest.skip('no resolve_player entry in tools.render_frames')
    try:
        player = player_fn('gat')
    except Exception as exc:  # harness fights us -> synthetic tests stand
        pytest.skip(f'gat 1 player did not resolve: {exc}')
    image_map = {'receptor': (0, 1.0, 1.0), 'tap': (1, _SPRITE_W, _SPRITE_H)}
    for t in (2.0, 20.0, 40.0):
        _, _, count, report = nf.feed_notes(player, t, image_map)
        print(f'gat1 t={t}: items={count} report={report}')
    assert count >= 0


def test_stealthglow_emits_an_additive_second_pass():
    # A stealthed note (alpha 0) is still visible as LIGHT: the fill drops but
    # an additive glow pass at `glow` strength survives, tinted by glow_rgb.
    views = [_note_view(col=0, lx=0.0, y=50.0, alpha=0.0, glow=0.75,
                        glow_rgb=(1.0, 0.5, 0.25))]
    ctx = _ctx(views)
    u, f, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)

    assert report['taps'] == 0, 'the blanked fill does not draw'
    assert report['glows'] == 1
    assert count == 4 + 1

    i = 4  # after the receptors
    assert int(u[i * nf.FEED_U_STRIDE + 3]) == 1, 'additive flag set'
    row = f[i * nf.FEED_F_STRIDE:(i + 1) * nf.FEED_F_STRIDE]
    assert row[nf._F_OPACITY] == pytest.approx(0.75)
    assert tuple(row[nf._F_TINT:nf._F_TINT + 3]) == pytest.approx(
        (1.0, 0.5, 0.25))


def test_a_lit_note_draws_both_fill_and_glow():
    views = [_note_view(col=0, lx=0.0, y=50.0, alpha=1.0, glow=0.5)]
    ctx = _ctx(views)
    u, _f, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)
    assert (report['taps'], report['glows']) == (1, 1)
    assert count == 4 + 2
    flags = [int(u[i * nf.FEED_U_STRIDE + 3]) for i in (4, 5)]
    assert flags == [0, 1], 'fill source-over, then the additive glow'


def test_head_sprite_resolves_by_column_and_state():
    # The raster cache keys heads on (col, state); the feed resolves the same
    # variant, most specific key first.
    box = (_SPRITE_W, _SPRITE_H)
    image_map = {'receptor': (100, 1.0, 1.0), 'tap': (200, *box),
                 'tap_1': (201, *box), 'tap_miss_tap': (210, *box),
                 'tap_2_miss_tap': (211, *box)}
    views = [_note_view(col=0, lx=0.0, y=50.0),                    # tap
             _note_view(col=1, lx=64.0, y=60.0),                   # tap_1
             _note_view(col=3, lx=192.0, y=70.0, miss=True),       # tap_miss_tap
             _note_view(col=2, lx=128.0, y=80.0, miss=True)]       # tap_2_miss_tap
    ctx = _ctx(views)
    u, _f, count, report = nf.feed_from_context(ctx, image_map)
    assert report['taps'] == 4
    drawn = [int(u[i * nf.FEED_U_STRIDE + 1]) for i in range(4, count)]
    assert drawn == [200, 201, 210, 211]


def test_press_hide_drops_a_pressed_head():
    # Under press_hide a tap disappears once pressed; without it the head
    # keeps drawing. The feed must honour the same gate as the raster layer.
    view = _note_view(col=0, lx=0.0, y=50.0, state='tap', press_t=1.0)
    hidden = _ctx([view])
    hidden.player = SimpleNamespace(press_hide=True)
    hidden.t_now = 2.0
    _, _, _count, report = nf.feed_from_context(hidden, _IMAGE_MAP)
    assert report['taps'] == 0, 'pressed head hidden'

    shown = _ctx([view])
    shown.player = SimpleNamespace(press_hide=True)
    shown.t_now = 0.5
    _, _, _count, report = nf.feed_from_context(shown, _IMAGE_MAP)
    assert report['taps'] == 1, 'not yet pressed - still drawn'


def test_ln_body_emits_a_quad_per_path_segment():
    # A ribbon IS a quad strip: one item per segment, each rotated to that
    # segment's angle, so no separate Lines tier is needed.
    import numpy as _np
    path = (_np.array([64.0, 64.0, 64.0]), _np.array([40.0, 90.0, 150.0]))
    views = [_note_view(col=1, lx=64.0, y=150.0, is_ln=True, y_end=40.0,
                        body_path=path)]
    u, _f, _count, report = nf.feed_from_context(_ctx(views), _IMAGE_MAP)
    assert report['ln_body_segments'] == 2, 'three samples -> two segments'
    body = [int(u[i * nf.FEED_U_STRIDE + 1]) for i in (5, 6)]
    assert body == [240, 240]


def test_ln_body_narrows_where_it_dives_toward_the_camera():
    # body_scale is the per-sample depth foreshortening; a segment whose
    # sample scales down must emit a narrower quad.
    import numpy as _np
    path = (_np.array([64.0, 64.0]), _np.array([40.0, 140.0]))
    views = [_note_view(col=1, lx=64.0, y=150.0, is_ln=True, y_end=40.0,
                        body_path=path, body_scale=_np.array([0.5, 0.5]))]
    _u, f, _count, report = nf.feed_from_context(_ctx(views), _IMAGE_MAP)
    assert report['ln_body_segments'] == 1
    row = f[5 * nf.FEED_F_STRIDE:6 * nf.FEED_F_STRIDE]
    mat = _np.asarray(row[nf._F_MAT:nf._F_MAT + 9], dtype=float).reshape(3, 3)
    # The segment runs straight down, rotated -90+90=0... width rides m01/m11.
    width = _np.hypot(mat[0, 0], mat[1, 0])
    assert width == pytest.approx(64.0 * 0.5, abs=1e-3)


# ── screen -> design conversion ──────────────────────────────────────

def _notitg_ctx(note_views=(), chart_rect=(100.0, 40.0, 1750.0, 900.0)):
    """A ctx shaped like the real NotITG one: `field_geometry` has already
    stretched the engine's 64-design-px grid onto the chart rect, so every
    lane/judge value the emitter reads is in SCREEN px."""
    x, y, w, h = chart_rect
    kx, ky = w / 640.0, h / 480.0
    lane_w = 64.0 * kx
    x0 = x + (320.0 - 64.0 * 4 / 2.0) * kx
    ctx = _ctx(list(note_views), keycount=4, lane_w=lane_w,
               judge_y=y + 115.0 * ky)
    ctx.chart_rect = chart_rect
    ctx.lane_x = lambda col: x0 + col * lane_w
    ctx.lane_center = lambda col: x0 + col * lane_w + lane_w / 2.0
    return ctx


def test_design_converts_screen_geometry_back_to_the_documents_space():
    # The consuming doc's screen is 640x480 and the executor stretches it onto
    # the chart rect at present time; without the conversion that stretch is
    # applied TWICE and every note lands scaled + offset by the rect ratio.
    ctx = _notitg_ctx()
    _u, f, n, _r = nf.feed_from_context(ctx, _IMAGE_MAP, design=(640, 480))
    mat9 = f.reshape(n, nf.FEED_F_STRIDE)[0][nf._F_MAT:nf._F_MAT + 9]
    cx, cy = _apply_mat(mat9, 0.5, 0.5)
    # Column 0's centre on the engine's 64px grid, at the reverse receptor row.
    assert cx == pytest.approx(320.0 - 128.0 + 32.0)
    assert cy == pytest.approx(115.0)
    # The notch spans _RECEPTOR_LANE_FRAC of a 64-design-px lane.
    x0, _ = _apply_mat(mat9, 0.0, 0.5)
    x1, _ = _apply_mat(mat9, 1.0, 0.5)
    assert (x1 - x0) == pytest.approx(64.0 * nf._RECEPTOR_LANE_FRAC)


def test_no_design_leaves_the_ctx_space_untouched():
    # A consumer whose document IS the ctx's screen passes no design size and
    # must get the mat3s verbatim.
    ctx = _notitg_ctx()
    _u, f, n, _r = nf.feed_from_context(ctx, _IMAGE_MAP)
    mat9 = f.reshape(n, nf.FEED_F_STRIDE)[0][nf._F_MAT:nf._F_MAT + 9]
    cx, _cy = _apply_mat(mat9, 0.5, 0.5)
    assert cx == pytest.approx(ctx.lane_center(0))


def test_design_conversion_preserves_a_projective_placement():
    # The bottom row rides through, so a 3D-modded head's homography converts
    # like any other placement instead of collapsing to an affine one.
    mat = (2.0, 0.0, 10.0, 0.0, 3.0, 20.0, 0.001, 0.002, 1.0)
    converted = nf._to_design_mat((0.5, 0.25, -50.0, -10.0), mat)
    assert converted[6:] == mat[6:]
    # A point maps the same as applying the affine after the original mat.
    x, y, w = (mat[0] * 1.0 + mat[1] * 1.0 + mat[2],
               mat[3] * 1.0 + mat[4] * 1.0 + mat[5],
               mat[6] * 1.0 + mat[7] * 1.0 + mat[8])
    want = (0.5 * (x / w) - 50.0, 0.25 * (y / w) - 10.0)
    gx, gy, gw = (converted[0] + converted[1] + converted[2],
                  converted[3] + converted[4] + converted[5],
                  converted[6] + converted[7] + converted[8])
    assert (gx / gw, gy / gw) == pytest.approx(want)
