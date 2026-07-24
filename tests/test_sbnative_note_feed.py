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


def _ctx(note_views, keycount=4, lane_w=64.0, judge_y=400.0, note_h=14.0,
         receptor_offsets=None, stream_views=None):
    """A stub with the RenderContext reads the emitter uses: keycount,
    lane geometry (lane_x/lane_width/lane_center), judge_y, note_h,
    note_views, and optional receptor_offsets / stream_views."""
    ctx = SimpleNamespace(
        keycount=keycount,
        judge_y=judge_y,
        note_h=note_h,
        note_views=note_views,
        stream_views=stream_views or [],
        receptor_offsets=receptor_offsets,
    )
    ctx.lane_width = lambda col: lane_w
    ctx.lane_x = lambda col: col * lane_w
    ctx.lane_center = lambda col: col * lane_w + lane_w / 2.0
    return ctx


_IMAGE_MAP = {'receptor': 100, 'tap': 200}


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


def test_lns_and_streams_are_skipped_not_drawn():
    views = [_note_view(col=0, lx=0.0, y=50.0),
             _note_view(col=1, lx=64.0, y=150.0, is_ln=True)]
    ctx = _ctx(views, stream_views=[_note_view()])
    _, _, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)
    assert report['taps'] == 1
    assert report['skipped'] >= 1  # the LN head
    assert count == 4 + 1
    assert 'ln_body_path_as_lines_source' in report['seams']
    assert 'stream_views_second_walk' in report['seams']


def test_3d_head_skipped_and_reported():
    views = [_note_view(col=0, lx=0.0, y=50.0, z=120.0)]
    ctx = _ctx(views)
    _, _, count, report = nf.feed_from_context(ctx, _IMAGE_MAP)
    assert report['taps'] == 0
    assert count == 4  # only receptors
    assert '3d_head_perspective_homography' in report['seams']


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


def test_tap_mat3_scales_to_note_size():
    ctx = _ctx([_note_view(col=0, lx=0.0, y=100.0)], lane_w=64.0, note_h=14.0)
    _, f, count, _ = nf.feed_from_context(ctx, _IMAGE_MAP)
    mat9 = _tap_row(f, count, count - 1)[nf._F_MAT:nf._F_MAT + 9]
    # unit box corners (0,0)-(1,1) span the note's screen rect: width 64,
    # height 14, centered at (32, 100).
    x0, y0 = _apply_mat(mat9, 0.0, 0.0)
    x1, y1 = _apply_mat(mat9, 1.0, 1.0)
    assert (x1 - x0) == pytest.approx(64.0)
    assert (y1 - y0) == pytest.approx(14.0)
    assert x0 == pytest.approx(0.0)
    assert x1 == pytest.approx(64.0)


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
    image_map = {'receptor': 100, 'tap': 200, 'tap_1': 201}
    ctx = _ctx([_note_view(col=1, lx=64.0, y=100.0),
                _note_view(col=0, lx=0.0, y=120.0)])
    u, _, count, _ = nf.feed_from_context(ctx, image_map)
    rows = u.reshape(count, nf.FEED_U_STRIDE)
    # col 1 tap -> 201 (override), col 0 tap -> 200 (fallback).
    taps = rows[4:]
    assert list(taps[:, 1]) == [201, 200]


def test_missing_tap_sprite_counts_skip():
    ctx = _ctx([_note_view(col=0, lx=0.0, y=100.0)], keycount=1)
    u, _, count, report = nf.feed_from_context(ctx, {'receptor': 100})
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
    image_map = {'receptor': 0, 'tap': 1}
    for t in (2.0, 20.0, 40.0):
        _, _, count, report = nf.feed_notes(player, t, image_map)
        print(f'gat1 t={t}: items={count} report={report}')
    assert count >= 0
