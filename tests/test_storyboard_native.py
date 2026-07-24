"""The Drawable core's Python surface (storyboard_native): Seam A doc
building, Seam B flat-buffer schedules, and the ordering semantics the
type sheet (.claude/plans/drawable-ir.md) freezes. Rust unit tests own
the fine-grained ordering rules; these guard the boundary contract.
"""
import numpy as np
import pytest

sn = pytest.importorskip('storyboard_native')


def _frames(evaluator, t):
    u_raw, f_raw, _uf_raw, n = evaluator.frame(t)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
    return u, f


def _frames3(evaluator, t):
    """Like _frames but also returns the flat uniform-value buffer."""
    u_raw, f_raw, uf_raw, n = evaluator.frame(t)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
    uf = np.frombuffer(uf_raw, dtype=np.float32)
    return u, f, uf


def test_schedule_round_trip_orders_and_samples():
    b = sn.DocBuilder(640.0, 480.0)
    fade = b.channel([0.0, 2.0], [0.0, 1.0], [2.0, 0.0], 0.0)
    slot = b.drawable(640.0, 480.0, True, False)
    b.item(0, sn.SRC_IMAGE, 3, opacity_id=fade, opacity_rest=0.0)
    b.snapshot(0, slot)
    b.item(0, sn.SRC_DRAWABLE, slot)
    ev = b.finish()

    u, f = _frames(ev, 1.0)
    kinds = u[:, 0].tolist()
    assert kinds == [sn.OP_BEGIN, sn.OP_BLIT, sn.OP_COPY, sn.OP_BLIT,
                     sn.OP_END]
    # The image blit samples the fade ramp (0 -> 1 over 2s) at t=1.
    assert f[1][9] == pytest.approx(0.5)
    # The slot blit reads the snapshotted drawable.
    assert (u[3][1], u[3][2]) == (sn.SRC_DRAWABLE, slot)


def test_referenced_drawable_composes_before_the_screen():
    b = sn.DocBuilder(640.0, 480.0)
    sub = b.drawable(64.0, 64.0, False, False)
    b.item(sub, sn.SRC_IMAGE, 7)
    b.item(0, sn.SRC_DRAWABLE, sub)
    ev = b.finish()

    u, _f = _frames(ev, 0.0)
    begins = u[u[:, 0] == sn.OP_BEGIN][:, 1].tolist()
    assert begins == [sub, 0]


def test_hidden_item_emits_no_op():
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0, visible_id=-1, visible_rest=0.0)
    ev = b.finish()
    u, _f = _frames(ev, 0.0)
    assert u[:, 0].tolist() == [sn.OP_BEGIN, sn.OP_END]


def test_shader_uniform_values_ride_the_side_buffer():
    b = sn.DocBuilder(640.0, 480.0)
    sh = b.shader('frag-src', None, ['strength', 'phase'])
    # A plain fill binds no shader; a shaded fill binds two uniforms, one
    # a ramp (0 -> 4 over 2s) sampled at t=1, one a constant.
    b.item(0, sn.SRC_FILL, 0)
    ramp = b.channel([0.0, 2.0], [0.0, 4.0], [2.0, 0.0], 0.0)
    b.item(0, sn.SRC_FILL, 0)
    b.item_shader(0, sh)
    b.item_uniform(0, 0, ramp, 0.0)   # strength <- ramp
    b.item_uniform(0, 1, -1, 9.0)     # phase <- constant 9
    ev = b.finish()

    u, _f, uf = _frames3(ev, 1.0)
    blits = u[u[:, 0] == sn.OP_BLIT]
    assert len(blits) == 2
    # Plain blit: no shader, zero uniform count.
    assert blits[0][5] == 0 and blits[0][9] == 0
    # Shaded blit: shader+1, offset/count into the uniform buffer.
    assert blits[1][5] == sh + 1
    off, cnt = int(blits[1][8]), int(blits[1][9])
    assert cnt == 2
    assert uf[off] == pytest.approx(2.0)   # ramp at t=1
    assert uf[off + 1] == pytest.approx(9.0)


def test_clip_id_rides_lane_six():
    b = sn.DocBuilder(640.0, 480.0)
    rect = b.clip_rect(0.0, 0.0, 320.0, 240.0)
    poly = b.clip_polygon([0.0, 0.0, 10.0, 0.0, 10.0, 10.0])
    b.item(0, sn.SRC_FILL, 0)
    b.item_clip(0, rect)
    b.item(0, sn.SRC_FILL, 0)
    b.item_clip(0, poly)
    ev = b.finish()

    u, _f = _frames(ev, 0.0)
    clips = u[u[:, 0] == sn.OP_BLIT][:, 6].tolist()
    assert clips == [rect + 1, poly + 1]


def test_frame_with_feeds_ingests_soa_and_matches_static():
    b = sn.DocBuilder(640.0, 480.0)
    notes = b.drawable(640.0, 480.0, False, True)  # dynamic
    b.item(0, sn.SRC_DRAWABLE, notes)
    ev = b.finish()

    # Two fed items in the frozen SoA layout: u32 stride 4, f32 stride 14.
    fu = ev.feed_u_stride
    ff = ev.feed_f_stride
    assert (fu, ff) == (4, 14)
    ADDITIVE = 1
    u = np.zeros((2, fu), dtype=np.uint32)
    f = np.zeros((2, ff), dtype=np.float32)
    # item 0: image 5, additive, opacity 0.5, positioned at (10, 20).
    u[0] = [sn.SRC_IMAGE, 5, 0, ADDITIVE]
    f[0, :6] = [10.0, 20.0, 1.0, 1.0, 0.0, 0.5]
    # item 1: image 6, plain, opacity 1.0.
    u[1] = [sn.SRC_IMAGE, 6, 0, 0]
    f[1, :6] = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    u_raw, f_raw, _uf_raw, n = ev.frame_with_feeds(
        0.0, [notes], [2], u.tobytes(), f.tobytes())
    uu = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, ev.u_stride)
    fu_out = np.frombuffer(f_raw, dtype=np.float32).reshape(n, ev.f_stride)

    blits = uu[uu[:, 0] == sn.OP_BLIT]
    # Two fed image blits, then the screen's drawable blit consuming them.
    assert [(int(r[1]), int(r[2])) for r in blits] == [
        (sn.SRC_IMAGE, 5), (sn.SRC_IMAGE, 6), (sn.SRC_DRAWABLE, notes)]
    # The additive flag reached the blit's blend lane (lane 4).
    assert blits[0][4] == 1 and blits[1][4] == 0
    # The fed position/opacity survived: item 0's mat3 tx/ty = 10/20,
    # opacity 0.5 (f lane 9).
    fed0 = fu_out[uu[:, 0] == sn.OP_BLIT][0]
    assert fed0[2] == pytest.approx(10.0)   # mat[0][2] = x
    assert fed0[5] == pytest.approx(20.0)   # mat[1][2] = y
    assert fed0[9] == pytest.approx(0.5)    # opacity

    # frame() with no feeds leaves the dynamic drawable empty (only the
    # screen's consuming blit, which reads the retained/empty target).
    u2, _f2 = _frames(ev, 0.0)
    static_blits = u2[u2[:, 0] == sn.OP_BLIT]
    assert [(int(r[1]), int(r[2])) for r in static_blits] == [
        (sn.SRC_DRAWABLE, notes)]
