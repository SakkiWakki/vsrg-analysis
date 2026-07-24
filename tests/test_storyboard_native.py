"""The Drawable core's Python surface (storyboard_native): Seam A doc
building, Seam B flat-buffer schedules, and the ordering semantics the
type sheet (.claude/plans/drawable-ir.md) freezes. Rust unit tests own
the fine-grained ordering rules; these guard the boundary contract.
"""
import numpy as np
import pytest

sn = pytest.importorskip('storyboard_native')


def _frames(evaluator, t):
    u_raw, f_raw, n = evaluator.frame(t)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
    return u, f


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
