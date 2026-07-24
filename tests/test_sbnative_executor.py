"""Golden-pixel tests for the Seam-B raster executor
(analysis/player/render/storyboard/executor.py).

Offscreen QPainter/QImage: build DrawSchedule records with the real
storyboard_native DocBuilder + Evaluator, run them through
RasterExecutor, and assert concrete pixel outcomes for the four
behaviours the port must preserve:

  1. painter's algorithm - a later blit covers an earlier one;
  2. the monitor-class regression in miniature - Snapshot a drawable
     into a slot, paint a fullscreen black Fill over it, then blit the
     slot: the pre-curtain content shows, not black;
  3. feedback - a persistent (Retain) drawable keeps content across two
     execute() calls;
  4. an opacity channel ramp changes pixel intensity between two times.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

sn = pytest.importorskip("storyboard_native")
pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage

from analysis.player.render.storyboard.executor import RasterExecutor


def _frames(evaluator, t):
    u_raw, f_raw, _uf_raw, n = evaluator.frame(t)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
    return u, f


def _solid(w, h, color):
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(color)
    return img


def _rgb(img, x, y):
    c = img.pixelColor(x, y)
    return (c.red(), c.green(), c.blue())


def _alpha(img, x, y):
    return img.pixelColor(x, y).alpha()


def test_painters_algorithm_later_blit_covers_earlier():
    b = sn.DocBuilder(4.0, 4.0)
    # Two opaque fullscreen images: red first, green second. Scale a 1x1
    # source up to cover the 4x4 screen.
    b.item(0, sn.SRC_IMAGE, 0, sx_rest=4.0, sy_rest=4.0)
    b.item(0, sn.SRC_IMAGE, 1, sx_rest=4.0, sy_rest=4.0)
    ev = b.finish()

    images = {
        0: _solid(1, 1, QColor(255, 0, 0, 255)),
        1: _solid(1, 1, QColor(0, 255, 0, 255)),
    }
    ex = RasterExecutor(images, [(4.0, 4.0)])
    u, f = _frames(ev, 0.0)
    screen = ex.execute(u, f)

    assert _rgb(screen, 2, 2) == (0, 255, 0)  # green (second) wins


def test_snapshot_shows_pre_curtain_content_not_black():
    # The monitor-class regression, in miniature:
    #   1. draw an image onto the screen
    #   2. Snapshot the screen into a slot (captures the image)
    #   3. paint a fullscreen BLACK Fill (the curtain)
    #   4. blit the slot back on top -> shows the pre-curtain image.
    b = sn.DocBuilder(4.0, 4.0)
    slot = b.drawable(4.0, 4.0, True, False)
    b.item(0, sn.SRC_IMAGE, 0, sx_rest=4.0, sy_rest=4.0)  # the image
    b.snapshot(0, slot)                                    # capture at position
    b.item(0, sn.SRC_FILL, 0, sx_rest=4.0, sy_rest=4.0)    # black curtain (tint defaults white...)
    b.item(0, sn.SRC_DRAWABLE, slot)                       # slot back on top
    ev = b.finish()

    images = {0: _solid(1, 1, QColor(40, 120, 200, 255))}
    ex = RasterExecutor(images, [(4.0, 4.0), (4.0, 4.0)])
    u, f = _frames(ev, 0.0)

    # The Fill tint defaults to white; force it black so the curtain is
    # genuinely black. Rewrite the tint lanes of the Fill op in-place.
    fill_rows = [i for i in range(u.shape[0])
                 if u[i, 0] == sn.OP_BLIT and u[i, 1] == sn.SRC_FILL]
    assert fill_rows, "expected a Fill blit op"
    f = f.copy()
    for i in fill_rows:
        f[i, 10:13] = 0.0  # tint rgb -> black

    screen = ex.execute(u, f)
    # The slot was snapshotted BEFORE the curtain, so the top blit paints
    # the original image, not black.
    assert _rgb(screen, 2, 2) == (40, 120, 200)


def test_persistent_drawable_retains_across_executes_feedback():
    # A persistent (Retain) drawable keeps its content between calls. We
    # paint into a Retain slot on frame 1 and only READ it (blit to
    # screen) on frame 2 without repainting it - the content survives.
    build = sn.DocBuilder(4.0, 4.0)
    slot = build.drawable(4.0, 4.0, True, False)  # persistent -> Retain
    build.item(slot, sn.SRC_IMAGE, 0, sx_rest=4.0, sy_rest=4.0)
    build.item(0, sn.SRC_DRAWABLE, slot)
    ev = build.finish()

    images = {0: _solid(1, 1, QColor(200, 50, 50, 255))}
    ex = RasterExecutor(images, [(4.0, 4.0), (4.0, 4.0)])

    u, f = _frames(ev, 0.0)
    screen1 = ex.execute(u, f)
    assert _rgb(screen1, 2, 2) == (200, 50, 50)

    # Second execute of the SAME schedule: the slot's Retain BEGIN keeps
    # last frame's pixels; even though the image is re-blitted here, the
    # retention is what makes a feedback chain stable. Verify the slot
    # target still holds content and the screen still shows it.
    screen2 = ex.execute(u, f)
    assert _rgb(screen2, 2, 2) == (200, 50, 50)
    assert _alpha(ex._targets[slot], 2, 2) == 255


def test_persistent_retain_survives_when_not_repainted():
    # Sharper feedback check: frame 1 fills the Retain slot; frame 2 uses
    # a DIFFERENT schedule that never draws into the slot, only reads it.
    # Retain means the slot still carries frame 1's pixels.
    b1 = sn.DocBuilder(4.0, 4.0)
    slot = b1.drawable(4.0, 4.0, True, False)
    b1.item(slot, sn.SRC_IMAGE, 0, sx_rest=4.0, sy_rest=4.0)
    ev1 = b1.finish()

    b2 = sn.DocBuilder(4.0, 4.0)
    slot2 = b2.drawable(4.0, 4.0, True, False)  # same id (1)
    b2.item(0, sn.SRC_DRAWABLE, slot2)          # read the slot onto the screen
    ev2 = b2.finish()
    assert slot == slot2

    images = {0: _solid(1, 1, QColor(10, 220, 30, 255))}
    ex = RasterExecutor(images, [(4.0, 4.0), (4.0, 4.0)])

    u1, f1 = _frames(ev1, 0.0)
    ex.execute(u1, f1)  # paints the slot, screen untouched

    u2, f2 = _frames(ev2, 0.0)
    screen = ex.execute(u2, f2)  # reads the retained slot
    assert _rgb(screen, 2, 2) == (10, 220, 30)


def test_opacity_ramp_changes_intensity_between_times():
    # An opacity channel ramps 0 -> 1 over 2s. A blit of a white image
    # over a black screen should be dimmer at t=0.5 than at t=1.5.
    b = sn.DocBuilder(4.0, 4.0)
    fade = b.channel([0.0, 2.0], [0.0, 1.0], [2.0, 0.0], 0.0)
    b.item(0, sn.SRC_IMAGE, 0, sx_rest=4.0, sy_rest=4.0,
           opacity_id=fade, opacity_rest=0.0)
    ev = b.finish()

    images = {0: _solid(1, 1, QColor(255, 255, 255, 255))}

    ex_a = RasterExecutor(dict(images), [(4.0, 4.0)])
    early = ex_a.execute(*_frames(ev, 0.5))
    v_early = early.pixelColor(2, 2).red()

    ex_b = RasterExecutor(dict(images), [(4.0, 4.0)])
    late = ex_b.execute(*_frames(ev, 1.5))
    v_late = late.pixelColor(2, 2).red()

    assert v_early < v_late
    # 0.25 opacity over opaque black -> ~64; 0.75 -> ~191 (premultiplied).
    assert v_early == pytest.approx(64, abs=6)
    assert v_late == pytest.approx(191, abs=6)
