"""Golden-pixel tests for the Seam-B GL executor
(analysis/player/render/storyboard/gl_executor.py).

The GL twin of test_sbnative_executor.py: build the SAME DrawSchedule
records with the real storyboard_native DocBuilder + Evaluator, run them
through GLExecutor on an offscreen GL context, and assert the same pixel
outcomes as the raster golden tests for the behaviours the port must
preserve GL-side:

  1. painter's algorithm - a later blit covers an earlier one;
  2. the monitor-class regression in miniature - Snapshot a drawable into
     a slot, paint a fullscreen black Fill over it, then blit the slot:
     the pre-curtain content shows, not black;
  3. an opacity channel ramp changes pixel intensity between two times.

Every test needs a current GL 3+ context. The module-scoped ``gl``
fixture (the QOffscreenSurface pattern from test_notitg_shader_bridge.py)
skips the whole module when no context can be made, so a headless box
without GL stays green (all-skipped).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

sn = pytest.importorskip("storyboard_native")
pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage

from analysis.player.render.storyboard.gl_executor import GLExecutor


@pytest.fixture(scope="module")
def gl(_qapp):
    from PySide6.QtGui import (QOffscreenSurface, QOpenGLContext,
                               QSurfaceFormat)
    fmt = QSurfaceFormat()
    fmt.setMajorVersion(3)
    fmt.setMinorVersion(2)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    context = QOpenGLContext()
    context.setFormat(fmt)
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    if not (context.create() and surface.isValid()
            and context.makeCurrent(surface)):
        pytest.skip("no OpenGL context on this platform")
    yield context
    context.doneCurrent()


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


def test_painters_algorithm_later_blit_covers_earlier(gl):
    # Two opaque fullscreen images: red first, green second. Scale a 1x1
    # source up to cover the screen - green (second) must win.
    b = sn.DocBuilder(4.0, 4.0)
    b.item(0, sn.SRC_IMAGE, 0, sx_rest=4.0, sy_rest=4.0)
    b.item(0, sn.SRC_IMAGE, 1, sx_rest=4.0, sy_rest=4.0)
    ev = b.finish()

    images = {
        0: _solid(1, 1, QColor(255, 0, 0, 255)),
        1: _solid(1, 1, QColor(0, 255, 0, 255)),
    }
    ex = GLExecutor(images, [(4.0, 4.0)])
    u, f = _frames(ev, 0.0)
    screen = ex.execute(u, f)

    assert not ex.broken
    assert _rgb(screen, 2, 2) == (0, 255, 0)


def test_snapshot_shows_pre_curtain_content_not_black(gl):
    # The monitor-class regression, GL-side:
    #   1. draw an image onto the screen
    #   2. Snapshot the screen into a slot (captures the image)
    #   3. paint a fullscreen BLACK Fill (the curtain)
    #   4. blit the slot back on top -> shows the pre-curtain image.
    b = sn.DocBuilder(4.0, 4.0)
    slot = b.drawable(4.0, 4.0, True, False)
    b.item(0, sn.SRC_IMAGE, 0, sx_rest=4.0, sy_rest=4.0)   # the image
    b.snapshot(0, slot)                                    # capture at position
    b.item(0, sn.SRC_FILL, 0, sx_rest=4.0, sy_rest=4.0)    # curtain (tint set black below)
    b.item(0, sn.SRC_DRAWABLE, slot)                       # slot back on top
    ev = b.finish()

    images = {0: _solid(1, 1, QColor(40, 120, 200, 255))}
    ex = GLExecutor(images, [(4.0, 4.0), (4.0, 4.0)])
    u, f = _frames(ev, 0.0)

    # The Fill tint defaults to white; force it black so the curtain is
    # genuinely black.
    fill_rows = [i for i in range(u.shape[0])
                 if u[i, 0] == sn.OP_BLIT and u[i, 1] == sn.SRC_FILL]
    assert fill_rows, "expected a Fill blit op"
    f = f.copy()
    for i in fill_rows:
        f[i, 10:13] = 0.0  # tint rgb -> black

    screen = ex.execute(u, f)
    assert not ex.broken
    # The slot was snapshotted BEFORE the curtain, so the top blit paints
    # the original image, not black.
    assert _rgb(screen, 2, 2) == pytest.approx((40, 120, 200), abs=2)


def test_opacity_ramp_changes_intensity_between_times(gl):
    # An opacity channel ramps 0 -> 1 over 2s. A white image over the
    # (opaque black) screen is dimmer at t=0.5 than at t=1.5.
    b = sn.DocBuilder(4.0, 4.0)
    fade = b.channel([0.0, 2.0], [0.0, 1.0], [2.0, 0.0], 0.0)
    b.item(0, sn.SRC_IMAGE, 0, sx_rest=4.0, sy_rest=4.0,
           opacity_id=fade, opacity_rest=0.0)
    ev = b.finish()

    images = {0: _solid(1, 1, QColor(255, 255, 255, 255))}

    ex_a = GLExecutor(dict(images), [(4.0, 4.0)])
    early = ex_a.execute(*_frames(ev, 0.5))
    v_early = early.pixelColor(2, 2).red()

    ex_b = GLExecutor(dict(images), [(4.0, 4.0)])
    late = ex_b.execute(*_frames(ev, 1.5))
    v_late = late.pixelColor(2, 2).red()

    assert not ex_a.broken and not ex_b.broken
    assert v_early < v_late
    # 0.25 opacity over black -> ~64; 0.75 -> ~191 (premultiplied), same
    # arithmetic as the raster golden.
    assert v_early == pytest.approx(64, abs=8)
    assert v_late == pytest.approx(191, abs=8)
