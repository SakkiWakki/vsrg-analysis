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


def _frames_uf(evaluator, t):
    u_raw, f_raw, uf_raw, n = evaluator.frame(t)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
    uf = np.frombuffer(uf_raw, dtype=np.float32)
    return u, f, uf


def _paint_fill_tint(u, f, rgb):
    """Return a writable copy of f with every SRC_FILL blit's tint set to
    rgb (the DocBuilder defaults a Fill's tint to white)."""
    f = f.copy()
    for i in range(u.shape[0]):
        if u[i, 0] == sn.OP_BLIT and u[i, 1] == sn.SRC_FILL:
            f[i, 10:13] = rgb
    return f


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


# --- B2 additions: clip consumption, Lines source, uniform arrival ---


def test_clipped_fullscreen_fill_colors_only_the_clip_region():
    # A green fullscreen Fill clipped to a centered rect (2,2)-(6,6) on an
    # 8x8 screen. The screen clears opaque black, so pixels inside the clip
    # go green and everything outside stays the cleared black. The clip
    # shape is in the TARGET's logical units.
    b = sn.DocBuilder(8.0, 8.0)
    clip = b.clip_rect(2.0, 2.0, 6.0, 6.0)
    b.item(0, sn.SRC_FILL, 0, sx_rest=8.0, sy_rest=8.0)
    b.item_clip(0, clip)
    ev = b.finish()

    u, f = _frames(ev, 0.0)
    f = _paint_fill_tint(u, f, (0.0, 1.0, 0.0))
    ex = RasterExecutor({}, [(8.0, 8.0)], clips=[("rect", 2.0, 2.0, 6.0, 6.0)])
    screen = ex.execute(u, f)

    assert _rgb(screen, 4, 4) == (0, 255, 0)   # inside the clip: green
    assert _rgb(screen, 0, 0) == (0, 0, 0)     # outside: cleared black
    assert _rgb(screen, 7, 7) == (0, 0, 0)


def test_polygon_clip_confines_fill_to_the_triangle():
    # A blue fullscreen Fill clipped to a right-triangle covering the
    # top-left half: the opposite corner stays black.
    tri = [(1.0, 1.0), (14.0, 1.0), (1.0, 14.0)]
    b = sn.DocBuilder(16.0, 16.0)
    clip = b.clip_polygon([c for xy in tri for c in xy])
    b.item(0, sn.SRC_FILL, 0, sx_rest=16.0, sy_rest=16.0)
    b.item_clip(0, clip)
    ev = b.finish()

    u, f = _frames(ev, 0.0)
    f = _paint_fill_tint(u, f, (0.0, 0.0, 1.0))
    ex = RasterExecutor({}, [(16.0, 16.0)], clips=[("poly", tri)])
    screen = ex.execute(u, f)

    assert _rgb(screen, 3, 3) == (0, 0, 255)   # inside the triangle: blue
    assert _rgb(screen, 13, 13) == (0, 0, 0)   # opposite corner: black


def test_lines_source_strokes_a_diagonal_polyline():
    # A red polyline from (2,2) to (13,13) on a 16x16 screen, drawn under
    # an identity transform. Vertices arrive via the `lines` ctor arg; the
    # tint is the stroke color. On-diagonal pixels light red, an
    # off-diagonal corner stays the cleared black.
    b = sn.DocBuilder(16.0, 16.0)
    b.item(0, sn.SRC_LINES, 5, sx_rest=1.0, sy_rest=1.0)
    ev = b.finish()

    u, f = _frames(ev, 0.0)
    f = f.copy()
    for i in range(u.shape[0]):
        if u[i, 0] == sn.OP_BLIT and u[i, 1] == sn.SRC_LINES:
            f[i, 10:13] = (1.0, 0.0, 0.0)  # red stroke
    verts = np.array([[2.0, 2.0], [13.0, 13.0]], dtype=np.float32)
    ex = RasterExecutor({}, [(16.0, 16.0)], lines={5: verts})
    screen = ex.execute(u, f)

    on = screen.pixelColor(7, 7)
    assert on.red() > 120 and on.green() < 80   # on the diagonal: red
    assert _rgb(screen, 2, 13) == (0, 0, 0)     # off-diagonal corner: black

    # set_lines swaps to the anti-diagonal; the old corner now lights.
    ex.set_lines(5, np.array([[2.0, 13.0], [13.0, 2.0]], dtype=np.float32))
    screen2 = ex.execute(u, f)
    anti = screen2.pixelColor(2, 13)
    assert anti.red() > 120 and anti.green() < 80


def test_shader_uniform_values_arrive_bound_per_shader_id():
    # A shaded Fill binds two uniforms (amp=2.5, freq=9.0). Their sampled
    # values ride the third `uf` buffer; the executor stashes the last-seen
    # dict per shader id (introspectable). The raster backend still draws
    # unshaded - only the binding is asserted here.
    b = sn.DocBuilder(4.0, 4.0)
    sh = b.shader("void main(){}", None, ["amp", "freq"])
    b.item(0, sn.SRC_FILL, 0, sx_rest=4.0, sy_rest=4.0)
    b.item_shader(0, sh)
    b.item_uniform(0, 0, -1, 2.5)
    b.item_uniform(0, 1, -1, 9.0)
    ev = b.finish()

    u, f, uf = _frames_uf(ev, 0.0)
    ex = RasterExecutor({}, [(4.0, 4.0)])
    ex.execute(u, f, uf)
    assert ex.shader_uniforms[sh] == pytest.approx([2.5, 9.0], abs=1e-4)

    # Without the uf buffer, nothing is stashed (the arg is optional).
    ex_no_uf = RasterExecutor({}, [(4.0, 4.0)])
    ex_no_uf.execute(u, f)
    assert ex_no_uf.shader_uniforms == {}


def test_shader_uniform_channel_ramp_arrives_time_sampled():
    # Uniforms are sampled per frame from their channel: an amp channel
    # ramping 0 -> 4 over 2s arrives as ~1.0 at t=0.5 and ~3.0 at t=1.5.
    b = sn.DocBuilder(4.0, 4.0)
    sh = b.shader("void main(){}", None, ["amp"])
    amp = b.channel([0.0, 2.0], [0.0, 4.0], [2.0, 0.0], 0.0)
    b.item(0, sn.SRC_FILL, 0, sx_rest=4.0, sy_rest=4.0)
    b.item_shader(0, sh)
    b.item_uniform(0, 0, amp, 0.0)
    ev = b.finish()

    ex_a = RasterExecutor({}, [(4.0, 4.0)])
    ex_a.execute(*_frames_uf(ev, 0.5))
    ex_b = RasterExecutor({}, [(4.0, 4.0)])
    ex_b.execute(*_frames_uf(ev, 1.5))

    assert ex_a.shader_uniforms[sh][0] == pytest.approx(1.0, abs=0.1)
    assert ex_b.shader_uniforms[sh][0] == pytest.approx(3.0, abs=0.1)
