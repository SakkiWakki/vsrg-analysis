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

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter

from analysis.player.render.storyboard.executor import (  # noqa: E402
    CLEAR_OPAQUE, CLEAR_TRANSPARENT, RasterExecutor, SCREEN_ID)


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


# --- D1 additions: set_clear override + set_drawable_image ingestion ---


def _screen_only_doc(w, h):
    """A doc whose screen root (id 0) just draws a fullscreen white image,
    so the ONLY thing that decides a background pixel's color is the
    screen's clear mode."""
    b = sn.DocBuilder(float(w), float(h))
    b.item(0, sn.SRC_IMAGE, 0, sx_rest=float(w), sy_rest=float(h))
    return b.finish()


def test_set_clear_makes_the_screen_transparent_off_the_content():
    # The screen root is minted OpaqueBlack: a corner the fullscreen image
    # does not fully cover is opaque black by default. set_clear to
    # TransparentBlack makes that same corner transparent (alpha 0) - the
    # black-chart-region fix in miniature (the composite overlays instead
    # of covering).
    ev = _screen_only_doc(4, 4)
    # A 2x2 image placed only over the top-left quadrant leaves the rest
    # to the clear color. Shrink the fullscreen blit to a 2x2 sub-quad by
    # rewriting its mat3 scale so (3,3) is uncovered.
    u, f = _frames(ev, 0.0)
    f = f.copy()
    for i in range(u.shape[0]):
        if u[i, 0] == sn.OP_BLIT and u[i, 1] == sn.SRC_IMAGE:
            f[i, 0] = 2.0   # m00: cover only 2 px wide
            f[i, 4] = 2.0   # m11: cover only 2 px tall
    images = {0: _solid(1, 1, QColor(255, 255, 255, 255))}

    opaque = RasterExecutor(dict(images), [(4.0, 4.0)])
    opaque.set_clear(SCREEN_ID, CLEAR_OPAQUE)
    scr_op = opaque.execute(u, f)
    assert _alpha(scr_op, 3, 3) == 255           # opaque clear
    assert _rgb(scr_op, 3, 3) == (0, 0, 0)       # ...and black

    transparent = RasterExecutor(dict(images), [(4.0, 4.0)])
    transparent.set_clear(SCREEN_ID, CLEAR_TRANSPARENT)
    scr_tr = transparent.execute(u, f)
    assert _alpha(scr_tr, 3, 3) == 0             # uncovered corner: clear
    assert _rgb(scr_tr, 0, 0) == (255, 255, 255)  # the image still draws


def test_set_drawable_image_feeds_command_less_field_drawable():
    # A command-less field drawable (non-persistent, non-dynamic) carries
    # no content of its own; the screen reads it via SRC_DRAWABLE. Feeding
    # a QImage with set_drawable_image makes that blit draw the fed pixels;
    # un-seeding drops them (the drawable is non-persistent - it holds only
    # what is fed this frame).
    b = sn.DocBuilder(4.0, 4.0)
    field = b.drawable(4.0, 4.0, False, False)   # command-less field scope
    b.item(0, sn.SRC_DRAWABLE, field, sx_rest=1.0, sy_rest=1.0)
    ev = b.finish()

    ex = RasterExecutor({}, [(4.0, 4.0), (4.0, 4.0)])
    ex.set_clear(SCREEN_ID, CLEAR_TRANSPARENT)
    u, f = _frames(ev, 0.0)

    # No content yet: the field read resolves to nothing, screen stays clear.
    assert _alpha(ex.execute(u, f), 2, 2) == 0

    content = _solid(4, 4, QColor(0, 200, 0, 255))
    ex.set_drawable_image(field, content)
    fed = ex.execute(u, f)
    assert _rgb(fed, 2, 2) == (0, 200, 0)        # the fed field pixels show

    # Un-seed: the stale target is dropped, the field reads empty again.
    ex.set_drawable_image(field, None)
    assert _alpha(ex.execute(u, f), 2, 2) == 0


# --- E1: source-space normalization (the zoom fix) ---


def test_oversized_content_normalizes_to_the_logical_box():
    # THE ZOOM FIX (drawable-ir.md rule 5): a 1280x960 image injected into a
    # 640x480 field drawable covers that drawable's LOGICAL box regardless of
    # its pixel size. A fullscreen SRC_DRAWABLE blit (identity mat3, scaled to
    # the field's 640x480 logical box) then shows the WHOLE image scaled down,
    # not the top-left quarter. Without normalization the field read spanned
    # 1280x960 source-logical units, so a 640x480 screen showed only a
    # quadrant - the "too zoomed in" bug.
    b = sn.DocBuilder(640.0, 480.0)
    field = b.drawable(640.0, 480.0, False, False)  # command-less field scope
    # Identity mat3: the field's 640x480 LOGICAL box maps 1:1 onto the 640x480
    # screen. The backing image is 1280x960 px - normalization must map the
    # whole image across that logical box (not span 1280x960 source units).
    b.item(0, sn.SRC_DRAWABLE, field, sx_rest=1.0, sy_rest=1.0)
    ev = b.finish()

    # A 1280x960 image: left half red, right half green; top half distinct
    # from bottom via a blue tint band, so we can prove the WHOLE image maps
    # into the screen (all four quadrants land where they should).
    content = QImage(1280, 960, QImage.Format.Format_ARGB32_Premultiplied)
    content.fill(QColor(0, 0, 0, 255))
    p = QPainter(content)
    p.fillRect(0, 0, 640, 960, QColor(255, 0, 0, 255))       # left half red
    p.fillRect(640, 0, 640, 960, QColor(0, 255, 0, 255))     # right half green
    p.fillRect(0, 0, 1280, 20, QColor(0, 0, 255, 255))       # a thin top band blue
    p.end()

    ex = RasterExecutor({}, [(640.0, 480.0), (640.0, 480.0)])
    ex.set_clear(SCREEN_ID, CLEAR_TRANSPARENT)
    ex.set_drawable_image(field, content)
    u, f = _frames(ev, 0.0)
    screen = ex.execute(u, f)

    # The screen is 640x480. The whole 1280x960 image is squeezed into it:
    # a point in the LEFT half of the screen samples the image's red left
    # half; a point in the RIGHT half samples the green right half. If the
    # blit were "too zoomed in" (1280 logical units into 640 px) the right
    # half of the screen would sample the image's HORIZONTAL CENTER, still
    # red - so green on the right proves the fix.
    assert _rgb(screen, 100, 240) == (255, 0, 0)    # left third: red
    assert _rgb(screen, 540, 240) == (0, 255, 0)    # right third: green
    # The top blue band (top ~2% of the image) maps to the top ~2% of the
    # screen (~10 px), proving vertical normalization too.
    assert _rgb(screen, 320, 4) == (0, 0, 255)      # near the very top: blue
    assert _rgb(screen, 320, 240) in ((255, 0, 0), (0, 255, 0))  # mid: not blue


# --- A4: retain decay (engine PreserveTexture accumulate-with-decay) ---

# Hand-built Seam-B records: a decay-only frame needs a Retain BEGIN with NO
# item, which the current DocBuilder does not emit for an item-less drawable,
# so these rows are built directly.
#
# Strides and lane offsets come from the record mirror, NEVER restated. A
# literal `_F_STRIDE = 20` here indexed past the end of its own row the moment
# the doc grew the box lanes, and the failure surfaced inside the executor
# rather than in the test that was wrong.
from analysis.player.render.storyboard import record as _rec  # noqa: E402

_CLEAR_RETAIN_CODE = _rec.CLEAR_RETAIN


def _rec_row(kind, a=0, b=0, mat=None, opacity=1.0, tint=(1.0, 1.0, 1.0)):
    u = np.zeros(_rec.U_STRIDE, dtype=np.uint32)
    u[_rec.U_KIND], u[_rec.U_A], u[_rec.U_B] = kind, a, b
    f = np.zeros(_rec.F_STRIDE, dtype=np.float32)
    f[:9] = mat if mat is not None else [1, 0, 0, 0, 1, 0, 0, 0, 1]
    f[_rec.F_OPACITY] = opacity
    f[_rec.F_TINT:_rec.F_TINT + 3] = tint
    # A record the evaluator builds carries "keep the natural box" on the
    # size lanes; zeros would mean an explicit zero-size draw.
    f[_rec.F_SIZE:_rec.F_SIZE + 2] = _rec.SIZE_NATURAL
    return u, f


def _stack(rows):
    us, fs = zip(*rows)
    return np.array(us, dtype=np.uint32), np.array(fs, dtype=np.float32)


def _scale_mat(sx, sy):
    return [sx, 0, 0, 0, sy, 0, 0, 0, 1]


def _paint_slot_records(slot):
    # BEGIN(slot, Transparent) - white fullscreen image - END, then
    # BEGIN(screen, Retain) - blit the slot - END. Frame 1 primes the slot.
    return _stack([
        _rec_row(sn.OP_BEGIN, a=slot, b=0),
        _rec_row(sn.OP_BLIT, a=sn.SRC_IMAGE, b=0, mat=_scale_mat(4.0, 4.0)),
        _rec_row(sn.OP_END, a=slot),
        _rec_row(sn.OP_BEGIN, a=SCREEN_ID, b=0),
        _rec_row(sn.OP_BLIT, a=sn.SRC_DRAWABLE, b=slot, mat=_scale_mat(4.0, 4.0)),
        _rec_row(sn.OP_END, a=SCREEN_ID),
    ])


def _decay_only_records(slot):
    # BEGIN(slot, RETAIN) with NO item (the slot's content just decays),
    # END, then composite it over an OPAQUE-BLACK screen so the decayed
    # ALPHA reads out as a dimmer RGB intensity (white * 1/8 over black).
    return _stack([
        _rec_row(sn.OP_BEGIN, a=slot, b=_CLEAR_RETAIN_CODE),
        _rec_row(sn.OP_END, a=slot),
        _rec_row(sn.OP_BEGIN, a=SCREEN_ID, b=1),  # OpaqueBlack
        _rec_row(sn.OP_BLIT, a=sn.SRC_DRAWABLE, b=slot, mat=_scale_mat(4.0, 4.0)),
        _rec_row(sn.OP_END, a=SCREEN_ID),
    ])


def test_retain_decay_leaves_one_eighth_after_three_frames():
    slot = 1
    images = {0: _solid(1, 1, QColor(255, 255, 255, 255))}
    ex = RasterExecutor(images, [(4.0, 4.0), (4.0, 4.0)])
    ex.set_decay(slot, 0.5)

    # Prime the slot with full-intensity white (over opaque black -> 255).
    u0, f0 = _paint_slot_records(slot)
    primed = ex.execute(u0, f0)
    assert _rgb(primed, 2, 2) == (255, 255, 255)
    assert _alpha(ex._targets[slot], 2, 2) == 255

    # Decay-only three times: the slot's alpha fades 1 -> 1/8; composited
    # over opaque black the white shows as ~1/8 intensity (~32).
    u1, f1 = _decay_only_records(slot)
    for _ in range(3):
        screen = ex.execute(u1, f1)
    assert _alpha(ex._targets[slot], 2, 2) == pytest.approx(32, abs=4)
    assert screen.pixelColor(2, 2).red() == pytest.approx(32, abs=6)


def test_retain_decay_default_factor_persists_forever():
    # Default (no set_decay / 1.0) keeps today's persist-forever behavior:
    # a decay-only frame leaves the primed content untouched.
    slot = 1
    images = {0: _solid(1, 1, QColor(255, 255, 255, 255))}
    ex = RasterExecutor(images, [(4.0, 4.0), (4.0, 4.0)])

    ex.execute(*_paint_slot_records(slot))
    u1, f1 = _decay_only_records(slot)
    for _ in range(3):
        screen = ex.execute(u1, f1)
    assert _alpha(ex._targets[slot], 2, 2) == 255   # unchanged: no decay
    assert _rgb(screen, 2, 2) == (255, 255, 255)


# --- A5: half-texel uv insets (filter-bleed guard) ---


def test_half_texel_inset_moves_edges_inward_and_guards_inversion():
    from analysis.player.render.storyboard.executor import _half_texel_inset

    # A full-texture sample of a 100x50 image insets 0.5 px on each edge.
    full = _half_texel_inset(QRectF(0.0, 0.0, 100.0, 50.0), 100, 50)
    assert full.left() == pytest.approx(0.5)
    assert full.top() == pytest.approx(0.5)
    assert full.right() == pytest.approx(99.5)
    assert full.bottom() == pytest.approx(49.5)

    # A sub-1px-wide window would invert; it is left untouched (the guard).
    tiny = QRectF(10.0, 10.0, 0.4, 0.4)
    assert _half_texel_inset(tiny, 100, 100) == tiny


def test_half_texel_inset_still_renders_the_image():
    # Smoke: the inset never blanks a normal blit. A 2x2 red/green split
    # scaled fullscreen still draws (the inset only trims filter bleed).
    b = sn.DocBuilder(4.0, 4.0)
    b.item(0, sn.SRC_IMAGE, 0, sx_rest=4.0, sy_rest=4.0)
    ev = b.finish()
    content = QImage(2, 2, QImage.Format.Format_ARGB32_Premultiplied)
    content.fill(QColor(255, 0, 0, 255))
    content.setPixelColor(1, 0, QColor(0, 255, 0, 255))
    content.setPixelColor(1, 1, QColor(0, 255, 0, 255))
    ex = RasterExecutor({0: content}, [(4.0, 4.0)])
    screen = ex.execute(*_frames(ev, 0.0))
    assert _alpha(screen, 1, 2) == 255            # opaque content, not blank
    assert _rgb(screen, 0, 2)[0] > 120            # left column still reddish


# --- the box lanes: origin / absolute size / scale-to-fit ---------------

def test_an_item_draws_about_its_origin_not_its_top_left():
    # The origin shifts the draw box before the transform (SM's
    # translate(-origin*w, -origin*h)). Reading the logical box straight
    # hangs every centred actor down-right by half its own size, which is
    # what this backend did for as long as it restated the record layout
    # and so never saw the origin lane at all.
    b = sn.DocBuilder(8.0, 8.0)
    b.item(0, sn.SRC_IMAGE, 0, x_rest=4.0, y_rest=4.0)
    b.item_box(0, origin_x=0.5, origin_y=0.5)
    ev = b.finish()

    ex = RasterExecutor({0: _solid(4, 4, QColor(0, 0, 255, 255))}, [(8.0, 8.0)])
    ex.set_clear(SCREEN_ID, CLEAR_TRANSPARENT)   # else the opaque clear is the alpha
    screen = ex.execute(*_frames(ev, 0.0))
    # Centred on (4, 4): the 4x4 image spans x,y in [2, 6).
    assert _alpha(screen, 3, 3) == 255
    assert _alpha(screen, 5, 5) == 255
    assert _alpha(screen, 7, 7) == 0    # would be covered if top-left anchored


def test_an_absolute_size_replaces_the_natural_box():
    # zoomto/setsize REPLACE the natural basis rather than scaling it, so a
    # 2x2 image sized to 8x8 covers the target.
    b = sn.DocBuilder(8.0, 8.0)
    b.item(0, sn.SRC_IMAGE, 0)
    b.item_box(0, size_x_rest=8.0, size_y_rest=8.0)
    ev = b.finish()

    ex = RasterExecutor({0: _solid(2, 2, QColor(0, 200, 0, 255))}, [(8.0, 8.0)])
    screen = ex.execute(*_frames(ev, 0.0))
    assert _rgb(screen, 7, 7) == (0, 200, 0)


def test_a_zero_size_item_draws_nothing():
    # A zero on a size lane is an explicit zero-size draw, not "natural" -
    # only a negative lane means natural. A storyboard rect at zoomto(0, 0)
    # relies on this; drawing it at the target's box would cover the screen.
    b = sn.DocBuilder(8.0, 8.0)
    b.item(0, sn.SRC_FILL, 0)
    b.item_box(0, size_x_rest=0.0, size_y_rest=0.0)
    ev = b.finish()

    ex = RasterExecutor({}, [(8.0, 8.0)])
    ex.set_clear(SCREEN_ID, CLEAR_TRANSPARENT)
    screen = ex.execute(*_frames(ev, 0.0))
    assert _alpha(screen, 4, 4) == 0


def test_a_curtain_fill_covers_its_target():
    # A fill has no texture to size from, so its box is the TARGET's. A unit
    # box drew every AFT-rig curtain one design pixel wide, masking nothing.
    b = sn.DocBuilder(8.0, 8.0)
    b.item(0, sn.SRC_FILL, 0)
    ev = b.finish()

    ex = RasterExecutor({}, [(8.0, 8.0)])
    ex.set_clear(SCREEN_ID, CLEAR_TRANSPARENT)
    screen = ex.execute(*_frames(ev, 0.0))
    assert _alpha(screen, 7, 7) == 255
