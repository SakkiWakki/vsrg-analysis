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

from analysis.player.render.storyboard import gl_executor as _gl
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


def _frames_uf(evaluator, t):
    u_raw, f_raw, uf_raw, n = evaluator.frame(t)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
    uf = np.frombuffer(uf_raw, dtype=np.float32)
    return u, f, uf


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


def _upload_texture(image):
    """Upload a QImage to a plain GL texture via the Qt GL functions and
    return (id, w, h). Mirrors a renderer capture FBO's ``texture()`` for the
    external-binding test - the executor never owns or deletes it."""
    from PySide6.QtGui import QOpenGLContext
    from analysis.player.render.storyboard.gl_executor import _upload_image
    gf = QOpenGLContext.currentContext().extraFunctions()
    return _upload_image(gf, image)   # (texture id, w, h)


def test_oversized_bound_texture_normalizes_to_the_logical_box(gl):
    # THE ZOOM FIX (drawable-ir.md rule 5), GL-side: a 1280x960 texture bound
    # (external, non-owned) as a 640x480 field drawable's content covers that
    # drawable's LOGICAL box regardless of its pixel size. A fullscreen
    # (identity mat3) SRC_DRAWABLE blit shows the WHOLE texture scaled into the
    # 640x480 screen, not the top-left quarter.
    b = sn.DocBuilder(640.0, 480.0)
    field = b.drawable(640.0, 480.0, False, False)  # command-less field scope
    b.item(0, sn.SRC_DRAWABLE, field, sx_rest=1.0, sy_rest=1.0)
    ev = b.finish()

    # A 1280x960 texture: left half red, right half green (like the raster
    # golden). Qt paints y-up into GL textures relative to a QImage's y-down,
    # but a left/right split is orientation-independent, so the horizontal
    # halves prove normalization without a flip caveat.
    content = QImage(1280, 960, QImage.Format.Format_ARGB32_Premultiplied)
    content.fill(QColor(0, 0, 0, 255))
    from PySide6.QtGui import QPainter
    p = QPainter(content)
    p.fillRect(0, 0, 640, 960, QColor(255, 0, 0, 255))       # left half red
    p.fillRect(640, 0, 640, 960, QColor(0, 255, 0, 255))     # right half green
    p.end()
    tex, tw, thh = _upload_texture(content)

    ex = GLExecutor({}, [(640.0, 480.0), (640.0, 480.0)])
    ex.set_drawable_texture(field, tex, tw, thh)
    u, f = _frames(ev, 0.0)
    screen = ex.execute(u, f)

    assert not ex.broken
    # Left third of the screen samples the image's red left half; right third
    # the green right half. Zoomed-in (1280 source units into 640 px) would
    # keep the whole screen red - green on the right proves the fix.
    assert _rgb(screen, 100, 240) == (255, 0, 0)
    assert _rgb(screen, 540, 240) == (0, 255, 0)


def test_bound_external_texture_draws_and_unbinds(gl):
    # GL BINDING (spec 2): set_drawable_texture binds an external capture
    # texture as a command-less field drawable's content; a SRC_DRAWABLE blit
    # samples it directly (no readback, no upload). Un-binding (texture 0)
    # drops it so the field reads empty again.
    b = sn.DocBuilder(4.0, 4.0)
    field = b.drawable(4.0, 4.0, False, False)
    b.item(0, sn.SRC_DRAWABLE, field, sx_rest=1.0, sy_rest=1.0)
    ev = b.finish()

    content = _solid(4, 4, QColor(0, 200, 0, 255))
    tex, tw, thh = _upload_texture(content)

    ex = GLExecutor({}, [(4.0, 4.0), (4.0, 4.0)])
    u, f = _frames(ev, 0.0)

    # Nothing bound: the field read draws nothing, so the screen keeps its
    # cleared color (OpaqueBlack - GLExecutor has no set_clear knob).
    assert _rgb(ex.execute(u, f), 2, 2) == (0, 0, 0)

    ex.set_drawable_texture(field, tex, tw, thh)
    drawn = ex.execute(u, f)
    assert not ex.broken
    assert _rgb(drawn, 2, 2) == (0, 200, 0)   # the bound texture shows

    # Un-bind (texture 0): the field reads empty, the screen is black again.
    ex.set_drawable_texture(field, 0, 0, 0)
    assert _rgb(ex.execute(u, f), 2, 2) == (0, 0, 0)


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


# --- A4: retain decay (GL constant-alpha modulate) ---

# Taken from the executor rather than restated: a hand-built record must
# match the stride the evaluator emits, and a stale copy silently indexes
# past the end of the row.
_U_STRIDE = _gl._U_STRIDE_LANES
_F_STRIDE = _gl._F_STRIDE_LANES
_CLEAR_RETAIN_CODE = 2


def _rec_row(kind, a=0, b=0, mat=None, opacity=1.0, tint=(1.0, 1.0, 1.0)):
    u = np.zeros(_U_STRIDE, dtype=np.uint32)
    u[0], u[1], u[2] = kind, a, b
    f = np.zeros(_F_STRIDE, dtype=np.float32)
    f[:9] = mat if mat is not None else [1, 0, 0, 0, 1, 0, 0, 0, 1]
    f[9] = opacity
    f[10:13] = tint
    # Negative size = "use the source's natural box"; a zeroed lane would
    # mean an explicit zero-size draw, which is correctly invisible.
    f[_gl._F_SIZE:_gl._F_SIZE + 2] = -1.0
    return u, f


def _stack(rows):
    us, fs = zip(*rows)
    return np.array(us, dtype=np.uint32), np.array(fs, dtype=np.float32)


def _scale(sx, sy):
    return [sx, 0, 0, 0, sy, 0, 0, 0, 1]


def _paint_slot(slot):
    # Prime the persistent slot with a full white image, blit onto screen.
    return _stack([
        _rec_row(sn.OP_BEGIN, a=slot, b=0),
        _rec_row(sn.OP_BLIT, a=sn.SRC_IMAGE, b=0, mat=_scale(4.0, 4.0)),
        _rec_row(sn.OP_END, a=slot),
        _rec_row(sn.OP_BEGIN, a=0, b=1),  # screen OpaqueBlack
        _rec_row(sn.OP_BLIT, a=sn.SRC_DRAWABLE, b=slot, mat=_scale(4.0, 4.0)),
        _rec_row(sn.OP_END, a=0),
    ])


def _decay_only(slot):
    # Retain BEGIN with no item (content fades), read over opaque black.
    return _stack([
        _rec_row(sn.OP_BEGIN, a=slot, b=_CLEAR_RETAIN_CODE),
        _rec_row(sn.OP_END, a=slot),
        _rec_row(sn.OP_BEGIN, a=0, b=1),
        _rec_row(sn.OP_BLIT, a=sn.SRC_DRAWABLE, b=slot, mat=_scale(4.0, 4.0)),
        _rec_row(sn.OP_END, a=0),
    ])


def test_retain_decay_leaves_one_eighth_after_three_frames(gl):
    slot = 1
    images = {0: _solid(1, 1, QColor(255, 255, 255, 255))}
    ex = GLExecutor(images, [(4.0, 4.0), (4.0, 4.0)])
    ex.set_decay(slot, 0.5)

    primed = ex.execute(*_paint_slot(slot))
    assert not ex.broken
    assert primed.pixelColor(2, 2).red() == pytest.approx(255, abs=2)

    u1, f1 = _decay_only(slot)
    for _ in range(3):
        screen = ex.execute(u1, f1)
    # 0.5 ** 3 = 1/8 white over black -> ~32.
    assert screen.pixelColor(2, 2).red() == pytest.approx(32, abs=8)


def test_retain_decay_default_persists_forever(gl):
    slot = 1
    images = {0: _solid(1, 1, QColor(255, 255, 255, 255))}
    ex = GLExecutor(images, [(4.0, 4.0), (4.0, 4.0)])

    ex.execute(*_paint_slot(slot))
    u1, f1 = _decay_only(slot)
    for _ in range(3):
        screen = ex.execute(u1, f1)
    assert not ex.broken
    assert screen.pixelColor(2, 2).red() == pytest.approx(255, abs=2)


# --- B10: rect clips via glScissor (mirrors the raster clip golden) ---


def test_clipped_fullscreen_fill_colors_only_the_clip_region(gl):
    # A green fullscreen Fill clipped to a centered rect (2,2)-(6,6) on an
    # 8x8 screen (OpaqueBlack). Inside the clip goes green; outside stays
    # cleared black. The clip shape is in the TARGET's logical units and is
    # consumed GL-side as a glScissor rect (mirrors the raster golden).
    b = sn.DocBuilder(8.0, 8.0)
    clip = b.clip_rect(2.0, 2.0, 6.0, 6.0)
    b.item(0, sn.SRC_FILL, 0, sx_rest=8.0, sy_rest=8.0)
    b.item_clip(0, clip)
    ev = b.finish()

    u, f = _frames(ev, 0.0)
    f = f.copy()
    for i in range(u.shape[0]):
        if u[i, 0] == sn.OP_BLIT and u[i, 1] == sn.SRC_FILL:
            f[i, 10:13] = (0.0, 1.0, 0.0)
    ex = GLExecutor({}, [(8.0, 8.0)], clips=[("rect", 2.0, 2.0, 6.0, 6.0)])
    screen = ex.execute(u, f)

    assert not ex.broken
    assert _rgb(screen, 4, 4) == (0, 255, 0)   # inside the clip: green
    assert _rgb(screen, 0, 0) == (0, 0, 0)     # outside: cleared black
    assert _rgb(screen, 7, 7) == (0, 0, 0)


def test_rect_clip_scissor_is_not_y_flipped(gl):
    # A clip covering the logical TOP band (y in [0,4)) must fill the TOP of
    # the (y-down content) output image, not the bottom - locks the scissor
    # y-flip (GL scissor is y-up, FBO content y-down).
    b = sn.DocBuilder(8.0, 8.0)
    clip = b.clip_rect(0.0, 0.0, 8.0, 4.0)
    b.item(0, sn.SRC_FILL, 0, sx_rest=8.0, sy_rest=8.0)
    b.item_clip(0, clip)
    ev = b.finish()

    u, f = _frames(ev, 0.0)
    f = f.copy()
    for i in range(u.shape[0]):
        if u[i, 0] == sn.OP_BLIT and u[i, 1] == sn.SRC_FILL:
            f[i, 10:13] = (0.0, 1.0, 0.0)
    ex = GLExecutor({}, [(8.0, 8.0)], clips=[("rect", 0.0, 0.0, 8.0, 4.0)])
    screen = ex.execute(u, f)

    assert not ex.broken
    assert _rgb(screen, 4, 1) == (0, 255, 0)   # top band: green
    assert _rgb(screen, 4, 6) == (0, 0, 0)     # bottom: cleared black


def test_poly_clip_draws_unclipped_todo(gl):
    # A 'poly' clip is a logged-once TODO GL-side: the fill draws UNCLIPPED
    # (never black / crash) - the raster QPainterPath clip is the reference.
    tri = [(1.0, 1.0), (7.0, 1.0), (1.0, 7.0)]
    b = sn.DocBuilder(8.0, 8.0)
    clip = b.clip_polygon([c for xy in tri for c in xy])
    b.item(0, sn.SRC_FILL, 0, sx_rest=8.0, sy_rest=8.0)
    b.item_clip(0, clip)
    ev = b.finish()

    u, f = _frames(ev, 0.0)
    f = f.copy()
    for i in range(u.shape[0]):
        if u[i, 0] == sn.OP_BLIT and u[i, 1] == sn.SRC_FILL:
            f[i, 10:13] = (0.0, 0.0, 1.0)
    ex = GLExecutor({}, [(8.0, 8.0)], clips=[("poly", tri)])
    screen = ex.execute(u, f)

    assert not ex.broken
    # Unclipped: the far corner (outside the triangle) is still filled.
    assert _rgb(screen, 6, 6) == (0, 0, 255)


# --- B7: per-item GL shaders (the monitor / lumikey tier) ---


def _shaded_doc(uniform_names):
    b = sn.DocBuilder(4.0, 4.0)
    sh = b.shader("dummy", None, uniform_names)   # id only; source via set_shaders
    b.item(0, sn.SRC_IMAGE, 0, sx_rest=4.0, sy_rest=4.0)
    b.item_shader(0, sh)
    if uniform_names:
        b.item_uniform(0, 0, -1, 1.0)             # tint_r = 1.0 (rest)
    return b.finish(), sh


_TINT_FRAG = ("uniform sampler2D sampler0;\n"
              "uniform float tint_r;\n"
              "void main(){ gl_FragColor = vec4(tint_r, 0.0, 0.0, 1.0); }\n")

_BROKEN_FRAG = ("uniform sampler2D sampler0;\n"
                "void main(){ this is not valid glsl @@@ }\n")


def test_per_item_shader_changes_pixels(gl):
    # A white image blit shaded through a trivial frag that outputs red
    # (uniform tint_r = 1.0). The shader path must change the pixels: the
    # screen shows red, not the source white.
    ev, sh = _shaded_doc(["tint_r"])
    images = {0: _solid(1, 1, QColor(255, 255, 255, 255))}
    ex = GLExecutor(images, [(4.0, 4.0)])
    ex.set_shaders([(_TINT_FRAG, None, ["tint_r"])])

    u, f, uf = _frames_uf(ev, 0.0)
    screen = ex.execute(u, f, uf)
    assert not ex.broken
    assert _rgb(screen, 2, 2) == pytest.approx((255, 0, 0), abs=4)


def test_broken_shader_degrades_to_unshaded_not_black(gl):
    # A frag that fails to build must degrade to an UNSHADED blit (the
    # plain textured program), never black / crash. The white source shows.
    ev, sh = _shaded_doc([])
    images = {0: _solid(1, 1, QColor(255, 255, 255, 255))}
    ex = GLExecutor(images, [(4.0, 4.0)])
    ex.set_shaders([(_BROKEN_FRAG, None, [])])

    u, f, uf = _frames_uf(ev, 0.0)
    screen = ex.execute(u, f, uf)
    assert not ex.broken
    assert _rgb(screen, 2, 2) == pytest.approx((255, 255, 255), abs=4)


def test_overscan_margin_content_survives_a_shifted_copy(gl):
    """A field capture is WINDOW-sized plus overscan MARGINS holding content
    the mods pushed outside the playfield; the uv sub-rect names the chart-rect
    region that maps onto the drawable's logical box.

    A copy that shifts the field must be able to bring that margin content on
    screen - the engine re-renders a proxy, so content outside the playfield is
    exactly what a shifted copy reveals. Sampling only the sub-rect discards it
    and the copy shows a hard-edged crop at the playfield boundary.
    """
    b = sn.DocBuilder(640.0, 480.0)
    field = b.drawable(640.0, 480.0, False, False)
    # A copy shifted LEFT by half the box: content living in the capture's
    # RIGHT margin lands inside the screen.
    b.item(0, sn.SRC_DRAWABLE, field, x_rest=-320.0, sx_rest=1.0, sy_rest=1.0)
    ev = b.finish()

    # Capture: 1280x480. The chart rect is its LEFT half (uv 0..0.5); the right
    # half is overscan margin. Playfield content green, margin content red.
    content = QImage(1280, 480, QImage.Format.Format_ARGB32_Premultiplied)
    content.fill(QColor(0, 0, 0, 255))
    from PySide6.QtGui import QPainter
    p = QPainter(content)
    p.fillRect(0, 0, 640, 480, QColor(0, 200, 0, 255))      # playfield
    p.fillRect(640, 0, 640, 480, QColor(200, 0, 0, 255))    # margin
    p.end()
    tex, tw, thh = _upload_texture(content)

    ex = GLExecutor({}, [(640.0, 480.0), (640.0, 480.0)])
    ex.set_drawable_texture(field, tex, tw, thh, (0.0, 0.0, 0.5, 1.0))
    screen = ex.execute(*_frames(ev, 0.0))

    assert not ex.broken
    # Shifted left by 320: the playfield's right half occupies screen x<320,
    # and the MARGIN content should occupy x>320.
    assert _rgb(screen, 160, 240) == (0, 200, 0), 'playfield content shifted in'
    assert _rgb(screen, 480, 240) == (200, 0, 0), (
        'margin content must survive the copy - a sub-rect-only sample '
        'crops everything outside the playfield')


def test_content_overflowing_a_drawable_is_clipped_at_its_box(gl):
    """A drawable that is a BEGIN target renders into an FBO of its LOGICAL
    size, so content drawn past that box is destroyed - not merely hidden.

    Here a sub-drawable's image is twice its box and offset, so half of it
    falls outside. Blitting the sub-drawable at 1:1 then shows the clipped
    result: the overflow is gone even though the destination had room for it.
    This is why a drawable must be sized to the content it holds.
    """
    b = sn.DocBuilder(64.0, 64.0)
    sub = b.drawable(32.0, 32.0, False, False)
    # A 1x1 white source scaled to 64x64 inside a 32x32 drawable: the right
    # and bottom halves overflow the box.
    b.item(sub, sn.SRC_IMAGE, 0, sx_rest=64.0, sy_rest=64.0)
    # Draw the sub-drawable into the screen at its natural 32x32 logical box,
    # then again shifted so any surviving overflow would be visible.
    b.item(0, sn.SRC_DRAWABLE, sub, sx_rest=1.0, sy_rest=1.0)
    ev = b.finish()

    images = {0: _solid(1, 1, QColor(255, 255, 255, 255))}
    ex = GLExecutor(images, [(64.0, 64.0), (32.0, 32.0)])
    screen = ex.execute(*_frames(ev, 0.0))

    assert not ex.broken
    # Inside the sub-drawable's box: painted.
    assert _rgb(screen, 16, 16) == (255, 255, 255)
    # Past its box, still inside the SCREEN: the overflow was clipped away by
    # the sub-drawable's own FBO, so nothing survives to composite here.
    assert _rgb(screen, 48, 48) == (0, 0, 0), (
        'content overflowing a drawable is destroyed at its box - the '
        'drawable must be sized to what it draws')
