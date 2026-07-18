"""GL capture backend: FBO slots, textured-quad instance blits, AFT
snapshots, and raster-equivalence of the composite step.

Runs the same composite scenarios through RasterCaptureBackend (into a
QPixmap host) and GLCaptureBackend (into an offscreen FBO host) and
probes matching pixels. Skips when the platform can't provide a GL 3+
context (CI without EGL/GLX). Blits of untransformed captures are
texel-aligned, so probes away from shape edges compare exactly; blended
and transformed probes allow a small filtering tolerance."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PySide6')

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (QColor, QOffscreenSurface, QOpenGLContext,
                           QPainter, QPixmap, QSurfaceFormat, QTransform)
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLPaintDevice

from analysis.player.render import gl_capture
from analysis.player.render.capture import RasterCaptureBackend
from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.qt_renderer import QtPlayerRenderer

W, H = 200, 150
CHART = QRectF(0, 0, 160, 150)


@pytest.fixture(scope='module')
def gl():
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
        pytest.skip('no OpenGL context on this platform')
    if context.format().majorVersion() < 3:
        pytest.skip('OpenGL context below 3.x')
    yield context
    context.doneCurrent()


class _GlHost:
    """An FBO-backed host painter standing in for the canvas widget."""

    def __init__(self):
        self.fbo = QOpenGLFramebufferObject(
            W, H, QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        assert self.fbo.bind()
        self.device = QOpenGLPaintDevice(W, H)
        self.painter = QPainter(self.device)
        self.painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.painter.fillRect(0, 0, W, H, QColor(0, 0, 0))

    def finish(self):
        self.painter.end()
        image = self.fbo.toImage()
        self.fbo.release()
        return image


class _RasterHost:
    def __init__(self):
        self.pixmap = QPixmap(W, H)
        self.pixmap.fill(QColor(0, 0, 0))
        self.painter = QPainter(self.pixmap)
        self.painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def finish(self):
        self.painter.end()
        return self.pixmap.toImage()


def _probe(image, x, y):
    c = QColor(image.pixel(x, y))
    return c.red(), c.green(), c.blue()


def _close(a, b, tol):
    return all(abs(p - q) <= tol for p, q in zip(a, b))


def _paint_field(backend, host, color=QColor(200, 40, 40)):
    """Capture a solid rectangle at (20, 30)-(120, 100) into 'field'."""
    painter = backend.open('field', host.painter, W, H)
    painter.fillRect(20, 30, 100, 70, color)
    return backend.close('field')


def _run_scenario(backend, host, scenario):
    handle = _paint_field(backend, host)
    scenario(backend, host.painter, handle)
    return host.finish()


def _both_backends(gl_unused, scenario):
    raster = _run_scenario(RasterCaptureBackend(), _RasterHost(), scenario)
    fbo = _run_scenario(gl_capture.GLCaptureBackend(), _GlHost(), scenario)
    return raster, fbo


# -- backend selection ----------------------------------------------------

def test_usable_requires_gl_painter(gl):
    host = _GlHost()
    assert gl_capture.usable(host.painter)
    host.finish()
    raster = _RasterHost()
    assert not gl_capture.usable(raster.painter)
    raster.finish()
    assert not gl_capture.usable(None)


def test_renderer_selects_gl_backend_on_gl_painter(gl):
    r = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    host = _GlHost()
    r._select_capture_backend(host.painter)
    assert isinstance(r._capture, gl_capture.GLCaptureBackend)
    host.finish()
    raster = _RasterHost()
    r._prev_screen = object()
    r._select_capture_backend(raster.painter)
    assert isinstance(r._capture, RasterCaptureBackend)
    # Switching backends drops retained handles from the other one.
    assert r._prev_screen is None
    raster.finish()


# -- blit equivalence -----------------------------------------------------

def test_identity_blit_matches_raster(gl):
    def scenario(backend, painter, handle):
        backend.blit(painter, handle, CHART)
    raster, fbo = _both_backends(gl, scenario)
    for x, y in ((70, 60), (25, 35), (115, 95), (10, 10), (150, 140)):
        assert _probe(fbo, x, y) == _probe(raster, x, y), (x, y)


def test_translated_scaled_blit_matches_raster(gl):
    transform = QTransform().translate(30, 10).scale(0.5, 0.5)
    def scenario(backend, painter, handle):
        backend.blit(painter, handle, CHART, transform=transform)
    raster, fbo = _both_backends(gl, scenario)
    # Rect maps to (40, 25)-(90, 60); probe interiors + outside, with
    # tolerance for the backends' different edge filtering.
    for x, y in ((60, 40), (45, 30), (85, 55), (100, 70), (20, 20)):
        assert _close(_probe(fbo, x, y), _probe(raster, x, y), 8), (x, y)


def test_opacity_blend_matches_raster(gl):
    def scenario(backend, painter, handle):
        backend.blit(painter, handle, CHART, opacity=0.5)
    raster, fbo = _both_backends(gl, scenario)
    assert _close(_probe(fbo, 70, 60), _probe(raster, 70, 60), 2)
    assert _probe(raster, 70, 60)[0] == pytest.approx(100, abs=3)


def test_clip_rect_scissors_blit(gl):
    clip = QRectF(0, 0, 60, 150)
    def scenario(backend, painter, handle):
        backend.blit(painter, handle, clip)
    raster, fbo = _both_backends(gl, scenario)
    for image in (raster, fbo):
        assert _probe(image, 40, 60)[0] > 150   # inside clip: red
        assert _probe(image, 80, 60) == (0, 0, 0)  # clipped away


def test_src_box_clips_source(gl):
    box = QRectF(20, 30, 50, 70)
    def scenario(backend, painter, handle):
        backend.blit(painter, handle, CHART, src_box=box)
    raster, fbo = _both_backends(gl, scenario)
    for image in (raster, fbo):
        assert _probe(image, 40, 60)[0] > 150   # inside the box
        assert _probe(image, 100, 60) == (0, 0, 0)  # outside the box


def test_fill_covers_clip_at_position(gl):
    def scenario(backend, painter, handle):
        with backend.blits(painter, QRectF(0, 0, 60, 150)) as batch:
            batch.blit(handle)
            batch.fill((0.0, 1.0, 0.0), 1.0)
    raster, fbo = _both_backends(gl, scenario)
    for image in (raster, fbo):
        assert _probe(image, 40, 60) == (0, 255, 0)   # curtain over blit
        assert _probe(image, 80, 60) == (0, 0, 0)     # outside clip


# -- screen-composite lifecycle on GL -------------------------------------

def _ctx(t):
    player = SimpleNamespace(W=W, H=H)
    return SimpleNamespace(t_now=float(t), player=player,
                           chart_rect=(0, 0, 160, 150))


def _screen_frame(*scopes):
    return EffectFrame(fields=tuple((None, 1.0, s) for s in scopes))


def _gl_renderer(host):
    r = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    r._select_capture_backend(host.painter)
    assert isinstance(r._capture, gl_capture.GLCaptureBackend)
    return r


def test_gl_screen_capture_taken_at_blit_and_presented(gl):
    """The GL composite mirrors the raster lifecycle: node capture
    snapshots mid-blit (pre-node green, never post-node red), the
    present hands the full composite to the host."""
    host = _GlHost()
    r = _gl_renderer(host)
    ctx = _ctx(1.0)
    frame = _screen_frame('screen')
    r._sync_prev_screen(ctx)
    target = r._begin_screen_composite(frame, ctx, host.painter)
    assert target is not None and target is not host.painter
    target.fillRect(0, 0, 160, 150, QColor(10, 200, 30))
    r._blit_field_instances(frame, ctx, target)
    target.fillRect(0, 0, 160, 150, QColor(200, 10, 10))
    r._end_screen_composite(host.painter, ctx)
    image = host.finish()
    assert r._prev_screen is not None
    assert r._prev_screen_t == pytest.approx(1.0)
    assert _probe(image, 80, 75)[0] > 150  # host: post-node red on top


def test_gl_screen_prev_feeds_back_one_frame(gl):
    """'screen_prev' skips the unprimed first frame, then blits the
    previous frame's retained capture texture."""
    frame = _screen_frame('screen_prev')

    host = _GlHost()
    r = _gl_renderer(host)
    ctx = _ctx(1.0)
    r._sync_prev_screen(ctx)
    target = r._begin_screen_composite(frame, ctx, host.painter)
    target.fillRect(0, 0, 160, 150, QColor(0, 0, 120))
    r._blit_field_instances(frame, ctx, target)
    r._end_screen_composite(host.painter, ctx)
    host.finish()
    assert r._prev_screen is not None

    host2 = _GlHost()
    ctx2 = _ctx(1.008)
    r._sync_prev_screen(ctx2)
    target2 = r._begin_screen_composite(frame, ctx2, host2.painter)
    r._blit_field_instances(frame, ctx2, target2)
    r._end_screen_composite(host2.painter, ctx2)
    image2 = host2.finish()
    assert _probe(image2, 80, 75)[2] > 80  # previous frame's blue


def test_gl_field_capture_nests_inside_screen_composite(gl):
    """The field slot opens while the screen slot's painter is live
    (nested native brackets); its capture blits into the composite."""
    host = _GlHost()
    r = _gl_renderer(host)
    ctx = _ctx(2.0)
    frame = EffectFrame(fields=((None, 1.0, 'field'),
                                (None, 1.0, 'screen')))
    r._sync_prev_screen(ctx)
    target = r._begin_screen_composite(frame, ctx, host.painter)
    fp = r._begin_field_capture(frame, ctx, target)
    fp.fillRect(30, 40, 60, 50, QColor(240, 240, 20))
    r._end_field_capture()
    r._blit_field_instances(frame, ctx, target)
    r._end_screen_composite(host.painter, ctx)
    image = host.finish()
    assert _probe(image, 60, 65) == (240, 240, 20)
    assert _probe(image, 10, 10) == (0, 0, 0)


def test_gl_unified_shader_stage_runs_passes_over_post_slot(gl):
    """With the GL backend active the shader capture is the 'post'
    slot and the passes run straight over its FBO into the host - no
    second capture painter."""
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    host = _GlHost()
    r = _gl_renderer(host)
    r.shader_pipeline = ShaderGLPipeline()
    ctx = _ctx(3.0)
    frame = EffectFrame(
        shaders=(('invert', {'u_strength': (1.0, 0.0, 0.0)}),))
    cp = r._begin_shader_capture(frame, ctx, host.painter)
    assert cp is not None and cp is not host.painter
    cp.fillRect(0, 0, W, H, QColor(255, 0, 0))
    r._end_shader_capture(frame, ctx)
    image = host.finish()
    assert _probe(image, 80, 75) == (0, 255, 255)  # inverted red


def test_gl_unified_stage_blits_unshaded_when_nothing_runnable(gl):
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    host = _GlHost()
    r = _gl_renderer(host)
    r.shader_pipeline = ShaderGLPipeline()
    ctx = _ctx(3.0)
    frame = EffectFrame(shaders=(('no_such_shader', {}),))
    cp = r._begin_shader_capture(frame, ctx, host.painter)
    cp.fillRect(0, 0, W, H, QColor(20, 220, 40))
    r._end_shader_capture(frame, ctx)
    image = host.finish()
    assert _probe(image, 80, 75) == (20, 220, 40)


def test_gl_ln_ribbon_survives_mod_slam_coordinates(gl):
    """Mod-slam spine samples (x ~ 1e5) used to overflow the GL paint
    engine's +/-32767 concave-fill limit: the visible ribbon vanished
    for the slam frame with a 'Painter path exceeds' warning. The
    sample clamp keeps the in-view ribbon drawn and the engine quiet."""
    from PySide6.QtCore import qInstallMessageHandler

    import numpy as np

    from analysis.player.render.layers.notes import _draw_ln_body_stroke

    warnings = []
    qInstallMessageHandler(lambda _mode, _ctx, msg: warnings.append(msg))
    try:
        host = _GlHost()
        pm = QPixmap(20, 20)
        pm.fill(QColor(200, 30, 200))
        ctx = SimpleNamespace(
            player=SimpleNamespace(W=W, H=H, skin='bar'),
            sprite_cache=SimpleNamespace(get=lambda *a, **k: pm),
            lane_width=lambda _col: 20)
        n = SimpleNamespace(
            body_path=(np.array([100.0, 100000.0]),
                       np.array([10.0, 140.0])),
            col=0, is_roll=False)
        _draw_ln_body_stroke(ctx, host.painter, n, 0.0, 150.0, 'normal')
        image = host.finish()
    finally:
        qInstallMessageHandler(None)
    assert not [w for w in warnings if '32767' in w]
    assert _probe(image, 150, 11) == (200, 30, 200)


def test_gl_snapshot_freelist_recycles_released_textures(gl):
    host = _GlHost()
    backend = gl_capture.GLCaptureBackend()
    painter = backend.open('screen', host.painter, W, H)
    painter.fillRect(0, 0, W, H, QColor(50, 60, 70))
    with backend.blits(painter, CHART):
        first = backend.snapshot('screen')
    backend.release(first)
    with backend.blits(painter, CHART):
        second = backend.snapshot('screen')
    assert second.fbo is first.fbo  # recycled, not reallocated
    backend.retain(second)
    backend.release(second)
    backend.release(second)
    with backend.blits(painter, CHART):
        third = backend.snapshot('screen')
    assert third.fbo is second.fbo
    backend.close('screen')
    host.finish()
