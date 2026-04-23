"""Test harness for sidebar rendering, backend-agnostic.

Existing sidebar tests assert on specific ``QPainter`` method calls via
a ``_FakeSctx`` mock, which couples them to the current backend. Phase
2 will introduce an ``OpenGLSidebarRenderer`` that produces visually
equivalent output via a ``QOpenGLPaintDevice`` instead of a raw
``QPainter`` on a widget backing store.

This module provides the pieces tests need to stay honest across both
backends:

  - :func:`render_sidebar_section_to_image` — render a ``draw(sctx)``
    callable into a fresh ``QImage`` through a named backend. Backend
    ``"qimage"`` is the always-available QPainter-on-QImage path;
    ``"opengl"`` is a Phase 2 QPainter-on-QOpenGLPaintDevice path. The
    ``"opengl"`` backend returns ``None`` when GL isn't usable (CI, no
    display) so parametrized tests auto-skip without a hard dependency.

  - :func:`raster_similarity` — structural similarity between two
    ``QImage`` outputs. Returns a float in ``[0, 1]`` where ``1.0`` is
    pixel-identical. Tests assert ``>= 0.98`` or similar; the exact
    threshold depends on how much subpixel font hinting differs between
    paint devices. Golden references are stored per-backend to keep the
    metric robust.

  - :func:`golden_path` — resolve a golden-image path for a test name;
    :func:`assert_matches_golden` — compare a ``QImage`` to its golden
    with a tolerance. Environment ``UPDATE_GOLDENS=1`` writes the
    current output to the golden path instead of asserting.

Design constraints:

  - Golden files live under ``tests/goldens/sidebar/<backend>/<name>.png``.
    Backends with rendering differences (GL vs. CPU subpixel) keep
    separate goldens; similarity between the two is asserted by tests
    that care.

  - The harness never imports the Player or a real sidebar — it wires a
    minimal ``SidebarContext`` directly. Tests that want to render a
    full component call ``draw_component_in_sidebar`` via the wrapper.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


_GOLDENS_ROOT = Path(__file__).resolve().parent / 'goldens' / 'sidebar'


# ── Backend constants ──────────────────────────────────────────────

BACKEND_QIMAGE  = 'qimage'    # QPainter on QImage (CPU)
BACKEND_OPENGL  = 'opengl'    # QPainter on QOpenGLPaintDevice via FBO

ALL_BACKENDS = (BACKEND_QIMAGE, BACKEND_OPENGL)


# ── Rendering ──────────────────────────────────────────────────────

def render_sidebar_section_to_image(
    draw_fn: Callable,
    *,
    backend: str = BACKEND_QIMAGE,
    width: int = 240,
    height: int = 200,
    player=None,
):
    """Render ``draw_fn(sctx)`` into a fresh ``QImage`` through the
    requested backend. Returns the image, or ``None`` if ``backend`` is
    unavailable on this host (so parametrized tests can skip).

    ``draw_fn`` receives a :class:`~analysis.player.hud.sidebar_api.SidebarContext`
    pointing at ``(0, 0, width, height)``; primitives operate in pixel
    coordinates. The sidebar's inset (``theme.SIDEBAR_INSET``) still
    applies to ``col_x``/``col_w``.
    """
    from PySide6.QtGui import QImage, QPainter, QColor
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication([])

    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 255))

    paint_device_factory = _PAINT_DEVICE_FACTORIES.get(backend)
    if paint_device_factory is None:
        raise ValueError(f'unknown backend: {backend!r}')

    device_info = paint_device_factory(image, width, height)
    if device_info is None:
        return None   # backend unavailable

    painter, cleanup = device_info
    try:
        _paint_through(painter, draw_fn, width, height, player=player)
    finally:
        painter.end()
        cleanup()
    return image


def _paint_through(painter, draw_fn, width, height, *, player):
    """Run ``draw_fn`` inside a properly-configured SidebarContext on
    ``painter``. Splitting this out keeps the backend-specific setup
    in the factory and the sidebar-wiring in one place."""
    from analysis.player.hud.sidebar_api import SidebarContext
    from types import SimpleNamespace

    render_ctx = SimpleNamespace(player=player if player is not None
                                 else SimpleNamespace(hud=SimpleNamespace(
                                     add_hitbox=lambda *a, **kw: None)))

    renderer = _font_renderer(painter)
    sctx = SidebarContext(render_ctx, painter, renderer,
                          sidebar_x=0, sidebar_w=width, y=0,
                          measure_only=False)
    draw_fn(sctx)


def _font_renderer(painter):
    """Build a minimal object satisfying the SidebarContext's renderer
    contract (``.big_font`` and ``.font``). Production uses a shared
    ``_Renderer`` instance; tests build their own to avoid pulling in
    the whole player renderer."""
    from PySide6.QtGui import QFont
    from types import SimpleNamespace
    font = painter.font()
    big = QFont(font)
    big.setPointSize(max(font.pointSize() + 2, 12))
    return SimpleNamespace(font=font, big_font=big)


# ── Paint device factories (per backend) ───────────────────────────

def _qimage_device(image, width, height):
    from PySide6.QtGui import QPainter
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    # No cleanup beyond painter.end() (handled by caller).
    return painter, lambda: None


def _opengl_device(image, width, height):
    """QPainter on a ``QOpenGLPaintDevice`` backed by an FBO.

    Returns ``None`` if a GL context can't be created on this host --
    CI without a display and X servers without GLX both trigger this.
    """
    try:
        from PySide6.QtGui import (QPainter, QOffscreenSurface,
                                   QOpenGLContext, QSurfaceFormat)
        from PySide6.QtOpenGL import (QOpenGLFramebufferObject,
                                      QOpenGLFramebufferObjectFormat,
                                      QOpenGLPaintDevice)
        from PySide6.QtCore import QSize
    except ImportError:
        return None

    fmt = QSurfaceFormat()
    fmt.setMajorVersion(3)
    fmt.setMinorVersion(2)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)

    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    if not surface.isValid():
        return None

    context = QOpenGLContext()
    context.setFormat(fmt)
    if not context.create():
        return None
    if not context.makeCurrent(surface):
        return None

    fbo_fmt = QOpenGLFramebufferObjectFormat()
    fbo_fmt.setAttachment(
        QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
    fbo = QOpenGLFramebufferObject(QSize(width, height), fbo_fmt)
    if not fbo.bind():
        return None

    device = QOpenGLPaintDevice(QSize(width, height))
    painter = QPainter(device)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

    # Hold references in the closure so GC doesn't drop `device`, `fbo`,
    # `context` or `surface` while the painter is still attached. Qt
    # fires "Cannot destroy paint device that is being painted" if the
    # device outlives its painter only in C++ lifetime.
    kept = {'device': device, 'fbo': fbo,
            'context': context, 'surface': surface}

    def cleanup():
        # Readback the FBO into the caller's QImage before we tear
        # down the context. toImage() copies GPU->CPU.
        gl_image = kept['fbo'].toImage()
        from PySide6.QtGui import QPainter
        out_painter = QPainter(image)
        out_painter.drawImage(0, 0, gl_image)
        out_painter.end()

        kept['fbo'].release()
        kept['context'].doneCurrent()
        # Now safe to drop references; order is "drawing target last".
        kept.clear()

    return painter, cleanup


_PAINT_DEVICE_FACTORIES = {
    BACKEND_QIMAGE: _qimage_device,
    BACKEND_OPENGL: _opengl_device,
}


# ── Similarity ─────────────────────────────────────────────────────

def raster_similarity(a, b) -> float:
    """Per-pixel similarity between two QImages. Returns a float in
    ``[0, 1]``: ``1.0`` means identical within integer rounding,
    ``0.0`` means every pixel is maximally different.

    Metric: mean of ``1 - (channel_diff / 255)`` across all pixels
    and RGBA channels. Cheap, stable, and "good enough" for
    asserting that two rasters produced by different backends
    represent the same drawing (similarity > ~0.95 means the visible
    output matches).

    Images must have the same size; mismatched sizes return 0.0.
    """
    if a is None or b is None:
        return 0.0
    if a.width() != b.width() or a.height() != b.height():
        return 0.0
    from PySide6.QtGui import QImage
    aa = a.convertToFormat(QImage.Format.Format_ARGB32)
    bb = b.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = aa.width(), aa.height()
    total_abs_diff = 0
    pixel_count = w * h * 4
    # Direct bytes iteration is substantially faster than QImage.pixel()
    # per pixel -- for a 240x200 image that's ~47000 px / 188000 chans.
    aa_bytes = bytes(aa.constBits()[: w * h * 4])
    bb_bytes = bytes(bb.constBits()[: w * h * 4])
    for pa, pb in zip(aa_bytes, bb_bytes):
        d = pa - pb
        total_abs_diff += d if d >= 0 else -d
    return 1.0 - (total_abs_diff / (255.0 * pixel_count))


# ── Goldens ────────────────────────────────────────────────────────

def golden_path(name: str, *, backend: str = BACKEND_QIMAGE) -> Path:
    """Path under ``tests/goldens/sidebar/<backend>/<name>.png``."""
    return _GOLDENS_ROOT / backend / f'{name}.png'


def assert_matches_golden(image, name: str, *,
                          backend: str = BACKEND_QIMAGE,
                          min_similarity: float = 0.99) -> None:
    """Compare ``image`` to the golden saved for ``name`` under the
    given backend. With ``UPDATE_GOLDENS=1`` in the environment, writes
    the golden instead of asserting -- useful for the first run after a
    deliberate rendering change.

    Raises ``AssertionError`` if the similarity is below
    ``min_similarity`` or the golden doesn't exist.
    """
    path = golden_path(name, backend=backend)
    if os.environ.get('UPDATE_GOLDENS'):
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(path), 'PNG')
        return

    assert path.exists(), (
        f'golden missing: {path}. Run with UPDATE_GOLDENS=1 to create it.')

    from PySide6.QtGui import QImage
    golden = QImage()
    if not golden.load(str(path), 'PNG'):
        raise AssertionError(f'failed to load golden {path}')
    sim = raster_similarity(image, golden)
    assert sim >= min_similarity, (
        f'raster similarity {sim:.4f} below threshold {min_similarity} '
        f'for {name} ({backend}). Review with UPDATE_GOLDENS=1 if intended.')


# ── Text-presence (for draw_* assertions that used to inspect mocks) ─

def image_contains_any_nonbackground(image, rect) -> bool:
    """True iff any pixel in ``rect`` (x, y, w, h) differs from the
    image's background color (assumed to be the pixel at (0, 0)).

    Replaces ``assert 'Background' in sctx.texts`` style assertions
    with a raster-based equivalent: 'the draw-fn drew *something* in
    this row's bounding box'. Tests that need to verify *which* text
    was drawn can pixel-compare against a golden reference instead.
    """
    if image is None:
        return False
    x, y, w, h = rect
    bg = image.pixel(0, 0)
    for px in range(max(0, x), min(image.width(), x + w)):
        for py in range(max(0, y), min(image.height(), y + h)):
            if image.pixel(px, py) != bg:
                return True
    return False
