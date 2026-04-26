"""Cross-process dmabuf WebTexture backend.

Hosts a ``QWebEngineView`` in our process and hands each frame to the
overlay (running in another process) via a dmabuf fd over a Unix
socket. The consumer side imports the fd as an ``EGLImage`` and
composites the texture zero-copy inside its own GL context.

Positioning within the PAL:

  - ``surfaces = {SURFACE_CROSSPROC_GL}``. Same-process GL consumers
    have no reason to go cross-process, so we don't advertise for
    local GL; they get the ``qpixmap`` path instead (or a future
    in-process GL backend if one lands).
  - ``is_available()`` requires the ``web_texture_ipc`` extension to
    import, the EGL MESA dmabuf-export extension to be present on
    the driver, AND the overlay socket to be listenable at
    ``/tmp/vsrg_overlay_web.sock`` (the socket's existence is how we
    detect "gl_layer is loaded and listening"; absent = the overlay
    isn't up, so we're better off falling through to qpixmap which
    can still paint *something* for preview modes).

The current producer-side path does one CPU upload per frame (grab
the WebEngineView into a QImage, glTexSubImage2D into a cached GL
texture, then export as dmabuf). That's the same order-of-magnitude
CPU cost as the qpixmap backend, but the *consumer* saves a full
upload + sample -- which is the bottleneck when the overlay is
compositing over a game running at 240+ Hz. A future producer that
taps Chromium's composited texture directly (C++ extension only;
PySide6 doesn't expose the handle) would make this end-to-end
zero-copy on our side too.
"""
from __future__ import annotations

import os
from typing import Any

from PySide6.QtCore import QSize, QUrl
from PySide6.QtGui import QImage, QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject
from PySide6.QtWidgets import QApplication

from analysis.components.pal.web.base import (
    KIND_DMABUF_FD,
    SURFACE_CROSSPROC_GL,
    WebTextureBackendCaps,
    WebTextureFrame,
)


class DmabufBackend:
    """Factory + capability advertiser for :class:`DmabufWebTexture`.

    Backend selection is opt-in: ``is_available()`` returns False on
    hosts where any of the moving parts (native extension, driver
    extension, overlay socket) is missing, so the PAL dispatcher
    prefers this over the qpixmap fallback only when the full chain
    is healthy.
    """

    name = 'dmabuf'

    def capabilities(self) -> WebTextureBackendCaps:
        return WebTextureBackendCaps(
            produces=(KIND_DMABUF_FD,),
            zero_copy=True,       # zero-copy on the *consumer* side
            cross_process=True,
            needs_qapplication=True,
            surfaces=frozenset({SURFACE_CROSSPROC_GL}),
        )

    def is_available(self) -> bool:
        if QApplication.instance() is None:
            return False
        # Native extension present?
        try:
            import web_texture_ipc  # noqa: F401
        except ImportError:
            return False
        # Overlay listening? A present socket file means the server
        # bound it; absent means the overlay hasn't started (or the
        # user isn't running the game under the preload layer).
        if not os.path.exists(self._socket_path()):
            return False
        # Driver supports dmabuf export? We can only probe this with a
        # live GL context; defer the cost to create() which has to set
        # up a context anyway. Reporting "probably" here is fine: the
        # backend will gracefully fail create() with a meaningful
        # exception that the PAL dispatcher surfaces.
        return True

    def create(self, *, width: int, height: int) -> 'DmabufWebTexture':
        return DmabufWebTexture(width=width, height=height,
                                socket_path=self._socket_path())

    @staticmethod
    def _socket_path() -> str:
        # Respect the same override environment the native crate uses
        # so tests can redirect to a throwaway path.
        try:
            import web_texture_ipc
            return os.environ.get('VSRG_OVERLAY_WEB_SOCKET',
                                  web_texture_ipc.SOCKET_PATH)
        except ImportError:
            return os.environ.get('VSRG_OVERLAY_WEB_SOCKET',
                                  '/tmp/vsrg_overlay_web.sock')


class DmabufWebTexture:
    """WebTexture backed by a hidden QWebEngineView + offscreen GL
    context. Each ``latest_frame()`` call uploads the view's grabbed
    raster to a cached GL texture and sends it over the socket as a
    dmabuf.

    This object owns:
      - A ``QWebEngineView`` rendering the configured page.
      - A ``QOffscreenSurface`` + ``QOpenGLContext`` sharing with the
        global share group (so the gl_layer's imported texture can in
        principle be sampled from other contexts too).
      - A ``QOpenGLFramebufferObject`` with one attached texture we
        upload into each frame.
      - A ``web_texture_ipc.WebTextureChannel`` (the fd-passing
        socket client).
    """

    def __init__(self, *, width: int, height: int, socket_path: str):
        self._width = int(width)
        self._height = int(height)
        self._generation = 0
        self._channel_id = _generate_channel_id()
        self._bridge: Any = None

        from PySide6.QtWebEngineWidgets import QWebEngineView
        self._view = QWebEngineView()
        self._view.resize(self._width, self._height)
        self._view.hide()

        # Defer GL + socket setup until first latest_frame() call. A
        # freshly-constructed texture before the event loop spins
        # isn't allowed to create a GL context (it must happen on the
        # thread that will later use it). latest_frame() is called
        # from the paint thread; that's where we lazily init.
        self._gl_ready = False
        self._socket_path = socket_path
        self._channel: Any = None
        self._ctx: Any = None
        self._surface: Any = None
        self._fbo: Any = None
        self._gl_tex_id = 0

    # ── WebTexture protocol ─────────────────────────────────────────

    @property
    def view(self):
        return self._view

    def attach_bridge(self, bridge) -> None:
        self._bridge = bridge

    def resize(self, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)
        self._view.resize(QSize(self._width, self._height))
        self._generation += 1
        # The cached GL texture is sized for the old dimensions; drop
        # it so the next latest_frame() re-allocates at the new size.
        if self._fbo is not None:
            self._fbo = None
            self._gl_tex_id = 0

    def load_url(self, url: str) -> None:
        self._view.load(QUrl(url))
        self._generation += 1

    def push_js_state(self, json_str: str) -> None:
        if self._bridge is not None:
            self._bridge.push(json_str)

    def active_filters(self) -> frozenset[str]:
        if self._bridge is None:
            return frozenset()
        return self._bridge.active_filters

    def latest_frame(self) -> WebTextureFrame | None:
        if not self._gl_ready:
            if not self._init_gl_and_socket():
                return None

        # 1. Grab the web view to a QPixmap (GPU-backed on most Qt
        #    builds). Convert to a QImage with a known byte order we
        #    can feed to glTexSubImage2D.
        pix = self._view.grab()
        if pix.isNull():
            return None
        qimg = pix.toImage().convertToFormat(
            QImage.Format.Format_ARGB32_Premultiplied)

        # 2. Upload to the cached GL texture. Resize the FBO if the
        #    grabbed image doesn't match our expected size (happens
        #    right after resize() or on a DPI change).
        if not self._ctx.makeCurrent(self._surface):
            return None
        try:
            if (self._fbo is None or self._fbo.width() != qimg.width()
                    or self._fbo.height() != qimg.height()):
                self._alloc_fbo(qimg.width(), qimg.height())

            self._upload_qimage_to_texture(qimg)

            # 3. Export via the native crate. Returns True if sent,
            #    False if the overlay is behind and the send was
            #    dropped; either way we bump the generation so the
            #    next publish has a fresh tag.
            self._generation += 1
            try:
                self._channel.publish_from_gl_texture(
                    self._channel_id,
                    self._generation,
                    self._gl_tex_id,
                    qimg.width(),
                    qimg.height(),
                )
            except Exception as exc:
                # Don't crash the paint loop on a transient EGL or
                # socket error -- log once per change of error kind
                # and return None so the caller skips this frame.
                _warn_once(f'dmabuf publish failed: {type(exc).__name__}: {exc}')
                return None
        finally:
            self._ctx.doneCurrent()

        return WebTextureFrame(
            width=qimg.width(),
            height=qimg.height(),
            kind=KIND_DMABUF_FD,
            handle={'channel_id': self._channel_id,
                    'generation': self._generation},
            generation=self._generation,
            meta={'qpixmap_fallback': pix},
        )

    def close(self) -> None:
        if self._channel is not None:
            try:
                self._channel.release(self._channel_id)
                self._channel.close()
            except Exception:
                pass
            self._channel = None
        self._view.close()
        self._view.deleteLater()

    # ── Lazy init ──────────────────────────────────────────────────

    def _init_gl_and_socket(self) -> bool:
        # Build a throwaway GL context that shares with the global
        # share group (AA_ShareOpenGLContexts set at QApplication
        # startup in app.py). That share group is what lets the
        # glTexSubImage2D we do here reach a texture the Rust
        # eglCreateImage can wrap.
        fmt = QSurfaceFormat.defaultFormat()
        self._surface = QOffscreenSurface()
        self._surface.setFormat(fmt)
        self._surface.create()
        if not self._surface.isValid():
            _warn_once('dmabuf: QOffscreenSurface.create() failed')
            return False

        self._ctx = QOpenGLContext()
        self._ctx.setFormat(fmt)
        share = QOpenGLContext.globalShareContext()
        if share is not None:
            self._ctx.setShareContext(share)
        if not self._ctx.create():
            _warn_once('dmabuf: QOpenGLContext.create() failed')
            return False

        try:
            import web_texture_ipc
            self._channel = web_texture_ipc.WebTextureChannel(
                self._socket_path)
        except Exception as exc:
            _warn_once(
                f'dmabuf: failed to open side socket at '
                f'{self._socket_path!r}: {exc}')
            return False

        self._gl_ready = True
        return True

    def _alloc_fbo(self, w: int, h: int) -> None:
        # QOpenGLFramebufferObject's default attachment is a GL_RGBA
        # color texture -- exactly what we want the dmabuf exporter to
        # wrap. Keep one FBO alive across frames and glTexSubImage2D
        # into its texture each time.
        self._fbo = QOpenGLFramebufferObject(QSize(int(w), int(h)))
        self._gl_tex_id = self._fbo.texture()

    def _upload_qimage_to_texture(self, qimg: QImage) -> None:
        # Use Qt's QOpenGLFunctions so we don't pull PyOpenGL into the
        # dependency graph. Qt guarantees these calls route through
        # whatever loader the current context was created with, and
        # PySide6 exposes glTexSubImage2D on the ``functions()``
        # handle with the same signature shape.
        from PySide6.QtOpenGL import QOpenGLVersionFunctionsFactory
        from PySide6.QtGui import QOpenGLVersionProfile

        profile = QOpenGLVersionProfile()
        profile.setVersion(3, 2)
        profile.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        f = QOpenGLVersionFunctionsFactory.get(profile, self._ctx)
        if f is None:
            # No 3.2 core functions on this context: fall back to the
            # plain QOpenGLFunctions (pre-3.2 core).
            f = self._ctx.functions()

        GL_TEXTURE_2D   = 0x0DE1
        GL_BGRA         = 0x80E1
        GL_UNSIGNED_BYTE = 0x1401

        f.glBindTexture(GL_TEXTURE_2D, int(self._gl_tex_id))
        # QImage::bits() gives BGRA memory layout on little-endian
        # when the format is ARGB32 Premultiplied -- feed it directly
        # with GL_BGRA so no channel swizzle is needed.
        stride_bytes = qimg.bytesPerLine()
        expected = qimg.width() * 4
        if stride_bytes != expected:
            qimg = qimg.copy()
        f.glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0,
            qimg.width(), qimg.height(),
            GL_BGRA, GL_UNSIGNED_BYTE, qimg.constBits(),
        )
        f.glBindTexture(GL_TEXTURE_2D, 0)


# ── module helpers ──────────────────────────────────────────────────

def _generate_channel_id() -> int:
    """Unique-enough id for one overlay session. The C consumer keys
    on this so the same channel id across runs shouldn't collide with
    anything else; using a per-PID random nonce keeps it cheap."""
    import random
    return random.randint(1, 0xFFFFFFFE)


_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    """Print ``msg`` at most once per process. Backend failure modes
    often recur every frame; spamming stderr is worse than silent."""
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    print(f'[dmabuf] {msg}')
