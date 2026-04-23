"""CPU-readback WebTexture backend.

Always-available fallback: owns a hidden ``QWebEngineView`` rendering at
its native resolution, grabs the widget as a ``QPixmap`` on demand, and
returns it as a :class:`WebTextureFrame` tagged :data:`KIND_QPIXMAP`.

Trade-offs:
- Pros: Works on any Qt host (QWidget, QOpenGLWidget, QML via QSGPixmap).
  No GL context dependency. Transparent background preserved.
- Cons: One CPU readback per frame grab. Fine for 30 Hz replay overlays;
  becomes expensive above ~120 Hz or for large (>1080p) overlays.

This backend is the one Phase 1 ships. Phase 2 introduces a scene-graph
/ GL backend registered ahead of this one so local-GL surfaces avoid
readback; this backend stays registered as the universal fallback.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, QUrl
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from analysis.components.pal.web.base import (
    KIND_QPIXMAP,
    SURFACE_LOCAL_CPU,
    SURFACE_LOCAL_GL,
    WebTextureBackendCaps,
    WebTextureFrame,
)


class QPixmapBackend:
    """Factory + capability advertiser for :class:`QPixmapWebTexture`."""

    name = 'qpixmap'

    def capabilities(self) -> WebTextureBackendCaps:
        # Advertise both local_cpu and local_gl: the GL-accelerated
        # scene-graph backend (when we add it in phase 2) will outrank
        # us on local_gl via zero-copy, but until it exists we cover
        # both so a QOpenGLWidget host still gets frames (just with a
        # readback).
        return WebTextureBackendCaps(
            produces=(KIND_QPIXMAP,),
            zero_copy=False,
            cross_process=False,
            needs_qapplication=True,
            surfaces=frozenset({SURFACE_LOCAL_CPU, SURFACE_LOCAL_GL}),
        )

    def is_available(self) -> bool:
        # We need a live QApplication (QWebEngineView won't construct
        # without one) and the WebEngine module must be importable.
        if QApplication.instance() is None:
            return False
        try:
            import PySide6.QtWebEngineWidgets  # noqa: F401
        except ImportError:
            return False
        return True

    def create(self, *, width: int, height: int) -> 'QPixmapWebTexture':
        return QPixmapWebTexture(width=width, height=height)


class QPixmapWebTexture:
    """Live off-screen ``QWebEngineView`` that exposes its latest frame
    as a :data:`KIND_QPIXMAP`-tagged :class:`WebTextureFrame`.

    The view is created hidden; it still composites (Qt's WebEngine
    keeps the page's layer tree live as long as the widget exists), so
    ``grab()`` captures an up-to-date raster. We rebuild the pixmap on
    demand rather than maintaining a render-on-paint cache because
    overlays can repaint at ~60 Hz while the consumer may only ask for
    a frame at 30 Hz; letting WebEngine paint freely and pulling on
    demand keeps CPU cost bounded by the consumer's cadence.

    Shim injection (``shim.js`` + QWebChannel) is owned by the
    :mod:`plugins.unsafe.tosu_overlay.view` layer's ``TosuOverlayView``
    class today; this texture-level wrapper is protocol-agnostic and
    doesn't know about tosu specifically. ``push_js_state`` / filter
    tracking are wired via an optional ``OverlayBridge`` the caller can
    attach after construction.
    """

    def __init__(self, *, width: int, height: int):
        from PySide6.QtWebEngineWidgets import QWebEngineView
        self._view = QWebEngineView()
        # Hide off-screen so the view is composited but not painted to
        # the screen. ``setAttribute(Qt.WA_DontShowOnScreen)`` would
        # also work but leaves the widget un-realized on some styles;
        # a shown-but-off-screen widget is the more portable path.
        self._view.resize(width, height)
        self._view.hide()

        self._width = int(width)
        self._height = int(height)
        self._generation = 0
        self._filters: frozenset[str] = frozenset()
        # Optional caller-attached bridge: see ``attach_bridge``.
        self._bridge = None

    # ── Widget hooks for plugin wiring ─────────────────────────────
    # The texture is protocol-agnostic. Callers (e.g. the tosu
    # plugin) attach their own shim and QWebChannel bridge to the
    # underlying view and tell us where to route state pushes by
    # calling ``attach_bridge``.

    @property
    def view(self):
        """The underlying ``QWebEngineView``. Callers attach scripts,
        web channels, bridges, etc. to this before loading a URL."""
        return self._view

    def attach_bridge(self, bridge) -> None:
        """Register the protocol bridge (exposes ``push`` and
        ``active_filters``). After this, :meth:`push_js_state` and
        :meth:`active_filters` route through it."""
        self._bridge = bridge

    # ── WebTexture protocol ────────────────────────────────────────

    def resize(self, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)
        self._view.resize(QSize(self._width, self._height))
        # Bump generation so the consumer re-uploads even if the page
        # hasn't actually repainted yet.
        self._generation += 1

    def load_url(self, url: str) -> None:
        self._view.load(QUrl(url))
        self._generation += 1

    def push_js_state(self, json_str: str) -> None:
        if self._bridge is not None:
            self._bridge.push(json_str)

    def push_precise_state(self, json_str: str) -> None:
        # The precise channel is pushed via a direct runJavaScript call
        # on the page -- the bridge only owns the v1+v2 main channel.
        safe = json_str.replace('\\', '\\\\').replace('`', '\\`')
        self._view.page().runJavaScript(
            f'window._tosuPushPrecise && window._tosuPushPrecise(`{safe}`);')

    def active_filters(self) -> frozenset[str]:
        if self._bridge is None:
            return frozenset()
        return self._bridge.active_filters

    def latest_frame(self) -> WebTextureFrame | None:
        pix: QPixmap = self._view.grab()
        if pix.isNull():
            return None
        self._generation += 1
        return WebTextureFrame(
            width=pix.width(),
            height=pix.height(),
            kind=KIND_QPIXMAP,
            handle=pix,
            generation=self._generation,
        )

    def close(self) -> None:
        self._view.close()
        self._view.deleteLater()
