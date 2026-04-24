"""Native Qt replay-player canvas.

Backed by ``QOpenGLWidget``: QPainter draws still work unchanged
(Qt routes them through ``QOpenGLPaintDevice`` internally) and the
widget is now a usable compositor target for GPU-backed overlay
textures produced by the web-texture PAL's GL backend. CPU-path
components still render identically -- the rasterization happens
through ``QOpenGLPaintDevice`` rather than a widget backing store,
and the output is pixel-equivalent per ``tests/test_sidebar_output.py``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from analysis.player.render.qt_renderer import QtPlayerRenderer


class PlayerCanvas(QOpenGLWidget):
    def __init__(self, player, parent=None):
        super().__init__(parent)
        self.player = player
        self.renderer = QtPlayerRenderer(player.plugins)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 400)
        # Mouse tracking lets move events fire without a button held.
        # Needed so the layout-edit cursor (grab hand) can update based
        # on hover, and so drag-in-progress updates even if the user's
        # pointer briefly leaves the sidebar region.
        self.setMouseTracking(True)
        # Force the native GL surface to be realized before we're added to
        # a tab's layout: without this, QTabWidget's first reparent of a
        # QOpenGLWidget triggers a top-level window recomposition on some
        # compositors (X11/Wayland/XWayland), which looks like the main
        # window briefly closing and reopening. Creating the surface up
        # front keeps it tied to this widget's native handle.
        self.setAttribute(Qt.WA_NativeWindow, True)

    def resizeEvent(self, ev):
        self.player.W = max(200, int(self.width()))
        self.player.H = max(200, int(self.height()))
        super().resizeEvent(ev)

    def paintEvent(self, _ev):
        self.player.W = max(200, int(self.width()))
        self.player.H = max(200, int(self.height()))
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, False)
            self.renderer.draw(self.player, painter, self.player.t)
        finally:
            painter.end()

    def mousePressEvent(self, ev):
        self.setFocus(Qt.MouseFocusReason)
        pos = ev.position() if hasattr(ev, 'position') else ev.pos()
        if self.player.handle_mouse_down(int(pos.x()), int(pos.y())):
            self.update()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        # Update the cursor to signal draggable / resizable regions in
        # edit mode. Also important when the canvas is the top-level
        # event target (e.g. player launched stand-alone rather than
        # inside PlayerTab), because the tab's eventFilter isn't
        # involved in that path.
        self._update_cursor(ev)
        super().mouseMoveEvent(ev)

    def _update_cursor(self, ev):
        hud = getattr(self.player, 'hud', None)
        if hud is None or not hud.edit_mode:
            if self.cursor().shape() != Qt.ArrowCursor:
                self.setCursor(Qt.ArrowCursor)
            return
        # During an active drag, keep the closed hand. During an active
        # resize, keep the diagonal resize cursor. Otherwise hover-test
        # against the hitbox list — a drag grab or resize handle wins.
        if hud.drag_key is not None:
            self.setCursor(Qt.ClosedHandCursor)
            return
        if hud.resize_key is not None:
            self.setCursor(Qt.SizeFDiagCursor)
            return
        pos = ev.position() if hasattr(ev, 'position') else ev.pos()
        x, y = int(pos.x()), int(pos.y())
        shape = Qt.ArrowCursor
        for rect, action, _payload in reversed(hud.hitboxes):
            rx, ry, rw, rh = rect
            if not (rx <= x <= rx + rw and ry <= y <= ry + rh):
                continue
            if action == 'begin_resize_section':
                shape = Qt.SizeFDiagCursor
                break
            if action == 'begin_drag_section':
                shape = Qt.OpenHandCursor
                break
        if self.cursor().shape() != shape:
            self.setCursor(shape)
