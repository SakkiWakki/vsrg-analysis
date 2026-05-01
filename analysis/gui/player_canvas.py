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

from analysis.gui import paint_profiler
from analysis.player.render.qt_renderer import QtPlayerRenderer


class PlayerCanvas(QOpenGLWidget):
    # TODO: Port to Component API
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
        paint_profiler.begin_frame()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, False)
            self.renderer.draw(self.player, painter, self.player.t)
        finally:
            painter.end()
            paint_profiler.end_frame()

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
        # edit mode.
        self._update_cursor(ev)
        super().mouseMoveEvent(ev)

    @staticmethod
    def _event_xy(ev) -> tuple[int, int]:
        pos = ev.position() if hasattr(ev, 'position') else ev.pos()
        return int(pos.x()), int(pos.y())

    @staticmethod
    def _rect_contains(rect, x, y):
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def _cursor_shape_for_event(self, ev):
        hud = getattr(self.player, 'hud', None)

        if hud is None or not hud.edit_mode:
            return Qt.ArrowCursor
        if hud.drag_key is not None:
            return Qt.ClosedHandCursor
        if hud.resize_key is not None:
            return Qt.SizeFDiagCursor

        x, y = self._event_xy(ev)

        for rect, action, _payload in reversed(hud.hitboxes):
            if not self._rect_contains(rect, x, y):
                continue
            if action == 'begin_resize_section':
                return Qt.SizeFDiagCursor
            if action == 'begin_drag_section':
                return Qt.OpenHandCursor

        return Qt.ArrowCursor


    def _update_cursor(self, ev):
        shape = self._cursor_shape_for_event(ev)

        if self.cursor().shape() != shape:
            self.setCursor(shape)

