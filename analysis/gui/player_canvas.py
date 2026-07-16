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

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from analysis.gui import paint_profiler
from analysis.player.render.qt_renderer import QtPlayerRenderer

# Below this inter-swap interval the compositor clearly isn't
# vsync-throttling us (broken driver, offscreen, VM); rescheduling
# immediately would busy-spin the GUI thread, so defer instead. The
# threshold must sit below the refresh period of any real display
_SPIN_GUARD_S = 0.0005
_SPIN_DEFER_MS = 2


class PlayerCanvas(QOpenGLWidget):
    # TODO: Port to Component API
    def __init__(self, player, parent=None, *, swap_paced=True):
        super().__init__(parent)
        self.player = player
        self.renderer = QtPlayerRenderer(player.plugins)
        self._last_swap = 0.0
        if swap_paced:
            # Presentation-driven render loop: schedule the next paint
            # each time a frame is handed to the compositor, so paints
            # run once per displayed frame at a fixed phase instead of
            # beating a wall-clock timer against the display's refresh.
            # Queued so the update lands on the next event-loop pass
            # rather than re-entering the swap.
            self.frameSwapped.connect(self._schedule_next_frame,
                                      Qt.ConnectionType.QueuedConnection)
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

    def _schedule_next_frame(self):
        # The chain intentionally dies while paused (nothing animates;
        # input handlers repaint on demand) and restarts from the next
        # update() -- unpause, seek, expose, or any handled input.
        if self.player.paused:
            return
        now = time.monotonic()
        throttled = now - self._last_swap >= _SPIN_GUARD_S
        self._last_swap = now
        if throttled:
            self.update()
        else:
            QTimer.singleShot(_SPIN_DEFER_MS, self.update)

    def showEvent(self, ev):
        # Hidden widgets don't paint, so the swap chain dies on tab
        # switch / minimize; re-arm it on expose.
        self.update()
        super().showEvent(ev)

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

