"""Native Qt replay-player canvas."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget

from analysis.player.render.qt_renderer import QtPlayerRenderer


class PlayerCanvas(QWidget):
    def __init__(self, player, parent=None):
        super().__init__(parent)
        self.player = player
        self.renderer = QtPlayerRenderer(player.plugins)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 400)

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
