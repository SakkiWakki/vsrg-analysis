"""In-game overlay window for osu! mania live stats.

A frameless, always-on-top Qt window that displays combo / accuracy /
UR and a small offset histogram while the player is actively in
gameplay. Hides itself outside of ``GameState.play`` (menu, results,
pause, song select) so it doesn't cover UI the player is trying to
interact with.

## Design notes

**Positioning.** osu! on Linux runs under wine, and the common modern
Linux session is Wayland — where no public API lets one process read
another's window geometry. Rather than ship platform-specific glue
that only works under X11+xdotool, the overlay is a free-floating
window the user positions and sizes once; we persist the geometry
per-screen-resolution via ``QSettings`` so fullscreen/borderless/
windowed all keep independent layouts. In practice this is what
streamers already do with OBS widgets.

**Drag/resize.** Frameless windows don't get a native title bar or
edge grips, so we roll our own: a drag handle at the top that moves
the window on mouse drag, and a ``QSizeGrip`` in the corner. Both are
optional — the user can disable the handle to click through the top
strip, although in v1 we always show it to keep the UX obvious.

**Visibility.** One ``QTimer`` drives refresh *and* show/hide: if the
latest ``LiveSnapshot`` reports ``in_gameplay=False`` we ``hide()``;
otherwise we ``show()`` and update the labels/figure. Hiding instead
of destroying keeps QSettings-persisted geometry stable across
plays.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPalette
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QSizeGrip, QVBoxLayout,
                               QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


_SETTINGS_ORG = 'etterna-analysis'
_SETTINGS_APP = 'osu_live_overlay'


class _DragHandle(QWidget):
    """Top strip that drags the parent window on mouse drag. Frameless
    windows have no native title bar, so we synthesize one."""

    def __init__(self, target: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self._target = target
        self._press_global: QPoint | None = None
        self._press_window_pos: QPoint | None = None
        self.setFixedHeight(18)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30, 220))
        self.setPalette(pal)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_global = ev.globalPosition().toPoint()
            self._press_window_pos = self._target.pos()
            ev.accept()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._press_global is None or self._press_window_pos is None:
            return
        delta = ev.globalPosition().toPoint() - self._press_global
        self._target.move(self._press_window_pos + delta)
        ev.accept()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        self._press_global = None
        self._press_window_pos = None
        ev.accept()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30, 220))
        p.setPen(QColor(200, 200, 200))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '⋯ drag ⋯')


class OsuLiveOverlay(QWidget):
    """Top-level overlay window. Create once and keep a reference (Qt
    will garbage-collect top-level widgets without a parent)."""

    def __init__(self, client=None, *, interval_ms: int = 100):
        super().__init__(None,
                         Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # ShowWithoutActivating so clicking the overlay doesn't steal
        # focus from osu! — essential for an always-on-top widget on
        # top of a game that consumes every input event.
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle('osu! live overlay')

        if client is None:
            from plugins.unsafe.osu_live.client import get_client
            client = get_client()
        self._client = client

        # ── Layout ────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._drag = _DragHandle(self)
        root.addWidget(self._drag)

        body = QWidget(self)
        body.setAutoFillBackground(True)
        body_pal = body.palette()
        body_pal.setColor(QPalette.ColorRole.Window, QColor(20, 20, 20, 200))
        body.setPalette(body_pal)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 6, 8, 6)
        body_layout.setSpacing(4)
        root.addWidget(body, 1)

        self._stats = QLabel('— — —', body)
        f = self._stats.font()
        f.setPointSize(14)
        f.setBold(True)
        self._stats.setFont(f)
        self._stats.setStyleSheet('color: #f5f5f5;')
        body_layout.addWidget(self._stats)

        self._map = QLabel('', body)
        self._map.setStyleSheet('color: #aaa;')
        body_layout.addWidget(self._map)

        # Small embedded histogram of hit offsets.
        self._fig = Figure(figsize=(3.0, 1.2), facecolor='#111111')
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet('background: transparent;')
        body_layout.addWidget(self._canvas, 1)

        # Size grip pinned bottom-right, inside a row that lets the
        # grip hug the corner without needing an absolute layout.
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        grip = QSizeGrip(body)
        grip_row.addWidget(grip, 0, Qt.AlignmentFlag.AlignBottom
                           | Qt.AlignmentFlag.AlignRight)
        body_layout.addLayout(grip_row)

        # ── Restore geometry ──────────────────────────────────────
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        geom = self._settings.value(self._geometry_key())
        if geom is not None:
            self.restoreGeometry(geom)
        else:
            self.resize(320, 200)

        # ── Refresh timer ─────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _geometry_key(self) -> str:
        screen = self.screen()
        if screen is None:
            return 'geometry/default'
        sz = screen.size()
        return f'geometry/{sz.width()}x{sz.height()}'

    def closeEvent(self, ev):
        self._settings.setValue(self._geometry_key(), self.saveGeometry())
        self._timer.stop()
        super().closeEvent(ev)

    def moveEvent(self, ev):
        self._settings.setValue(self._geometry_key(), self.saveGeometry())
        super().moveEvent(ev)

    def resizeEvent(self, ev):
        self._settings.setValue(self._geometry_key(), self.saveGeometry())
        super().resizeEvent(ev)

    # ── Refresh ───────────────────────────────────────────────────

    def _refresh(self):
        snap = self._client.snapshot()
        if not snap.in_gameplay:
            if self.isVisible():
                self.hide()
            return
        if not self.isVisible():
            # ShowWithoutActivating flag ensures this doesn't pull
            # focus away from osu!.
            self.show()

        ur = _unstable_rate_ms(snap)
        self._stats.setText(
            f'{snap.combo}x   {snap.accuracy:.2f}%   UR {ur:.1f}'
        )
        self._map.setText(snap.map_title or '')
        self._redraw_hist(snap)

    def _redraw_hist(self, snap):
        import numpy as np
        self._ax.clear()
        offsets_ms = (np.asarray(snap.offsets, dtype=float) * 1000.0
                      if len(snap.offsets) else np.zeros(0))
        if len(offsets_ms):
            # Clip to a ±100 ms window — anything outside is either a
            # miss-adjacent click or stale data; squishing it into the
            # edges helps the histogram stay legible.
            clipped = np.clip(offsets_ms, -100.0, 100.0)
            self._ax.hist(clipped, bins=40, range=(-100, 100),
                          color='#4aa3ff', edgecolor='none')
        self._ax.set_facecolor('#111111')
        self._ax.set_xlim(-100, 100)
        self._ax.axvline(0, color='#888', linewidth=0.8, alpha=0.6)
        for spine in self._ax.spines.values():
            spine.set_color('#333')
        self._ax.tick_params(colors='#888', labelsize=7)
        self._ax.set_yticks([])
        self._fig.tight_layout(pad=0.2)
        self._canvas.draw_idle()


def _unstable_rate_ms(snap) -> float:
    """Compute UR (10× stdev of hit offsets in ms).

    The native reader doesn't publish UR directly, and tosu's field
    may be zero for the first few hits. Fall back to computing from
    the offsets array so the overlay reads something meaningful
    regardless of source.
    """
    if snap.unstable_rate and snap.unstable_rate > 0:
        return float(snap.unstable_rate)
    import numpy as np
    if len(snap.offsets) < 2:
        return 0.0
    offsets_ms = np.asarray(snap.offsets, dtype=float) * 1000.0
    return float(10.0 * np.std(offsets_ms))


_overlay_instance: OsuLiveOverlay | None = None


def open_overlay() -> None:
    """Toggle the overlay window. Called from the library toolbar."""
    global _overlay_instance
    if _overlay_instance is not None:
        # Toggle off if already open.
        try:
            _overlay_instance.close()
        except RuntimeError:
            pass
        _overlay_instance = None
        return
    _overlay_instance = OsuLiveOverlay()
    # Show immediately if already in gameplay; otherwise the refresh
    # timer will show it on the first in-gameplay tick.
    snap = _overlay_instance._client.snapshot()
    if snap.in_gameplay:
        _overlay_instance.show()
