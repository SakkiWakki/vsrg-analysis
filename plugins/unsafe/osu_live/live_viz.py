"""Qt widget that periodically rebuilds a Matplotlib figure from a
:class:`TosuClient` snapshot and swaps it onto an embedded canvas.

One widget per viz panel; all share the same singleton client. Each
widget holds its own ``QTimer`` so refresh rate can be per-viz (a
heavy plot can poll slower than a tiny one).

Rebuild strategy: the whole figure is discarded each tick and the
provided ``build_fn`` produces a fresh one. Wasteful versus
``set_data``, but lets us reuse the existing static viz builders
unchanged. Fine for v1; revisit if CPU becomes an issue.
"""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class LiveFigureWidget(QWidget):
    """Hosts a FigureCanvas that rebuilds on a timer.

    ``build_fn`` is a callable accepting a ``replay``-shaped dict and
    returning a fresh ``matplotlib.figure.Figure``. Called every
    ``interval_ms`` from the Qt event loop."""

    def __init__(self, build_fn, client=None, *, interval_ms: int = 100,
                 parent=None):
        super().__init__(parent)
        if client is None:
            from plugins.unsafe.osu_live.tosu_client import get_client
            client = get_client()
        self._client = client
        self._build_fn = build_fn

        self._status = QLabel('Connecting to tosu…')
        self._canvas: FigureCanvasQTAgg | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._status)
        self._layout = layout

        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    def closeEvent(self, ev):
        self._timer.stop()
        super().closeEvent(ev)

    def _refresh(self):
        snap = self._client.snapshot()
        if not snap.connected and len(snap.offsets) == 0:
            self._set_status(
                'Not connected to tosu (is it running on 24050?)')
            return
        if len(snap.offsets) == 0:
            self._set_status(
                f'Connected — waiting for hits  |  map: {snap.map_title}')
            return

        try:
            fig = self._build_fn(snap.as_replay_dict())
        except Exception as exc:
            self._set_status(f'viz error: {exc}')
            return

        # Swap canvases. Tearing the old one down before building the
        # new keeps peak memory low — matplotlib figures aren't tiny.
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
            self._canvas = None

        self._canvas = FigureCanvasQTAgg(fig)
        self._layout.addWidget(self._canvas, 1)

        live = (
            f'{snap.map_title}  |  '
            f'combo {snap.combo}/{snap.max_combo}  '
            f'acc {snap.accuracy:.2f}%  UR {snap.unstable_rate:.1f}  '
            f'({len(snap.offsets)} hits)'
        )
        self._set_status(live)

    def _set_status(self, text: str):
        self._status.setText(text)
