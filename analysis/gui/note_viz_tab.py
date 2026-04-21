"""Embedded scrollable note visualizer tab."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QSlider)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

from analysis.gui.widgets import _viz_toolbar


class NoteVizTab(QWidget):
    """Scrollable note visualizer with a slider, embedded."""
    def __init__(self, replay, game='etterna', od=None, judge=None, on_play=None):
        super().__init__()
        from matplotlib.figure import Figure
        from analysis.viz.note_visualizer import (render_chart, _judgment_counts,
                                      _legend_axes)
        from analysis.core import gui_adapter as gui_mod
        self.replay = replay
        self.game = game
        cfg = gui_mod.get(game).note_viz_config(replay, judge=judge, od=od)
        self.windows = cfg['windows']
        self.unit_label = cfg['unit_label']
        self.rpm = cfg['rows_per_ms']
        self.win = cfg['win']
        self._render_chart = render_chart

        fig = Figure(figsize=(10, 10))
        fig.patch.set_facecolor('#121212')
        gs = fig.add_gridspec(1, 4, width_ratios=[3, 0.05, 1, 0.05], wspace=0.1)
        self.ax = fig.add_subplot(gs[0, 0])
        legend = fig.add_subplot(gs[0, 2])
        counts = _judgment_counts(replay, self.windows)
        _legend_axes(legend, self.windows, counts)

        self.total = int(replay['noterows'].max()) + 1 if len(replay['noterows']) else 1000
        self._draw(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(NavToolbar(self.canvas, self))
        layout.addWidget(self.canvas, 1)

        ctl = QHBoxLayout()
        self.pos_lbl = QLabel('pos: 0')
        ctl.addWidget(self.pos_lbl)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, max(1, self.total - self.win))
        slider.setSingleStep(max(1, self.win // 40))
        slider.valueChanged.connect(self._on_slide)
        ctl.addWidget(slider, 1)
        layout.addLayout(ctl)

        if on_play is not None:
            layout.addLayout(_viz_toolbar(on_play))

    def _draw(self, start):
        self.ax.clear()
        self._render_chart(self.replay, window_units=self.win, start=int(start),
                           ax=self.ax, windows=self.windows,
                           unit_label=self.unit_label, rows_per_ms=self.rpm)

    def _on_slide(self, v):
        self.pos_lbl.setText(f'pos: {v}')
        self._draw(v)
        self.canvas.draw_idle()
