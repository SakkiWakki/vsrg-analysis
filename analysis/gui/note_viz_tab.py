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
        from analysis.viz.note_visualizer import (render_chart, etterna_windows,
                                      osu_mania_windows, _judgment_counts,
                                      _legend_axes, effective_osu_od)
        self.replay = replay
        self.game = game
        if game == 'osu':
            base_od = od if od is not None else float(replay.get('od', 8.0))
            mods = int(replay.get('mods', 0))
            eff_od = effective_osu_od(base_od, mods)
            self.windows = osu_mania_windows(od=eff_od)
            self.unit_label = f'time (ms)  —  OD {eff_od:.1f}'
            self.rpm = None
            self.win = 8000
        else:
            j = judge or 'J4'
            self.windows = etterna_windows(j)
            self.unit_label = f'noterow  —  {j}'
            self.rpm = 0.37
            self.win = 2400
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
