"""The combined summary grid. Users can pick which plots to include."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QCheckBox, QDialog, QDialogButtonBox,
                               QSizePolicy, QScrollArea)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

from analysis.gui.widgets import WheelToScroll as _WheelToScroll

from analysis.viz.plots import (plot_full_report, FULL_REPORT_PLOTS,
                         FULL_REPORT_DEFAULT_SELECTION)


class _PlotPickerDialog(QDialog):
    def __init__(self, parent, current_selection):
        super().__init__(parent)
        self.setWindowTitle('Customize full report')
        v = QVBoxLayout(self)
        v.addWidget(QLabel('Plots to include:'))
        self.checks = {}
        for key, label in FULL_REPORT_PLOTS:
            cb = QCheckBox(label)
            cb.setChecked(key in current_selection)
            self.checks[key] = cb
            v.addWidget(cb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def selection(self):
        return [k for k, cb in self.checks.items() if cb.isChecked()]


class FullReportTab(QWidget):
    """Full report with a Customize button and bottom-left Play button."""

    def __init__(self, replay, on_play=None):
        super().__init__()
        self.replay = replay
        self.on_play = on_play
        self.selection = list(FULL_REPORT_DEFAULT_SELECTION)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Vertical-only scroll so tall custom selections don't clip.
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.canvas_host = QWidget()
        self.canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_host_layout = QVBoxLayout(self.canvas_host)
        self.canvas_host_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll.setWidget(self.canvas_host)
        self._layout.addWidget(self.scroll, 1)
        self._wheel_filter = _WheelToScroll(self.scroll)

        # Bottom-right action bar: Customize + Play replay.
        bot = QHBoxLayout()
        bot.setContentsMargins(4, 2, 4, 2)
        bot.addStretch(1)
        cust = QPushButton('Customize plots…')
        cust.clicked.connect(lambda _checked=False: self._customize())
        bot.addWidget(cust)
        if on_play is not None:
            play = QPushButton('▶ Play replay')
            play.clicked.connect(lambda _checked=False: on_play())
            bot.addWidget(play)
        self._layout.addLayout(bot)

        self._rebuild()

    def _rebuild(self):
        for i in reversed(range(self.canvas_host_layout.count())):
            w = self.canvas_host_layout.itemAt(i).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        fig = plot_full_report(self.replay, show=False, selection=self.selection)
        canvas = FigureCanvas(fig)
        # Enforce a minimum canvas height proportional to the figure so the
        # plots stay legible; scroll area kicks in when this exceeds viewport.
        h_in = fig.get_figheight()
        dpi = fig.get_dpi()
        canvas.setMinimumHeight(int(h_in * dpi * 0.6))
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.installEventFilter(self._wheel_filter)
        toolbar = NavToolbar(canvas, self)
        self.canvas_host_layout.addWidget(toolbar)
        self.canvas_host_layout.addWidget(canvas, 1)

    def _customize(self):
        dlg = _PlotPickerDialog(self, self.selection)
        if dlg.exec() == QDialog.Accepted:
            self.selection = dlg.selection()
            self._rebuild()


def build(replay, game='etterna', on_play=None, **_):
    w = FullReportTab(replay, on_play=on_play)
    w._has_play_btn = on_play is not None
    return w


def register(add):
    add('Full report (all plots)', build, category='widget')
