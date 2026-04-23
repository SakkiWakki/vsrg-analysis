from __future__ import annotations

from dataclasses import dataclass

from analysis.components.api import DataNotAvailable, GameState, SURFACE_VIZ
from analysis.components.gui_backend import PlayerDataAnalysis
from analysis.plugins.host_api import PluginConfig


VIZ_ATTACH_LIBRARY_TAB = 'library_tab'
VIZ_ATTACH_WINDOW = 'window'
VIZ_CATEGORY_CHART = 'chart'
VIZ_CATEGORY_WIDGET = 'widget'

LEFT_CLR = '#4fc3f7'
RIGHT_CLR = '#ff8a65'
MS = 1000.0

_FIGURE_CLASS = None
_NOTE_VIZ_TAB = None
_QDIALOG_ACCEPTED = None
_QCHECKBOX = None
_QDIALOG = None
_QDIALOG_BUTTON_BOX = None
_QHBOX_LAYOUT = None
_QLABEL = None
_QPUSH_BUTTON = None
_QSCROLL_AREA = None
_QSIZE_POLICY = None
_QVB_LAYOUT = None
_QWIDGET = None
_QT = None
_FIGURE_CANVAS = None
_NAV_TOOLBAR = None
_WHEEL_TO_SCROLL = None


@dataclass(frozen=True)
class VizFields:
    category: str = VIZ_CATEGORY_CHART
    attach: str = VIZ_ATTACH_LIBRARY_TAB


class VizDataSource:
    _FIELDS = frozenset({'game', 'keycount'})

    def __init__(self, replay: dict, *, game: str):
        self._replay = replay
        self._game = str(game)

    def supports(self, field: str) -> bool:
        return str(field) in self._FIELDS

    def game(self) -> str:
        return self._game

    def keycount(self) -> int:
        if 'keycount' in self._replay:
            return int(self._replay['keycount'])
        cols = self._replay.get('columns')
        if cols is None or len(cols) == 0:
            return 4
        return int(cols.max()) + 1

    def combo(self) -> int:
        raise DataNotAvailable('combo')

    def accuracy(self) -> float:
        raise DataNotAvailable('accuracy')

    def judgment_windows(self):
        raise DataNotAvailable('judgment_windows')

    def judgment_counts(self):
        raise DataNotAvailable('judgment_counts')

    def judgment_colors(self):
        raise DataNotAvailable('judgment_colors')

    def judge_label(self) -> str:
        raise DataNotAvailable('judge_label')

    def game_memory(self):
        return None

    def t_now(self) -> float:
        raise DataNotAvailable('t_now')

    def play_rate(self) -> float:
        raise DataNotAvailable('play_rate')

    def paused(self) -> bool:
        raise DataNotAvailable('paused')

    def note_count(self) -> int:
        misses = self._replay.get('misses')
        if misses is None:
            return 0
        return int(len(misses))

    def sv_enabled(self) -> bool:
        raise DataNotAvailable('sv_enabled')

    def sv_suspended(self) -> bool:
        raise DataNotAvailable('sv_suspended')

    def sv_sections(self) -> list:
        raise DataNotAvailable('sv_sections')

    def skin(self) -> str:
        raise DataNotAvailable('skin')

    def press_hide(self) -> bool:
        raise DataNotAvailable('press_hide')

    def scroll_mode(self) -> str:
        raise DataNotAvailable('scroll_mode')

    def scroll_value(self) -> float:
        raise DataNotAvailable('scroll_value')

    def effective_scroll_ms(self) -> float:
        raise DataNotAvailable('effective_scroll_ms')

    def layer_visible(self, layer: str) -> bool:
        raise DataNotAvailable('layer_visible')

    def layer_tree(self):
        raise DataNotAvailable('layer_tree')


class ReplayDictState:
    def __init__(self, replay: dict):
        self._replay = replay

    def _require(self, key: str):
        value = self._replay.get(key)
        if value is None:
            raise DataNotAvailable(f'replay.{key}')
        return value

    def _clean_mask(self):
        misses = self._require('misses')
        return ~misses

    def offsets(self):
        return self._require('offsets')

    def offsets_clean(self):
        return self.offsets()[self._clean_mask()]

    def columns(self):
        return self._require('columns')

    def columns_clean(self):
        return self.columns()[self._clean_mask()]

    def noterows(self):
        noterows = self._replay.get('noterows')
        if noterows is not None:
            return noterows
        return self._require('times')

    def noterows_clean(self):
        return self.noterows()[self._clean_mask()]

    def misses(self):
        return self._require('misses')

    def notetypes(self):
        return self._require('notetypes')

    def keycount(self) -> int:
        if 'keycount' in self._replay:
            return int(self._replay['keycount'])
        cols = self._replay.get('columns')
        if cols is None or len(cols) == 0:
            return 4
        return int(cols.max()) + 1

    def game(self) -> str:
        return str(self._replay.get('game', 'unknown'))


def new_figure(w=10, h=5):
    figure_class = _FIGURE_CLASS
    if figure_class is None:
        _prepare_chart_runtime()
        figure_class = _FIGURE_CLASS
    fig = figure_class(figsize=(w, h))
    fig.patch.set_facecolor('#121212')
    ax = fig.add_subplot(111)
    return fig, ax


def col_colors(keycount: int) -> list[str]:
    import matplotlib

    if keycount <= 1:
        return ['#80deea']
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        'hand',
        [LEFT_CLR, '#ffffff', RIGHT_CLR],
    )
    return [
        matplotlib.colors.to_hex(cmap(i / (keycount - 1)))
        for i in range(keycount)
    ]


class VizContext:
    surface = SURFACE_VIZ

    def __init__(self, component_key: str, replay: dict, *, game: str,
                 on_play=None, entry: dict | None = None):
        self._key = str(component_key)
        self.region = VIZ_ATTACH_LIBRARY_TAB
        self.measure_only = False
        self.data = VizDataSource(replay, game=game)
        self.replay = ReplayDictState(replay)
        self.analysis = PlayerDataAnalysis()
        self.hud_flags = None
        self.config = PluginConfig(self._key)
        self.game = str(game)
        self.entry = entry
        self.on_play = on_play
        self._result = None

    def set_result(self, result) -> None:
        self._result = result

    def figure(self, fig) -> None:
        self._result = fig

    def widget(self, widget) -> None:
        self._result = widget

    def result(self):
        return self._result

    def build_note_visualizer(self, *, od=None, judge=None):
        note_viz_tab = _NOTE_VIZ_TAB
        if note_viz_tab is None:
            _prepare_widget_runtime()
            note_viz_tab = _NOTE_VIZ_TAB

        kwargs = {}
        if od is not None:
            kwargs['od'] = od
        if judge is not None:
            kwargs['judge'] = judge
        widget = note_viz_tab(self.replay._replay, game=self.game,
                              on_play=self.on_play, **kwargs)
        widget._has_play_btn = self.on_play is not None
        return widget

    def build_full_report(self):
        raise DataNotAvailable('build_full_report')

    def build_selectable_figure_widget(self, *,
                                       options: list[tuple[str, str]],
                                       default_selection: list[str],
                                       build_figure):
        return _SelectableFigureWidget(
            options=options,
            default_selection=default_selection,
            build_figure=build_figure,
            on_play=self.on_play,
        ).widget()

    def noterows_to_seconds(self, rows):
        replay = self.replay._replay
        if replay.get('chart_path'):
            return rows.astype(float) / 1000.0
        bpms = replay.get('bpms')
        sm_offset = replay.get('sm_offset', 0.0)
        if bpms is not None:
            from analysis.games.etterna.sm_chart import row_to_time
            import numpy as np

            return np.array([
                row_to_time(int(row), bpms, sm_offset)
                for row in rows
            ])
        return rows.astype(float) / 96.0


class _SelectionDialog:
    def __init__(self, parent, current_selection, options):
        self._dialog = _QDIALOG(parent)
        self._dialog.setWindowTitle('Customize plots')
        layout = _QVB_LAYOUT(self._dialog)
        layout.addWidget(_QLABEL('Plots to include:'))
        self._checks = {}
        for key, label in options:
            checkbox = _QCHECKBOX(label)
            checkbox.setChecked(key in current_selection)
            self._checks[key] = checkbox
            layout.addWidget(checkbox)
        buttons = _QDIALOG_BUTTON_BOX(
            _QDIALOG_BUTTON_BOX.Ok | _QDIALOG_BUTTON_BOX.Cancel
        )
        buttons.accepted.connect(self._dialog.accept)
        buttons.rejected.connect(self._dialog.reject)
        layout.addWidget(buttons)

    def exec(self):
        return self._dialog.exec()

    def selection(self):
        return [key for key, checkbox in self._checks.items() if checkbox.isChecked()]


class _SelectableFigureWidget:
    def __init__(self, *, options, default_selection, build_figure, on_play=None):
        self._widget = _QWIDGET()
        self._options = list(options)
        self._build_figure = build_figure
        self._on_play = on_play
        self._selection = list(default_selection)

        layout = _QVB_LAYOUT(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = _QSCROLL_AREA()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(_QT.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(_QT.ScrollBarAsNeeded)

        self._canvas_host = _QWIDGET()
        self._canvas_host.setSizePolicy(_QSIZE_POLICY.Expanding, _QSIZE_POLICY.Expanding)
        self._canvas_layout = _QVB_LAYOUT(self._canvas_host)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._canvas_host)
        layout.addWidget(self._scroll, 1)
        self._wheel_filter = _WHEEL_TO_SCROLL(self._scroll)

        bottom = _QHBOX_LAYOUT()
        bottom.setContentsMargins(4, 2, 4, 2)
        bottom.addStretch(1)
        customize = _QPUSH_BUTTON('Customize plots…')
        customize.clicked.connect(lambda _checked=False: self._customize())
        bottom.addWidget(customize)
        if on_play is not None:
            play = _QPUSH_BUTTON('▶ Play replay')
            play.clicked.connect(lambda _checked=False: on_play())
            bottom.addWidget(play)
        layout.addLayout(bottom)
        self._widget._has_play_btn = on_play is not None
        self._rebuild()

    def _customize(self):
        dialog = _SelectionDialog(self._widget, self._selection, self._options)
        if dialog.exec() == _QDIALOG_ACCEPTED:
            self._selection = dialog.selection()
            self._rebuild()

    def _rebuild(self):
        for index in reversed(range(self._canvas_layout.count())):
            item = self._canvas_layout.itemAt(index)
            widget = item.widget()
            if widget is None:
                continue
            widget.setParent(None)
            widget.deleteLater()
        fig = self._build_figure(list(self._selection))
        canvas = _FIGURE_CANVAS(fig)
        canvas.setMinimumHeight(int(fig.get_figheight() * fig.get_dpi() * 0.6))
        canvas.installEventFilter(self._wheel_filter)
        self._canvas_layout.addWidget(_NAV_TOOLBAR(canvas, self._widget))
        self._canvas_layout.addWidget(canvas, 1)

    def widget(self):
        return self._widget


def bridge_into_viz_registry(components) -> list[tuple[str, object, str]]:
    out: list[tuple[str, object, str]] = []
    eligible = components.components_for(
        SURFACE_VIZ,
        data_source_fields=VizDataSource._FIELDS,
    )
    for comp in eligible:
        manifest = comp.manifest
        fields = manifest.plugin_fields.get('viz', VizFields())

        def _builder(replay, game='unknown', on_play=None, entry=None,
                     _comp=comp, _fields=fields):
            if _fields.category == VIZ_CATEGORY_CHART:
                _prepare_chart_runtime()
            elif _fields.category == VIZ_CATEGORY_WIDGET:
                _prepare_widget_runtime()
            ctx = VizContext(
                _comp.manifest.key,
                replay,
                game=game,
                on_play=on_play,
                entry=entry,
            )
            _comp.draw(ctx)
            return ctx.result()

        out.append((manifest.name, _builder, fields.category))
    return out


def _prepare_chart_runtime() -> None:
    global _FIGURE_CLASS

    if _FIGURE_CLASS is not None:
        return
    from matplotlib.figure import Figure

    _FIGURE_CLASS = Figure


def _prepare_widget_runtime() -> None:
    global _NOTE_VIZ_TAB
    global _QDIALOG_ACCEPTED
    global _QCHECKBOX
    global _QDIALOG
    global _QDIALOG_BUTTON_BOX
    global _QHBOX_LAYOUT
    global _QLABEL
    global _QPUSH_BUTTON
    global _QSCROLL_AREA
    global _QSIZE_POLICY
    global _QVB_LAYOUT
    global _QWIDGET
    global _QT
    global _FIGURE_CANVAS
    global _NAV_TOOLBAR
    global _WHEEL_TO_SCROLL

    if _NOTE_VIZ_TAB is not None:
        return
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QCheckBox,
        QDialog,
        QDialogButtonBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

    from analysis.gui.note_viz_tab import NoteVizTab
    from analysis.gui.widgets import WheelToScroll

    _NOTE_VIZ_TAB = NoteVizTab
    _QDIALOG_ACCEPTED = QDialog.Accepted
    _QCHECKBOX = QCheckBox
    _QDIALOG = QDialog
    _QDIALOG_BUTTON_BOX = QDialogButtonBox
    _QHBOX_LAYOUT = QHBoxLayout
    _QLABEL = QLabel
    _QPUSH_BUTTON = QPushButton
    _QSCROLL_AREA = QScrollArea
    _QSIZE_POLICY = QSizePolicy
    _QVB_LAYOUT = QVBoxLayout
    _QWIDGET = QWidget
    _QT = Qt
    _FIGURE_CANVAS = FigureCanvas
    _NAV_TOOLBAR = NavToolbar
    _WHEEL_TO_SCROLL = WheelToScroll
