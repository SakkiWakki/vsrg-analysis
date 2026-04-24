from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QComboBox, QCheckBox, QTreeWidget, QHeaderView, QToolButton, QMenu,
)

from analysis.gui.settings import get_settings
from analysis.gui.replay_cache import ReplayCache
from analysis.gui.library.model import LibraryQuery
from analysis.gui.library.tree import LibraryTreeController
from analysis.gui.library.jobs import LibraryJobRunner
from analysis.gui.library.openers import LibraryOpeners
from analysis.gui.library.plugin_actions import PluginActionsController
from analysis.gui.library.context_menu import LibraryContextMenu
from analysis.gui.plugins_dialog import PluginsDialog
from analysis.gui.game_settings_dialog import GameSettingsDialog
from analysis.gui.paths_dialog import PathsDialog



class LibraryTab(QWidget):
    def __init__(self, add_tab):
        super().__init__()
        self._add_tab = add_tab
        self.library = []
        self._replay_cache = ReplayCache(max_size=16)

        self.query = LibraryQuery(self)
        self.jobs = LibraryJobRunner(self)
        self.openers = LibraryOpeners(self)
        self.context_menu = LibraryContextMenu(self)
        self.plugin_actions = PluginActionsController(self)

        self._build_ui()
        self.tree_ctl = LibraryTreeController(self.tree, self)

        self._load_saved_settings()
        self.plugin_actions.rebuild()

        QTimer.singleShot(200, self.load_library)

    # ---------- public lifecycle ----------

    def active_workers(self):
        return self.jobs.active_workers()

    def persist_settings(self):
        s = get_settings()
        s.setValue('library/game', self.game_cb.currentText())
        s.setValue('library/sort', self.sort_cb.currentText())
        s.setValue('library/desc', self.desc_cbx.isChecked())
        s.setValue('library/group', self.group_cbx.isChecked())
        s.setValue('library/keys', self.keys_cb.currentText())
        s.setValue('library/min_wife', self.min_wife_edit.text())
        s.setValue('library/viz', self.viz_cb.currentText())
        s.setValue(
            'library/default_scroll_ms',
            self.default_scroll_edit.text().strip() or '400',
        )

    # ---------- UI ----------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.addLayout(self._build_top_bar())
        root.addLayout(self._build_filter_bar())

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([
            'game', 'K', 'song', 'pack', 'diff',
            'rate', 'acc%', 'grade', 'date',
        ])
        self._col_sort_keys = [
            'game', 'keys', 'song', 'pack', 'steps',
            'rate', 'wife', 'grade', 'date',
        ]
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(False)
        self.tree.itemDoubleClicked.connect(
            lambda *_: self.openers.open_viz(self.selected_entry())
        )
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.context_menu.open)
        header = self.tree.header()
        for i, w in enumerate([70, 40, 420, 180, 80, 55, 70, 60, 140]):
            self.tree.setColumnWidth(i, w)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self._on_header_clicked)
        root.addWidget(self.tree, 1)

        root.addLayout(self._build_action_bar())

    def _build_top_bar(self):
        row = QHBoxLayout()

        row.addWidget(QLabel('Search:'))
        self.filter_edit = QLineEdit()
        self.filter_edit.textChanged.connect(self.refresh_tree)
        row.addWidget(self.filter_edit, 1)

        scan_btn = QPushButton('Check for new replays')
        scan_btn.clicked.connect(self.load_library)
        row.addWidget(scan_btn)

        for label, cb in [
            ('Games…', self.open_game_settings_dialog),
            ('Plugins…', self.open_plugins_dialog),
            ('Paths…', self.open_paths_dialog),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(cb)
            row.addWidget(btn)

        self._plugin_actions_btn = QToolButton()
        self._plugin_actions_btn.setText('Plugin actions')
        self._plugin_actions_btn.setPopupMode(QToolButton.InstantPopup)
        self._plugin_actions_menu = QMenu(self._plugin_actions_btn)
        self._plugin_actions_btn.setMenu(self._plugin_actions_menu)
        self._plugin_actions_btn.setVisible(False)
        row.addWidget(self._plugin_actions_btn)

        return row

    def _build_filter_bar(self):
        row = QHBoxLayout()

        row.addWidget(QLabel('Game:'))
        self.game_cb = QComboBox()
        self.game_cb.addItems(['all', 'etterna', 'osu'])
        row.addWidget(self.game_cb)

        row.addWidget(QLabel('Sort:'))
        self.sort_cb = QComboBox()
        self.sort_cb.addItems([
            'recent', 'date', 'wife', 'song', 'pack', 'rate',
            'keys', 'game', 'grade', 'maxcombo', 'overall_ssr',
        ])
        row.addWidget(self.sort_cb)

        self.desc_cbx = QCheckBox('desc')
        self.desc_cbx.setChecked(True)
        row.addWidget(self.desc_cbx)

        self.group_cbx = QCheckBox('group by song')
        self.group_cbx.setChecked(True)
        row.addWidget(self.group_cbx)

        row.addWidget(QLabel('K:'))
        self.keys_cb = QComboBox()
        self.keys_cb.setEditable(True)
        self.keys_cb.addItems(['any', *(str(k) for k in range(4, 11))])
        row.addWidget(self.keys_cb)

        row.addWidget(QLabel('min acc%:'))
        self.min_wife_edit = QLineEdit('0')
        self.min_wife_edit.setMaximumWidth(60)
        self.min_wife_edit.textChanged.connect(self.refresh_tree)
        row.addWidget(self.min_wife_edit)

        row.addStretch(1)
        return row
    
    def _run_selected(self, action: str) -> None:
        entry = self.selected_entry()
        if entry:
            self.openers.run(action, entry)

    def _build_action_bar(self):
        import analysis.viz.plugins as viz_pkg

        row = QHBoxLayout()
        row.addWidget(QLabel('Visualization:'))

        self.viz_cb = QComboBox()
        names = [name for name, _, _ in viz_pkg.all_visualizations()]
        self.viz_cb.addItems(names)
        if 'Full report (all plots)' in names:
            self.viz_cb.setCurrentText('Full report (all plots)')
        row.addWidget(self.viz_cb, 1)

        for label, action in [
            ('Open', 'visualize'),
            ('HTML report', 'html_report'),
            ('▶ Play replay', 'play'),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _checked=False, action=action: self._run_selected(action)
            )
            row.addWidget(btn)

        row.addWidget(QLabel('Default scroll (ms):'))
        self.default_scroll_edit = QLineEdit('400')
        self.default_scroll_edit.setMaximumWidth(70)
        row.addWidget(self.default_scroll_edit)

        row.addStretch(1)
        self.status_lbl = QLabel('')
        row.addWidget(self.status_lbl)
        return row

    def _load_saved_settings(self):
        s = get_settings()
        self.game_cb.setCurrentText(s.value('library/game', 'all'))
        self.sort_cb.setCurrentText(s.value('library/sort', 'recent'))
        self.desc_cbx.setChecked(s.value('library/desc', True, type=bool))
        self.group_cbx.setChecked(s.value('library/group', True, type=bool))
        self.keys_cb.setCurrentText(s.value('library/keys', 'any'))
        self.min_wife_edit.setText(s.value('library/min_wife', '0'))
        self.default_scroll_edit.setText(
            s.value('library/default_scroll_ms', '400')
        )
        saved_viz = s.value('library/viz')
        if saved_viz and self.viz_cb.findText(saved_viz) >= 0:
            self.viz_cb.setCurrentText(saved_viz)

    # ---------- high-level commands ----------

    def load_library(self, refresh=False):
        self.jobs.load_library(refresh=refresh)

    def on_library_loaded(self, lib):
        self.library = lib
        self.query.set_library(lib)
        self.refresh_tree()

    def refresh_tree(self, *_):
        self.tree_ctl.render(self.query.rows())

    def selected_entry(self):
        return self.tree_ctl.selected_entry(parent=self)

    def open_plugins_dialog(self):
        PluginsDialog(self).exec()

    def open_paths_dialog(self):
        dlg = PathsDialog(self)
        if dlg.exec():
            self.load_library(refresh=True)

    def open_game_settings_dialog(self):
        dlg = GameSettingsDialog(self, on_rebuild=self.jobs.rebuild_game)
        dlg.exec()
        self.load_library()

    def _on_header_clicked(self, col_idx):
        if col_idx < 0 or col_idx >= len(self._col_sort_keys):
            return
        key = self._col_sort_keys[col_idx]
        if self.sort_cb.currentText() == key:
            self.desc_cbx.setChecked(not self.desc_cbx.isChecked())
        else:
            self.sort_cb.setCurrentText(key)