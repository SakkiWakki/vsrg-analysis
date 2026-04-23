"""Library tab: score tree + filters + context menu + open-viz / open-player
flows. Built tabs are handed back to the host via an add_tab callback so this
module doesn't need a reference to the MainWindow."""
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QLineEdit, QComboBox,
                               QCheckBox, QTreeWidget, QTreeWidgetItem,
                               QMessageBox, QFileDialog, QHeaderView, QMenu,
                               QProgressDialog, QToolButton)

from analysis.core import game as game_mod
from analysis.core import gui_adapter as gui_mod
from analysis.gui.settings import get_settings, load_player_settings
from analysis.gui.loaders import Worker
from analysis.gui.replay_cache import ReplayCache
from analysis.gui.widgets import MplTab, HtmlTab, _viz_toolbar
from analysis.gui.player_tab import PlayerTab
import analysis.viz.plugins as _viz_pkg


def _all_viz():
    return _viz_pkg.all_visualizations()


def _filedialog_save_html(parent, default_name):
    p, _ = QFileDialog.getSaveFileName(parent, 'Save HTML', default_name,
                                        'HTML (*.html)')
    return p or None


class LibraryTab(QWidget):
    def __init__(self, add_tab):
        """add_tab(widget, title, closable=True) — callback to add a new tab
        to the host QTabWidget."""
        super().__init__()
        self._add_tab = add_tab
        self.library = []
        self._replay_cache = ReplayCache(max_size=16)
        self._scan_worker = None
        self._viz_workers = []

        self._build_ui()
        QTimer.singleShot(200, self._load_library)

    # ---------- workers ----------
    def active_workers(self):
        ws = [self._scan_worker] + list(self._viz_workers)
        return [w for w in ws if w is not None]

    # ---------- persistence ----------
    def persist_settings(self):
        s = get_settings()
        s.setValue('library/game', self.game_cb.currentText())
        s.setValue('library/sort', self.sort_cb.currentText())
        s.setValue('library/desc', self.desc_cbx.isChecked())
        s.setValue('library/group', self.group_cbx.isChecked())
        s.setValue('library/keys', self.keys_cb.currentText())
        s.setValue('library/min_wife', self.min_wife_edit.text())
        s.setValue('library/viz', self.viz_cb.currentText())
        s.setValue('library/default_scroll_ms',
                   self.default_scroll_edit.text().strip() or '400')

    # ---------- UI ----------
    def _build_ui(self):
        v = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel('Search:'))
        self.filter_edit = QLineEdit()
        self.filter_edit.textChanged.connect(self._refresh_tree)
        row1.addWidget(self.filter_edit, 1)
        # Built-in toolbar actions first. Plugin-contributed actions live
        # behind one menu button so discovery does not keep appending
        # extra top-level toolbar controls.
        b = QPushButton('Check for new replays')
        b.clicked.connect(self._load_library)
        row1.addWidget(b)
        for label, cb in [('Games…', self._open_game_settings_dialog),
                          ('Plugins…', self._open_plugins_dialog),
                          ('Paths…', self._open_paths_dialog)]:
            b = QPushButton(label); b.clicked.connect(cb); row1.addWidget(b)
        self._plugin_actions_btn = QToolButton()
        self._plugin_actions_btn.setText('Plugin actions')
        self._plugin_actions_btn.setPopupMode(QToolButton.InstantPopup)
        self._plugin_actions_menu = QMenu(self._plugin_actions_btn)
        self._plugin_actions_btn.setMenu(self._plugin_actions_menu)
        self._plugin_actions_btn.setVisible(False)
        row1.addWidget(self._plugin_actions_btn)
        # Discover plugins up-front so bundle-contributed library-toolbar
        # buttons appear on first paint.
        from analysis.player.plugin_loader import PluginManager
        PluginManager.discover()
        from analysis.gui.library_actions import get_registry
        self._rebuild_plugin_actions()
        self._library_actions_unsub = get_registry().subscribe(
            self._rebuild_plugin_actions)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel('Game:'))
        self.game_cb = QComboBox(); self.game_cb.addItems(['all', 'etterna', 'osu'])
        self.game_cb.currentTextChanged.connect(self._refresh_tree)
        row2.addWidget(self.game_cb)
        row2.addWidget(QLabel('Sort:'))
        self.sort_cb = QComboBox()
        self.sort_cb.addItems(['recent', 'date', 'wife', 'song', 'pack', 'rate',
                                'keys', 'game', 'grade', 'maxcombo', 'overall_ssr'])
        self.sort_cb.currentTextChanged.connect(self._refresh_tree)
        row2.addWidget(self.sort_cb)
        self.desc_cbx = QCheckBox('desc'); self.desc_cbx.setChecked(True)
        self.desc_cbx.stateChanged.connect(self._refresh_tree)
        row2.addWidget(self.desc_cbx)
        self.group_cbx = QCheckBox('group by song'); self.group_cbx.setChecked(True)
        self.group_cbx.stateChanged.connect(self._refresh_tree)
        row2.addWidget(self.group_cbx)
        row2.addWidget(QLabel('K:'))
        self.keys_cb = QComboBox()
        self.keys_cb.addItems(['any', '4', '5', '6', '7', '8', '9', '10'])
        self.keys_cb.currentTextChanged.connect(self._refresh_tree)
        row2.addWidget(self.keys_cb)
        row2.addWidget(QLabel('min acc%:'))
        self.min_wife_edit = QLineEdit('0')
        self.min_wife_edit.setMaximumWidth(60)
        self.min_wife_edit.textChanged.connect(self._refresh_tree)
        row2.addWidget(self.min_wife_edit)
        row2.addStretch(1)
        v.addLayout(row2)

        cols = ['game', 'K', 'song', 'pack', 'diff', 'rate', 'acc%',
                'grade', 'date']
        self._col_sort_keys = ['game', 'keys', 'song', 'pack', 'steps',
                               'rate', 'wife', 'grade', 'date']
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(cols)
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(False)
        self.tree.itemDoubleClicked.connect(lambda *_: self._open_viz(self._selected_entry()))
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_context_menu)
        hdr = self.tree.header()
        for i, w in enumerate([70, 40, 420, 180, 80, 55, 70, 60, 140]):
            self.tree.setColumnWidth(i, w)
        hdr.setSectionResizeMode(2, QHeaderView.Interactive)
        hdr.setSectionsClickable(True)
        hdr.sectionClicked.connect(self._on_header_clicked)
        v.addWidget(self.tree, 1)

        acts = QHBoxLayout()
        acts.addWidget(QLabel('Visualization:'))
        self.viz_cb = QComboBox()
        names = [n for n, _, _ in _all_viz()]
        for name in names:
            self.viz_cb.addItem(name)
        if 'Full report (all plots)' in names:
            self.viz_cb.setCurrentText('Full report (all plots)')
        acts.addWidget(self.viz_cb, 1)
        for label, cb in [('Open', lambda: self._open_viz(self._selected_entry())),
                          ('HTML report', self._html_selected),
                          ('▶ Play replay', self._play_selected)]:
            b = QPushButton(label); b.clicked.connect(cb); acts.addWidget(b)

        acts.addWidget(QLabel('Default scroll (ms):'))
        self.default_scroll_edit = QLineEdit('400')
        self.default_scroll_edit.setMaximumWidth(70)
        self.default_scroll_edit.setToolTip(
            'Default scroll speed (ms from screen top to judgment line) for new player tabs')
        acts.addWidget(self.default_scroll_edit)

        acts.addStretch(1)
        self.status_lbl = QLabel('')
        acts.addWidget(self.status_lbl)
        v.addLayout(acts)

        s = get_settings()
        self.game_cb.setCurrentText(s.value('library/game', 'all'))
        self.sort_cb.setCurrentText(s.value('library/sort', 'recent'))
        self.desc_cbx.setChecked(s.value('library/desc', True, type=bool))
        self.group_cbx.setChecked(s.value('library/group', True, type=bool))
        self.keys_cb.setCurrentText(s.value('library/keys', 'any'))
        self.min_wife_edit.setText(s.value('library/min_wife', '0'))
        saved_viz = s.value('library/viz')
        if saved_viz and self.viz_cb.findText(saved_viz) >= 0:
            self.viz_cb.setCurrentText(saved_viz)
        self.default_scroll_edit.setText(s.value('library/default_scroll_ms', '400'))

    # ---------- plugins dialog ----------
    def _open_plugins_dialog(self):
        from analysis.gui.plugins_dialog import PluginsDialog
        PluginsDialog(self).exec()

    # ---------- plugin toolbar actions ----------
    def _rebuild_plugin_actions(self):
        """Rebuild the plugin-contributed action menu.

        Called at build time and any time the registry changes. We
        rebuild rather than diff because the set is small and registration
        order is the intended visual order.
        """
        menu = getattr(self, '_plugin_actions_menu', None)
        btn = getattr(self, '_plugin_actions_btn', None)
        if menu is None or btn is None:
            return
        menu.clear()
        from analysis.gui.library_actions import get_registry
        actions = get_registry().actions()
        for action in actions:
            # Capture the callback in the closure; the action object may
            # be replaced if the plugin re-registers with the same key.
            cb = action.callback
            item = menu.addAction(action.label)
            item.triggered.connect(
                lambda _checked=False, fn=cb: self._invoke_plugin_action(fn))
        btn.setVisible(bool(actions))

    def _invoke_plugin_action(self, fn):
        try:
            fn()
        except Exception as exc:
            QMessageBox.warning(
                self, 'Plugin action failed',
                f'A plugin-contributed toolbar action raised an error:\n{exc}')

    # ---------- paths dialog ----------
    def _open_paths_dialog(self):
        from analysis.gui.paths_dialog import PathsDialog
        dlg = PathsDialog(self)
        if dlg.exec():
            # Rescan against the new library root. Using refresh=True so the
            # cache is rebuilt against the updated paths.
            self._load_library(refresh=True)

    # ---------- library scan ----------
    def _load_library(self, refresh=False):
        """Incremental library load. `refresh=True` forces a rebuild of
        every enabled game; per-game rebuilds go through `_rebuild_game`
        on the Refresh-games dropdown."""
        # Guard against re-entry: QThread aborts the process if a running
        # thread's Python wrapper is garbage-collected. Before we overwrite
        # `self._scan_worker`, make sure the old one has finished.
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self.status_lbl.setText('already scanning — please wait')
            return
        from analysis.core.search import build_library
        self.status_lbl.setText('scanning…')

        def job(progress):
            return build_library(refresh=refresh, progress=progress)

        self._scan_worker = Worker(job)
        self._scan_worker.progress.connect(self.status_lbl.setText)
        self._scan_worker.done.connect(self._on_library_loaded)
        self._scan_worker.failed.connect(lambda e: QMessageBox.critical(self, 'scan failed', e))
        self._scan_worker.start()

    def _open_game_settings_dialog(self):
        from analysis.gui.game_settings_dialog import GameSettingsDialog
        dlg = GameSettingsDialog(self, on_rebuild=self._rebuild_game)
        dlg.exec()
        # The dialog may have toggled `library.enabled_games` — reload
        # so entries from newly-disabled games drop out (and newly-enabled
        # games are pulled in from their caches).
        self._load_library()

    def _rebuild_game(self, name):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self.status_lbl.setText('already scanning — please wait')
            return
        self.status_lbl.setText(f'rebuilding {name}…')

        def job(progress):
            from analysis.core.search import build_library
            adapter = game_mod.get(name)
            adapter.rebuild(progress=progress)
            # Reload the full library from every adapter's cache so the
            # tree shows the fresh entries plus whatever the other games
            # have cached.
            return build_library(progress=progress)

        self._scan_worker = Worker(job)
        self._scan_worker.progress.connect(self.status_lbl.setText)
        self._scan_worker.done.connect(self._on_library_loaded)
        self._scan_worker.failed.connect(
            lambda e: QMessageBox.critical(self, 'rebuild failed', e))
        self._scan_worker.start()

    def _on_library_loaded(self, lib):
        self.library = lib
        from collections import Counter
        counts = Counter(e['game'] for e in lib)
        parts = [f'{counts.get(name, 0)} {adapter.label.split()[0]}'
                 for name, adapter in gui_mod.all_games().items()]
        self.status_lbl.setText(f'{len(lib)} entries ({", ".join(parts)})')
        self._refresh_tree()

    # ---------- context menu ----------
    def _tree_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        entry = item.data(0, Qt.UserRole)
        if not entry:
            return
        m = QMenu(self.tree)
        a_play = m.addAction('▶ Watch replay')
        a_viz = m.addAction('Analyze (open visualization)')
        m.addSeparator()
        a_html = m.addAction('HTML report')
        m.addSeparator()
        a_copy = m.addAction('Copy replay path')
        a_copy_chart = m.addAction('Copy chart path') if entry.get('chart_path') else None
        a_open_folder = m.addAction('Open containing folder')
        chosen = m.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is a_play:
            self._open_player_for(entry)
        elif chosen is a_viz:
            self._open_viz(entry)
        elif chosen is a_html:
            self._html_selected()
        elif chosen is a_copy:
            QApplication.clipboard().setText(entry.get('replay_path', ''))
        elif a_copy_chart is not None and chosen is a_copy_chart:
            QApplication.clipboard().setText(entry.get('chart_path', ''))
        elif chosen is a_open_folder:
            target = entry.get('chart_path') or entry.get('replay_path', '')
            if not target:
                return
            folder = str(Path(target).parent)
            if sys.platform.startswith('linux'):
                subprocess.Popen(['xdg-open', folder])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            elif sys.platform.startswith('win'):
                subprocess.Popen(['explorer', folder])

    def _on_header_clicked(self, col_idx):
        if col_idx < 0 or col_idx >= len(self._col_sort_keys):
            return
        key = self._col_sort_keys[col_idx]
        if self.sort_cb.currentText() == key:
            self.desc_cbx.setChecked(not self.desc_cbx.isChecked())
        else:
            self.sort_cb.setCurrentText(key)

    def _refresh_tree(self, *_):
        from analysis.core.search import search
        self.tree.clear()
        if not self.library:
            return
        try:
            min_w = float(self.min_wife_edit.text()) / 100.0
        except ValueError:
            min_w = 0
        game = self.game_cb.currentText()
        res = search(self.library,
                     query=self.filter_edit.text() or None,
                     game=None if game == 'all' else game,
                     min_wife=min_w,
                     sort=self.sort_cb.currentText(),
                     descending=self.desc_cbx.isChecked(),
                     limit=None)
        kv = self.keys_cb.currentText()
        if kv != 'any':
            try:
                k = int(kv)
                res = [e for e in res if e.get('keycount') == k]
            except ValueError:
                pass

        def vals(e):
            return [e['game'],
                    str(e.get('keycount') or '?'),
                    (e.get('song') or '')[:200],
                    (e.get('pack') or '')[:80],
                    e.get('steps', '') or '',
                    f"{e.get('rate', 1):.2f}",
                    f"{e.get('wife', 0) * 100:.2f}",
                    e.get('grade', '') or '',
                    (e.get('datetime') or '')[:19]]

        max_rows = 5000
        if not self.group_cbx.isChecked():
            for e in res[:max_rows]:
                it = QTreeWidgetItem(vals(e))
                it.setData(0, Qt.UserRole, e)
                self.tree.addTopLevelItem(it)
            return

        groups, order = {}, []
        for e in res:
            k = (e['game'], (e.get('song') or '').strip())
            if k not in groups:
                groups[k] = []; order.append(k)
            groups[k].append(e)

        shown = 0
        for k in order:
            if shown >= max_rows:
                break
            children = groups[k]
            if len(children) == 1:
                e = children[0]
                it = QTreeWidgetItem(vals(e))
                it.setData(0, Qt.UserRole, e)
                self.tree.addTopLevelItem(it)
                shown += 1
                continue
            best = max(children, key=lambda x: x.get('wife', 0))
            pv = vals(best)
            pv[2] = f'[{len(children)}]  {pv[2]}'
            pit = QTreeWidgetItem(pv)
            pit.setData(0, Qt.UserRole, best)
            self.tree.addTopLevelItem(pit)
            for e in children:
                cit = QTreeWidgetItem(vals(e))
                cit.setData(0, Qt.UserRole, e)
                pit.addChild(cit)
            shown += 1 + len(children)

    def _selected_entry(self):
        items = self.tree.selectedItems()
        if not items:
            QMessageBox.warning(self, 'no selection', 'pick a score')
            return None
        return items[0].data(0, Qt.UserRole)

    # ---------- library-level mutations (main thread only) ----------
    def _maybe_backfill_entry(self, entry, rep):
        # Propagate chart_path from the parsed replay so the adapter has
        # something to read from; the adapter itself decides whether any
        # field needs filling in.
        if rep.get('chart_path') and not entry.get('chart_path'):
            entry['chart_path'] = rep['chart_path']
        if gui_mod.get(entry['game']).enrich_entry(entry):
            self._persist_library()
            self._refresh_tree()

    def _persist_library(self):
        # Library is split across per-adapter caches. Ask every registered
        # adapter to persist its own subset; each adapter filters
        # `self.library` for the game it owns.
        for adapter in game_mod.all_games().values():
            try:
                adapter.save_cached(self.library)
            except Exception:
                pass

    # ---------- open player / viz / html ----------
    def _open_player_for(self, entry, rep=None):
        try:
            default_ms = float(self.default_scroll_edit.text().strip() or 400)
        except ValueError:
            default_ms = 400.0
        get_settings().setValue('library/default_scroll_ms', default_ms)
        # Scroll mode is validated against the replay's game inside
        # load_player_settings, so a saved 'cmod' under an osu replay is
        # transparently replaced with that game's default here.
        scroll_mode = load_player_settings(entry['game'])['scroll_mode']

        title_song = (entry.get('song') or Path(entry['replay_path']).name)[:40]
        dlg = QProgressDialog(f'Loading {title_song}…', None, 0, 0, self)
        dlg.setWindowTitle('Replay')
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()

        def job(progress):
            progress('parsing replay…')
            r = rep if rep is not None else self._replay_cache.get(entry)
            progress('resolving chart/audio…')
            bpms, sm_off, audio = gui_mod.get(entry['game']).resolve_chart_context(
                r, entry=entry, progress=progress)
            return (r, bpms, sm_off, audio)

        worker = Worker(job)
        self._viz_workers.append(worker)

        def on_progress(msg):
            dlg.setLabelText(f'{title_song}\n{msg}')

        def _close_dlg():
            dlg.close()
            dlg.deleteLater()

        def on_done(payload):
            from analysis.core import game as game_mod
            r, bpms, sm_off, audio = payload
            self._maybe_backfill_entry(entry, r)
            # Etterna stores the score's music rate in Etterna.xml; osu! stores
            # rate-like mods on the replay itself (handled in osu_replay.py).
            # Pass it through so the audio engine resamples to match the rate
            # the replay was actually played at — otherwise the audio plays at
            # 1.0x against a chart that was originally at e.g. 1.15x.
            rate = float(entry.get('rate') or 1.0)
            extra = game_mod.get(entry['game']).player_tab_kwargs(
                r, entry, (bpms, sm_off, audio))
            tab = PlayerTab(r, game=entry['game'], audio_path=audio,
                            scroll_ms=default_ms, scroll_mode=scroll_mode,
                            play_rate=rate, **extra)
            title = (entry.get('song') or 'play')[:40]
            self._add_tab(tab, f'▶ {title}')
            if worker in self._viz_workers:
                self._viz_workers.remove(worker)
            _close_dlg()

        def on_failed(tb):
            _close_dlg()
            QMessageBox.warning(self, 'Failed to load replay', tb)
            if worker in self._viz_workers:
                self._viz_workers.remove(worker)

        worker.progress.connect(on_progress)
        worker.done.connect(on_done)
        worker.failed.connect(on_failed)
        worker.start()

    def _html_selected(self):
        e = self._selected_entry()
        if not e:
            return
        from analysis.viz.plots import generate_html_report
        rep = self._replay_cache.get(e)
        out = _filedialog_save_html(self, f'report_{e["game"]}.html')
        if not out:
            return
        generate_html_report(rep, score_meta=e, output_path=out)
        title = (e.get('song') or 'report')[:40]
        self._add_tab(HtmlTab(out), f'📄 {title}')

    def _play_selected(self):
        e = self._selected_entry()
        if not e:
            return
        self._open_player_for(e)

    def _open_viz(self, entry, viz_name=None):
        if not entry:
            return
        name = viz_name or self.viz_cb.currentText()
        match = next(((vn, builder, category)
                      for vn, builder, category in _all_viz() if vn == name),
                     None)
        if match is None:
            QMessageBox.warning(self, 'unknown visualization', name)
            return
        _vn, builder, category = match

        title_song = (entry.get('song') or Path(entry['replay_path']).name)[:40]
        tab_title = f'📊 {name} — {title_song}'

        dlg = QProgressDialog(f'Processing {title_song}…', None, 0, 0, self)
        dlg.setWindowTitle(name)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()

        def job(progress):
            progress('parsing replay…')
            rep = self._replay_cache.get(entry)
            if category == 'chart':
                progress(f'rendering {name}…')
                result = builder(rep, game=entry['game'], entry=entry)
                return ('chart', rep, result)
            return ('widget', rep, None)

        worker = Worker(job)
        self._viz_workers.append(worker)

        def on_progress(msg):
            dlg.setLabelText(f'{title_song}\n{msg}')

        def _close_dlg():
            dlg.close()
            dlg.deleteLater()

        def on_done(payload):
            kind, rep, prebuilt = payload
            self._maybe_backfill_entry(entry, rep)
            on_play = lambda e=entry, r=rep: self._open_player_for(e, rep=r)
            if kind == 'chart':
                real = MplTab(prebuilt, on_play=on_play)
            else:
                try:
                    result = builder(rep, game=entry['game'], on_play=on_play, entry=entry)
                except TypeError:
                    result = builder(rep, game=entry['game'], entry=entry)
                if isinstance(result, QWidget) and not getattr(result, '_has_play_btn', False):
                    wrapper = QWidget()
                    wl = QVBoxLayout(wrapper)
                    wl.setContentsMargins(0, 0, 0, 0)
                    wl.addWidget(result, 1)
                    wl.addLayout(_viz_toolbar(on_play))
                    real = wrapper
                else:
                    real = result
            self._add_tab(real, tab_title)
            if worker in self._viz_workers:
                self._viz_workers.remove(worker)
            _close_dlg()

        def on_failed(tb):
            _close_dlg()
            QMessageBox.warning(self, f'Failed to load {name}', tb)
            if worker in self._viz_workers:
                self._viz_workers.remove(worker)

        worker.progress.connect(on_progress)
        worker.done.connect(on_done)
        worker.failed.connect(on_failed)
        worker.start()
