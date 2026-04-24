from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QMessageBox, QProgressDialog, QFileDialog

from analysis.core import game as game_mod
from analysis.core import gui_adapter as gui_mod
from analysis.gui.player_tab import PlayerTab
from analysis.gui.settings import get_settings, load_player_settings
from analysis.gui.widgets import MplTab, HtmlTab, _viz_toolbar
from analysis.gui.loaders import Worker


def _save_html_path(parent, default_name):
    path, _ = QFileDialog.getSaveFileName(
        parent, 'Save HTML', default_name, 'HTML (*.html)'
    )
    return path or None


class LibraryOpeners:
    def __init__(self, tab):
        self.tab = tab

    def play_selected(self):
        entry = self.tab.selected_entry()
        if entry:
            self.open_player_for(entry)

    def html_selected(self):
        entry = self.tab.selected_entry()
        if not entry:
            return

        from analysis.viz.plots import generate_html_report

        replay = self.tab._replay_cache.get(entry)
        out = _save_html_path(self.tab, f'report_{entry["game"]}.html')
        if not out:
            return

        generate_html_report(replay, score_meta=entry, output_path=out)
        title = (entry.get('song') or 'report')[:40]
        self.tab._add_tab(HtmlTab(out), f'📄 {title}')

    def open_player_for(self, entry, rep=None):
        default_ms = self._default_scroll_ms()
        get_settings().setValue('library/default_scroll_ms', default_ms)

        scroll_mode = load_player_settings(entry['game'])['scroll_mode']
        title_song = (entry.get('song') or Path(entry['replay_path']).name)[:40]

        dlg = self._progress_dialog(f'Loading {title_song}…', 'Replay')

        def job(progress):
            progress('parsing replay…')
            replay = rep if rep is not None else self.tab._replay_cache.get(entry)
            progress('resolving chart/audio…')
            bpms, sm_off, audio = gui_mod.get(entry['game']).resolve_chart_context(
                replay, entry=entry, progress=progress
            )
            return replay, bpms, sm_off, audio

        worker = self.tab.jobs.track(Worker(job))
        worker.progress.connect(lambda msg: dlg.setLabelText(f'{title_song}\n{msg}'))
        worker.done.connect(
            lambda payload, w=worker: self._finish_open_player(
                entry, payload, dlg, w, default_ms, scroll_mode
            )
        )
        worker.failed.connect(
            lambda tb, w=worker: self._fail_dialog(
                dlg, w, 'Failed to load replay', tb
            )
        )
        worker.start()

    def open_viz(self, entry, viz_name=None):
        if not entry:
            return

        import analysis.viz.plugins as viz_pkg

        name = viz_name or self.tab.viz_cb.currentText()
        match = next(
            ((vn, builder, category)
             for vn, builder, category in viz_pkg.all_visualizations()
             if vn == name),
            None,
        )
        if match is None:
            QMessageBox.warning(self.tab, 'unknown visualization', name)
            return

        _vn, builder, category = match
        title_song = (entry.get('song') or Path(entry['replay_path']).name)[:40]
        tab_title = f'📊 {name} — {title_song}'
        dlg = self._progress_dialog(f'Processing {title_song}…', name)

        def job(progress):
            progress('parsing replay…')
            replay = self.tab._replay_cache.get(entry)
            if category == 'chart':
                progress(f'rendering {name}…')
                return 'chart', replay, builder(replay, game=entry['game'], entry=entry)
            return 'widget', replay, None

        worker = self.tab.jobs.track(Worker(job))
        worker.progress.connect(lambda msg: dlg.setLabelText(f'{title_song}\n{msg}'))
        worker.done.connect(
            lambda payload, w=worker: self._finish_open_viz(
                entry, payload, dlg, w, builder, tab_title
            )
        )
        worker.failed.connect(
            lambda tb, w=worker: self._fail_dialog(dlg, w, f'Failed to load {name}', tb)
        )
        worker.start()

    def _finish_open_player(self, entry, payload, dlg, worker, default_ms, scroll_mode):
        replay, bpms, sm_off, audio = payload
        self._maybe_backfill_entry(entry, replay)

        rate = float(entry.get('rate') or 1.0)
        extra = game_mod.get(entry['game']).player_tab_kwargs(
            replay, entry, (bpms, sm_off, audio)
        )
        tab = PlayerTab(
            replay,
            game=entry['game'],
            audio_path=audio,
            scroll_ms=default_ms,
            scroll_mode=scroll_mode,
            play_rate=rate,
            **extra,
        )
        title = (entry.get('song') or 'play')[:40]
        self.tab._add_tab(tab, f'▶ {title}')
        self._close_worker_dialog(dlg, worker)

    def _finish_open_viz(self, entry, payload, dlg, worker, builder, tab_title):
        kind, replay, prebuilt = payload
        self._maybe_backfill_entry(entry, replay)

        on_play = lambda e=entry, r=replay: self.open_player_for(e, rep=r)

        if kind == 'chart':
            widget = MplTab(prebuilt, on_play=on_play)
        else:
            try:
                result = builder(
                    replay,
                    game=entry['game'],
                    on_play=on_play,
                    entry=entry,
                )
            except TypeError:
                result = builder(replay, game=entry['game'], entry=entry)

            if isinstance(result, QWidget) and not getattr(result, '_has_play_btn', False):
                wrapper = QWidget()
                layout = QVBoxLayout(wrapper)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(result, 1)
                layout.addLayout(_viz_toolbar(on_play))
                widget = wrapper
            else:
                widget = result

        self.tab._add_tab(widget, tab_title)
        self._close_worker_dialog(dlg, worker)

    def _maybe_backfill_entry(self, entry, replay):
        if replay.get('chart_path') and not entry.get('chart_path'):
            entry['chart_path'] = replay['chart_path']

        if gui_mod.get(entry['game']).enrich_entry(entry):
            self._persist_library()
            self.tab.refresh_tree()

    def _persist_library(self):
        for adapter in game_mod.all_games().values():
            try:
                adapter.save_cached(self.tab.library)
            except Exception:
                pass

    def _default_scroll_ms(self) -> float:
        try:
            return float(self.tab.default_scroll_edit.text().strip() or 400)
        except ValueError:
            return 400.0

    def _progress_dialog(self, text, title):
        dlg = QProgressDialog(text, None, 0, 0, self.tab)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()
        return dlg

    def _close_worker_dialog(self, dlg, worker):
        dlg.close()
        dlg.deleteLater()
        self.tab.jobs.untrack(worker)

    def _fail_dialog(self, dlg, worker, title, text):
        self._close_worker_dialog(dlg, worker)
        QMessageBox.warning(self.tab, title, text)