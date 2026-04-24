from __future__ import annotations

from collections import Counter

from PySide6.QtWidgets import QMessageBox

from analysis.core import game as game_mod
from analysis.core import gui_adapter as gui_mod
from analysis.gui.loaders import Worker


class LibraryJobRunner:
    def __init__(self, tab):
        self.tab = tab
        self._scan_worker = None
        self._workers: list[Worker] = []

    def active_workers(self):
        workers = [self._scan_worker] + list(self._workers)
        return [w for w in workers if w is not None]

    def track(self, worker: Worker) -> Worker:
        self._workers.append(worker)
        return worker

    def untrack(self, worker: Worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def load_library(self, *, refresh=False) -> None:
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self.tab.status_lbl.setText('already scanning — please wait')
            return

        from analysis.core.search import build_library

        self.tab.status_lbl.setText('scanning…')

        def job(progress):
            return build_library(refresh=refresh, progress=progress)

        worker = Worker(job)
        self._scan_worker = worker
        worker.progress.connect(self.tab.status_lbl.setText)
        worker.done.connect(self._on_library_loaded)
        worker.failed.connect(
            lambda e: QMessageBox.critical(self.tab, 'scan failed', e)
        )
        worker.start()

    def rebuild_game(self, name: str) -> None:
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self.tab.status_lbl.setText('already scanning — please wait')
            return

        self.tab.status_lbl.setText(f'rebuilding {name}…')

        def job(progress):
            from analysis.core.search import build_library

            adapter = game_mod.get(name)
            adapter.rebuild(progress=progress)
            return build_library(progress=progress)

        worker = Worker(job)
        self._scan_worker = worker
        worker.progress.connect(self.tab.status_lbl.setText)
        worker.done.connect(self._on_library_loaded)
        worker.failed.connect(
            lambda e: QMessageBox.critical(self.tab, 'rebuild failed', e)
        )
        worker.start()

    def _on_library_loaded(self, library):
        self.tab.on_library_loaded(library)

        counts = Counter(e['game'] for e in library)
        parts = [
            f'{counts.get(name, 0)} {adapter.label.split()[0]}'
            for name, adapter in gui_mod.all_games().items()
        ]
        self.tab.status_lbl.setText(
            f'{len(library)} entries ({", ".join(parts)})'
        )