from __future__ import annotations

from collections import Counter
from typing import Callable, Any

from PySide6.QtWidgets import QMessageBox

from analysis.core import game as game_mod
from analysis.core import manifest as manifest_mod
from analysis.gui.loaders import Worker
from analysis.gui.loading_dialog import LoadingDialog

from analysis.core.search import build_library

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

    def run_dialog_job(
        self,
        *,
        title: str,
        label: str,
        job: Callable[[Callable[[str], None]], Any],
        on_done: Callable[[Any], None],
        error_title: str,
        label_prefix: str | None = None,
    ) -> Worker:
        dlg = LoadingDialog(label, title, self.tab,
                            label_prefix=label_prefix)
        worker = self.track(Worker(job))
        dlg.attach(worker)
        dlg.show()

        worker.done.connect(
            lambda payload, w=worker, d=dlg: self._finish_dialog_job(
                d, w, on_done, payload
            )
        )
        worker.failed.connect(
            lambda tb, w=worker, d=dlg: self._fail_dialog_job(
                d, w, error_title, tb
            )
        )
        # A cancelled worker emits neither done nor failed; `finished`
        # (the QThread signal) fires regardless, so tracking releases
        # once the thread actually exits.
        worker.finished.connect(lambda w=worker: self.untrack(w))
        worker.start()
        return worker

    def _finish_dialog_job(self, dlg, worker, on_done, payload) -> None:
        try:
            on_done(payload)
        finally:
            self._close_dialog_job(dlg, worker)

    def _fail_dialog_job(self, dlg, worker, title: str, text: str) -> None:
        self._close_dialog_job(dlg, worker)
        QMessageBox.warning(self.tab, title, text)

    def _close_dialog_job(self, dlg, worker) -> None:
        dlg.finish()
        self.untrack(worker)

    # Existing library scan/rebuild jobs can keep living here too.

    def load_library(self, *, refresh=False) -> None:
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self.tab.status_lbl.setText('already scanning ; please wait')
            return

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
            self.tab.status_lbl.setText('already scanning ; please wait')
            return

        self.tab.status_lbl.setText(f'rebuilding {name}…')

        def job(progress):

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
        # Display label is derived from the manifest's first path field
        # (e.g. "Etterna install folder" -> "Etterna").
        parts = [
            f'{counts.get(name, 0)} {manifest.path_fields[0].label.split()[0]}'
            for name, manifest in manifest_mod.all_manifests().items()
        ]
        self.tab.status_lbl.setText(
            f'{len(library)} entries ({", ".join(parts)})'
        )