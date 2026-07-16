"""Reusable busy dialog for background Workers.

Deliberately dumb: it owns no job logic and knows nothing about what
the worker computes. It provides presentation (label + busy bar) and
default handlers -- every way of dismissing it (Cancel button, title
bar X, Esc) cancels the attached worker, whose late result is then
discarded by the Worker itself. New loading windows should reuse this
instead of hand-wiring QProgressDialog + close behavior each time.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog


class LoadingDialog(QProgressDialog):
    def __init__(self, label: str, title: str, parent=None, *,
                 label_prefix: str | None = None):
        super().__init__(label, 'Cancel', 0, 0, parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumDuration(0)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self._prefix = label_prefix
        self._worker = None
        self._dismissed = False
        # Dismissal routes differ per input: the Cancel button emits
        # `canceled`, Esc goes through reject(), the title-bar X through
        # closeEvent(). All funnel into _dismiss(), which is idempotent
        # so overlapping routes (X also emits canceled) fire it once.
        self.canceled.connect(self._dismiss)

    def attach(self, worker) -> None:
        """Bind the worker this dialog reports on: progress messages
        update the label, and dismissing the dialog cancels it."""
        self._worker = worker
        worker.progress.connect(self.set_message)

    def set_message(self, msg: str) -> None:
        text = f'{self._prefix}\n{msg}' if self._prefix else str(msg)
        self.setLabelText(text)

    def finish(self) -> None:
        """Programmatic close on job completion: no cancel involved."""
        self._dismissed = True
        self.close()
        self.deleteLater()

    def reject(self) -> None:
        self._dismiss()
        super().reject()

    def closeEvent(self, ev) -> None:
        self._dismiss()
        super().closeEvent(ev)

    def _dismiss(self) -> None:
        if self._dismissed:
            return
        self._dismissed = True
        if self._worker is not None:
            self._worker.cancel()
        self.close()
        self.deleteLater()
