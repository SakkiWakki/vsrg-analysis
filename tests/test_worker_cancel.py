"""Worker cooperative cancellation + LoadingDialog default handlers."""
import os
import threading
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtWidgets import QApplication

from analysis.gui.loaders import Worker
from analysis.gui.loading_dialog import LoadingDialog


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _drain(app, worker, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert not worker.isRunning(), 'worker did not exit'


def test_cancel_interrupts_job_at_next_progress(app):
    started = threading.Event()

    def job(progress):
        for _ in range(10_000):
            progress('working')
            started.set()
            time.sleep(0.001)
        return 'finished'

    results, failures = [], []
    worker = Worker(job)
    worker.done.connect(results.append)
    worker.failed.connect(failures.append)
    worker.start()
    assert started.wait(2.0)

    worker.cancel()
    _drain(app, worker)
    assert results == []
    assert failures == []


def test_cancelled_result_is_discarded(app):
    release = threading.Event()

    def job(progress):
        release.wait(2.0)
        return 'late result'   # produced after cancel, no progress calls

    results = []
    worker = Worker(job)
    worker.done.connect(results.append)
    worker.start()
    worker.cancel()
    release.set()
    _drain(app, worker)
    assert results == []


def test_uncancelled_worker_still_delivers(app):
    worker = Worker(lambda progress: 'ok')
    results = []
    worker.done.connect(results.append)
    worker.start()
    _drain(app, worker)
    assert results == ['ok']


def test_dialog_dismissal_cancels_attached_worker(app):
    class FakeWorker:
        def __init__(self):
            self.cancelled = False
            self.progress = type('S', (), {'connect': lambda *a: None})()

        def cancel(self):
            self.cancelled = True

    for dismiss in ('close', 'reject', 'canceled'):
        dlg = LoadingDialog('loading...', 'Test')
        fake = FakeWorker()
        dlg.attach(fake)
        match dismiss:
            case 'close':
                dlg.close()          # title-bar X
            case 'reject':
                dlg.reject()         # Esc
            case 'canceled':
                dlg.canceled.emit()  # Cancel button
        assert fake.cancelled, f'{dismiss} did not cancel'
        assert dlg.isHidden(), f'{dismiss} did not hide'
        app.processEvents()   # flush the queued deleteLater


def test_dialog_finish_does_not_cancel(app):
    class FakeWorker:
        def __init__(self):
            self.cancelled = False
            self.progress = type('S', (), {'connect': lambda *a: None})()

        def cancel(self):
            self.cancelled = True

    dlg = LoadingDialog('loading...', 'Test')
    fake = FakeWorker()
    dlg.attach(fake)
    dlg.finish()
    assert not fake.cancelled
    assert dlg.isHidden()
    app.processEvents()   # flush the queued deleteLater
