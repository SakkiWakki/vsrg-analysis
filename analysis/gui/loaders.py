"""Replay loading + chart/audio resolution + background Worker thread.
Nothing here touches Qt widgets except QThread."""
import traceback

from PySide6.QtCore import QThread, Signal

from analysis.core import game as game_mod


class JobCancelled(Exception):
    """Raised inside a job's thread when the worker was cancelled.
    Cancellation is cooperative: it fires at the job's next progress()
    call, so jobs stay cancel-responsive by reporting progress."""


class Worker(QThread):
    done = Signal(object)
    progress = Signal(str)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._cancelled = False

    def cancel(self):
        """Request cooperative cancellation. The job is interrupted at
        its next progress() call; a result it still manages to produce
        is discarded rather than emitted."""
        self._cancelled = True

    def run(self):
        def report(msg):
            if self._cancelled:
                raise JobCancelled()
            self.progress.emit(msg)

        try:
            result = self.fn(report)
            if not self._cancelled:
                self.done.emit(result)
        except JobCancelled:
            pass
        except Exception:
            if not self._cancelled:
                self.failed.emit(traceback.format_exc())


def load_replay(path, game, chart_path=None):
    return game_mod.get(game).parse_replay(path, chart_path=chart_path)
