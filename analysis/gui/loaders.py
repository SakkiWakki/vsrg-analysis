"""Replay loading + chart/audio resolution + background Worker thread.
Nothing here touches Qt widgets except QThread."""
import traceback

from PySide6.QtCore import QThread, Signal

from analysis.core import game as game_mod


class Worker(QThread):
    done = Signal(object)
    progress = Signal(str)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            result = self.fn(lambda m: self.progress.emit(m))
            self.done.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


def load_replay(path, game, chart_path=None):
    return game_mod.get(game).parse_replay(path, chart_path=chart_path)
