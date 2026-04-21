"""PySide6 GUI for Etterna + osu!mania replay analysis.
Every action opens in an embedded in-app tab instead of a separate window.

This module is the app shell only — the real logic lives in:
  theme.py         dark palette + QSS
  widgets.py       JumpSlider, MplTab, HtmlTab, _viz_toolbar
  loaders.py       Worker thread + replay/chart/audio resolvers
  replay_cache.py  LRU parsed-replay cache
  note_viz_tab.py  NoteVizTab
  player_tab.py    PlayerTab
  library_tab.py   LibraryTab (tree + filters + open flows)
"""
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from analysis.gui.settings import get_settings
from analysis.gui.theme import apply_dark_palette
from analysis.gui.library_tab import LibraryTab

# Re-export for backward compatibility (note_viewer plugin imports NoteVizTab
# from analysis.gui.app).
from analysis.gui.note_viz_tab import NoteVizTab  # noqa: F401


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Replay Analyzer')
        s = get_settings()
        geom = s.value('window/geometry')
        if geom is not None:
            self.restoreGeometry(geom)
        else:
            self.resize(1300, 820)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.setCentralWidget(self.tabs)

        self.library_tab = LibraryTab(add_tab=self._add_tab)
        self._add_tab(self.library_tab, 'Library', closable=False)

    def closeEvent(self, ev):
        s = get_settings()
        s.setValue('window/geometry', self.saveGeometry())
        self.library_tab.persist_settings()
        for w in self.library_tab.active_workers():
            if w.isRunning():
                w.quit()
                w.wait(2000)
        for i in range(self.tabs.count()):
            widget = self.tabs.widget(i)
            if hasattr(widget, 'cleanup'):
                try: widget.cleanup()
                except Exception: pass
        super().closeEvent(ev)

    def _add_tab(self, widget, title, closable=True):
        idx = self.tabs.addTab(widget, title)
        if not closable:
            self.tabs.tabBar().setTabButton(
                idx, self.tabs.tabBar().ButtonPosition.RightSide, None)
        self.tabs.setCurrentIndex(idx)
        return idx

    def _close_tab(self, idx):
        w = self.tabs.widget(idx)
        if hasattr(w, 'cleanup'):
            try:
                w.cleanup()
            except Exception:
                pass
        self.tabs.removeTab(idx)


def main():
    app = QApplication(sys.argv)
    apply_dark_palette(app)
    # First-run: prompt for install paths before building the main window so
    # the library scan that kicks off on window open sees the user's choices.
    from analysis.gui.paths_dialog import prompt_if_first_run
    prompt_if_first_run()
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
