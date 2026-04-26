"""PySide6 GUI for Etterna + osu!mania replay analysis.
Every action opens in an embedded in-app tab instead of a separate window.

This module is the app shell only ; the real logic lives in:
  theme.py         dark palette + QSS
  widgets.py       JumpSlider, MplTab, HtmlTab, _viz_toolbar
  loaders.py       Worker thread + replay/chart/audio resolvers
  replay_cache.py  LRU parsed-replay cache
  note_viz_tab.py  NoteVizTab
  player_tab.py    PlayerTab
  library_tab.py   LibraryTab (tree + filters + open flows)
"""
import sys

# Python 3.12 compatibility shim for PySide6 + six.
#
# shiboken's feature_imported hook runs on every import. On Python 3.12
# it eventually calls inspect.getsourcefile(module), which walks into
# importlib._bootstrap._module_repr_from_spec and asks the loader for a
# `_path` attribute. six's `_SixMetaPathImporter` (the loader behind
# `six.moves`) doesn't have one, and the whole chain raises
# AttributeError: '_SixMetaPathImporter' object has no attribute '_path'.
#
# matplotlib → python-dateutil → six.moves is a common import chain, so
# the crash hits before the app even opens a window. Patch before
# importing PySide6 so the shiboken hook never sees an unpatched loader.
import six as _six
_cls = _six._SixMetaPathImporter
if not hasattr(_cls, 'get_filename'):
    _cls.get_filename = lambda self, fullname: None
if not hasattr(_cls, '_path'):
    _cls._path = None
del _six, _cls

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from analysis.gui.settings import get_settings
from analysis.gui.theme import apply_dark_palette
from analysis.gui.library.tab import LibraryTab

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
        # QOpenGLWidget children (PlayerCanvas) need a native window in the
        # widget hierarchy before being added to a tab, otherwise the first
        # reparent forces a top-level compositor re-present that looks like
        # the main window minimizing. Giving the tab widget's internal stack
        # a native window handle lets GL children slot in without the flash.
        self.tabs.setAttribute(Qt.WA_NativeWindow, True)
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


def _apply_default_gl_format():
    """Set a process-wide default OpenGL surface format.

    ``PlayerCanvas`` is a ``QOpenGLWidget``; setting the format before
    ``QApplication`` constructs means every widget shares the same GL
    profile. Keeping this explicit also gives us a single place to
    later opt into shared contexts for the web-texture PAL's GL
    backend (``QOpenGLContext::setShareContext`` on the Chromium
    offscreen context).
    """
    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setMajorVersion(3)
    fmt.setMinorVersion(2)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    # Enable resource sharing between contexts so offscreen producers
    # (Chromium compositor, any future GL-backed WebTexture) can hand
    # textures to the player canvas without readback.
    fmt.setOption(QSurfaceFormat.FormatOption.ResetNotification)
    QSurfaceFormat.setDefaultFormat(fmt)


def main():
    _apply_default_gl_format()
    # ``AA_ShareOpenGLContexts`` makes every QOpenGLWidget share its GL
    # context with the global one. Needed so our future GL web-texture
    # backend can upload into a context whose textures the canvas can
    # sample. Must be set before ``QApplication`` is constructed.
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtWidgets import QApplication as _QA
    _QA.setAttribute(_Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    app = QApplication(sys.argv)
    apply_dark_palette(app)
    # Install the QSettings-backed path-overrides shopkeeper so headless
    # core modules (find_etterna_dirs, find_osu_dirs, ...) read user
    # overrides without importing Qt themselves.
    from analysis.gui.path_overrides_qt import install as _install_overrides
    _install_overrides()
    # First-run: prompt for install paths before building the main window so
    # the library scan that kicks off on window open sees the user's choices.
    from analysis.gui.paths_dialog import prompt_if_first_run
    prompt_if_first_run()
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
