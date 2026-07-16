"""MainWindow tab de-duplication: opening an action for an entry that
already has a tab focuses that tab instead of building a duplicate."""
import pytest

from PySide6.QtWidgets import QApplication, QWidget


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _window(app):
    # Build MainWindow without the library scan / first-run dialog side
    # effects by constructing the bare QMainWindow shell the dedup logic
    # lives on. `_add_tab` / `_focus_tab` only touch `self.tabs`.
    from analysis.gui.app import MainWindow
    from PySide6.QtWidgets import QMainWindow, QTabWidget

    win = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(win)
    win.tabs = QTabWidget()
    return win


def test_focus_existing_tab_by_key(app):
    win = _window(app)
    a, b = QWidget(), QWidget()
    win._add_tab(a, 'A', key=('play', '/x'))
    win._add_tab(b, 'B', key=('viz', '/x', 'errorbar'))
    assert win.tabs.currentIndex() == 1

    assert win._focus_tab(('play', '/x')) is True
    assert win.tabs.currentIndex() == 0
    assert win.tabs.count() == 2   # no new tab


def test_no_match_returns_false(app):
    win = _window(app)
    win._add_tab(QWidget(), 'A', key=('play', '/x'))
    assert win._focus_tab(('play', '/other')) is False


def test_none_key_never_matches(app):
    win = _window(app)
    win._add_tab(QWidget(), 'A')          # keyless tab
    assert win._focus_tab(None) is False


def test_distinct_kinds_do_not_collide(app):
    win = _window(app)
    win._add_tab(QWidget(), 'play', key=('play', '/x'))
    win._add_tab(QWidget(), 'viz', key=('viz', '/x', 'errorbar'))
    assert win._focus_tab(('play', '/x')) is True
    assert win.tabs.currentIndex() == 0
    assert win._focus_tab(('viz', '/x', 'errorbar')) is True
    assert win.tabs.currentIndex() == 1


def test_closed_tab_key_is_stale(app):
    win = _window(app)
    win._add_tab(QWidget(), 'A', key=('play', '/x'))
    win.tabs.removeTab(0)
    assert win._focus_tab(('play', '/x')) is False


def test_tab_key_helper_none_without_replay_path():
    from analysis.gui.library.entry_actions.base import EntryActionBase
    action = EntryActionBase(tab=None)
    assert action.tab_key('play', {}) is None
    assert action.tab_key('play', {'replay_path': '/x'}) == ('play', '/x')
    assert action.tab_key('viz', {'replay_path': '/x'}, 'errorbar') \
        == ('viz', '/x', 'errorbar')
