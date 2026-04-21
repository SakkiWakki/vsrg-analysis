"""Shared pytest setup.

QSettings is global per (org, app). Tests must not clobber the real user's
settings — so we point QSettings at an isolated INI file in a temp dir and
reset the `get_settings()` module cache around each test.
"""
import os
import sys
import types
from pathlib import Path

import pytest

# `analysis.games.osu.replay` imports `osrparse` at module load. That dep
# isn't installed in every dev venv and these tests only exercise the path
# override helpers, so stub it with an empty module if missing.
if 'osrparse' not in sys.modules:
    try:
        import osrparse  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType('osrparse')
        stub.Replay = type('Replay', (), {})
        stub.GameMode = type('GameMode', (), {'MANIA': 3})
        stub.Key = type('Key', (), {'M1': 1, 'M2': 2, 'K1': 4, 'K2': 8,
                                      'K3': 16, 'K4': 32, 'K5': 64})
        sys.modules['osrparse'] = stub


@pytest.fixture(scope='session', autouse=True)
def _qapp():
    """QDialog widgets need a QApplication. Create one once for the session —
    reinstantiating it per-test tends to crash under offscreen platforms."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(['test'])
    yield app


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path, monkeypatch, _qapp):
    from PySide6.QtCore import QSettings
    # QSettings honors XDG_CONFIG_HOME on Linux under the native format, so
    # redirecting that is the most reliable way to isolate tests from both
    # the real user's config and from prior tests.
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg'))
    (tmp_path / 'xdg').mkdir()

    from analysis.gui import settings as settings_mod
    settings_mod._cached = None
    # Clear any leftover from prior tests (belt-and-braces: if QSettings was
    # already cached globally by Qt, wipe the keys we care about).
    s = QSettings(settings_mod._ORG, settings_mod._APP)
    s.clear()
    s.sync()
    settings_mod._cached = None
    yield
    s = QSettings(settings_mod._ORG, settings_mod._APP)
    s.clear()
    s.sync()
    settings_mod._cached = None
