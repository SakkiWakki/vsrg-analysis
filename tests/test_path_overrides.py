"""Tests for the path-overrides shopkeeper (`analysis.core.path_overrides`)
and its Qt-backed implementation."""
from __future__ import annotations

import pytest

from analysis.core import path_overrides


# ---- in-memory backend (for the shopkeeper itself) -------------------------


class _MemBackend:
    """Trivial dict-backed backend ; the shopkeeper's contract is just
    `get(key) -> str|None` + `set(key, value|None)`."""
    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        if value is None:
            self.store.pop(key, None)
        else:
            self.store[key] = value


@pytest.fixture
def mem_backend():
    """Replace whatever the conftest installed with a per-test in-memory
    backend ; restored on teardown."""
    prev = path_overrides._BACKEND
    backend = _MemBackend()
    path_overrides.set_backend(backend)
    try:
        yield backend
    finally:
        path_overrides.set_backend(prev)


# ---- shopkeeper contract ----------------------------------------------------


def test_get_returns_none_with_no_backend():
    prev = path_overrides._BACKEND
    path_overrides.set_backend(None)
    try:
        assert path_overrides.get('paths/anything') is None
    finally:
        path_overrides.set_backend(prev)


def test_set_is_no_op_with_no_backend():
    """Headless callers shouldn't be writing paths ; set() with no
    backend just drops the call without raising."""
    prev = path_overrides._BACKEND
    path_overrides.set_backend(None)
    try:
        path_overrides.set('paths/etterna_root', '/tmp/x')  # no raise
    finally:
        path_overrides.set_backend(prev)


def test_get_set_roundtrip(mem_backend):
    assert path_overrides.get('paths/etterna_root') is None
    path_overrides.set('paths/etterna_root', '/tmp/etterna')
    assert path_overrides.get('paths/etterna_root') == '/tmp/etterna'


def test_set_none_clears(mem_backend):
    path_overrides.set('paths/etterna_root', '/tmp/etterna')
    path_overrides.set('paths/etterna_root', None)
    assert path_overrides.get('paths/etterna_root') is None


def test_set_empty_string_clears(mem_backend):
    """Empty input is normalized to None so callers can do
    `override or autodetect()` without an extra emptiness check."""
    path_overrides.set('paths/etterna_root', '/tmp/etterna')
    path_overrides.set('paths/etterna_root', '')
    assert path_overrides.get('paths/etterna_root') is None


def test_get_strips_whitespace(mem_backend):
    """Whitespace-only stored values should read back as None too --
    matches set('') behavior so the autodetect fallback always fires."""
    mem_backend.store['paths/etterna_root'] = '   '
    assert path_overrides.get('paths/etterna_root') is None


def test_keys_are_independent(mem_backend):
    """Setting one key must not bleed into another."""
    path_overrides.set('paths/etterna_root', '/tmp/etterna')
    path_overrides.set('paths/osu_root', '/tmp/osu')
    assert path_overrides.get('paths/etterna_root') == '/tmp/etterna'
    assert path_overrides.get('paths/osu_root') == '/tmp/osu'
    path_overrides.set('paths/etterna_root', None)
    assert path_overrides.get('paths/osu_root') == '/tmp/osu'


# ---- Qt-backed implementation ----------------------------------------------


def test_qt_backend_persists_through_qsettings():
    """The Qt backend is installed by the conftest. Writing through the
    shopkeeper should round-trip via the same QSettings the GUI reads."""
    from analysis.gui.settings import get_settings
    path_overrides.set('paths/test_qt_roundtrip', '/some/path')
    assert get_settings().value('paths/test_qt_roundtrip') == '/some/path'
    assert path_overrides.get('paths/test_qt_roundtrip') == '/some/path'


def test_qt_backend_clear_removes_key():
    from analysis.gui.settings import get_settings
    path_overrides.set('paths/test_qt_clear', '/some/path')
    path_overrides.set('paths/test_qt_clear', None)
    assert get_settings().value('paths/test_qt_clear') is None
    assert path_overrides.get('paths/test_qt_clear') is None
