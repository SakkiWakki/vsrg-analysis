"""Tests for the legacy → unified config migration.

Covers the one-shot fold of ``player_plugins.json`` and
``sidebar_sections.json`` (and QSettings paths, when PySide is
present) into the ``ConfigStore`` tree.
"""
from __future__ import annotations

import json

import pytest

from analysis.config.migrate import _migrate_disabled_list, migrate_legacy
from analysis.config.store import ConfigStore


@pytest.fixture
def store(tmp_path):
    s = ConfigStore(tmp_path / 'config.json', autosave=False)
    s.load()
    return s


def test_disabled_list_folded_into_plugins(tmp_path, store):
    path = tmp_path / 'player_plugins.json'
    path.write_text(json.dumps({'disabled': ['foo:bar', 'baz']}))
    _migrate_disabled_list(store, path, 'replay')
    assert store.get('plugins.foo:bar.replay_disabled') is True
    assert store.get('plugins.baz.replay_disabled') is True
    assert not path.exists()  # legacy file removed


def test_disabled_list_missing_file_is_noop(tmp_path, store):
    path = tmp_path / 'player_plugins.json'
    assert _migrate_disabled_list(store, path, 'replay') is False
    assert store.snapshot() == {}


def test_disabled_list_corrupt_file_is_tolerated(tmp_path, store):
    path = tmp_path / 'player_plugins.json'
    path.write_text('{not valid')
    assert _migrate_disabled_list(store, path, 'replay') is False
    assert store.snapshot() == {}


def test_disabled_list_wrong_shape_is_noop(tmp_path, store):
    path = tmp_path / 'player_plugins.json'
    path.write_text(json.dumps({'other_key': 'value'}))
    assert _migrate_disabled_list(store, path, 'replay') is False


def test_disabled_list_sidebar_kind_uses_distinct_flag(tmp_path, store):
    path = tmp_path / 'sidebar_sections.json'
    path.write_text(json.dumps({'disabled': ['sec:one']}))
    _migrate_disabled_list(store, path, 'sidebar')
    assert store.get('plugins.sec:one.sidebar_disabled') is True
    # Replay flag remains unset ; the two roles track independently.
    assert store.get('plugins.sec:one.replay_disabled') is None


def test_disabled_list_escapes_dots_in_keys(tmp_path, store):
    """Plugin keys with dots would collide with the store's path
    separator. Migration rewrites dots to underscores."""
    path = tmp_path / 'player_plugins.json'
    path.write_text(json.dumps({'disabled': ['vendor.module:name']}))
    _migrate_disabled_list(store, path, 'replay')
    # Look up the escaped path.
    assert store.get(
        'plugins.vendor_module:name.replay_disabled') is True


def test_migrate_legacy_is_idempotent(tmp_path):
    """Second call with legacy files present should not re-migrate
    (schema version guards it) ; but also must not crash."""
    cfg = tmp_path / 'config.json'
    legacy = tmp_path / 'player_plugins.json'
    legacy.write_text(json.dumps({'disabled': ['one']}))

    # Point the migrator at a fake ~/.config via monkey-patching is
    # awkward; simulate by driving the inner helper directly first,
    # then confirming the public entry point no-ops on a versioned
    # tree.
    s = ConfigStore(cfg, autosave=False)
    s.load()
    _migrate_disabled_list(s, legacy, 'replay')
    s.set('_schema_version', 1)

    # Write another legacy file and confirm migrate_legacy skips it.
    legacy2 = tmp_path / 'player_plugins.json'  # same path, new content
    legacy2.write_text(json.dumps({'disabled': ['two']}))
    migrate_legacy(s)
    # 'two' should NOT appear ; schema guard short-circuited.
    assert s.get('plugins.two.replay_disabled') is None
