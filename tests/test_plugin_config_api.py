"""Tests for :class:`PluginConfig` ; the scoped config handle plugins
use to persist their own settings into the unified JSON.

The interesting properties:

  * A plugin can read/write/subscribe to its own subtree.
  * Cross-window fanout: a write in one instance reaches subscribers
    on another instance of the same plugin.
  * Scoping: one plugin's handle can't reach another plugin's data.
  * Field paths in subscribers are relative to the plugin's root.
  * Snapshot is a flat copy.
"""
from __future__ import annotations

import pytest

from analysis.config.store import ConfigStore
from analysis.plugins.host_api import PluginConfig


@pytest.fixture
def store(tmp_path):
    s = ConfigStore(tmp_path / 'config.json', autosave=False)
    s.load()
    return s


def test_requires_plugin_key():
    with pytest.raises(ValueError):
        PluginConfig('')


def test_set_and_get_roundtrip(store):
    cfg = PluginConfig('bundle:name', config=store)
    cfg.set('volume', 0.5)
    assert cfg.get('volume') == 0.5


def test_get_default_on_missing(store):
    cfg = PluginConfig('bundle:name', config=store)
    assert cfg.get('missing') is None
    assert cfg.get('missing', 42) == 42


def test_writes_land_under_plugin_subtree(store):
    cfg = PluginConfig('bundle:name', config=store)
    cfg.set('volume', 0.5)
    # Raw tree path ; verifies the scoping layout the dialog/UI expects.
    assert store.get('plugins.bundle:name.settings.volume') == 0.5


def test_scoping_prevents_cross_plugin_reads(store):
    a = PluginConfig('bundle:a', config=store)
    b = PluginConfig('bundle:b', config=store)
    a.set('volume', 0.5)
    assert b.get('volume') is None  # b can't see a's setting


def test_dotted_field_paths(store):
    cfg = PluginConfig('bundle:name', config=store)
    cfg.set('colors.background', '#000')
    cfg.set('colors.text', '#fff')
    assert cfg.get('colors.background') == '#000'
    assert cfg.get('colors') == {'background': '#000', 'text': '#fff'}


def test_subscribe_reports_relative_field(store):
    cfg = PluginConfig('bundle:name', config=store)
    events = []
    cfg.subscribe(lambda f, o, n: events.append((f, o, n)))
    cfg.set('volume', 0.5)
    cfg.set('colors.fg', '#fff')
    assert [f for f, _, _ in events] == ['volume', 'colors.fg']


def test_cross_window_propagation(store):
    """Two handles for the same plugin ; a write in one reaches a
    subscriber on the other. This is the user-visible promise: a
    plugin's other-window instance updates when config changes."""
    a = PluginConfig('bundle:name', config=store)
    b = PluginConfig('bundle:name', config=store)
    seen = []
    b.subscribe(lambda f, o, n: seen.append((f, n)))
    a.set('volume', 0.7)
    assert seen == [('volume', 0.7)]


def test_unrelated_plugins_dont_cross_fire(store):
    a = PluginConfig('bundle:a', config=store)
    b = PluginConfig('bundle:b', config=store)
    seen_a = []
    a.subscribe(lambda f, o, n: seen_a.append(f))
    b.set('volume', 0.5)
    assert seen_a == []


def test_unsubscribe_stops_firing(store):
    cfg = PluginConfig('bundle:name', config=store)
    events = []
    h = cfg.subscribe(lambda f, o, n: events.append(f))
    cfg.set('a', 1)
    assert cfg.unsubscribe(h) is True
    cfg.set('b', 2)
    assert events == ['a']


def test_delete_field(store):
    cfg = PluginConfig('bundle:name', config=store)
    cfg.set('volume', 0.5)
    assert cfg.delete('volume') is True
    assert cfg.get('volume') is None


def test_snapshot_returns_flat_copy(store):
    cfg = PluginConfig('bundle:name', config=store)
    cfg.set('volume', 0.5)
    cfg.set('muted', False)
    snap = cfg.snapshot()
    assert snap == {'volume': 0.5, 'muted': False}
    snap['volume'] = 1.0  # mutating the copy doesn't touch store
    assert cfg.get('volume') == 0.5


def test_snapshot_when_no_settings_yet(store):
    cfg = PluginConfig('bundle:never_written', config=store)
    assert cfg.snapshot() == {}


def test_plugin_key_with_dots_is_escaped(store):
    """A bundle author who picks ``vendor.module:name`` as a key would
    collide with the store's path separator. Escaping keeps fields
    namespaced correctly under a flat subtree."""
    cfg = PluginConfig('vendor.module:name', config=store)
    cfg.set('volume', 0.3)
    # Dots in the key become underscores; settings live flat under
    # that escaped key.
    assert store.get('plugins.vendor_module:name.settings.volume') == 0.3


def test_set_and_get_require_field(store):
    cfg = PluginConfig('bundle:name', config=store)
    with pytest.raises(ValueError):
        cfg.set('', 'x')
    with pytest.raises(ValueError):
        cfg.delete('')
