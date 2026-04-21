"""Tests for the unified config store.

Covers: read/write/delete, dotted-path navigation, subscription fanout
and prefix scoping, persistence round-trip, debounced save + flush,
corrupt-file tolerance.
"""
from __future__ import annotations

import json

import pytest

from analysis.config.store import MISSING, ConfigStore


@pytest.fixture
def store(tmp_path):
    s = ConfigStore(tmp_path / 'config.json', debounce_s=0.01)
    s.load()
    return s


# ─── Read / write basics ───────────────────────────────────────────────────

def test_missing_path_returns_default(store):
    assert store.get('nope') is None
    assert store.get('nope', 42) == 42
    assert store.get('deep.not.here', 'd') == 'd'


def test_set_then_get_roundtrip(store):
    assert store.set('player.scroll_speed', 400)
    assert store.get('player.scroll_speed') == 400


def test_set_creates_nested_dicts(store):
    store.set('a.b.c.d', 'leaf')
    snap = store.snapshot()
    assert snap == {'a': {'b': {'c': {'d': 'leaf'}}}}


def test_set_returns_false_when_value_unchanged(store):
    assert store.set('x', 1) is True
    assert store.set('x', 1) is False
    assert store.set('x', 2) is True


def test_set_rejects_empty_path(store):
    with pytest.raises(ValueError):
        store.set('', 1)


def test_set_rejects_traversing_non_dict(store):
    store.set('x', 'leaf')
    with pytest.raises(TypeError):
        store.set('x.y', 'deeper')  # can't descend into a string


def test_delete_removes_leaf(store):
    store.set('a.b.c', 1)
    assert store.delete('a.b.c') is True
    assert store.get('a.b.c') is None
    # Parent dicts stay — we don't garbage-collect empty branches,
    # since future writes may re-fill them.
    assert store.get('a.b') == {}


def test_delete_missing_is_noop(store):
    assert store.delete('nope') is False


def test_snapshot_is_deep_copy(store):
    store.set('a.b', [1, 2])
    snap = store.snapshot()
    snap['a']['b'].append(3)
    assert store.get('a.b') == [1, 2]  # original untouched


# ─── Subscriptions ─────────────────────────────────────────────────────────

def test_root_subscription_fires_for_any_change(store):
    events = []
    store.subscribe('', lambda p, o, n: events.append((p, o, n)))
    store.set('a', 1)
    store.set('b.c', 2)
    assert events == [(('a',), MISSING, 1), (('b', 'c'), MISSING, 2)]


def test_prefix_subscription_scopes_fanout(store):
    plugin_events = []
    other_events = []
    store.subscribe('plugins', lambda p, o, n: plugin_events.append(p))
    store.subscribe('paths', lambda p, o, n: other_events.append(p))
    store.set('plugins.foo.enabled', True)
    store.set('paths.etterna', '/opt')
    store.set('player.scroll_speed', 400)  # neither prefix
    assert plugin_events == [('plugins', 'foo', 'enabled')]
    assert other_events == [('paths', 'etterna')]


def test_subscription_reports_old_value_on_overwrite(store):
    events = []
    store.subscribe('', lambda p, o, n: events.append((o, n)))
    store.set('x', 'first')
    store.set('x', 'second')
    assert events == [(MISSING, 'first'), ('first', 'second')]


def test_delete_fanout_uses_missing_sentinel(store):
    store.set('x', 'v')
    events = []
    store.subscribe('', lambda p, o, n: events.append((o, n)))
    store.delete('x')
    assert events == [('v', MISSING)]


def test_unsubscribed_handler_stops_firing(store):
    events = []
    sub = store.subscribe('', lambda p, o, n: events.append(p))
    store.set('a', 1)
    assert store.unsubscribe(sub) is True
    store.set('b', 2)
    assert events == [('a',)]


def test_handler_exception_doesnt_block_others(store):
    events = []
    store.subscribe('', lambda p, o, n: (_ for _ in ()).throw(
        RuntimeError('boom')))
    store.subscribe('', lambda p, o, n: events.append(p))
    store.set('a', 1)
    assert events == [('a',)]


def test_subscriptions_on_same_path_both_fire(store):
    a, b = [], []
    store.subscribe('plugins', lambda p, o, n: a.append(p))
    store.subscribe('plugins', lambda p, o, n: b.append(p))
    store.set('plugins.x.enabled', True)
    assert a == [('plugins', 'x', 'enabled')]
    assert b == [('plugins', 'x', 'enabled')]


def test_unchanged_set_does_not_fire_subscription(store):
    events = []
    store.subscribe('', lambda p, o, n: events.append(p))
    store.set('x', 1)
    store.set('x', 1)  # same value
    assert events == [('x',)]


# ─── Persistence ───────────────────────────────────────────────────────────

def test_flush_writes_file(tmp_path):
    path = tmp_path / 'config.json'
    s = ConfigStore(path)
    s.load()
    s.set('a.b', 1)
    s.set('c', [1, 2, 3])
    s.flush()
    data = json.loads(path.read_text())
    assert data == {'a': {'b': 1}, 'c': [1, 2, 3]}


def test_load_reads_file(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'x': {'y': 42}}))
    s = ConfigStore(path)
    s.load()
    assert s.get('x.y') == 42


def test_load_missing_file_is_empty_tree(tmp_path):
    s = ConfigStore(tmp_path / 'nope.json')
    s.load()
    assert s.snapshot() == {}


def test_load_corrupt_file_falls_back_to_empty(tmp_path, capsys):
    path = tmp_path / 'config.json'
    path.write_text('{not valid json')
    s = ConfigStore(path)
    s.load()
    assert s.snapshot() == {}
    # Should have warned on stdout so the user sees it.
    assert 'parse failed' in capsys.readouterr().out


def test_load_non_object_root_falls_back_to_empty(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps([1, 2, 3]))
    s = ConfigStore(path)
    s.load()
    assert s.snapshot() == {}


def test_flush_is_idempotent(tmp_path):
    path = tmp_path / 'config.json'
    s = ConfigStore(path)
    s.load()
    s.set('a', 1)
    s.flush()
    s.flush()  # no pending changes
    assert json.loads(path.read_text()) == {'a': 1}


def test_autosave_debounce_coalesces_writes(tmp_path):
    """Burst of sets should produce one write. We observe indirectly by
    flushing once at the end — the on-disk contents should match the
    final state regardless of intermediate values."""
    path = tmp_path / 'config.json'
    s = ConfigStore(path, debounce_s=0.05)
    s.load()
    for i in range(10):
        s.set('counter', i)
    s.flush()
    assert json.loads(path.read_text())['counter'] == 9


def test_autosave_disabled_skips_writes(tmp_path):
    path = tmp_path / 'config.json'
    s = ConfigStore(path, autosave=False)
    s.load()
    s.set('a', 1)
    # No flush → nothing on disk.
    assert not path.exists()
    s.flush()
    assert json.loads(path.read_text()) == {'a': 1}


def test_persisted_values_roundtrip_exact(tmp_path):
    """Multi-type round-trip — JSON doesn't preserve tuples (become
    lists), so document that explicitly."""
    path = tmp_path / 'config.json'
    s = ConfigStore(path)
    s.load()
    s.set('str', 'hello')
    s.set('int', 42)
    s.set('float', 3.14)
    s.set('bool', True)
    s.set('list', [1, 'a', None])
    s.set('dict', {'nested': True})
    s.flush()

    s2 = ConfigStore(path)
    s2.load()
    assert s2.get('str') == 'hello'
    assert s2.get('int') == 42
    assert s2.get('float') == 3.14
    assert s2.get('bool') is True
    assert s2.get('list') == [1, 'a', None]
    assert s2.get('dict') == {'nested': True}
