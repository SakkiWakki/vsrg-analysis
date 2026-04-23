"""Tests for the osu! live polling client (native-only path).

The native Rust extension is mocked out so CI doesn't need osu! running.
Focus: GameMemoryState construction from raw native data, thread lifecycle,
and graceful handling of a missing/unavailable native reader.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from analysis.components.api import GameMemoryState
from plugins.unsafe.osu_live.client import (
    OsuLiveClient,
    _raw_to_game_memory,
    get_client,
    reset_for_tests,
)


def _raw(in_gameplay=True, combo=42, max_combo=100, accuracy=0.987,
         hit_300=10, hit_100=2, hit_50=0, hit_miss=0,
         hit_geki=0, hit_katu=0, hit_errors_ms=None,
         map_md5='abc123', map_title='Blue Zenith'):
    return {
        'in_gameplay': in_gameplay,
        'combo': combo,
        'max_combo': max_combo,
        'accuracy': accuracy,
        'hit_300': hit_300,
        'hit_100': hit_100,
        'hit_50': hit_50,
        'hit_miss': hit_miss,
        'hit_geki': hit_geki,
        'hit_katu': hit_katu,
        'hit_errors_ms': list(hit_errors_ms or []),
        'map_md5': map_md5,
        'map_title': map_title,
    }


def test_raw_to_game_memory_maps_all_fields():
    snap = _raw_to_game_memory(_raw(
        combo=42, accuracy=0.987, hit_errors_ms=[5, -3, 10],
        map_title='Blue Zenith'))
    assert isinstance(snap, GameMemoryState)
    assert snap.combo == 42
    assert snap.accuracy == pytest.approx(0.987)
    assert snap.hit_errors_ms == (5, -3, 10)
    assert snap.map_title == 'Blue Zenith'
    assert snap.in_gameplay is True


def test_raw_to_game_memory_hit_errors_is_tuple():
    snap = _raw_to_game_memory(_raw(hit_errors_ms=[1, 2, 3]))
    assert isinstance(snap.hit_errors_ms, tuple)


def test_raw_to_game_memory_empty_errors():
    snap = _raw_to_game_memory(_raw(hit_errors_ms=None))
    assert snap.hit_errors_ms == ()


def test_raw_to_game_memory_missing_fields_default_safely():
    snap = _raw_to_game_memory({})
    assert snap.combo == 0
    assert snap.accuracy == pytest.approx(0.0)
    assert snap.in_gameplay is False
    assert snap.hit_errors_ms == ()
    assert snap.map_md5 == ''
    assert snap.map_title == ''


def test_initial_snapshot_is_none():
    c = OsuLiveClient()
    assert c.snapshot() is None


def test_snapshot_is_none_when_native_unavailable():
    with patch.dict('sys.modules', {'osu_memory_native': None}):
        c = OsuLiveClient()
        result = c._poll()
    assert result is None


def test_snapshot_is_none_when_osu_not_running():
    native = MagicMock()
    native.find_osu_pid.return_value = None
    with patch.dict('sys.modules', {'osu_memory_native': native}):
        c = OsuLiveClient()
        result = c._poll()
    assert result is None


def test_poll_resolves_handle_on_first_call():
    native = MagicMock()
    native.find_osu_pid.return_value = 1234
    native.resolve.return_value = MagicMock()
    native.read_state.return_value = _raw()
    with patch.dict('sys.modules', {'osu_memory_native': native}):
        c = OsuLiveClient()
        result = c._poll()
    assert isinstance(result, GameMemoryState)
    assert c._pid == 1234


def test_poll_resets_handle_on_pid_change():
    native = MagicMock()
    native.find_osu_pid.side_effect = [1234, 5678]
    handle_a = MagicMock()
    handle_b = MagicMock()
    native.resolve.side_effect = [handle_a, handle_b]
    native.read_state.return_value = _raw()
    with patch.dict('sys.modules', {'osu_memory_native': native}):
        c = OsuLiveClient()
        c._poll()
        assert c._pid == 1234
        c._poll()
        assert c._pid == 5678


def test_poll_returns_none_and_clears_handle_on_read_error():
    native = MagicMock()
    native.find_osu_pid.return_value = 1234
    native.resolve.return_value = MagicMock()
    native.read_state.side_effect = OSError('stale pointer')
    with patch.dict('sys.modules', {'osu_memory_native': native}):
        c = OsuLiveClient()
        result = c._poll()
    assert result is None
    assert c._handle is None


def test_poll_returns_none_on_resolve_failure():
    native = MagicMock()
    native.find_osu_pid.return_value = 1234
    native.resolve.side_effect = OSError('signatures stale')
    with patch.dict('sys.modules', {'osu_memory_native': native}):
        c = OsuLiveClient()
        result = c._poll()
    assert result is None


def test_thread_lifecycle_starts_and_stops():
    native = MagicMock()
    native.find_osu_pid.return_value = None
    with patch.dict('sys.modules', {'osu_memory_native': native}):
        c = OsuLiveClient(poll_hz=200.0)
        c.start()
        time.sleep(0.05)
        assert c._thread is not None and c._thread.is_alive()
        c.stop(timeout=0.5)
    assert c._thread is None or not c._thread.is_alive()


def test_start_is_idempotent():
    native = MagicMock()
    native.find_osu_pid.return_value = None
    with patch.dict('sys.modules', {'osu_memory_native': native}):
        c = OsuLiveClient(poll_hz=200.0)
        c.start()
        t1 = c._thread
        c.start()
        assert c._thread is t1
        c.stop(timeout=0.5)


def test_stop_is_idempotent():
    c = OsuLiveClient()
    c.stop(timeout=0.1)
    c.stop(timeout=0.1)


def test_get_client_returns_singleton():
    reset_for_tests()
    try:
        with patch.dict('sys.modules', {'osu_memory_native': MagicMock()}):
            a = get_client()
            b = get_client()
        assert a is b
    finally:
        reset_for_tests()


def test_game_memory_state_is_frozen():
    snap = _raw_to_game_memory(_raw())
    with pytest.raises((AttributeError, TypeError)):
        snap.combo = 999
