"""Tests for the osu!-live polling client.

We stub the fetcher so CI doesn't need a data source (native or tosu)
running. Focus: payload → snapshot mapping, append semantics across
ticks, map-change reset, disconnect resilience. The fetcher contract
is source-agnostic — whether the bytes came from the native Rust
reader or from an HTTP GET, the dict shape fed to ``_build_snapshot``
is identical, so these tests cover both paths.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from plugins.unsafe.osu_live.client import (LiveSnapshot, OsuLiveClient)

def effect(param1, param2):
    effect_f = lambda frames : {
        # Transform function using param1 and param2
    }
    return effect_f

# To use, call effect_f(frames) -> effected frames

def _payload(md5='abc', hits=None, combo=0, acc=0.0, ur=0.0,
             keycount=4, mode='mania', title='Test Map',
             h300=0, h100=0, h50=0, h0=0):
    return {
        'beatmap': {
            'md5': md5,
            'title': title,
            'mode': {'name': mode},
            'stats': {'cs': keycount},
        },
        'play': {
            'hits': {'300': h300, '100': h100, '50': h50, '0': h0},
            'hitErrorArray': list(hits or []),
            'combo': {'current': combo, 'max': combo},
            'accuracy': acc,
            'unstableRate': ur,
        },
    }


def _stub(payloads):
    """Build a fetcher that returns each payload in sequence,
    looping on the last one once exhausted."""
    i = {'n': 0}

    def fetch(url):
        n = i['n']
        if n < len(payloads) - 1:
            i['n'] += 1
        return payloads[n]
    return fetch


def test_initial_snapshot_is_disconnected():
    c = OsuLiveClient(fetch=lambda u: (_ for _ in ()).throw(OSError('no')))
    snap = c.snapshot()
    assert snap.connected is False
    assert len(snap.offsets) == 0


def test_build_snapshot_maps_headline_fields():
    c = OsuLiveClient(fetch=_stub([_payload(
        hits=[5.0, -3.0], combo=42, acc=98.7, ur=12.3,
        h300=10, h100=2, h50=1, h0=0, title='Blue Zenith')]))
    snap = c._build_snapshot(_stub([_payload(
        hits=[5.0, -3.0], combo=42, acc=98.7, ur=12.3,
        h300=10, h100=2, h50=1, title='Blue Zenith')])('x'))
    assert snap.connected is True
    assert snap.combo == 42
    assert snap.accuracy == pytest.approx(98.7)
    assert snap.unstable_rate == pytest.approx(12.3)
    assert snap.hits_300 == 10
    assert snap.map_title == 'Blue Zenith'


def test_offsets_are_converted_ms_to_seconds():
    c = OsuLiveClient(fetch=_stub([_payload(hits=[10.0, -20.0])]))
    snap = c._build_snapshot(_payload(hits=[10.0, -20.0]))
    # tosu reports ms; our convention is seconds.
    assert np.allclose(snap.offsets, [0.010, -0.020])


def _tick(c, payload):
    """Mirror one poll-loop iteration: build a snapshot and publish
    it. Lets tests advance the client's state the same way the
    background thread would."""
    snap = c._build_snapshot(payload)
    c._snapshot = snap
    return snap


def test_subsequent_ticks_append_new_hits_only():
    """Tosu's hitErrorArray grows monotonically within one play. Each
    poll should only surface the tail we haven't seen."""
    c = OsuLiveClient(fetch=lambda u: None)
    _tick(c, _payload(hits=[1.0, 2.0]))
    snap = _tick(c, _payload(hits=[1.0, 2.0, 3.0]))
    assert len(snap.offsets) == 3
    assert np.allclose(snap.offsets, [0.001, 0.002, 0.003])


def test_map_change_resets_accumulated_arrays():
    c = OsuLiveClient(fetch=lambda u: None)
    _tick(c, _payload(md5='one', hits=[1.0, 2.0]))
    # Second payload is a different map — arrays should reset, not
    # concatenate.
    snap = _tick(c, _payload(md5='two', hits=[9.0]))
    assert len(snap.offsets) == 1
    assert snap.offsets[0] == pytest.approx(0.009)


def test_disconnect_keeps_previous_arrays():
    """One bad fetch shouldn't wipe the visible viz."""
    c = OsuLiveClient(fetch=lambda u: None)
    _tick(c, _payload(hits=[1.0, 2.0]))
    disc = c._disconnected('network error')
    assert disc.connected is False
    assert len(disc.offsets) == 2


def test_columns_are_synthesized_when_absent():
    """Tosu v2 doesn't publish per-hit column for mania. We round-robin
    so hand-split viz have some signal. Not true column data."""
    c = OsuLiveClient(fetch=lambda u: None)
    snap = c._build_snapshot(_payload(keycount=4, hits=[1.0, 2.0, 3.0, 4.0]))
    assert len(snap.columns) == 4
    # Round-robin across the 4 lanes.
    assert list(snap.columns) == [0, 1, 2, 3]


def test_snapshot_as_replay_dict_has_expected_keys():
    snap = LiveSnapshot(connected=True)
    d = snap.as_replay_dict()
    for k in ('offsets', 'columns', 'noterows',
              'notetypes', 'misses', 'keycount'):
        assert k in d


def test_thread_lifecycle_starts_and_stops_cleanly():
    """End-to-end-ish: start the thread with a fast stub, verify
    snapshots become connected, stop the thread, verify it exits."""
    payloads = [_payload(hits=[1.0, 2.0], combo=10)]
    c = OsuLiveClient(poll_hz=200.0, fetch=_stub(payloads))
    c.start()
    try:
        # Spin until we see a connected snapshot, or time out.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if c.snapshot().connected:
                break
            time.sleep(0.01)
        assert c.snapshot().connected is True
    finally:
        c.stop(timeout=0.5)
    # Thread should have exited.
    assert c._thread is None or not c._thread.is_alive()


def test_poll_loop_handles_fetch_exception_without_dying():
    def bad(url):
        raise ConnectionRefusedError('tosu not running')

    c = OsuLiveClient(poll_hz=200.0, fetch=bad)
    c.start()
    try:
        time.sleep(0.05)
        assert c.snapshot().connected is False
        # Thread still alive despite repeated exceptions.
        assert c._thread is not None and c._thread.is_alive()
    finally:
        c.stop(timeout=0.5)


def test_start_is_idempotent():
    c = OsuLiveClient(fetch=lambda u: _payload())
    c.start()
    t1 = c._thread
    c.start()  # second call should not spin a new thread
    assert c._thread is t1
    c.stop(timeout=0.5)


def test_stop_is_idempotent():
    c = OsuLiveClient(fetch=lambda u: _payload())
    c.start()
    c.stop(timeout=0.5)
    c.stop(timeout=0.5)  # must not raise
