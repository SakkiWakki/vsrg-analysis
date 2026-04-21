"""HTTP polling client for tosu's ``/json/v2`` endpoint.

Runs a daemon thread that polls tosu at a configurable rate and
maintains a growing ``replay``-shaped dict plus a few live headline
fields (combo, accuracy, UR). Viz plugins call :func:`snapshot` per
frame to get the current state.

Why polling, not WebSocket? For a sidebar viz refreshed at the Qt
tick rate, a 30Hz local HTTP poll is trivially fast, and uses only
``urllib`` (stdlib) so we don't pull in ``websocket-client`` just for
this. Upgrade to WebSocket if latency ever matters.

Why a singleton client? Multiple viz panels shouldn't each spawn a
poller. One thread, many subscribers.

Thread safety: the poller maintains a private mutable state dict and
publishes immutable snapshots by copying arrays into fresh NumPy
arrays under a lock. Readers never see a half-updated state.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np


DEFAULT_URL = 'http://127.0.0.1:24050/json/v2'
DEFAULT_POLL_HZ = 30.0


@dataclass
class LiveSnapshot:
    """Immutable view of tosu state at one instant.

    ``offsets``/``columns``/``noterows``/``notetypes``/``misses`` have the
    same shapes a parsed replay dict would, so they can be fed into the
    existing viz helpers (``_common.clean_arrays`` etc.). The live
    fields are what tosu reports about the session as a whole."""
    connected: bool
    map_title: str = ''
    combo: int = 0
    max_combo: int = 0
    accuracy: float = 0.0
    unstable_rate: float = 0.0
    hits_300: int = 0
    hits_100: int = 0
    hits_50: int = 0
    hits_miss: int = 0
    # Growing arrays — oldest hits first, index = hit order.
    offsets: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64))
    columns: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32))
    noterows: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64))
    notetypes: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32))
    misses: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool))
    keycount: int = 4

    def as_replay_dict(self) -> dict:
        """Shape this snapshot as a replay dict for the existing viz
        helpers. Only the keys the viz actually consume are populated."""
        return {
            'offsets': self.offsets,
            'columns': self.columns,
            'noterows': self.noterows,
            'notetypes': self.notetypes,
            'misses': self.misses,
            'keycount': self.keycount,
            'game': 'osu',
        }


class TosuClient:
    """Background poller. Call :meth:`start` to spin up the thread;
    call :meth:`snapshot` any time (cheap). :meth:`stop` joins the
    thread and is safe to call multiple times."""

    def __init__(self, url: str = DEFAULT_URL,
                 poll_hz: float = DEFAULT_POLL_HZ, *,
                 fetch=None):
        self._url = str(url)
        self._interval = 1.0 / max(1.0, float(poll_hz))
        # Injectable fetcher so tests can stub HTTP without bringing up
        # an actual server. Takes (url) -> dict (parsed JSON).
        self._fetch = fetch or _default_fetch
        # RLock because _build_snapshot reads the previous snapshot
        # under the same lock as callers that hold it while replacing.
        self._lock = threading.RLock()
        self._snapshot = LiveSnapshot(connected=False)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # State the poller keeps between ticks — the most recent
        # hitErrorArray length we've consumed, so we only append new
        # hits. Also the currently-seen map_md5 so a map change resets
        # the arrays.
        self._seen_hits = 0
        self._map_md5 = ''

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name='TosuClient', daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
            self._thread = None

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            return self._snapshot

    # ── Internals ─────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                payload = self._fetch(self._url)
                snap = self._build_snapshot(payload)
            except Exception as exc:
                # Don't spam on every tick; only log the first failure
                # of a streak by remembering the last state.
                snap = self._disconnected(str(exc))
            with self._lock:
                self._snapshot = snap
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self._interval - elapsed))

    def _disconnected(self, reason: str) -> LiveSnapshot:
        # Keep accumulated hits on transient errors so one dropped
        # frame doesn't wipe the viz. Only the ``connected`` flag and
        # live stats reset.
        with self._lock:
            prev = self._snapshot
        return LiveSnapshot(
            connected=False,
            map_title=prev.map_title,
            offsets=prev.offsets,
            columns=prev.columns,
            noterows=prev.noterows,
            notetypes=prev.notetypes,
            misses=prev.misses,
            keycount=prev.keycount,
        )

    def _build_snapshot(self, payload: dict) -> LiveSnapshot:
        """Fold a tosu JSON payload into a :class:`LiveSnapshot`.

        Tosu's shape is versioned (``/json/v2``); we touch only a
        small, documented subset. The structure below matches
        https://github.com/tosuapp/tosu v2 API as of 2026.
        """
        beatmap = payload.get('beatmap') or {}
        play = payload.get('play') or {}
        hits = play.get('hits') or {}
        hit_error_array = play.get('hitErrorArray') or []
        mode = (beatmap.get('mode') or {}).get('name') or ''
        title = (beatmap.get('titleUnicode')
                 or beatmap.get('title') or '')
        # tosu v2 ships stats.cs as either a scalar (older) or
        # ``{'original': N, 'converted': N}`` (current). Accept both.
        cs_raw = (beatmap.get('stats') or {}).get('cs', 4)
        if isinstance(cs_raw, dict):
            cs_raw = cs_raw.get('converted', cs_raw.get('original', 4))
        keycount = int(cs_raw) if mode == 'mania' else 4

        # Map-change detection: prefer ``checksum`` (tosu v2), fall back to
        # the older ``md5`` alias, then mapset ``id``.
        md5 = str(beatmap.get('checksum') or beatmap.get('md5')
                  or beatmap.get('id') or '')
        map_changed = md5 != self._map_md5
        if map_changed:
            self._map_md5 = md5
            self._seen_hits = 0

        # hitErrorArray grows monotonically within a single play;
        # reset when the map changes.
        new_hits = hit_error_array[self._seen_hits:] if not map_changed \
            else hit_error_array
        self._seen_hits = len(hit_error_array)

        with self._lock:
            prev = self._snapshot

        if map_changed:
            offsets = np.asarray(new_hits, dtype=np.float64) / 1000.0
            columns = (np.arange(len(new_hits), dtype=np.int32)
                       % max(1, keycount))
            noterows = np.arange(len(new_hits), dtype=np.int64)
            notetypes = np.zeros(len(new_hits), dtype=np.int32)
            misses = np.zeros(len(new_hits), dtype=bool)
        else:
            # Append. offsets from tosu are milliseconds; our viz
            # convention is seconds.
            new_offsets = np.asarray(new_hits, dtype=np.float64) / 1000.0
            offsets = np.concatenate([prev.offsets, new_offsets])
            # Tosu doesn't publish per-hit column. Best we can do for
            # v1 is round-robin the columns so hand-split viz (drift,
            # per_column) have *some* signal; replace with real column
            # data when we pick it up from a richer feed.
            n_prev = len(prev.columns)
            new_cols = np.arange(
                n_prev, n_prev + len(new_hits), dtype=np.int32) % keycount
            columns = np.concatenate([prev.columns, new_cols])
            noterows = np.concatenate([
                prev.noterows,
                np.arange(n_prev, n_prev + len(new_hits), dtype=np.int64),
            ])
            notetypes = np.concatenate([
                prev.notetypes,
                np.zeros(len(new_hits), dtype=np.int32),
            ])
            misses = np.concatenate([
                prev.misses,
                np.zeros(len(new_hits), dtype=bool),
            ])

        return LiveSnapshot(
            connected=True,
            map_title=str(title),
            combo=int(play.get('combo', {}).get('current', 0) if
                      isinstance(play.get('combo'), dict)
                      else play.get('combo', 0)),
            max_combo=int(play.get('combo', {}).get('max', 0) if
                          isinstance(play.get('combo'), dict) else 0),
            accuracy=float(play.get('accuracy', 0.0)),
            unstable_rate=float(play.get('unstableRate', 0.0)),
            hits_300=int(hits.get('300', 0) or hits.get('h300', 0)),
            hits_100=int(hits.get('100', 0) or hits.get('h100', 0)),
            hits_50=int(hits.get('50', 0) or hits.get('h50', 0)),
            hits_miss=int(hits.get('0', 0) or hits.get('h0', 0)
                          or hits.get('miss', 0)),
            offsets=offsets,
            columns=columns,
            noterows=noterows,
            notetypes=notetypes,
            misses=misses,
            keycount=keycount,
        )


def _default_fetch(url: str) -> dict:
    """Perform one GET and return parsed JSON. Short timeout so a
    wedged server doesn't stall the poller."""
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=1.0) as r:
        raw = r.read()
    return json.loads(raw.decode('utf-8'))


# ─── Module-level singleton ────────────────────────────────────────────

_singleton: TosuClient | None = None
_singleton_lock = threading.Lock()


def get_client(url: str = DEFAULT_URL,
               poll_hz: float = DEFAULT_POLL_HZ) -> TosuClient:
    """Return the shared poller. The first caller's ``url``/``poll_hz``
    wins; subsequent calls with different values are ignored (they'd
    race, and the most common use is "give me the feed" regardless).
    To reconfigure, stop the existing client first."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = TosuClient(url=url, poll_hz=poll_hz)
            _singleton.start()
        return _singleton


def reset_for_tests() -> None:
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.stop()
        _singleton = None
