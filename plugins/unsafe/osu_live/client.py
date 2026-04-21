"""Background poller for live osu! mania state.

Two data sources are supported; the poller picks the first one that
works on each tick:

1. **Native reader** (``osu_memory_native`` Rust extension). Reads
   osu!'s process memory directly via ``process_vm_readv``. No server,
   no port, ~40 µs per read on the sample hardware. Preferred.
2. **HTTP fallback** (tosu's ``/json/v2`` endpoint). Used only when the
   native extension isn't built, the osu! signatures don't resolve
   (binary update), or the user explicitly configures it. Keeps things
   working while we re-derive signatures.

Both sources are normalized to a single internal dict shape before
``_build_snapshot`` parses them, so one code path produces the
``LiveSnapshot`` that viz panels consume.

Thread safety: the poller maintains private mutable state and
publishes immutable snapshots by copying arrays into fresh NumPy
arrays under an ``RLock``. Readers never see a half-updated state.
``RLock`` (vs ``Lock``) because ``_build_snapshot`` reads the previous
snapshot under the same lock callers hold while replacing.
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
    """Immutable view of osu! state at one instant.

    ``offsets``/``columns``/``noterows``/``notetypes``/``misses`` have the
    same shapes a parsed replay dict would, so they can be fed into the
    existing viz helpers (``_common.clean_arrays`` etc.). The scalar
    fields are what osu! reports about the session as a whole."""
    connected: bool
    map_title: str = ''
    # True only when osu! reports ``GameState.play`` (active gameplay,
    # not menu / results / song select / pause). Consumers that want to
    # hide UI during non-play states gate on this.
    in_gameplay: bool = False
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


class OsuLiveClient:
    """Background poller. Call :meth:`start` to spin up the thread;
    call :meth:`snapshot` any time (cheap). :meth:`stop` joins the
    thread and is safe to call multiple times.

    ``fetch`` is injectable for tests: a callable taking the URL and
    returning a dict in the tosu ``/json/v2`` shape (so the snapshot
    builder stays one implementation). If not provided, the default
    fetcher tries the native reader first, then falls back to HTTP.
    """

    def __init__(self, url: str = DEFAULT_URL,
                 poll_hz: float = DEFAULT_POLL_HZ, *,
                 fetch=None):
        self._url = str(url)
        self._interval = 1.0 / max(1.0, float(poll_hz))
        self._fetch = fetch or _build_default_fetcher()
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
            target=self._run, name='OsuLiveClient', daemon=True)
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
        """Fold a tosu-shaped JSON payload into a :class:`LiveSnapshot`.

        The payload's top-level keys (``beatmap``, ``play``, ...) match
        tosu's ``/json/v2`` regardless of which source produced it —
        the native adapter (:func:`_native_to_payload`) rewraps its
        flat output into the same shape before handing it to us.
        """
        beatmap = payload.get('beatmap') or {}
        play = payload.get('play') or {}
        hits = play.get('hits') or {}
        hit_error_array = play.get('hitErrorArray') or []
        # in_gameplay: native source sets play.inGameplay directly; tosu
        # HTTP reports it at top-level state.number == 2 (GameState.play).
        if 'inGameplay' in play:
            in_gameplay = bool(play.get('inGameplay'))
        else:
            state = payload.get('state') or {}
            state_num = state.get('number') if isinstance(state, dict) else None
            in_gameplay = int(state_num) == 2 if state_num is not None else False
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
            # Append. offsets from osu! are milliseconds; our viz
            # convention is seconds.
            new_offsets = np.asarray(new_hits, dtype=np.float64) / 1000.0
            offsets = np.concatenate([prev.offsets, new_offsets])
            # Per-hit column data isn't exposed by either source yet.
            # Round-robin across lanes so hand-split viz (drift,
            # per_column) have *some* signal; swap in real column
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
            in_gameplay=in_gameplay,
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


# ─── Sources ──────────────────────────────────────────────────────────────

def _build_default_fetcher():
    """Return a fetcher that prefers the native reader and falls back
    to HTTP if native is unavailable or the signatures fail to resolve.

    The returned callable takes the HTTP URL (used only on fallback)
    and returns a tosu-shaped payload dict. We construct it once per
    client so the native handle is cached across ticks — signature
    resolution is a 500ms pattern scan we don't want to repeat.
    """
    try:
        import osu_memory_native  # noqa: F401
    except ImportError:
        return _http_fetch

    # Closure-local state so successive ticks reuse the resolved
    # handle. We re-resolve lazily if osu! restarted (pid mismatch)
    # or if a read raises.
    state: dict = {'handle': None, 'pid': None}

    def _native_then_http(url: str) -> dict:
        import osu_memory_native as native
        # Re-resolve on PID change (osu! restart) or first call.
        pid = native.find_osu_pid()
        if pid is None:
            # osu! not running — native is useless, try HTTP in case
            # the user has tosu pointed at a different osu! (remote
            # stream setup, tourney client, etc.).
            return _http_fetch(url)
        if state['handle'] is None or state['pid'] != pid:
            try:
                state['handle'] = native.resolve(pid)
                state['pid'] = pid
            except OSError:
                # Signatures didn't resolve — osu! binary likely
                # updated ahead of our signatures.rs. Fall back.
                state['handle'] = None
                return _http_fetch(url)
        try:
            raw = native.read_state(state['handle'])
        except OSError:
            # A pointer chain went stale — drop the handle and retry
            # next tick. Fallback keeps the viz populated meanwhile.
            state['handle'] = None
            return _http_fetch(url)
        return _native_to_payload(raw)

    return _native_then_http


def _native_to_payload(raw: dict) -> dict:
    """Rewrap the native reader's flat dict into the tosu ``/json/v2``
    shape that :meth:`_build_snapshot` expects.
    """
    return {
        'beatmap': {
            'checksum': str(raw.get('map_md5') or ''),
            'title': str(raw.get('map_title') or ''),
            'stats': {'cs': float(raw.get('map_cs', 4.0) or 4.0)},
            # Mode name is derived from the ruleset id. Mania = 3.
            'mode': {'name': _MODE_NAMES.get(int(raw.get('mode', 3) or 0), '')},
        },
        'play': {
            'hits': {
                '300': int(raw.get('hit_300', 0) or 0),
                '100': int(raw.get('hit_100', 0) or 0),
                '50': int(raw.get('hit_50', 0) or 0),
                '0': int(raw.get('hit_miss', 0) or 0),
            },
            'hitErrorArray': list(raw.get('hit_errors_ms') or []),
            'combo': {
                'current': int(raw.get('combo', 0) or 0),
                'max': int(raw.get('max_combo', 0) or 0),
            },
            'accuracy': float(raw.get('accuracy', 0.0) or 0.0),
            # Native reader doesn't surface UR yet; leave as 0 and
            # compute from hit errors if the viz needs it.
            'unstableRate': 0.0,
            'inGameplay': bool(raw.get('in_gameplay', False)),
        },
    }


_MODE_NAMES = {0: 'osu', 1: 'taiko', 2: 'fruits', 3: 'mania'}


def _http_fetch(url: str) -> dict:
    """Perform one GET and return parsed JSON. Short timeout so a
    wedged server doesn't stall the poller."""
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=1.0) as r:
        raw = r.read()
    return json.loads(raw.decode('utf-8'))


# ─── Module-level singleton ────────────────────────────────────────────

_singleton: OsuLiveClient | None = None
_singleton_lock = threading.Lock()


def get_client(url: str = DEFAULT_URL,
               poll_hz: float = DEFAULT_POLL_HZ) -> OsuLiveClient:
    """Return the shared poller. The first caller's ``url``/``poll_hz``
    wins; subsequent calls with different values are ignored (they'd
    race, and the most common use is "give me the feed" regardless).
    To reconfigure, stop the existing client first."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = OsuLiveClient(url=url, poll_hz=poll_hz)
            _singleton.start()
        return _singleton


def reset_for_tests() -> None:
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.stop()
        _singleton = None
