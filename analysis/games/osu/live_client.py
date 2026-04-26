"""Background poller for live osu! state via the native memory reader.

Reads osu!'s process memory directly via the ``osu_memory_native`` Rust
extension and publishes immutable
:class:`~analysis.components.api.GameMemoryState` snapshots at a fixed
rate. When native is unavailable (not built, osu! not running, or
signatures stale after a binary update) the poller publishes ``None``
until the next successful read.

Lives in the host so sandboxed plugins don't need ``threading`` /
``ctypes`` access just to read live game memory: the polling thread,
native reader, and snapshot lock are owned here. Plugins read the
published snapshot via ``ctx.data.game_memory()`` (overlay surface) or
:func:`analysis.components.provider.current_game_memory` (legacy
overlay callbacks pre-Component-API port).

Polling is started lazily by :func:`start_polling`; once started, both
the memory provider (raw snapshot) and the overlay-state provider
(osu-specific :class:`OverlayGameState` derivation) are installed on
:mod:`analysis.components.provider` so any overlay component on any
surface picks up live data without further wiring.

Thread safety: mutable poller state is private; the published snapshot
is replaced atomically under a lock. Readers always see a consistent
value.
"""
from __future__ import annotations

import threading
import time

from analysis.components.api import GameMemoryState
from analysis.overlay.api import (PHASE_DISCONNECTED, PHASE_IDLE,
                                  PHASE_PLAYING, OverlayGameState)


DEFAULT_POLL_HZ = 30.0


class OsuLiveClient:
    """Background poller. Call :meth:`start` to spin up the thread and
    :meth:`snapshot` to read the latest state. :meth:`stop` joins the
    thread and is safe to call multiple times."""

    def __init__(self, poll_hz: float = DEFAULT_POLL_HZ):
        self._interval = 1.0 / max(1.0, float(poll_hz))
        self._lock = threading.Lock()
        self._snapshot: GameMemoryState | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._handle = None
        self._pid: int | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name='OsuLiveClient', daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def snapshot(self) -> GameMemoryState | None:
        with self._lock:
            return self._snapshot

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            snap = self._poll()
            with self._lock:
                self._snapshot = snap
            self._stop.wait(max(0.0, self._interval - (time.monotonic() - started)))

    def _poll(self) -> GameMemoryState | None:
        try:
            import osu_memory_native as native
        except ImportError:
            return None

        pid = native.find_osu_pid()
        if pid is None:
            self._handle = None
            return None

        if self._handle is None or self._pid != pid:
            try:
                self._handle = native.resolve(pid)
                self._pid = pid
            except OSError:
                self._handle = None
                return None

        try:
            raw = native.read_state(self._handle)
        except OSError:
            self._handle = None
            return None

        return _raw_to_game_memory(raw)


def _raw_to_game_memory(raw: dict) -> GameMemoryState:
    judgment_counts = {
        '300':  int(raw.get('hit_300',  0) or 0),
        '100':  int(raw.get('hit_100',  0) or 0),
        '50':   int(raw.get('hit_50',   0) or 0),
        'miss': int(raw.get('hit_miss', 0) or 0),
        'geki': int(raw.get('hit_geki', 0) or 0),
        'katu': int(raw.get('hit_katu', 0) or 0),
    }
    chart_meta = raw.get('chart_meta') or {}
    chart_stats = raw.get('chart_stats') or {}
    md5 = chart_meta.get('md5') or raw.get('map_md5') or ''
    title = chart_meta.get('title') or raw.get('map_title') or ''
    return GameMemoryState(
        in_gameplay=bool(raw.get('in_gameplay', False)),
        combo=int(raw.get('combo', 0) or 0),
        max_combo=int(raw.get('max_combo', 0) or 0),
        accuracy=float(raw.get('accuracy', 0.0) or 0.0),
        judgment_counts=judgment_counts,
        hit_errors_ms=tuple(int(e) for e in (raw.get('hit_errors_ms') or [])),
        map_md5=str(md5),
        map_title=str(title),
        extra={
            'chart_meta': dict(chart_meta),
            'chart_stats': dict(chart_stats),
        },
    )


def snapshot_to_overlay_state(snap: GameMemoryState | None) -> OverlayGameState:
    """Translate a native memory snapshot into game-agnostic overlay
    state. Returns a disconnected state when snap is None.

    osu-specific derivation kept next to the poller so the adapter is
    self-contained: starting polling installs both providers and any
    overlay component reading either one gets fresh data."""
    if snap is None:
        return OverlayGameState(game='osu', phase=PHASE_DISCONNECTED,
                                keycount=0)

    phase = PHASE_PLAYING if snap.in_gameplay else PHASE_IDLE
    offsets = tuple(e / 1000.0 for e in snap.hit_errors_ms)
    c = snap.judgment_counts
    judgments = (
        ('300',  c.get('300', 0)),
        ('geki', c.get('geki', 0)),
        ('200',  c.get('katu', 0)),
        ('100',  c.get('100', 0)),
        ('50',   c.get('50', 0)),
        ('miss', c.get('miss', 0)),
    )
    return OverlayGameState(
        game='osu',
        phase=phase,
        song_id=snap.map_md5,
        song_title=snap.map_title,
        keycount=4,
        combo=snap.combo,
        max_combo=snap.max_combo,
        accuracy=snap.accuracy,
        judgments=judgments,
        hit_offsets_s=offsets,
        hit_lanes=(),
        pressed_lanes=(),
    )


# ── Singleton + provider wiring ─────────────────────────────────────


_singleton: OsuLiveClient | None = None
_singleton_lock = threading.Lock()


def get_client(poll_hz: float = DEFAULT_POLL_HZ) -> OsuLiveClient:
    """Return the shared poller. The first caller's ``poll_hz`` wins;
    subsequent calls with different values are ignored. To reconfigure,
    stop the existing client first."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = OsuLiveClient(poll_hz=poll_hz)
            _singleton.start()
        return _singleton


def start_polling(poll_hz: float = DEFAULT_POLL_HZ) -> OsuLiveClient:
    """Start the host-owned osu memory poller and install the overlay
    state providers so any overlay component picks up live data
    transparently. Idempotent."""
    client = get_client(poll_hz=poll_hz)
    from analysis.components.provider import (
        set_game_memory_provider, set_game_state_provider)

    def _memory_provider():
        return client.snapshot()

    def _overlay_state_provider():
        return snapshot_to_overlay_state(client.snapshot())

    set_game_memory_provider(_memory_provider)
    set_game_state_provider(_overlay_state_provider)
    return client


def reset_for_tests() -> None:
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.stop()
        _singleton = None
