"""POSIX shared-memory publisher for the gamescope external overlay.

Binary contract lives at
``analysis/games/osu/gamescope_overlay/shm_layout.h``.
The two must stay in sync; to catch drift we write a ``magic`` and
``version`` word and the consumer rejects mismatches.

Why shm rather than Unix socket?
--------------------------------
The overlay renders at 60 Hz and would otherwise have to make a
read/recv syscall every frame just to pick up values that change
~30× per second. A single mmap of /dev/shm/osu_live_overlay lets
the consumer dereference a struct pointer — zero syscalls per
frame. Writer updates are infrequent (poller is 30 Hz) and cheap
(one ``struct.pack_into``). The seqlock keeps readers from ever
seeing a torn write without needing a futex.
"""
from __future__ import annotations

import mmap
import os
import struct
import threading
import time

import numpy as np

from plugins.unsafe.osu_live.client import LiveSnapshot, get_client


# ─── Constants mirrored from shm_layout.h ────────────────────────────────

_SHM_PATH = '/dev/shm/osu_live_overlay'
_MAGIC    = 0x4F53554C  # 'OSUL'
_VERSION  = 1
_TITLE_LEN = 128
_HIST_BINS = 41    # ±100 ms in 5 ms bins

# Layout must match the C struct byte-for-byte. Little-endian '<',
# packed with explicit padding bytes.
_LAYOUT = struct.Struct(
    '<'
    'I I I I'            # magic, version, seq, _pad0
    'B B B B'            # connected, in_gameplay, keycount, _pad1
    'i i'                # combo, max_combo
    'i i i i'            # hits_300, hits_100, hits_50, hits_miss
    'f f'                # accuracy, unstable_rate
    f'{_HIST_BINS}I'     # histogram bins
    f'{_TITLE_LEN}s'     # map_title (null-padded utf-8)
)
_TOTAL_SIZE = _LAYOUT.size

# ─── Publisher ───────────────────────────────────────────────────────────


class OsuLiveShmPublisher:
    """Background thread that polls the shared ``OsuLiveClient`` and
    mirrors each snapshot into /dev/shm/osu_live_overlay.

    Call :meth:`start` once per process; the publisher survives
    plugin reloads because :func:`get_publisher` is a module-level
    singleton gated by a lock.
    """

    def __init__(self, client=None, *, publish_hz: float = 30.0):
        self._client = client or get_client()
        self._interval = 1.0 / max(1.0, publish_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._mm: mmap.mmap | None = None
        self._seq = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # /dev/shm is a tmpfs on every modern Linux, so opening a
        # regular file there gives us the same kernel-backed page
        # cache that shm_open() would. Using os.open + ftruncate
        # avoids pulling in posix_ipc as a dependency.
        fd = os.open(_SHM_PATH, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, _TOTAL_SIZE)
        self._fd = fd
        self._mm = mmap.mmap(fd, _TOTAL_SIZE,
                             flags=mmap.MAP_SHARED,
                             prot=mmap.PROT_READ | mmap.PROT_WRITE)
        # Zero the region and write the header so a consumer that
        # attaches before our first publish still sees a valid
        # magic/version (and seq=0, so it waits for the first real
        # update rather than reading garbage fields).
        self._mm[:] = b'\x00' * _TOTAL_SIZE
        struct.pack_into('<II', self._mm, 0, _MAGIC, _VERSION)

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name='OsuLiveShmPublisher', daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
            self._thread = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        # Intentionally do NOT unlink the shm segment — leave it
        # around so the overlay can come up/down independently.

    # ── Internals ────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._publish(self._client.snapshot())
            except Exception:
                # Never kill the thread on publish errors — the
                # overlay just sees stale data until we recover.
                pass
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self._interval - elapsed))

    def _publish(self, snap: LiveSnapshot) -> None:
        mm = self._mm
        if mm is None:
            return

        # Seqlock: bump to odd, write fields, bump to next even.
        # Readers retry if they see odd or if seq changed mid-read.
        self._seq = (self._seq + 1) & 0xFFFFFFFF  # odd now
        struct.pack_into('<I', mm, 8, self._seq)

        hist = _offsets_to_histogram(snap.offsets)
        title = snap.map_title.encode('utf-8')[: _TITLE_LEN - 1]

        _LAYOUT.pack_into(
            mm, 0,
            _MAGIC, _VERSION, self._seq, 0,              # header
            1 if snap.connected else 0,                  # connected
            1 if snap.in_gameplay else 0,                # in_gameplay
            int(snap.keycount) & 0xFF,                   # keycount
            0,                                           # _pad1
            int(snap.combo),
            int(snap.max_combo),
            int(snap.hits_300),
            int(snap.hits_100),
            int(snap.hits_50),
            int(snap.hits_miss),
            float(snap.accuracy),
            float(_unstable_rate_ms(snap)),
            *hist,
            title,
        )

        self._seq = (self._seq + 1) & 0xFFFFFFFF  # even now
        struct.pack_into('<I', mm, 8, self._seq)


def _offsets_to_histogram(offsets: np.ndarray) -> list[int]:
    """Bucket hit offsets (seconds) into 41 bins spanning ±100 ms.

    Returns a ``list[int]`` of length :data:`_HIST_BINS` so
    :data:`_LAYOUT` can splat it with ``*hist``. We clip out-of-range
    offsets to the edge bins so a stray late hit doesn't disappear.
    """
    bins = [0] * _HIST_BINS
    if offsets is None or len(offsets) == 0:
        return bins
    # Convert s → ms, clip to [-100, 100], map to 0..40 bin index.
    ms = np.asarray(offsets, dtype=np.float64) * 1000.0
    ms = np.clip(ms, -100.0, 100.0)
    idx = ((ms + 100.0) / 5.0).astype(np.int32)
    idx = np.clip(idx, 0, _HIST_BINS - 1)
    # np.bincount is faster than a Python loop for >~30 hits; for
    # short arrays it's the same. Always use it for a simple path.
    counts = np.bincount(idx, minlength=_HIST_BINS)
    for i in range(_HIST_BINS):
        bins[i] = int(counts[i])
    return bins


def _unstable_rate_ms(snap: LiveSnapshot) -> float:
    """UR fallback identical to overlay.py's: prefer the source
    value, else 10× stdev of the ms hit offsets."""
    if snap.unstable_rate and snap.unstable_rate > 0:
        return float(snap.unstable_rate)
    if len(snap.offsets) < 2:
        return 0.0
    ms = np.asarray(snap.offsets, dtype=np.float64) * 1000.0
    return float(10.0 * np.std(ms))


# ─── Singleton ───────────────────────────────────────────────────────────

_publisher: OsuLiveShmPublisher | None = None
_lock = threading.Lock()


def get_publisher() -> OsuLiveShmPublisher:
    """Return the process-wide publisher, starting it on first call."""
    global _publisher
    with _lock:
        if _publisher is None:
            _publisher = OsuLiveShmPublisher()
            _publisher.start()
        return _publisher
