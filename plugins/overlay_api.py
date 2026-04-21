"""Python client for the in-game gamescope overlay renderer.

Any plugin that wants to draw HUD widgets over osu! (or any other
game we run through gamescope) publishes via :class:`OverlayPublisher`.
The C binary at ``analysis/games/osu/gamescope_overlay/osu_overlay``
attaches to the shm region, reads the widget array each frame, and
renders — it knows nothing about the plugin's game semantics.

Binary contract: ``analysis/games/osu/gamescope_overlay/overlay_shm.h``.
All types, offsets, and sizes mirror that header exactly.

Usage::

    from plugins.overlay_api import OverlayPublisher, WHITE, BLUE

    pub = OverlayPublisher('osu_mania', width=2560, height=1440)
    pub.start()
    while playing:
        with pub.frame() as f:
            with f.group('panel'):
                f.rect('bg', 0.02, 0.02, 0.30, 0.18,
                       color=(10, 10, 15, 140))
                f.text('combo', f'{combo}X', 0.03, 0.04,
                       px_scale=2.5, color=WHITE)
        time.sleep(1 / 30)

Positions are normalized ``[0, 1]``. Each widget has a stable string
``id``; the publisher hashes it to a 32-bit int that the renderer uses
as the widget's identity in edit mode. Drag persistence is stored as a
single ``(dx, dy)`` *delta* per group (standalone widgets use their
own id as a singleton group key), so dragging the composite HUD moves
every child without needing to track each child's absolute position.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import struct
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


# ── Constants mirrored from overlay_shm.h ────────────────────────────

MAGIC        = 0x56524F56
VERSION      = 2
MAX_WIDGETS  = 32
TEXT_LEN     = 48

KIND_UNUSED = 0
KIND_RECT   = 1
KIND_TEXT   = 2

ANCHOR_TL = 0
ANCHOR_TR = 1
ANCHOR_BL = 2
ANCHOR_BR = 3
ANCHOR_C  = 4

# Widget struct: kind B, anchor B, _pad0 2s, widget_id I, group_id I,
#                x f, y f, w f, h f, color I, text 48s, px_scale f
_WIDGET_FMT = f'<B B 2s I I f f f f I {TEXT_LEN}s f'
_WIDGET_STRUCT = struct.Struct(_WIDGET_FMT)
_WIDGET_SIZE = _WIDGET_STRUCT.size

# Header: magic I, version I, seq I, n_widgets I,
#         edit_mode B, _pad0 3s, dragged_widget_id I, dragged_seq I, _pad1 I
_HEADER_STRUCT = struct.Struct('<I I I I B 3s I I I')
_HEADER_SIZE   = _HEADER_STRUCT.size

_TOTAL_SIZE = _HEADER_SIZE + MAX_WIDGETS * _WIDGET_SIZE


# ── Color helpers ────────────────────────────────────────────────────


def rgba(r: int, g: int, b: int, a: int = 255) -> int:
    """Pack four 0..255 components into the layout the C side reads
    (byte 0 = R, byte 3 = A).

    We store as a little-endian uint32 so ``color & 0xff`` is R, which
    matches how the C side decodes it via ``(c >> 0) & 0xff``."""
    return ((r & 0xff)
            | ((g & 0xff) << 8)
            | ((b & 0xff) << 16)
            | ((a & 0xff) << 24))


WHITE       = rgba(250, 250, 250)
BLACK_DIM   = rgba(10, 10, 15, 140)
BLUE_ACCENT = rgba(75, 164, 255, 230)
WARN_AMBER  = rgba(255, 180, 50)
HIST_BAR    = rgba(75, 164, 255, 230)


def widget_id(name: str) -> int:
    """FNV-1a 32-bit of ``name``. Stable across restarts, used as the
    key for persisting per-widget (or per-group) drag offsets."""
    h = 0x811c9dc5
    for b in name.encode('utf-8'):
        h ^= b
        h = (h * 0x01000193) & 0xffffffff
    # widget_id 0 is reserved (shm uses it as "no drag"), so bump
    # off-zero in the vanishingly unlikely case FNV hit exactly 0.
    return h or 0x811c9dc5


# ── Publisher ────────────────────────────────────────────────────────


class _FrameBuilder:
    """Temporary collector for widgets within one ``with pub.frame():``
    block. Writes happen on ``__exit__`` under the publisher's seqlock.

    Grouping: use ``with f.group('panel'): ...`` to stamp every widget
    emitted inside with the same ``group_id``. Grouped widgets drag
    together and share one persisted ``(dx, dy)`` offset — the user
    perceives them as one HUD element.
    """

    def __init__(self, publisher: 'OverlayPublisher'):
        self._pub = publisher
        self._widgets: list[tuple] = []
        self._group_stack: list[int] = [0]  # 0 = no group (standalone)

    @contextmanager
    def group(self, name: str) -> Iterator[None]:
        """Stamp every widget emitted inside this block with
        ``widget_id(name)`` as their ``group_id``. Drags hit-test on
        any widget in the group and move the whole group together."""
        gid = widget_id(name)
        self._group_stack.append(gid)
        try:
            yield
        finally:
            self._group_stack.pop()

    def _record(self, kind: int, id_str: str, x: float, y: float,
                w: float, h: float, color: int, text: str,
                px_scale: float, anchor: int) -> None:
        wid = widget_id(id_str)
        gid = self._group_stack[-1]
        # Remember the publisher's *baseline* (x, y) for this widget so
        # the drag-detect pass can compute (dx, dy) = (shm_x − baseline)
        # when the C side moves it.
        self._pub._remember_baseline(wid, float(x), float(y))
        # Mid-drag: if this widget belongs to the group the user is
        # dragging, the C side's shm (x, y) is fresher than anything
        # we could compute from baseline + persisted delta. Read it
        # back so our commit doesn't snap the widget to its old spot
        # every frame. On drag release, drag_active clears and the
        # normal baseline+delta path takes over — with the now-saved
        # delta reflecting the drop location.
        live_xy = self._pub._live_drag_xy(wid, gid)
        if live_xy is not None:
            x, y = live_xy
        else:
            delta_key = gid if gid != 0 else wid
            dx, dy = self._pub._group_delta(delta_key)
            x = float(x) + dx
            y = float(y) + dy
        text_bytes = text.encode('utf-8')[: TEXT_LEN - 1]
        self._widgets.append((
            int(kind) & 0xff,
            int(anchor) & 0xff,
            b'\x00\x00',
            int(wid) & 0xffffffff,
            int(gid) & 0xffffffff,
            float(x), float(y), float(w), float(h),
            int(color) & 0xffffffff,
            text_bytes + b'\x00' * (TEXT_LEN - len(text_bytes)),
            float(px_scale),
        ))

    def rect(self, id_str: str, x: float, y: float, w: float, h: float,
             *, color: int = BLACK_DIM, anchor: int = ANCHOR_TL) -> None:
        self._record(KIND_RECT, id_str, x, y, w, h, color, '',
                     1.0, anchor)

    def text(self, id_str: str, s: str, x: float, y: float,
             *, px_scale: float = 2.0, color: int = WHITE,
             anchor: int = ANCHOR_TL) -> None:
        # w/h are unused for text widgets — the renderer derives width
        # from string length * px_scale * 9 (glyph + kerning) so drag
        # hit-testing on the C side has a meaningful bounding box.
        self._record(KIND_TEXT, id_str, x, y, 0.0, 0.0, color, s,
                     px_scale, anchor)


class OverlayPublisher:
    """Producer for one overlay feed (one shm segment).

    Callers construct, :meth:`start`, then drive with ``with
    pub.frame(): ...`` inside their own update loop (or let the
    convenience :meth:`run_thread` spin one up).
    """

    def __init__(self, plugin_key: str, *, width: int = 2560,
                 height: int = 1440,
                 config_store=None,
                 config_prefix: str | None = None):
        self.plugin_key   = plugin_key
        self.width        = int(width)
        self.height       = int(height)
        self._shm_path    = f'/dev/shm/vsrg_overlay_{plugin_key}'
        self._fd: int | None = None
        self._mm: mmap.mmap | None = None
        self._seq         = 0
        self._lock        = threading.Lock()

        # Per-group drag offset. Key is group_id (or widget_id for
        # standalone widgets, which are treated as a singleton group
        # keyed by their own id). Value is (dx, dy) in normalized units.
        self._group_deltas: dict[int, tuple[float, float]] = {}
        # Baseline (pre-delta) normalized (x, y) we fed _record for each
        # widget this session. Used to back out the delta the C side
        # applied when it dragged a widget.
        self._baselines: dict[int, tuple[float, float]] = {}
        # widget_id -> group_id last seen, so _poll_drag_updates can
        # decide which delta bucket to persist into. Populated when the
        # Python side records a widget.
        self._widget_group: dict[int, int] = {}

        self._last_dragged_seq = 0
        # While a drag is in progress on the C side, these track
        # which group (or which widget, for standalone) must not be
        # overwritten by our commit path. None = no active drag.
        self._drag_active_group: int | None = None
        self._drag_active_widget: int = 0

        # Config persistence wiring. Pass a ConfigStore (typically
        # the app-wide singleton) to persist drag offsets across
        # runs. If None, drag still works at runtime but offsets
        # reset when the publisher restarts.
        self._cfg = config_store
        self._cfg_prefix = config_prefix or f'overlay.layouts.{plugin_key}'

        # Which (width × height) bucket inside the config we read
        # from / write to. Per-resolution since the user's ideal
        # layout at 1080p differs from 1440p.
        self._res_key = f'{self.width}x{self.height}'

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        if self._mm is not None:
            return
        fd = os.open(self._shm_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, _TOTAL_SIZE)
        self._fd = fd
        self._mm = mmap.mmap(fd, _TOTAL_SIZE,
                             flags=mmap.MAP_SHARED,
                             prot=mmap.PROT_READ | mmap.PROT_WRITE)
        # Zero + write magic/version so a consumer that attaches
        # before our first frame doesn't read garbage.
        self._mm[:] = b'\x00' * _TOTAL_SIZE
        struct.pack_into('<II', self._mm, 0, MAGIC, VERSION)

        self._load_deltas_from_config()

    def stop(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    # ── Frame context manager ──────────────────────────────────────

    @contextmanager
    def frame(self) -> Iterator[_FrameBuilder]:
        """Open a drawing frame. Widgets added inside the ``with``
        block are committed atomically on exit."""
        builder = _FrameBuilder(self)
        # Before the caller builds, refresh our view of C-side drags
        # so the new baseline + delta math reflects the latest edits.
        self._poll_drag_updates()
        yield builder
        self._commit(builder._widgets)

    # Called from _FrameBuilder._record.
    def _remember_baseline(self, wid: int, x: float, y: float) -> None:
        self._baselines[wid] = (float(x), float(y))

    def _group_delta(self, key: int) -> tuple[float, float]:
        return self._group_deltas.get(key, (0.0, 0.0))

    def _live_drag_xy(self, wid: int, gid: int) -> tuple[float, float] | None:
        """If a drag is currently in progress and this widget is part
        of the dragged group, return its current shm (x, y). Returns
        ``None`` otherwise, meaning the caller should use the normal
        baseline+delta path.

        Why: while the user is holding the mouse button, the C side
        owns every widget's position in the dragged group. If we
        wrote baseline+delta during that window, each frame would
        flash the widget back to its pre-drag spot.
        """
        active_gid = self._drag_active_group
        if active_gid is None:
            return None
        # gid==0 means "standalone"; the drag only owns that exact
        # widget's position, not siblings.
        if active_gid == 0:
            if wid != self._drag_active_widget:
                return None
        else:
            if gid != active_gid:
                return None
        mm = self._mm
        if mm is None:
            return None
        # Find the slot by wid. We can't cache a slot index because
        # the publisher may have re-ordered widgets between commits.
        for i in range(MAX_WIDGETS):
            off = _HEADER_SIZE + i * _WIDGET_SIZE
            slot_wid = struct.unpack_from('<I', mm, off + 4)[0]
            if slot_wid == wid:
                x, y = struct.unpack_from('<ff', mm, off + 12)
                return float(x), float(y)
        return None

    # ── Commit path ────────────────────────────────────────────────

    def _commit(self, widgets: list[tuple]) -> None:
        mm = self._mm
        if mm is None:
            return
        # Defensive clamp so a buggy caller that emits >MAX_WIDGETS
        # never overruns the shm region.
        n = max(0, min(len(widgets), MAX_WIDGETS))

        with self._lock:
            # Seqlock: bump to odd, write, bump to even.
            self._seq = (self._seq + 1) & 0xffffffff
            struct.pack_into('<I', mm, 8, self._seq)

            # Preserve C-owned header fields (edit_mode, dragged_*).
            # We only own magic/version/seq/n_widgets on this side.
            struct.pack_into('<II', mm, 0, MAGIC, VERSION)
            struct.pack_into('<I', mm, 12, n)

            for i in range(MAX_WIDGETS):
                off = _HEADER_SIZE + i * _WIDGET_SIZE
                if i < n:
                    _WIDGET_STRUCT.pack_into(mm, off, *widgets[i])
                    wid = widgets[i][3]
                    gid = widgets[i][4]
                    self._widget_group[wid] = gid
                else:
                    # Clear stale slots fully — otherwise a deleted
                    # widget would linger until the region is zeroed.
                    ctypes.memset(
                        ctypes.addressof(ctypes.c_char.from_buffer(mm, off)),
                        0, _WIDGET_SIZE)

            self._seq = (self._seq + 1) & 0xffffffff
            struct.pack_into('<I', mm, 8, self._seq)

    # ── Drag updates: C → us ───────────────────────────────────────

    def _poll_drag_updates(self) -> None:
        """Read the C side's drag state at the start of each frame.

        Two things happen here:

        1. **Live drag snapshot.** If ``drag_active`` is set, record
           which group (or which widget, for standalone) is being
           dragged so ``_record`` knows to leave those widgets'
           shm-owned positions alone during this frame's commit.

        2. **Release-time persistence.** The C side bumps
           ``dragged_seq`` once per drag, on ButtonRelease. When we
           observe the bump we walk the widget array, compute the
           final delta against the widget's baseline, and persist
           it. The drag is done — future frames go back to writing
           ``baseline + delta`` as usual.

        Header layout (see overlay_shm.h):
          0  magic  4  version  8  seq  12 n_widgets
          16 edit_mode 17 drag_active 20 dragged_widget_id 24 dragged_seq
        """
        mm = self._mm
        if mm is None:
            self._drag_active_group = None
            return

        drag_active = mm[17]
        dragged_widget_id = struct.unpack_from('<I', mm, 20)[0]
        dragged_seq = struct.unpack_from('<I', mm, 24)[0]

        if drag_active:
            # Find the group of the widget being dragged so every
            # sibling in the group dodges the overwrite in _record.
            gid = self._widget_group.get(dragged_widget_id, 0)
            self._drag_active_group = gid
            self._drag_active_widget = dragged_widget_id
        else:
            self._drag_active_group = None
            self._drag_active_widget = 0

        if dragged_seq == self._last_dragged_seq:
            return
        self._last_dragged_seq = dragged_seq

        # dragged_seq just advanced — a drag just ended. Capture the
        # final (x, y) for every widget that was moved, translate to
        # group-keyed deltas, and persist.
        for i in range(MAX_WIDGETS):
            off = _HEADER_SIZE + i * _WIDGET_SIZE
            try:
                slot = _WIDGET_STRUCT.unpack_from(mm, off)
            except struct.error:
                continue
            (kind, _anchor, _pad, wid, gid, x, y, _w, _h,
             _color, _text, _px) = slot
            if kind == KIND_UNUSED or wid == 0:
                continue
            baseline = self._baselines.get(wid)
            if baseline is None:
                continue
            delta_key = gid if gid != 0 else wid
            new_dx = float(x) - baseline[0]
            new_dy = float(y) - baseline[1]
            cur_dx, cur_dy = self._group_deltas.get(delta_key, (0.0, 0.0))
            if abs(new_dx - cur_dx) < 1e-5 and abs(new_dy - cur_dy) < 1e-5:
                continue
            self._group_deltas[delta_key] = (new_dx, new_dy)
            self._save_delta(delta_key, new_dx, new_dy, grouped=(gid != 0))

    # ── Config persistence ─────────────────────────────────────────

    def _load_deltas_from_config(self) -> None:
        if self._cfg is None:
            return
        # Group-keyed deltas (composite HUD pieces).
        groups = self._cfg.get(f'{self._cfg_prefix}.{self._res_key}.groups', {})
        if isinstance(groups, dict):
            for gid_str, delta in groups.items():
                try:
                    gid = int(gid_str)
                    dx = float(delta['dx'])
                    dy = float(delta['dy'])
                except (TypeError, ValueError, KeyError):
                    continue
                self._group_deltas[gid] = (dx, dy)
        # Per-widget deltas (standalone widgets — group_id == 0 on
        # the wire, but persisted under the widget's own id).
        widgets = self._cfg.get(f'{self._cfg_prefix}.{self._res_key}.widgets', {})
        if isinstance(widgets, dict):
            for wid_str, delta in widgets.items():
                try:
                    wid = int(wid_str)
                    dx = float(delta['dx'])
                    dy = float(delta['dy'])
                except (TypeError, ValueError, KeyError):
                    continue
                self._group_deltas[wid] = (dx, dy)

    def _save_delta(self, key: int, dx: float, dy: float,
                    *, grouped: bool) -> None:
        if self._cfg is None:
            return
        bucket = 'groups' if grouped else 'widgets'
        # JSON object keys must be strings — stringify the id.
        path = f'{self._cfg_prefix}.{self._res_key}.{bucket}.{key}'
        self._cfg.set(path, {'dx': float(dx), 'dy': float(dy)})

    # ── Convenience: run on a background thread ────────────────────

    def run_thread(self, build_fn, *, hz: float = 30.0,
                   name: str = 'OverlayPublisher') -> threading.Thread:
        """Spin up a daemon thread that calls ``build_fn(self)`` at
        ``hz``. ``build_fn`` is expected to use ``with self.frame():
        ...`` to emit widgets.
        """
        interval = 1.0 / max(1.0, hz)
        stop = threading.Event()

        def _loop():
            while not stop.is_set():
                t0 = time.monotonic()
                try:
                    build_fn(self)
                except Exception as exc:
                    # Publishers should not die silently; log but
                    # keep looping so a transient bug doesn't kill
                    # the overlay forever.
                    print(f'[overlay_api] build_fn raised: {exc}')
                elapsed = time.monotonic() - t0
                stop.wait(max(0.0, interval - elapsed))

        t = threading.Thread(target=_loop, name=name, daemon=True)
        t._stop_event = stop  # type: ignore[attr-defined]
        t.start()
        return t
