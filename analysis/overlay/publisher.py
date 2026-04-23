"""Python client for the in-game gamescope overlay renderer.

Trusted host code that wants to draw HUD widgets over osu! (or any other
game we run through gamescope) publishes via :class:`OverlayPublisher`.
The C binary at ``analysis/games/osu/gamescope_overlay/osu_overlay``
attaches to the shm region, reads the widget array each frame, and
renders — it knows nothing about the plugin's game semantics.

Binary contract: ``analysis/overlay/widgets/overlay_shm.h``.
All types, offsets, and sizes mirror that header exactly.

Low-level usage::

    from analysis.overlay.publisher import OverlayPublisher, WHITE, BLUE

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

Plugin authors normally use the higher-level ``overlay/`` bundle role
instead. The host owns the publisher and calls a sandbox-safe draw
function with an active frame::

    from analysis.overlay.api import BLACK_DIM, WHITE

    def draw(frame):
        with frame.group('panel'):
            frame.rect('bg', 0.02, 0.02, 0.30, 0.18, color=BLACK_DIM)
            frame.text('hello', 'HELLO', 0.03, 0.04, color=WHITE)

    def register_overlay(add):
        add('Hello HUD', draw, key='hello_hud', hz=30)

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
import sys
import threading
import time
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Callable, Iterator

from analysis.overlay.api import (
    ANCHOR_BL,
    ANCHOR_BR,
    ANCHOR_C,
    ANCHOR_TL,
    ANCHOR_TR,
    BLACK_DIM,
    BLUE_ACCENT,
    HIST_BAR,
    WARN_AMBER,
    WHITE,
    rgba,
    widget_id,
)


# ── Constants mirrored from overlay_shm.h ────────────────────────────

MAGIC        = 0x56524F56
VERSION      = 2
MAX_WIDGETS  = 128
TEXT_LEN     = 48

KIND_UNUSED = 0
KIND_RECT   = 1
KIND_TEXT   = 2

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


OverlayFrame = _FrameBuilder


class OverlayPublisher:
    """Producer for one overlay feed (one shm segment).

    Callers construct, :meth:`start`, then drive with ``with
    pub.frame(): ...`` inside their own update loop (or let the
    convenience :meth:`run_thread` spin one up).
    """

    def __init__(self, plugin_key: str, *, width: int = 2560,
                 height: int = 1440,
                 config_store=None,
                 config_prefix: str | None = None,
                 shm_path: str | None = None):
        self.plugin_key   = plugin_key
        self.width        = int(width)
        self.height       = int(height)
        # Single global shm by default — every plugin in the live
        # session draws into the same segment. Tests may pass a
        # per-test path to keep parallel runs from racing on it.
        self._shm_path    = shm_path or '/dev/shm/vsrg_overlay'
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
        if sys.platform == 'win32':
            tagname = os.path.basename(self._shm_path)
            self._fd = None
            self._mm = mmap.mmap(-1, _TOTAL_SIZE, tagname=tagname)
        else:
            fd = os.open(self._shm_path, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(fd, _TOTAL_SIZE)
            self._fd = fd
            self._mm = mmap.mmap(fd, _TOTAL_SIZE,
                                 flags=mmap.MAP_SHARED,
                                 prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self._mm[:] = b'\x00' * _TOTAL_SIZE
        struct.pack_into('<II', self._mm, 0, MAGIC, VERSION)
        self._load_deltas_from_config()

    def stop(self) -> None:
        if self._mm is not None:
            # Leave the feed in an empty-but-valid state so an attached
            # renderer clears any previously drawn widgets after the
            # publisher stops.
            self._commit([])
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
                    print(f'[overlay.publisher] build_fn raised: {exc}')
                elapsed = time.monotonic() - t0
                stop.wait(max(0.0, interval - elapsed))

        t = threading.Thread(target=_loop, name=name, daemon=True)
        t._stop_event = stop  # type: ignore[attr-defined]
        t.start()
        return t

    def run_draw_thread(self, draw_fn, *, hz: float = 30.0,
                        name: str = 'OverlayPublisher') -> threading.Thread:
        """Run a higher-level overlay draw function on a daemon thread.

        ``draw_fn(frame)`` receives an active :class:`OverlayFrame`; the
        publisher owns the frame context and atomic commit. This mirrors
        sidebar plugins more closely than :meth:`run_thread`, where the
        caller has to open ``with pub.frame()`` manually.
        """

        def _build(pub: 'OverlayPublisher') -> None:
            with pub.frame() as frame:
                draw_fn(frame)

        return self.run_thread(_build, hz=hz, name=name)


# Session-wide overlay refresh rate. Hardcoded — every spec drew at
# its own ``hz`` when each lived in its own publisher; now they share
# one draw thread, so the slowest plugin doesn't slow down the rest
# and the fastest doesn't oversample everyone else.
SESSION_HZ = 30.0

# Stable key for the single session publisher. Used as the config
# bucket prefix for persisted layouts.
SESSION_KEY = 'vsrg'


@dataclass
class OverlaySpec:
    key: str
    name: str
    draw: Callable[[OverlayFrame], None]
    hz: float = SESSION_HZ  # kept on the spec for back-compat; ignored at draw
    width: int = 2560
    height: int = 1440
    module: str = ''
    enabled: bool = True

    @property
    def feed_path(self) -> str:
        return '/dev/shm/vsrg_overlay'


class OverlayRegistry:
    """Registry for ``overlay/*.py`` plugin modules and unified components.

    Overlay modules expose ``register_overlay(add)`` and call
    ``add(name, draw_fn, key=..., hz=...)``. Unified components are
    bridged in via :func:`analysis.components.registry.bridge_into_overlay_registry`.

    All enabled specs share one :class:`OverlayPublisher` and one render
    thread. Each tick: open one frame, call every enabled spec's
    ``draw(frame)`` in registration order, commit. A spec that raises is
    caught and latched-off so a buggy plugin can't blank the overlay.
    """

    def __init__(self, config=None):
        from analysis.config import get_config
        self._config = config if config is not None else get_config()
        self._overlays: list[OverlaySpec] = []
        self._runtime: tuple[OverlayPublisher, threading.Thread] | None = None
        self._last_start: tuple[int | None, int | None, object] | None = None
        # Specs whose draw raised this session — kept off until restart
        # or until the user re-enables them through the dialog (mirrors
        # PluginManager._runtime_disabled).
        self._runtime_disabled: set[str] = set()
        # Diagnostic: print the spec list once on the first session tick
        # so logs reveal whether the bridge actually registered the
        # unified-component spec the user is hunting for.
        self._logged_first_tick = False
        self._config_sub = self._config.subscribe(
            'plugins', self._on_config_change)

    def add(self, name, draw, *, key=None, hz: float = SESSION_HZ,
            width: int = 2560, height: int = 1440, enabled: bool = True,
            module: str = '') -> None:
        spec_key = str(key) if key is not None else _default_overlay_key(
            module, name)
        self._overlays.append(OverlaySpec(
            key=spec_key,
            name=str(name),
            draw=draw,
            hz=float(hz),
            width=int(width),
            height=int(height),
            module=str(module),
            enabled=bool(enabled) and not self._is_disabled(spec_key),
        ))
        self._overlays.sort(key=lambda o: (o.name, o.key))

    def all_overlays(self) -> list[OverlaySpec]:
        return list(self._overlays)

    def get(self, key: str) -> OverlaySpec | None:
        key = str(key)
        for spec in self._overlays:
            if spec.key == key:
                return spec
        return None

    def start(self, key: str | None = None, *, width: int | None = None,
              height: int | None = None,
              config_store=None) -> OverlayPublisher:
        """Start the session publisher.

        ``key`` is accepted for back-compat but ignored — there is one
        publisher per session, not per plugin. Repeat calls return the
        same publisher.
        """
        if self._runtime is not None:
            return self._runtime[0]
        # Pick the canvas size from the first spec that declared one,
        # or the explicit override. All plugins share the same canvas
        # because they share the same shm.
        canvas_w = int(width) if width is not None else (
            self._overlays[0].width if self._overlays else 2560)
        canvas_h = int(height) if height is not None else (
            self._overlays[0].height if self._overlays else 1440)
        self._last_start = (width, height, config_store)
        pub = OverlayPublisher(
            SESSION_KEY,
            width=canvas_w,
            height=canvas_h,
            config_store=self._config if config_store is None else config_store,
        )
        pub.start()
        thread = pub.run_draw_thread(
            self._draw_session, hz=SESSION_HZ,
            name=f'OverlayPublisher:{SESSION_KEY}')
        self._runtime = (pub, thread)
        return pub

    def _draw_session(self, frame: OverlayFrame) -> None:
        """Render one frame across every enabled spec.

        Each spec's draw runs inside its own try/except so a single
        misbehaving plugin can't take the whole overlay down. A spec
        that raises is latched off for the rest of the session — same
        policy as :class:`PluginManager` for replay plugins.
        """
        from analysis import diag
        if not self._logged_first_tick:
            self._logged_first_tick = True
            diag.log('overlay.registry',
                     f'first session tick — specs: '
                     f'{[(s.key, s.enabled) for s in self._overlays]}')
        for spec in list(self._overlays):
            if not spec.enabled or spec.key in self._runtime_disabled:
                continue
            try:
                spec.draw(frame)
            except Exception as exc:
                import traceback
                self._runtime_disabled.add(spec.key)
                src = f' ({spec.module})' if spec.module else ''
                diag.log('overlay.registry',
                         f'plugin disabled: {spec.name}{src}: {exc}\n'
                         + traceback.format_exc())

    def stop(self, key: str | None = None) -> bool:
        """Stop the session publisher. ``key`` ignored (back-compat)."""
        if self._runtime is None:
            return False
        pub, thread = self._runtime
        self._runtime = None
        stop = getattr(thread, '_stop_event', None)
        if stop is not None:
            stop.set()
        if thread.is_alive():
            thread.join(timeout=1.0)
        pub.stop()
        return True

    def stop_all(self) -> None:
        self.stop()

    def set_enabled(self, key: str, enabled: bool) -> bool:
        key = str(key)
        if not any(o.key == key for o in self._overlays):
            return False
        if enabled and key in self._runtime_disabled:
            # User explicitly re-enabled — give it another shot.
            self._runtime_disabled.discard(key)
        self._config.set(
            f'plugins.{_escape_config_key(key)}.overlay_disabled',
            not bool(enabled))
        for spec in self._overlays:
            if spec.key == key:
                spec.enabled = bool(enabled)
        # Note: we do NOT stop the session when one spec is disabled —
        # other plugins keep drawing. The disabled spec is just skipped
        # in _draw_session next tick.
        return True

    def close(self) -> None:
        if self._config_sub is not None:
            self._config.unsubscribe(self._config_sub)
            self._config_sub = None
        self.stop()

    def _is_disabled(self, key: str) -> bool:
        return bool(self._config.get(
            f'plugins.{_escape_config_key(key)}.overlay_disabled', False))

    def _on_config_change(self, path, old, new) -> None:
        if len(path) < 3 or path[-1] != 'overlay_disabled':
            return
        escaped = path[1]
        disabled = bool(new) if new is not None else False
        for spec in self._overlays:
            if _escape_config_key(spec.key) != escaped:
                continue
            spec.enabled = not disabled
            if not disabled:
                # Re-enabling clears the runtime-fail latch; the next
                # tick will try the spec again.
                self._runtime_disabled.discard(spec.key)
            break


def discover_overlays(extra_paths=None, config=None) -> OverlayRegistry:
    """Discover every bundle ``overlay/`` module into a fresh registry.

    In addition to the legacy ``register_overlay(add)`` entry point,
    this also bridges in any unified components (see
    ``analysis/components``) whose manifest lists ``overlay`` as a
    supported surface and whose ``requires_data`` is satisfied by
    :class:`~analysis.components.overlay_backend.OverlayGameStateDataSource`.
    """
    from analysis.plugins import discover_bundles
    registry = OverlayRegistry(config=config)
    bundles = list(discover_bundles(extra_paths))
    for bundle in bundles:
        for mod in getattr(bundle, 'overlay_modules', []) or []:
            if not hasattr(mod, 'register_overlay'):
                continue
            module_name = f'{bundle.key}/{getattr(mod, "__name__", "")}'

            def add_overlay(name, draw, *, key=None, hz=30.0,
                            width=2560, height=1440, enabled=True,
                            _module=module_name):
                registry.add(name, draw, key=key, hz=hz, width=width,
                             height=height, enabled=enabled, module=_module)
            try:
                mod.register_overlay(add_overlay)
            except Exception as exc:
                print(f'overlay plugin register failed: {module_name}: {exc}')
    # Bridge unified components targeting the overlay surface.
    try:
        from analysis.components.registry import (
            bridge_into_overlay_registry,
            discover_from_bundles,
        )
        components = discover_from_bundles(bundles)
        bridge_into_overlay_registry(components, registry)
    except Exception as exc:
        print(f'component→overlay bridge failed: {exc}')
    return registry


def all_overlays(extra_paths=None) -> list[OverlaySpec]:
    return discover_overlays(extra_paths).all_overlays()


def _default_overlay_key(module: str, name: str) -> str:
    raw = (f'{module}:{name}' if module else str(name)).replace('/', ':')
    chars = []
    for ch in raw.lower():
        if ch.isalnum() or ch in ('_', '-', ':'):
            chars.append(ch)
        else:
            chars.append('_')
    key = ''.join(chars).strip('_')
    return key or f'overlay_{widget_id(raw):08x}'


def _escape_config_key(key: str) -> str:
    return str(key).replace('.', '_')
