"""Memory-safety tests for :mod:`analysis.overlay.publisher`.

The publisher writes into a fixed-size shared-memory region whose
layout is pinned by ``overlay_shm.h``. Any mismatch between the
struct format and the region size, or any off-by-one in slot
indexing, would corrupt adjacent slots ; and because the consumer is
a C binary, a miscompile shows up as a renderer crash rather than a
Python exception. So: pin every boundary as a test.

Covered:

* Struct sizes agree with the header's declared offsets (header + 32
  widget slots fit exactly in the region we mmap).
* Commit never writes past slot N-1, even when callers emit more
  widgets than ``MAX_WIDGETS``.
* ``_commit`` clears stale slots back to zero, so a shrinking frame
  can't leak the tail of a previous frame.
* Oversized text payloads are truncated, not overrun.
* Seqlock invariant: readers never see odd ``seq`` after a commit.
* Round-trip: widgets the publisher wrote re-parse to the same
  tuple the publisher computed internally.
* Group-delta persistence is keyed correctly (group-id bucket for
  grouped widgets, widget-id bucket for standalone).
* Polling a drag update that moves a widget produces the expected
  delta, not an absolute-position read.
"""
from __future__ import annotations

import os
import struct
import sys
import textwrap
import uuid

import pytest

# Every test in this module creates a /dev/shm segment via OverlayPublisher.
# On Windows the publisher uses a named mapping instead, exercised by the
# parallel test_win_overlay_api.py suite.
pytestmark = pytest.mark.skipif(
    sys.platform == 'win32',
    reason='uses /dev/shm; Windows publisher covered by test_win_overlay_api',
)

from analysis.overlay.publisher import (
    _HEADER_SIZE,
    _HEADER_STRUCT,
    _TOTAL_SIZE,
    _WIDGET_FMT,
    _WIDGET_SIZE,
    _WIDGET_STRUCT,
    ANCHOR_TL,
    BLACK_DIM,
    KIND_RECT,
    KIND_TEXT,
    KIND_WEB_TEXTURE,
    KIND_UNUSED,
    MAGIC,
    MAX_WIDGETS,
    TEXT_LEN,
    VERSION,
    WHITE,
    OverlayPublisher,
    discover_overlays,
    rgba,
    widget_id,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    """A publisher using an isolated /dev/shm path. Each test gets a
    unique shm_path so parallel test runs don't collide on the
    session-wide ``/dev/shm/vsrg_overlay`` segment."""
    key = f'test_overlay_{uuid.uuid4().hex[:12]}'
    shm_path = f'/dev/shm/vsrg_overlay_{key}'
    pub = OverlayPublisher(key, width=1920, height=1080, shm_path=shm_path)
    pub.start()
    try:
        yield pub
    finally:
        pub.stop()
        # /dev/shm is a tmpfs; clean up the backing file so repeated
        # test runs don't leave a puddle of stale segments behind.
        try:
            os.unlink(shm_path)
        except FileNotFoundError:
            pass


class _MemStore:
    """Tiny in-memory stand-in for ConfigStore. Mirrors the real
    store's dotted-path semantics (``get('a.b')`` reads the ``b``
    child of ``a``) so publisher-side lookups that read an entire
    subtree work here too."""

    def __init__(self):
        self._tree: dict = {}

    @staticmethod
    def _split(path: str) -> list[str]:
        return [p for p in path.split('.') if p]

    def get(self, path, default=None):
        node = self._tree
        for part in self._split(path):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path, value):
        parts = self._split(path)
        node = self._tree
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = value
        return True


# ── Struct/layout invariants ─────────────────────────────────────────


def test_widget_struct_size_matches_header_contract():
    # overlay_shm.h v3: 1(kind)+1(anchor)+2(pad)+4(wid)+4(gid)
    # +4(x)+4(y)+4(w)+4(h)+4(color)+48(text)+4(px_scale)
    # +4(channel_id)+4(generation) = 92 bytes.
    # If the Python format ever drifts from the C layout, commits
    # would write one offset low/high and corrupt the next slot.
    # The canonical value was captured by running a tiny C program
    # against overlay_shm.h; it reports sizeof(VsrgOverlayWidget)==92.
    assert _WIDGET_SIZE == 92, (
        f'_WIDGET_SIZE is {_WIDGET_SIZE}, expected 92 to match '
        f'VsrgOverlayWidget in overlay_shm.h')

    # Same check for the header. Mismatch here = first widget slot
    # lands at the wrong offset, corrupting every subsequent slot.
    assert _HEADER_SIZE == 32, (
        f'_HEADER_SIZE is {_HEADER_SIZE}, expected 32 to match '
        f'VsrgOverlayShm header in overlay_shm.h')


def test_header_plus_widgets_fills_total_region():
    # _TOTAL_SIZE is what the publisher ftruncate()s the shm to.
    # If this calculation is wrong, the last slot's bytes land
    # outside the mmap'd region and commit_into raises.
    assert _TOTAL_SIZE == _HEADER_SIZE + MAX_WIDGETS * _WIDGET_SIZE


def test_struct_format_strings_parse():
    # Pure sanity: struct.Struct would have thrown at import time
    # already, but make it an assertion so a format-string regression
    # gets a descriptive failure.
    assert _WIDGET_STRUCT.format == _WIDGET_FMT


# ── Commit bounds & slot clearing ────────────────────────────────────


def test_commit_caps_at_max_widgets(publisher):
    # Emit twice as many widgets as the region holds. The extras
    # must not be written beyond the slot array ; worst case would
    # be mmap.mmap.__setitem__ raising, better case we silently
    # truncate. Either way, no segfault, no past-end write.
    with publisher.frame() as f:
        for i in range(MAX_WIDGETS * 2):
            f.rect(f'r{i}', 0.1, 0.1, 0.05, 0.05, color=BLACK_DIM)

    mm = publisher._mm
    n = struct.unpack_from('<I', mm, 12)[0]
    assert n == MAX_WIDGETS, (
        f'n_widgets written as {n}, expected clamp to {MAX_WIDGETS}')

    # The byte one past the region end must not be writable via the
    # publisher's commit path ; it doesn't exist. Access here is
    # just to confirm the region is exactly _TOTAL_SIZE long.
    assert len(mm) == _TOTAL_SIZE


def test_commit_clears_stale_slots(publisher):
    # Frame 1 fills several slots with a recognizable color.
    with publisher.frame() as f:
        for i in range(5):
            f.rect(f'frame1_r{i}', 0.1, 0.1, 0.05, 0.05,
                   color=rgba(123, 45, 67, 200))

    # Frame 2 emits only one widget. The other four slots from
    # frame 1 must be zeroed, not left with stale data ; otherwise
    # the renderer would draw phantom widgets.
    with publisher.frame() as f:
        f.rect('frame2_only', 0.2, 0.2, 0.1, 0.1, color=WHITE)

    mm = publisher._mm
    for i in range(1, MAX_WIDGETS):
        off = _HEADER_SIZE + i * _WIDGET_SIZE
        slot_bytes = bytes(mm[off:off + _WIDGET_SIZE])
        assert slot_bytes == b'\x00' * _WIDGET_SIZE, (
            f'slot {i} not zeroed after shrinking frame: '
            f'first 16 bytes = {slot_bytes[:16]!r}')


def test_empty_frame_clears_all_slots(publisher):
    with publisher.frame() as f:
        f.rect('first', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)

    # Then an empty frame must fully zero the widget array.
    with publisher.frame() as f:
        pass

    mm = publisher._mm
    assert struct.unpack_from('<I', mm, 12)[0] == 0
    widget_bytes = bytes(mm[_HEADER_SIZE:])
    assert widget_bytes == b'\x00' * (_WIDGET_SIZE * MAX_WIDGETS)


# ── Payload truncation ───────────────────────────────────────────────


def test_long_text_is_truncated_without_overrun(publisher):
    # If the text payload overran its 48-byte field it would
    # scribble over px_scale (next field) and then into the next
    # slot. Force the condition and check the boundary.
    long_string = 'A' * (TEXT_LEN * 4)
    with publisher.frame() as f:
        f.text('long', long_string, 0.1, 0.1, px_scale=2.0)
        f.rect('after', 0.5, 0.5, 0.1, 0.1, color=BLACK_DIM)

    mm = publisher._mm
    slot0 = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE)
    text_bytes = slot0[10]
    # Null-terminator must be within the field.
    assert b'\x00' in text_bytes, 'text field is not null-terminated'
    # px_scale (slot0[11]) must be our requested 2.0, not garbage
    # from an overrunning text write.
    assert slot0[11] == pytest.approx(2.0), (
        'px_scale was clobbered ; text overran its field')

    # And the second slot must be intact.
    slot1 = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE + _WIDGET_SIZE)
    assert slot1[0] == KIND_RECT
    assert slot1[5] == pytest.approx(0.5)  # x


def test_unicode_text_truncates_on_byte_boundary(publisher):
    # Mid-codepoint truncation would break the C side's null-term
    # read at best; encode-then-truncate must cut on a UTF-8 safe
    # boundary (we don't check exact chars here, just that we don't
    # pack garbage into the field or raise during encoding).
    emoji_heavy = '★' * TEXT_LEN   # 3 bytes each → guaranteed overflow
    with publisher.frame() as f:
        f.text('stars', emoji_heavy, 0.1, 0.1)
    mm = publisher._mm
    slot = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE)
    text_bytes = slot[10]
    assert b'\x00' in text_bytes


# ── Seqlock + header integrity ───────────────────────────────────────


def test_seq_is_even_after_commit(publisher):
    # The seqlock contract: odd seq means a writer is mid-update. If
    # we ever leave seq odd, a reader would spin forever trying to
    # get a consistent read. Assert even after every commit.
    for _ in range(5):
        with publisher.frame() as f:
            f.rect('r', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)
        mm = publisher._mm
        seq = struct.unpack_from('<I', mm, 8)[0]
        assert seq % 2 == 0, f'seq left odd: {seq}'


def test_header_magic_and_version_preserved(publisher):
    with publisher.frame() as f:
        f.rect('r', 0.0, 0.0, 0.1, 0.1, color=BLACK_DIM)
    mm = publisher._mm
    magic, version = struct.unpack_from('<II', mm, 0)
    assert magic == MAGIC
    assert version == VERSION


# ── Round-trip through the shared region ─────────────────────────────


def test_widget_roundtrip_matches_input(publisher):
    rid = widget_id('round_trip')
    with publisher.frame() as f:
        f.rect('round_trip', 0.25, 0.5, 0.1, 0.2,
               color=rgba(10, 20, 30, 255), anchor=ANCHOR_TL)

    mm = publisher._mm
    n = struct.unpack_from('<I', mm, 12)[0]
    assert n == 1
    (kind, anchor, _pad, w_id, g_id, x, y, w, h,
     color, text, px_scale,
     channel_id, generation) = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE)
    assert kind == KIND_RECT
    assert anchor == ANCHOR_TL
    assert w_id == rid
    assert g_id == 0              # no group ; standalone
    assert x == pytest.approx(0.25)
    assert y == pytest.approx(0.5)
    assert w == pytest.approx(0.1)
    assert h == pytest.approx(0.2)
    assert color == rgba(10, 20, 30, 255)
    assert text.startswith(b'\x00')   # rect has no text
    assert px_scale == pytest.approx(1.0)
    # Non-web-texture widgets leave the side-socket fields at 0.
    assert channel_id == 0
    assert generation == 0


def test_group_id_flows_into_shm(publisher):
    gid = widget_id('panel')
    with publisher.frame() as f:
        with f.group('panel'):
            f.rect('r1', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)
            f.rect('r2', 0.2, 0.2, 0.1, 0.1, color=BLACK_DIM)

    mm = publisher._mm
    for i in range(2):
        slot = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE + i * _WIDGET_SIZE)
        assert slot[4] == gid, (
            f'slot {i} group_id = {slot[4]}, expected {gid}')


# ── Drag delta persistence ───────────────────────────────────────────


def test_drag_delta_persists_under_group_key(publisher):
    cfg = _MemStore()
    publisher._cfg = cfg

    # Emit a grouped widget at baseline (0.1, 0.1).
    with publisher.frame() as f:
        with f.group('panel'):
            f.rect('r', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)

    # Simulate the C side moving it to (0.3, 0.4) and bumping
    # dragged_seq on release so the next poll persists the delta.
    mm = publisher._mm
    # Widget slot 0 x/y offsets: kind(1)+anchor(1)+pad(2)+wid(4)+gid(4) = 12
    x_off = _HEADER_SIZE + 12
    struct.pack_into('<ff', mm, x_off, 0.3, 0.4)
    # Header: drag_active at 17 (cleared ; simulating post-release),
    # dragged_seq at 24 (bumped to 1 to trigger the capture path).
    mm[17] = 0
    struct.pack_into('<I', mm, 24, 1)

    # Next frame's open calls _poll_drag_updates. The delta should
    # be (0.3 - 0.1, 0.4 - 0.1) = (0.2, 0.3), keyed by the GROUP id,
    # not the widget id.
    with publisher.frame() as f:
        pass

    gid = widget_id('panel')
    width_key = f'{publisher.width}x{publisher.height}'
    delta_path = f'overlay.layouts.{publisher.plugin_key}.' \
                 f'{width_key}.groups.{gid}'
    stored = cfg.get(delta_path)
    assert stored is not None, (
        f'group delta was not persisted; tree = {cfg._tree}')
    assert stored['dx'] == pytest.approx(0.2)
    assert stored['dy'] == pytest.approx(0.3)


def test_standalone_drag_delta_persists_under_widget_key(publisher):
    cfg = _MemStore()
    publisher._cfg = cfg

    with publisher.frame() as f:
        f.rect('lone', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)

    mm = publisher._mm
    x_off = _HEADER_SIZE + 12
    struct.pack_into('<ff', mm, x_off, 0.15, 0.18)
    mm[17] = 0                          # drag_active = 0 (released)
    struct.pack_into('<I', mm, 24, 1)   # dragged_seq = 1

    with publisher.frame() as f:
        pass

    wid = widget_id('lone')
    width_key = f'{publisher.width}x{publisher.height}'
    delta_path = f'overlay.layouts.{publisher.plugin_key}.' \
                 f'{width_key}.widgets.{wid}'
    stored = cfg.get(delta_path)
    assert stored is not None
    assert stored['dx'] == pytest.approx(0.05, abs=1e-4)
    assert stored['dy'] == pytest.approx(0.08, abs=1e-4)


def test_active_drag_does_not_snap_widget_back(publisher):
    # While drag_active is set and the dragged widget's group is
    # known to the publisher, the commit path must read the widget's
    # *current* shm (x, y) rather than overwriting it with
    # baseline + persisted delta. If this regresses, the HUD will
    # visibly flicker/snap each frame during a drag.
    cfg = _MemStore()
    publisher._cfg = cfg

    # Frame 1: register a grouped widget at baseline (0.1, 0.1).
    # After this, the publisher knows wid → gid for lookup.
    with publisher.frame() as f:
        with f.group('panel'):
            f.rect('r', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)

    mm = publisher._mm
    # Simulate C mid-drag: move the widget to (0.42, 0.55) and set
    # drag_active=1, dragged_widget_id=<wid>. dragged_seq stays at
    # its previous value ; bump only happens on release.
    wid = widget_id('r')
    x_off = _HEADER_SIZE + 12
    struct.pack_into('<ff', mm, x_off, 0.42, 0.55)
    mm[17] = 1                          # drag_active = 1
    struct.pack_into('<I', mm, 20, wid)

    # Frame 2: publisher commits again. The dragged widget's shm
    # (x, y) must survive ; i.e. the publisher must read 0.42/0.55
    # back and write them unchanged, NOT re-stamp baseline+delta.
    with publisher.frame() as f:
        with f.group('panel'):
            f.rect('r', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)

    slot = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE)
    assert slot[5] == pytest.approx(0.42), (
        f'drag snap-back: x is {slot[5]}, expected 0.42 preserved')
    assert slot[6] == pytest.approx(0.55), (
        f'drag snap-back: y is {slot[6]}, expected 0.55 preserved')

    # And critically: delta was NOT persisted mid-drag. The C side
    # bumps dragged_seq only on release; before that, config must be
    # untouched.
    width_key = f'{publisher.width}x{publisher.height}'
    assert cfg.get(f'overlay.layouts.{publisher.plugin_key}.'
                   f'{width_key}.groups') in (None, {})


def test_load_deltas_applies_on_next_frame(publisher):
    cfg = _MemStore()
    gid = widget_id('panel')
    width_key = f'{publisher.width}x{publisher.height}'
    cfg.set(
        f'overlay.layouts.{publisher.plugin_key}.{width_key}.groups.{gid}',
        {'dx': 0.2, 'dy': 0.3})
    publisher._cfg = cfg
    publisher._load_deltas_from_config()

    with publisher.frame() as f:
        with f.group('panel'):
            f.rect('r', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)

    mm = publisher._mm
    slot = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE)
    # baseline (0.1, 0.1) + delta (0.2, 0.3) = (0.3, 0.4)
    assert slot[5] == pytest.approx(0.3)
    assert slot[6] == pytest.approx(0.4)


# ── Pathological inputs ──────────────────────────────────────────────


def test_noninteger_coords_do_not_corrupt_slot(publisher):
    # Floats from numpy or arithmetic can be inf/nan. The publisher
    # must not let those crash pack_into ; they'll pack as their IEEE
    # bit pattern, but we at least check the call completes and the
    # adjacent slot stays intact.
    import math
    with publisher.frame() as f:
        f.rect('weird', math.inf, math.nan, 0.1, 0.1, color=BLACK_DIM)
        f.rect('normal', 0.5, 0.5, 0.1, 0.1, color=BLACK_DIM)

    mm = publisher._mm
    slot1 = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE + _WIDGET_SIZE)
    assert slot1[5] == pytest.approx(0.5)
    assert slot1[6] == pytest.approx(0.5)


def test_widget_id_never_zero():
    # widget_id == 0 is reserved on the wire as "no widget dragged".
    # The FNV hash collides with 0 essentially never, but we still
    # guard against it ; regression would silently erase drag tracking
    # for that widget.
    for name in ['', 'a', 'b' * 512, '\x00null\x00bytes\x00', '星']:
        assert widget_id(name) != 0


def test_rgba_packing_byte_order():
    # Byte 0 must be R. If this ever flips to ABGR the renderer
    # would draw every widget in the wrong color.
    c = rgba(0xAA, 0xBB, 0xCC, 0xDD)
    assert (c >>  0) & 0xff == 0xAA
    assert (c >>  8) & 0xff == 0xBB
    assert (c >> 16) & 0xff == 0xCC
    assert (c >> 24) & 0xff == 0xDD


def test_commit_overwrites_deleted_widget_completely(publisher):
    # If slot-clearing memset were short, the first few bytes would
    # go to zero but old widget_id / text / px_scale would remain.
    # Check the *entire* slot region becomes zero when we shrink.
    with publisher.frame() as f:
        f.text('ghost', 'VISIBLE TEXT', 0.1, 0.1, px_scale=3.0)

    with publisher.frame() as f:
        pass   # zero widgets

    mm = publisher._mm
    off = _HEADER_SIZE
    assert bytes(mm[off:off + _WIDGET_SIZE]) == b'\x00' * _WIDGET_SIZE


# ── Header-only fields the C side owns ───────────────────────────────


def test_commit_preserves_edit_mode_and_dragged_fields(publisher):
    # The C side writes edit_mode, drag_active, dragged_widget_id,
    # dragged_seq independently of our commits. Our commit path must
    # *not* overwrite those ; we only own magic, version, seq, n_widgets.
    mm = publisher._mm
    # Simulate C writing the fields it owns.
    mm[16] = 1                                   # edit_mode = 1
    mm[17] = 1                                   # drag_active = 1
    struct.pack_into('<I', mm, 20, 42)           # dragged_widget_id = 42
    struct.pack_into('<I', mm, 24, 7)            # dragged_seq = 7

    with publisher.frame() as f:
        f.rect('r', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)

    assert mm[16] == 1, 'commit clobbered edit_mode'
    assert mm[17] == 1, 'commit clobbered drag_active'
    assert struct.unpack_from('<I', mm, 20)[0] == 42, \
        'commit clobbered dragged_widget_id'
    assert struct.unpack_from('<I', mm, 24)[0] == 7, \
        'commit clobbered dragged_seq'


def test_many_frames_no_monotonic_overflow_crash(publisher):
    # A long-running publisher will wrap uint32_t seq eventually. We
    # don't simulate the actual wrap (too slow) ; just force it into
    # the upper range and confirm we handle it without raising.
    publisher._seq = 0xfffffff0
    for _ in range(20):
        with publisher.frame() as f:
            f.rect('r', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)
    # Post-wrap, seq should still be even (invariant: even after
    # each commit).
    mm = publisher._mm
    seq = struct.unpack_from('<I', mm, 8)[0]
    assert seq % 2 == 0


# ── Overlay plugin role ──────────────────────────────────────────────


def test_sandboxed_overlay_role_registers_and_draws(tmp_path, monkeypatch):
    monkeypatch.setenv('EA_PLUGINS_PATH', str(tmp_path))
    bundle = tmp_path / 'overlay_demo'
    (bundle / 'overlay').mkdir(parents=True)
    (bundle / 'manifest.toml').write_text(
        'name = "overlay_demo"\nkey = "overlay_demo"\n')
    (bundle / 'overlay' / 'hud.py').write_text(textwrap.dedent('''
        from analysis.overlay.api import BLACK_DIM, WHITE

        def draw(frame):
            with frame.group('panel'):
                frame.rect('bg', 0.1, 0.2, 0.3, 0.4, color=BLACK_DIM)
                frame.text('label', 'SAFE', 0.12, 0.23, color=WHITE)

        def register_overlay(add):
            add('Demo HUD', draw, key='overlay_demo_hud', hz=12)
    '''))

    registry = discover_overlays()
    spec = registry.get('overlay_demo_hud')
    assert spec is not None
    assert spec.name == 'Demo HUD'
    assert spec.hz == pytest.approx(12.0)

    shm_path = f'/dev/shm/vsrg_overlay_overlay_demo_hud_{uuid.uuid4().hex[:8]}'
    pub = OverlayPublisher('overlay_demo_hud', width=1920, height=1080,
                           shm_path=shm_path)
    pub.start()
    try:
        with pub.frame() as frame:
            spec.draw(frame)
        mm = pub._mm
        assert struct.unpack_from('<I', mm, 12)[0] == 2
        slot0 = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE)
        slot1 = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE + _WIDGET_SIZE)
        assert slot0[0] == KIND_RECT
        assert slot1[0] == KIND_TEXT
        assert slot1[10].startswith(b'SAFE')
    finally:
        pub.stop()
        try:
            os.unlink(shm_path)
        except FileNotFoundError:
            pass


def test_disabling_overlay_skips_spec_but_keeps_publisher(tmp_path):
    """Disabling one spec must not stop the session publisher.

    In the single-shm world, every overlay shares one publisher and
    one render thread. Toggling one plugin off should keep every other
    plugin drawing ; the toggled spec just sits out of the merged
    draw on the next tick. This pins that behavior so a future refactor
    doesn't accidentally restore per-spec start/stop.
    """
    from analysis.config.store import ConfigStore
    from analysis.overlay.publisher import OverlayRegistry

    key = f'overlay_stop_{uuid.uuid4().hex[:8]}'
    store = ConfigStore(tmp_path / 'config.json', autosave=False)
    store.load()

    def draw(frame):
        frame.rect('r', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)

    running = OverlayRegistry(config=store)
    running.add('Stop test', draw, key=key, hz=1)

    toggler = OverlayRegistry(config=store)
    toggler.add('Stop test', draw, key=key, hz=1)
    try:
        assert running.get(key).enabled is True

        # Disable through the second registry ; the config change must
        # propagate to the running one via the subscription.
        assert toggler.set_enabled(key, False) is True
        assert running.get(key).enabled is False

        # Re-enable.
        assert toggler.set_enabled(key, True) is True
        assert running.get(key).enabled is True
    finally:
        toggler.close()
        running.close()


# ── v3: web-texture widgets + side-socket channel ──────────────────


def test_web_texture_widget_stamps_channel_and_generation(publisher):
    """``f.web_texture(...)`` emits a ``KIND_WEB_TEXTURE`` slot whose
    channel_id + generation match what the producer stamped. Those are
    the keys the overlay's EGLImage cache looks up to find the dmabuf
    imported out-of-band via the side socket."""
    with publisher.frame() as f:
        f.web_texture('stream', 0.1, 0.2, 0.3, 0.4,
                      channel_id=0xDEADBEEF, generation=42,
                      anchor=ANCHOR_TL)

    mm = publisher._mm
    slot = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE)
    (kind, _anchor, _pad, _wid, _gid, _x, _y, _w, _h,
     _color, _text, _px, channel_id, generation) = slot
    assert kind == KIND_WEB_TEXTURE
    assert channel_id == 0xDEADBEEF
    assert generation == 42


def test_web_texture_uses_web_texture_kind_constant():
    """Pin KIND_WEB_TEXTURE to the header's ``VSRG_OVERLAY_KIND_WEB_TEXTURE``
    value. Python sees it as 3; drift here = the publisher writing a
    kind byte that the C overlay interprets as unused / rect / text."""
    assert KIND_WEB_TEXTURE == 3


def test_rect_widget_leaves_web_texture_fields_zero(publisher):
    """Non-web-texture widgets must zero out the new tail fields.
    Otherwise a stale slot with garbage channel_id could accidentally
    cause the overlay to sample a random cached texture."""
    with publisher.frame() as f:
        f.rect('r', 0.1, 0.1, 0.1, 0.1, color=BLACK_DIM)

    mm = publisher._mm
    slot = _WIDGET_STRUCT.unpack_from(mm, _HEADER_SIZE)
    (kind, *_rest, channel_id, generation) = slot
    assert kind == KIND_RECT
    assert channel_id == 0
    assert generation == 0


def test_web_texture_ipc_header_struct_offsets_match_c():
    """Compile a tiny C probe against ``web_texture_ipc.h`` and verify
    ``sizeof(VsrgWebTexFrame)`` + the ``magic``/``version``/``kind``
    field offsets agree with what the Rust/Python producer will pack.

    Skipped if a C compiler isn't available (some CI shells); the Rust
    crate exposes a constant layout check that catches drift too."""
    import shutil
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')

    here = os.path.dirname(__file__)
    src = textwrap.dedent(f"""
        #include <stddef.h>
        #include <stdio.h>
        #include "{here}/../analysis/overlay/widgets/web_texture_ipc.h"
        int main(void) {{
            printf("size=%zu magic_off=%zu kind_off=%zu channel_off=%zu "
                   "modifier_off=%zu n_planes_off=%zu "
                   "offsets_off=%zu strides_off=%zu\\n",
                   sizeof(VsrgWebTexFrame),
                   offsetof(VsrgWebTexFrame, magic),
                   offsetof(VsrgWebTexFrame, kind),
                   offsetof(VsrgWebTexFrame, channel_id),
                   offsetof(VsrgWebTexFrame, modifier),
                   offsetof(VsrgWebTexFrame, n_planes),
                   offsetof(VsrgWebTexFrame, offsets),
                   offsetof(VsrgWebTexFrame, strides));
            return 0;
        }}
    """)
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        csrc = os.path.join(d, 'probe.c')
        exe  = os.path.join(d, 'probe')
        with open(csrc, 'w') as fh:
            fh.write(src)
        subprocess.check_call([cc, '-O0', csrc, '-o', exe])
        out = subprocess.check_output([exe], text=True).strip()

    # Parse key=value pairs
    fields = dict(tok.split('=') for tok in out.split())
    fields = {k: int(v) for k, v in fields.items()}
    # Canonical layout (verified once with a C probe, pinned as a
    # contract so future edits to the header can't silently shift
    # offsets and desync the Rust producer / overlay consumer).
    #   magic      u32 @ 0
    #   version    u32 @ 4
    #   kind       u32 @ 8
    #   channel_id u32 @ 12
    #   generation u32 @ 16
    #   width      u32 @ 20
    #   height     u32 @ 24
    #   format     u32 @ 28
    #   modifier   u64 @ 32
    #   n_planes   u32 @ 40
    #   _pad0      u32 @ 44
    #   offsets    u32[4] @ 48
    #   strides    u32[4] @ 64
    #   total size = 80
    assert fields['magic_off']    == 0
    assert fields['kind_off']     == 8
    assert fields['channel_off']  == 12
    assert fields['modifier_off'] == 32
    assert fields['n_planes_off'] == 40
    assert fields['offsets_off']  == 48
    assert fields['strides_off']  == 64
    assert fields['size']         == 80
