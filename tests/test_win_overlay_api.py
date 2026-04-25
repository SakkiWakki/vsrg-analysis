"""Windows counterpart to ``test_overlay_api`` ; covers the named-
mapping branch in ``OverlayPublisher.start`` and verifies that a second
opener of the same name sees the publisher's writes.

The Linux suite uses ``/dev/shm`` and is skipped here; everything
platform-independent (seqlock invariants, struct-size math, oversized
text truncation) is already covered over there and doesn't need a
per-OS duplicate.
"""
from __future__ import annotations

import mmap
import struct
import sys
import uuid

import pytest

from analysis.overlay.publisher import (
    MAGIC,
    MAX_WIDGETS,
    VERSION,
    WHITE,
    _HEADER_STRUCT,
    _TOTAL_SIZE,
    KIND_RECT,
    OverlayPublisher,
)

pytestmark = pytest.mark.skipif(
    sys.platform != 'win32',
    reason='named-mapping path is Windows-only',
)


@pytest.fixture
def publisher():
    # Unique shm_path basename per test so parallel runs don't collide.
    # ``start()`` uses the basename as the mapping tagname on Windows.
    key = f'test_{uuid.uuid4().hex[:12]}'
    pub = OverlayPublisher(key, width=1920, height=1080,
                           shm_path=f'vsrg_overlay_{key}')
    pub.start()
    try:
        yield pub
    finally:
        pub.stop()


def _open_mapping(tagname: str) -> mmap.mmap:
    # ``mmap(-1, ..., tagname=...)`` on Windows opens the existing named
    # mapping if one already exists (CreateFileMapping returns the handle
    # to the existing object when the name matches). ACCESS_READ is
    # enough for inspection.
    return mmap.mmap(-1, _TOTAL_SIZE, tagname=tagname,
                     access=mmap.ACCESS_READ)


def test_start_creates_named_mapping(publisher):
    # The publisher's ``_shm_path`` is the basename we passed; the
    # actual tagname is the same string (no prefix on Windows per
    # publisher.py).
    tag = publisher._shm_path  # noqa: SLF001 ; whitebox; tag is the contract.
    # Reopening by name must succeed ; that's the whole point of the
    # Windows path (one writer, many readers).
    view = _open_mapping(tag)
    try:
        magic, version, *_ = _HEADER_STRUCT.unpack_from(view, 0)
        assert magic == MAGIC
        assert version == VERSION
    finally:
        view.close()


def test_reader_sees_publisher_widgets(publisher):
    # Write one rect, then open a second view and decode the header.
    # n_widgets in the header tells the consumer how many slots are live.
    with publisher.frame() as f:
        f.rect('hello', 0.1, 0.1, 0.2, 0.05, color=WHITE)

    tag = publisher._shm_path  # noqa: SLF001
    view = _open_mapping(tag)
    try:
        magic, version, seq, n_widgets, *_ = _HEADER_STRUCT.unpack_from(view, 0)
        assert magic == MAGIC
        assert version == VERSION
        # After a successful commit ``seq`` is even (seqlock write-
        # released state); odd would mean the reader caught a torn write.
        assert seq % 2 == 0
        assert n_widgets == 1
    finally:
        view.close()


def test_second_opener_observes_count_changes(publisher):
    # Commit two frames with different widget counts and verify the
    # reader view reflects each one.
    tag = publisher._shm_path  # noqa: SLF001
    view = _open_mapping(tag)
    try:
        with publisher.frame() as f:
            f.rect('a', 0.0, 0.0, 0.1, 0.1)
            f.rect('b', 0.2, 0.0, 0.1, 0.1)
        _, _, _, n_widgets, *_ = _HEADER_STRUCT.unpack_from(view, 0)
        assert n_widgets == 2

        with publisher.frame() as f:
            f.rect('c', 0.0, 0.0, 0.1, 0.1)
        _, _, _, n_widgets, *_ = _HEADER_STRUCT.unpack_from(view, 0)
        assert n_widgets == 1
    finally:
        view.close()


def test_empty_frame_sets_zero_widgets(publisher):
    with publisher.frame():
        pass
    tag = publisher._shm_path  # noqa: SLF001
    view = _open_mapping(tag)
    try:
        _, _, _, n_widgets, *_ = _HEADER_STRUCT.unpack_from(view, 0)
        assert n_widgets == 0
    finally:
        view.close()


def test_stop_empties_frame_before_closing(publisher):
    # The DLL consumer polls forever; when the publisher stops, it
    # commits one last empty frame so an attached renderer clears
    # instead of latching whatever the last commit left behind.
    # Keep a reader open across the stop ; if we drop our handle first,
    # the named object is gone with nothing to re-open.
    with publisher.frame() as f:
        f.rect('a', 0.0, 0.0, 0.1, 0.1)

    tag = publisher._shm_path  # noqa: SLF001
    view = _open_mapping(tag)
    try:
        # Before stop: one widget is live.
        _, _, _, n_widgets, *_ = _HEADER_STRUCT.unpack_from(view, 0)
        assert n_widgets == 1

        publisher.stop()

        # After stop: empty frame committed; the renderer will draw
        # nothing on its next tick.
        _, _, _, n_widgets, *_ = _HEADER_STRUCT.unpack_from(view, 0)
        assert n_widgets == 0
    finally:
        view.close()
