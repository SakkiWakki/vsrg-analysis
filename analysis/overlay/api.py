"""Sandbox-safe helpers for gamescope overlay plugins.

Overlay plugin modules may import this file from sandboxed bundles. It
intentionally contains only pure constants and small helpers; the shm
publisher that touches ``/dev/shm`` lives in ``analysis.overlay.publisher`` and is
owned by the trusted host process.
"""
from __future__ import annotations


ANCHOR_TL = 0
ANCHOR_TR = 1
ANCHOR_BL = 2
ANCHOR_BR = 3
ANCHOR_C = 4


def rgba(r: int, g: int, b: int, a: int = 255) -> int:
    """Pack 0..255 RGBA components into the uint32 layout read by the C
    overlay renderer: byte 0 is R and byte 3 is A."""
    return ((int(r) & 0xff)
            | ((int(g) & 0xff) << 8)
            | ((int(b) & 0xff) << 16)
            | ((int(a) & 0xff) << 24))


WHITE = rgba(250, 250, 250)
BLACK_DIM = rgba(10, 10, 15, 140)
BLUE_ACCENT = rgba(75, 164, 255, 230)
WARN_AMBER = rgba(255, 180, 50)
HIST_BAR = rgba(75, 164, 255, 230)


def widget_id(name: str) -> int:
    """Stable FNV-1a 32-bit id used by the renderer for drag layout."""
    h = 0x811c9dc5
    for b in str(name).encode('utf-8'):
        h ^= b
        h = (h * 0x01000193) & 0xffffffff
    return h or 0x811c9dc5
