"""Platform abstraction layer for overlay hosting.

The sidebar is always Qt-in-process — nothing platform-specific about it.
The *overlay*, however, needs a way to get pixels on top of a running
game, and every OS offers a different contract for that:

    Linux + Gamescope: set GAMESCOPE_EXTERNAL_OVERLAY=1 on an X window;
        the current implementation ships a small C renderer that reads
        widgets from /dev/shm via mmap.
    Windows:           hypothetical DWM thumbnail / D3D overlay swapchain.
    macOS:             hypothetical Metal layer on the shared window server.
    (no overlay):      NullOverlayPlatform — overlay surface simply isn't
                       registered; components with 'overlay' in their
                       supported_surfaces silently skip it.

``OverlayPlatform`` is the contract every backend speaks. The component
overlay backend consumes ``OverlayFrame`` (a neutral command list) and
hands it to the active platform's ``submit_frame``; the platform
decides how to turn it into pixels. Adding a new OS means writing one
``OverlayPlatform`` implementation, not touching the drawing code.

The active platform is picked by :func:`detect` at import time. Override
via ``VSRG_OVERLAY_PLATFORM=<name>`` for tests or headless CI.
"""
from __future__ import annotations

import os
import sys

from analysis.components.pal.base import (
    OverlayFrame,
    OverlayHandle,
    OverlayPlatform,
    OverlayPlatformCapabilities,
)
from analysis.components.pal.null import NullOverlayPlatform


_FORCED_ENV = 'VSRG_OVERLAY_PLATFORM'


def detect() -> OverlayPlatform:
    """Return the best available :class:`OverlayPlatform` for this host.

    Priority:
      1. ``VSRG_OVERLAY_PLATFORM`` env var wins, if set to a known name.
      2. Linux + gamescope session detected → ``GamescopeOverlayPlatform``.
      3. Fall back to :class:`NullOverlayPlatform` — overlay disabled.

    The detection deliberately never *requires* gamescope at import
    time: headless CI, sidebar-only Macs, and unit tests all land on
    Null without noise.
    """
    forced = os.environ.get(_FORCED_ENV, '').strip().lower()
    if forced:
        plat = _load_named(forced)
        if plat is not None:
            return plat
        # Unknown name → honest warning, fall through to auto-detect.
        print(f'{_FORCED_ENV}={forced!r} not recognised; '
              f'falling back to auto-detect')

    if sys.platform.startswith('linux') and _gamescope_available():
        try:
            from analysis.components.pal.gamescope import (
                GamescopeOverlayPlatform,
            )
            return GamescopeOverlayPlatform()
        except Exception as exc:
            print(f'gamescope overlay platform failed to init: {exc}')

    return NullOverlayPlatform()


def _load_named(name: str) -> OverlayPlatform | None:
    if name in ('null', 'none', 'disabled'):
        return NullOverlayPlatform()
    if name == 'gamescope':
        try:
            from analysis.components.pal.gamescope import (
                GamescopeOverlayPlatform,
            )
            return GamescopeOverlayPlatform()
        except Exception as exc:
            print(f'gamescope platform requested but init failed: {exc}')
            return None
    return None


def _gamescope_available() -> bool:
    """Heuristic: gamescope advertises itself via env vars in its
    nested session. Checking for them avoids importing X11 / mmap /
    subprocess code on hosts that don't have gamescope at all."""
    return any(k in os.environ for k in (
        'GAMESCOPE_WAYLAND_DISPLAY', 'GAMESCOPE'))


__all__ = [
    'OverlayFrame',
    'OverlayHandle',
    'OverlayPlatform',
    'OverlayPlatformCapabilities',
    'NullOverlayPlatform',
    'detect',
]
