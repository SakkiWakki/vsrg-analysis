"""No-op overlay platform for hosts without an overlay capability.

Picked by :func:`analysis.components.pal.detect` when nothing better is
available (Windows/macOS today, any Linux without gamescope, CI). With
Null active, overlay-targeted components still register, but their
surface never mounts — the registry declines gracefully and the
sidebar side of any dual-surface component continues to work.
"""
from __future__ import annotations

from analysis.components.pal.base import (
    OverlayFrame,
    OverlayHandle,
    OverlayPlatformCapabilities,
)


class NullOverlayPlatform:
    def is_available(self) -> bool:
        return False

    def capabilities(self) -> OverlayPlatformCapabilities:
        return OverlayPlatformCapabilities(
            supports_input=False, supports_drag_edit=False)

    def setup(self, key: str, *, width: int, height: int) -> OverlayHandle:
        return OverlayHandle(key=str(key), width=int(width),
                             height=int(height), impl=None)

    def submit_frame(self, handle: OverlayHandle,
                     frame: OverlayFrame) -> None:
        pass  # drop on the floor

    def teardown(self, handle: OverlayHandle) -> None:
        pass
