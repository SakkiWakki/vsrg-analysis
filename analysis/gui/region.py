"""Compositional input-region system for the player tab.

The player tab is visually one canvas but semantically several regions
(the lanes, the sidebar, potentially future panels or pop-ups). Each
region owns the input that lands inside it. An ``InputRouter`` picks
the right region for a positional event and forwards it; events that
no region handles fall through to the tab's default handling.

This replaces the ad-hoc cursor-x check in ``eventFilter`` that had to
grow a new branch every time a new region appeared. New regions just
register with the router.

Shape:

  * ``Region`` is a lightweight protocol — any object with ``contains``
    and the optional ``on_wheel`` / ``on_mouse_down`` hooks works. No
    inheritance required.
  * ``InputRouter`` walks regions in registered order and returns True
    if one handled the event. The tab short-circuits on True.
  * Keyboard events are global (not positional) and stay on the tab —
    regions only see wheel + mouse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Region(Protocol):
    """A positional input region.

    ``contains(x, y)`` decides membership; the optional hooks return
    True when they've consumed the event. A region that doesn't
    implement a hook is treated as "didn't handle" for that event.
    """

    def contains(self, x: int, y: int) -> bool: ...


@dataclass
class InputRouter:
    """Dispatches positional events to the first region that contains
    the cursor. Regions are scanned in registration order; later
    registrations win for overlapping areas if you put them first — the
    common pattern is to register overlays before the base region they
    cover.
    """

    regions: list = None

    def __post_init__(self):
        if self.regions is None:
            self.regions = []

    def add(self, region) -> None:
        self.regions.append(region)

    def remove(self, region) -> bool:
        try:
            self.regions.remove(region)
            return True
        except ValueError:
            return False

    def region_at(self, x: int, y: int):
        for r in self.regions:
            try:
                if r.contains(int(x), int(y)):
                    return r
            except Exception:
                continue
        return None

    def dispatch_wheel(self, x, y, delta_y, modifiers) -> bool:
        r = self.region_at(x, y)
        fn = getattr(r, 'on_wheel', None) if r is not None else None
        if fn is None:
            return False
        try:
            return bool(fn(x, y, delta_y, modifiers))
        except Exception as exc:
            print(f'region wheel handler failed: {exc}')
            return False

    def dispatch_mouse_down(self, x, y, button, modifiers) -> bool:
        r = self.region_at(x, y)
        fn = getattr(r, 'on_mouse_down', None) if r is not None else None
        if fn is None:
            return False
        try:
            return bool(fn(x, y, button, modifiers))
        except Exception as exc:
            print(f'region mouse handler failed: {exc}')
            return False


class SidebarRegion:
    """Covers the painted HUD sidebar. Wheel scrolls the sidebar; mouse
    clicks dispatch to the player's hitbox table (``handle_mouse_down``
    already walks the HUD hitboxes registered during the last frame)."""

    def __init__(self, player):
        self.player = player

    @property
    def sidebar_x(self) -> int:
        from analysis.player import theme
        return self.player.W - theme.SIDEBAR_WIDTH

    def contains(self, x: int, y: int) -> bool:
        return x >= self.sidebar_x and 0 <= y <= self.player.H

    def on_wheel(self, x, y, delta_y, modifiers) -> bool:
        hud = self.player.hud
        hud.sidebar_scroll = max(
            0,
            min(hud.sidebar_scroll_max, hud.sidebar_scroll - delta_y // 3))
        return True

    def on_mouse_down(self, x, y, button, modifiers) -> bool:
        # Hitboxes populated by the last render frame include both the
        # painted sidebar controls and any plugin-registered buttons.
        # The player itself owns the hitbox-action dispatcher.
        return bool(self.player.handle_mouse_down(x, y))


class LanesRegion:
    """Covers the chart-space lanes. Wheel seeks the replay; mouse
    clicks fall through (lane-space interactions, if any, are handled
    via ``handle_mouse_down`` which is already consulted for lanes)."""

    def __init__(self, player, seek_fn):
        self.player = player
        self._seek = seek_fn

    def contains(self, x: int, y: int) -> bool:
        from analysis.player import theme
        sidebar_x = self.player.W - theme.SIDEBAR_WIDTH
        return 0 <= x < sidebar_x and 0 <= y <= self.player.H

    def on_wheel(self, x, y, delta_y, modifiers) -> bool:
        from PySide6.QtCore import Qt
        step = delta_y / 120.0 * 0.5
        if modifiers & Qt.ShiftModifier:
            step *= 10
        self._seek(step)
        return True

    def on_mouse_down(self, x, y, button, modifiers) -> bool:
        return bool(self.player.handle_mouse_down(x, y))
