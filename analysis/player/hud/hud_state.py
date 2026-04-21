"""Per-frame HUD state, separated from replay state.

``Player`` used to own the HUD's scroll offset, plugin-panel toggle, and
hitbox list alongside replay state like ``t`` / ``scroll_speed`` — which
conflated two very different lifecycles. Replay state is the source of
truth for what's being analyzed; HUD state is ephemeral overlay state
that only matters while the painted sidebar is on screen.

Split out so:

  * The replay core doesn't grow an attribute every time the HUD does.
  * Future HUD features (multiple panels, pop-ups, tooltips) have a
    stable container without reaching into ``Player``.
  * Plugins that only need HUD state don't have to grab the whole
    ``Player`` object.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HudState:
    # Sidebar scrolling. ``scroll`` is the vertical offset (px) applied
    # to top-pinned sections; bottom-pinned sections ignore it. ``max``
    # is recomputed each frame by the renderer after measuring content.
    sidebar_scroll: int = 0
    sidebar_scroll_max: int = 0

    # Collapsible plugins panel in the sidebar. Opened by the user via
    # the painted toggle; the renderer checks it to decide layout.
    plugin_panel_open: bool = False

    # Hitboxes registered during the frame's HUD draw. Rebuilt each
    # frame by the renderer (cleared at the start of ``_draw_hud``),
    # then consulted by ``Player.handle_mouse_down`` when Qt forwards a
    # click. Each entry: ``(rect, action, payload)``.
    hitboxes: list = field(default_factory=list)

    def clear_hitboxes(self) -> None:
        self.hitboxes.clear()

    def add_hitbox(self, rect, action, payload=None) -> None:
        self.hitboxes.append((tuple(rect), action, payload))
