"""Per-frame HUD state, separated from replay state.

``Player`` used to own the HUD's scroll offset, plugin-panel toggle, and
hitbox list alongside replay state like ``t`` / ``scroll_speed``, which
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
    # to top-pinned sections; bottom-pinned sections ignore it.
    # ``content_h`` is the top region's painted height, observed by the
    # renderer after each real draw (never trusted from plugin measure
    # hooks), and ``max`` derives from it -- so scrollability tracks
    # whatever the sections actually painted, one HUD render behind at
    # worst.
    sidebar_scroll: int = 0
    sidebar_scroll_max: int = 0
    sidebar_content_h: int = 0

    # Collapsible plugins panel in the sidebar. Opened by the user via
    # the painted toggle; the renderer checks it to decide layout.
    plugin_panel_open: bool = False
    layers_panel_open: bool = False

    # Key of the currently open flyout (sidebar section with an expanded
    # panel to the left of the sidebar), or None if no flyout is open.
    # One-at-a-time: opening a new flyout replaces the current one.
    open_flyout: str | None = None

    # Layout edit mode. When True, draggable sidebar components show an
    # outline + drag handle; left-click-drag moves them between the
    # sidepanel and the free (floating) region. Toggled by Shift+Tab
    # and by the "Edit layout" button in the sidebar header. Normal
    # button actions are suppressed while this is on so drags don't
    # accidentally trigger clicks.
    edit_mode: bool = False

    # Live drag state. Set when the user presses on a draggable
    # component in edit mode; cleared on release. The renderer reads
    # this to draw the ghost and the blue insertion line.
    #   drag_key:     section key being dragged
    #   drag_pointer: current (x, y) cursor px
    #   drag_origin:  (x, y) where the mouse-down landed, used for
    #                 click-vs-drag threshold
    #   drag_offset:  (dx, dy) from the dragged rect's top-left to
    #                 the pointer at mouse-down (so the ghost follows
    #                 the cursor in the same spot it was grabbed)
    #   drag_origin_region: 'sidepanel' or 'free'; where the drag
    #                 started, for the release-time routing rules
    drag_key: str | None = None
    drag_pointer: tuple = (0, 0)
    drag_origin: tuple = (0, 0)
    drag_offset: tuple = (0, 0)
    drag_origin_region: str | None = None

    # Live resize state: component key being resized + the (x, y) at
    # which the resize started + the original (w, h) at that moment.
    resize_key: str | None = None
    resize_origin: tuple = (0, 0)
    resize_origin_size: tuple = (0, 0)

    # Per-frame rect snapshots, populated by the renderer each frame
    # after _draw_hud. Used by drag-drop drop-order calculation so it
    # can insert a dropped section between the real neighbors under
    # the cursor instead of guessing from priority.
    #   frame_sidepanel_rects: {section_key: (x, y, w, h)}
    #   frame_free_rects:      {section_key: (x, y, w, h)}
    frame_sidepanel_rects: dict = field(default_factory=dict)
    frame_free_rects: dict = field(default_factory=dict)

    # Hitboxes registered during the frame's HUD draw. Rebuilt each
    # frame by the renderer (cleared at the start of ``_draw_hud``),
    # then consulted by ``Player.handle_mouse_down`` when Qt forwards a
    # click. Each entry: ``(rect, action, payload)``.
    hitboxes: list = field(default_factory=list)

    def clear_hitboxes(self) -> None:
        self.hitboxes.clear()

    def add_hitbox(self, rect, action, payload=None) -> None:
        self.hitboxes.append((tuple(rect), action, payload))
