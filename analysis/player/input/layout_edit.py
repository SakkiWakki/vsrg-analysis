from __future__ import annotations

from analysis.components.api import REGION_FREE
from analysis.player.render import theme


_ORDER_GAP = 10.0


def rect_contains(rect, x, y):
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def compute_drop_order(y, targets, window_h):
    if not targets:
        return None

    mids = [m for m, _ in targets]
    orders = [o for _, o in targets]

    if all(m is not None for m in mids):
        slot = next((i for i, mid in enumerate(mids) if y < mid), len(mids))
    else:
        frac = max(0.0, min(1.0, y / max(1, window_h)))
        slot = max(0, min(len(mids), int(round(frac * len(mids)))))

    bounds = [orders[0] - _ORDER_GAP, *orders, orders[-1] + _ORDER_GAP]
    return (bounds[slot] + bounds[slot + 1]) / 2.0


class LayoutEditController:
    def __init__(self, player):
        self.p = player

    def begin_drag_section(self, key, x, y, grab_rect):
        grab_x, grab_y, _gw, _gh = grab_rect
        hud = self.p.hud

        hud.drag_key = key
        hud.drag_pointer = (int(x), int(y))
        hud.drag_origin = (int(x), int(y))
        hud.drag_offset = (int(x) - int(grab_x), int(y) - int(grab_y))
        hud.drag_origin_region = self.p.plugins.sidebar.section_region(key)

    def begin_resize_section(self, key, x, y):
        section = self.p.plugins.sidebar.find_section(key)
        if section is None:
            return

        _, _, w, h = self.p.plugins.sidebar.section_free_rect(
            section,
            self.p.W,
            self.p.H,
        )

        self.p.hud.resize_key = key
        self.p.hud.resize_origin = (int(x), int(y))
        self.p.hud.resize_origin_size = (int(w), int(h))

    def handle_mouse_move(self, x, y):
        dragging = self.p.hud.drag_key is not None
        resizing = self.p.hud.resize_key is not None

        if dragging:
            self.p.hud.drag_pointer = (int(x), int(y))
        elif resizing:
            self.apply_resize(int(x), int(y))

        return dragging or resizing

    def apply_resize(self, x, y):
        hud = self.p.hud
        ox, oy = hud.resize_origin
        ow, oh = hud.resize_origin_size

        new_w = max(theme.FREE_MIN_W, ow + (x - ox))
        new_h = max(theme.FREE_MIN_H, oh + (y - oy))

        section = self.p.plugins.sidebar.find_section(hud.resize_key)
        if section is None:
            return

        rx, ry, _, _ = self.p.plugins.sidebar.section_free_rect(
            section,
            self.p.W,
            self.p.H,
        )
        self.p.plugins.sidebar.set_section_free_rect(
            section.key,
            rx,
            ry,
            new_w,
            new_h,
        )

    def handle_mouse_up(self, x, y):
        dragging = self.p.hud.drag_key is not None
        resizing = self.p.hud.resize_key is not None

        if dragging:
            self.finish_drag(int(x), int(y))
        elif resizing:
            self.p.hud.resize_key = None

        return dragging or resizing

    def finish_drag(self, x, y):
        key = self.p.hud.drag_key
        target_region = self.p.plugins.sidebar.region_for_x(x, self.p.W)

        if target_region != REGION_FREE:
            self.place_in_panel(key, y, target_region)
        else:
            self.place_in_free_region(key, x, y)

        self.p.hud.drag_key = None
        self.p.hud.drag_origin_region = None

    def place_in_panel(self, key, y, region):
        registry = self.p.plugins.sidebar
        registry.set_section_region(key, region)

        targets = registry.reorder_targets(
            key,
            region,
            self.p.hud.frame_sidepanel_rects,
        )
        new_order = compute_drop_order(y, targets, self.p.H)
        if new_order is not None:
            registry.set_section_order(key, new_order)

    def place_in_free_region(self, key, x, y):
        self.p.plugins.sidebar.set_section_region(key, 'free')

        section = self.p.plugins.sidebar.find_section(key)
        if section is None:
            return

        _, _, w, h = self.p.plugins.sidebar.section_free_rect(
            section,
            self.p.W,
            self.p.H,
        )
        dx, dy = self.p.hud.drag_offset

        new_x = max(0, min(self.p.W - w, x - dx))
        new_y = max(0, min(self.p.H - h, y - dy))

        self.p.plugins.sidebar.set_section_free_rect(key, new_x, new_y, w, h)

    def toggle_edit_mode(self):
        hud = self.p.hud
        hud.edit_mode = not hud.edit_mode

        if hud.edit_mode:
            return

        hud.drag_key = None
        hud.drag_origin_region = None
        hud.resize_key = None
