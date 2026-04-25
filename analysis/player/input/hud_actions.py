from __future__ import annotations

from analysis.player.input.layout_edit import rect_contains


class HudActionController:
    def __init__(self, player):
        self.p = player

    def handle_mouse_down(self, x, y):
        for rect, action, payload in reversed(self.p.hud.hitboxes):
            if not rect_contains(rect, x, y):
                continue

            edit_mode = self.p.hud.edit_mode
            is_drag_grab = edit_mode and action == 'begin_drag_section'
            is_resize_grab = edit_mode and action == 'begin_resize_section'
            is_dispatchable = not edit_mode or action == 'toggle_edit_mode'

            if is_drag_grab:
                self.p._begin_drag_section(payload, x, y, rect)
            elif is_resize_grab:
                self.p._begin_resize_section(payload, x, y)
            elif is_dispatchable:
                self.dispatch(action, payload)

            return True

        return False

    def dispatch(self, action, payload):
        match action:
            case 'toggle_plugin_panel':
                self.p.hud.plugin_panel_open = not self.p.hud.plugin_panel_open

            case 'toggle_plugin':
                self.p.plugins.toggle_enabled(payload)

            case 'scroll_nudge':
                self.p.nudge_scroll(payload)
                self.notify_scroll_change()

            case 'cycle_scroll_mode':
                self.cycle_scroll_mode()

            case 'rate_nudge':
                self.p.nudge_rate(payload)
                self.notify_scroll_change()

            case 'judge_nudge':
                self.p.nudge_judge(payload)
                self.notify_scroll_change()

            case 'cycle_game':
                self.p.cycle_game()
                self.notify_scroll_change()

            case 'cycle_sv_engine':
                sv = getattr(self.p, 'sv_render', None)
                if sv is not None and hasattr(sv, 'cycle_engine'):
                    sv.cycle_engine()
                    self.notify_scroll_change()

            case 'toggle_layer':
                self.p.plugins.layers.toggle(payload)

            case 'toggle_layers_panel':
                self.p.hud.layers_panel_open = not getattr(
                    self.p.hud,
                    'layers_panel_open',
                    False,
                )

            case 'toggle_flyout':
                self.p.hud.open_flyout = (
                    None if self.p.hud.open_flyout == payload else payload
                )

            case 'toggle_edit_mode':
                self.p.toggle_edit_mode()

            case (
                'toggle_sv'
                | 'cycle_skin'
                | 'toggle_press_hide'
                | 'toggle_pitch'
                | 'edit_scroll_value'
            ):
                self.notify_hud_action(action, payload)

    def cycle_scroll_mode(self):
        keys = self.p._available_mode_keys()
        if not keys:
            return

        cur = self.p.scroll_mode if self.p.scroll_mode in keys else keys[0]
        self.p.set_scroll_mode(keys[(keys.index(cur) + 1) % len(keys)])
        self.notify_scroll_change()

    def notify_scroll_change(self):
        self.p.events.emit('scroll_changed')

    def notify_hud_action(self, action, payload):
        self.p.events.emit('hud_action', (action, payload))
