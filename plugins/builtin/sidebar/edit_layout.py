"""Built-in sidebar section: toggle layout edit mode.

Renders a single button that flips ``player.hud.edit_mode``. In edit
mode, draggable sidebar components can be moved between the sidepanel
and the floating ("free") region by clicking and dragging. Shift+Tab
is the keyboard shortcut for the same toggle.
"""
from __future__ import annotations


_KEY = 'builtin:edit_layout'


def _draw(sctx):
    p = sctx.player
    label = 'Exit edit mode' if p.hud.edit_mode else 'Edit layout'
    sctx.draw_button(label, 'toggle_edit_mode')


def register_sidebar(add):
    # Pinned to the bottom, priority 850 — sits between Scroll (800)
    # and Options (900) so it's reachable without scrolling.
    add('Edit layout', _draw, priority=850, key=_KEY, pin_bottom=True)
