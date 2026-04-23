"""Built-in sidebar section: toggle layout edit mode.

Renders a single button that flips ``player.hud.edit_mode``. In edit
mode, draggable sidebar components can be moved between the sidepanel
and the floating ("free") region by clicking and dragging. Shift+Tab
is the keyboard shortcut for the same toggle.
"""
from __future__ import annotations

from analysis.components import Manifest, SURFACE_GUI
from plugins.builtin.sidepanel import SidebarFields


MANIFEST = Manifest(
    key='builtin:edit_layout',
    name='Edit layout',
    supported_surfaces={SURFACE_GUI},
    plugin_fields={
        'sidebar': SidebarFields(
            priority=850,
            pin_bottom=True,
        ),
    },
)


def _draw(ctx):
    label = 'Exit edit mode' if ctx.hud_flags.edit_mode else 'Edit layout'
    ctx.draw_button(label, 'toggle_edit_mode')


def register_components(add):
    add(MANIFEST, _draw)
