"""Built-in sidebar section: keyboard-shortcut hints."""
from __future__ import annotations

from analysis.components import Manifest, SURFACE_GUI
from plugins.builtin.sidepanel import SidebarFields


_HINTS = (
    'Space: pause',
    'L/R: seek',
    'Sh+L/R: seek10',
    'Up/Dn: scrollspd',
    '+/-: playspd',
    'M: mute',
    'Q: quit',
)

MANIFEST = Manifest(
    key='builtin:hints',
    name='Hints',
    supported_surfaces={SURFACE_GUI},
    plugin_fields={
        'sidebar': SidebarFields(
            priority=300,
            draggable=True,
            default_free_xy=(0.02, 0.55),
            default_size=(210, 160),
        ),
    },
)


def _draw(ctx):
    ctx.spacer()
    for h in _HINTS:
        ctx.draw_hint(h)


def register_components(add):
    add(MANIFEST, _draw)
