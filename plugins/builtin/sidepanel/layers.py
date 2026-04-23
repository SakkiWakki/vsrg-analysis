"""Built-in sidebar section: per-layer visibility toggles.

Mirrors the renderer's ordered layer list. Each row is a checkbox-backed
hitbox that dispatches ``toggle_layer`` with the layer name; the Player
flips the matching config key, and the built-in ``layer_visibility``
replay plugin republishes the map on the next frame.
"""
from __future__ import annotations

from analysis.components import Manifest, SURFACE_GUI
from analysis.player.render import theme
from plugins.builtin.sidepanel import SidebarFields


_LAYERS = (
    ('background',   'Background'),
    ('lanes',        'Lanes'),
    ('judgment',     'Judgment line'),
    ('notes',        'Notes'),
    ('chart_extras', 'Chart extras'),
    ('miss_holds',   'Miss holds'),
    ('ghost_taps',   'Ghost taps'),
)

_CHECKBOX_INSET_X = 6
_CHECKBOX_INSET_Y = 3
_LABEL_INDENT = 22
_ROW_PANEL_H = 18

MANIFEST = Manifest(
    key='builtin:layers',
    name='Layers',
    supported_surfaces={SURFACE_GUI},
    plugin_fields={
        'sidebar': SidebarFields(
            priority=450,
            draggable=True,
            default_free_xy=(0.02, 0.78),
            default_size=(210, 180),
        ),
    },
)


def _draw(ctx):
    open_ = ctx.hud_flags.layers_panel_open
    shown = sum(1 for key, _ in _LAYERS if ctx.data.layer_visible(key))

    ctx.spacer()
    ctx.draw_heading('Layers')
    ctx.draw_button(
        f'{"[-]" if open_ else "[+]"} {shown}/{len(_LAYERS)} visible',
        'toggle_layers_panel',
    )
    if not open_:
        return

    for key, label in _LAYERS:
        checked = ctx.data.layer_visible(key)
        row = (ctx.col_x, ctx.y, ctx.w, _ROW_PANEL_H)
        ctx.button_at(row, '', 'toggle_layer', key)
        ctx.checkbox(ctx.col_x + _CHECKBOX_INSET_X,
                     ctx.y + _CHECKBOX_INSET_Y,
                     checked=checked)
        color = (theme.COLOR_PLUGIN_ENABLED if checked
                 else theme.COLOR_PLUGIN_DISABLED)
        ctx.text(label, ctx.col_x + _LABEL_INDENT,
                 ctx.y + theme.TEXT_BASELINE_ROW, color)
        ctx.y += _ROW_PANEL_H


def register_components(add):
    add(MANIFEST, _draw)
