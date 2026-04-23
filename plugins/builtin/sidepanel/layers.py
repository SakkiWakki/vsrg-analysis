"""Built-in sidebar section: replay-layer visibility toggles."""
from __future__ import annotations

from analysis.components import Manifest, SURFACE_GUI
from analysis.player.render import theme
from plugins.builtin.sidepanel import SidebarFields

_CHECKBOX_INSET_X = 6
_CHECKBOX_INSET_Y = 3
_LABEL_INDENT = 22
_DEPTH_INDENT = 14
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
    layers = ctx.data.layer_tree()
    open_ = ctx.hud_flags.layers_panel_open
    listed = _listed_layers(layers)
    shown = sum(1 for state in listed if state.visible)

    ctx.spacer()
    ctx.draw_heading('Layers')
    ctx.draw_button(
        f'{"[-]" if open_ else "[+]"} {shown}/{len(listed)} visible',
        'toggle_layers_panel',
    )
    if not open_:
        return

    for state in listed:
        _draw_row(ctx, state)


def _listed_layers(states):
    out = []
    for state in states:
        if state.listed:
            out.append(state)
        out.extend(_listed_layers(state.children))
    return out


def _draw_row(ctx, state):
    checked = state.visible
    row = (0, ctx.y, ctx.w, _ROW_PANEL_H)
    indent = _LABEL_INDENT + (state.depth * _DEPTH_INDENT)
    ctx.button_at(row, '', 'toggle_layer', state.key)
    ctx.checkbox(_CHECKBOX_INSET_X,
                 ctx.y + _CHECKBOX_INSET_Y,
                 checked=checked)
    color = (theme.COLOR_PLUGIN_ENABLED if checked
             else theme.COLOR_PLUGIN_DISABLED)
    ctx.text(state.name, indent,
             ctx.y + theme.TEXT_BASELINE_ROW, color)
    ctx.y += _ROW_PANEL_H


def register_components(add):
    add(MANIFEST, _draw)
