"""Built-in sidebar section: per-layer visibility toggles.

Mirrors the renderer's ordered layer list. Each row is a checkbox-backed
hitbox that dispatches ``toggle_layer`` with the layer name; the Player
flips the matching config key, and the built-in ``layer_visibility``
replay plugin republishes the map on the next frame.
"""
from __future__ import annotations

from analysis.config import get_config
from analysis.player.render import theme


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


def _draw_layers_panel(sctx):
    p = sctx.player
    cfg = get_config()
    open_ = getattr(p.hud, 'layers_panel_open', False)

    sctx.spacer()
    shown = sum(1 for key, _ in _LAYERS
                if cfg.get(f'player.layer_visibility.{key}', True))
    sctx.draw_button(
        f'{"[-]" if open_ else "[+]"} Layers {shown}/{len(_LAYERS)}',
        'toggle_layers_panel',
    )
    if not open_:
        return

    for key, label in _LAYERS:
        checked = bool(cfg.get(f'player.layer_visibility.{key}', True))
        row = (sctx.col_x, sctx.y, sctx.col_w, _ROW_PANEL_H)
        sctx.add_hitbox(row, 'toggle_layer', key)
        sctx.checkbox(sctx.col_x + _CHECKBOX_INSET_X,
                      sctx.y + _CHECKBOX_INSET_Y,
                      checked=checked)
        color = (theme.COLOR_PLUGIN_ENABLED if checked
                 else theme.COLOR_PLUGIN_DISABLED)
        sctx.text(label, sctx.col_x + _LABEL_INDENT,
                  sctx.y + theme.TEXT_BASELINE_ROW, color)
        sctx.y += _ROW_PANEL_H


def register_sidebar(add):
    add('Layers', _draw_layers_panel, priority=450,
        key='builtin:layers',
        draggable=True, default_free_xy=(0.02, 0.78),
        default_size=(210, 180))
