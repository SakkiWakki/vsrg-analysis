"""Per-layer visibility toggles.

The Qt renderer draws in ordered layers (background, lanes, judgment,
notes, chart_extras, miss_holds, ghost_taps, hud). This plugin reads a
per-layer on/off map from the config store and publishes it into
``ctx.plugin_data['layer_visibility']`` once per frame, before any
layer draws. Missing keys default to visible, so an empty config
matches the legacy "draw everything" behavior.

Config path: ``player.layer_visibility.<layer_name>`` (bool).
"""
from __future__ import annotations

from analysis.player.plugin_api import Stage

_LAYERS = (
    'background', 'lanes', 'judgment', 'notes',
    'chart_extras', 'miss_holds', 'ghost_taps', 'hud',
)


def _build_visibility(config):
    out = {}
    for name in _LAYERS:
        out[name] = bool(config.get(
            f'player.layer_visibility.{name}', True))
    return out


def _draw(ctx, stage):
    from analysis.config import get_config
    ctx.plugin_data['layer_visibility'] = _build_visibility(get_config())


def register(add):
    add('Layer visibility', _draw, stages=(Stage.PRE_FRAME,),
        priority=0, key='builtin:layer_visibility')
