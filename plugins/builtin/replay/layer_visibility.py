"""Publish the current replay-layer tree into ``ctx.plugin_data``."""
from __future__ import annotations

from analysis.player.plugin_api import Stage

def _flatten(states, out):
    for state in states:
        out[state.key] = state.visible
        _flatten(state.children, out)


def _draw(ctx, stage):
    try:
        registry = ctx.player.plugins.layers
    except Exception:
        return
    tree = registry.layer_tree()
    ctx.plugin_data['layer_visibility_tree'] = tree
    flat = {}
    _flatten(tree, flat)
    ctx.plugin_data['layer_visibility'] = flat


def register(add):
    add('Layer visibility', _draw, stages=(Stage.PRE_FRAME,),
        priority=0, key='builtin:layer_visibility')
