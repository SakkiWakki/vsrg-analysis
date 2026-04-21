"""Sandboxed 'hello world' plugin.

Demonstrates:

  * The declarative component API — the plugin returns a ``Component``
    tree, the host renders it.
  * Per-plugin persisted config via ``host_api.plugin_config`` — the
    click counter below survives restarts, and updates to it from one
    window fan out to every other window live.
"""
from __future__ import annotations

import math

from analysis.plugins.host_api import plugin_config
from analysis.ui import Button, Column, Heading, Spacer, Text
from analysis.ui.render_sidebar import render


_KEY = 'sandboxed:hello'
_cfg = plugin_config(_KEY)


def _build():
    clicks = int(_cfg.get('clicks', 0))
    return Column((
        Spacer(),
        Heading('Sandboxed demo'),
        Text(f'pi = {math.pi:.4f}'),
        Text(f'clicks: {clicks}'),
        Button('hello!', 'example_sandboxed_hello'),
    ))


def _draw(sctx):
    render(sctx, _build())


def register_sidebar(add):
    add('Sandboxed demo', _draw, priority=550, key=_KEY)
