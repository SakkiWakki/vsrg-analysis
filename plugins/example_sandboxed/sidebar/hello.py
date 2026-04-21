"""Sandboxed 'hello world' plugin.

Demonstrates the declarative component API: the plugin returns a
``Component`` tree, the host renders it. Compared to the old imperative
style (``sctx.draw_*`` calls), the tree is trivially testable — a
plugin's unit tests can snapshot the tree without needing a live Qt
sidebar.
"""
from __future__ import annotations

import math

from analysis.ui import Button, Column, Heading, Spacer, Text
from analysis.ui.render_sidebar import render


def _build():
    return Column((
        Spacer(),
        Heading('Sandboxed demo'),
        Text(f'pi = {math.pi:.4f}'),
        Button('hello!', 'example_sandboxed_hello'),
    ))


def _draw(sctx):
    render(sctx, _build())


def register_sidebar(add):
    add('Sandboxed demo', _draw, priority=550, key='sandboxed:hello')
