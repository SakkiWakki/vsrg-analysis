"""Sandboxed 'hello world' plugin. Uses only host API + stdlib math."""
from __future__ import annotations

import math

from analysis.player import theme


def _draw(sctx):
    sctx.spacer()
    sctx.draw_heading('Sandboxed demo')
    sctx.draw_text(f'pi = {math.pi:.4f}', color=theme.BTN_FG)
    sctx.draw_button('hello!', 'example_sandboxed_hello')


def register_sidebar(add):
    add('Sandboxed demo', _draw, priority=550, key='sandboxed:hello')
