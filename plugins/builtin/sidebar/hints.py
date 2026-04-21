"""Built-in sidebar section: keyboard-shortcut hints."""
from __future__ import annotations


_HINTS = (
    'Space: pause',
    'L/R: seek',
    'Sh+L/R: seek10',
    'Up/Dn: scrollspd',
    '+/-: playspd',
    'M: mute',
    'Q: quit',
)


def _draw_hints(sctx):
    sctx.spacer()
    for h in _HINTS:
        sctx.draw_hint(h)


def register_sidebar(add):
    add('Hints', _draw_hints, priority=300, key='builtin:hints')
