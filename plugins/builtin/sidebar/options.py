"""Built-in sidebar section: per-replay display options. Rendered as a
flyout — the sidebar shows a one-line ``Options ▸`` button that opens a
panel to the left of the sidebar with the full control set."""
from __future__ import annotations


_KEY = 'builtin:options'


def _collapsed_header(sctx):
    p = sctx.player
    open_ = p.hud.open_flyout == _KEY
    sctx.draw_button(f'Options {"▾" if open_ else "▸"}',
                     'toggle_flyout', _KEY)


def _draw_flyout(sctx):
    p = sctx.player
    status = getattr(p, '_ui_status',
                     {'audio_ready': False, 'pitch_correct': True})

    if not p.sv_sections or p.sv_suspended():
        sv_label = 'SV: n/a'
        sv_enabled = False
    else:
        sv_label = f'SV: {"on" if p.sv_enabled else "off"}'
        sv_enabled = True
    sctx.draw_button(sv_label, 'toggle_sv', enabled=sv_enabled)

    sctx.draw_button(f'Skin: {p.skin}', 'cycle_skin')

    hits_label = f'Display hits: {"on" if p.press_hide else "off"}'
    sctx.draw_button(hits_label, 'toggle_press_hide')

    if status.get('audio_ready', False):
        pitch_label = ('Pitch-correct: '
                       f'{"on" if status.get("pitch_correct", True) else "off"}')
        pitch_enabled = True
    else:
        pitch_label = 'Pitch-correct: n/a'
        pitch_enabled = False
    sctx.draw_button(pitch_label, 'toggle_pitch', enabled=pitch_enabled)


def register_sidebar(add):
    add('Options', _collapsed_header, priority=900, key=_KEY,
        pin_bottom=True, draw_expanded=_draw_flyout)
