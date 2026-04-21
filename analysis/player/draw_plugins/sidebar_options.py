"""Built-in sidebar section: per-replay display options (SV / Skin / hits /
pitch-correct). Pinned to the bottom of the sidebar under Scroll."""
from __future__ import annotations


def _draw_options(sctx):
    p = sctx.player
    status = getattr(p, '_ui_status',
                     {'audio_ready': False, 'pitch_correct': True})

    sctx.draw_heading('Options')

    if not p.sv_sections or p.sv_suspended():
        sv_label = 'SV: n/a'
        sv_enabled = False
    else:
        sv_label = f'SV: {"on" if p.sv_enabled else "off"}'
        sv_enabled = True
    sctx.draw_button(sv_label, 'toggle_sv', enabled=sv_enabled)

    sctx.draw_button(f'Skin: {p.skin}', 'cycle_skin')

    # "Display hits: on" means hits are hidden after press (press_hide=True)
    # — the toggle reads intuitively ("yes, suppress hits on contact")
    # rather than mirroring the internal flag name.
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
    add('Options', _draw_options, priority=900, key='builtin:options',
        pin_bottom=True)
