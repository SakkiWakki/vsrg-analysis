"""Built-in sidebar section: time / speed / notes / keycount / SV / play state."""
from __future__ import annotations


def _draw_status(sctx):
    p = sctx.player
    if not p.sv_sections or p.sv_suspended():
        sv_line = 'SV: n/a'
    else:
        sv_line = 'SV: on' if p.sv_enabled else 'SV: off'
    for line in (
        f't = {sctx.render_ctx.t_now:+7.3f}s',
        f'speed = {p.play_rate:.2f}x',
        f'notes = {len(p.times)}',
        f'keycount = {p.keycount}',
        sv_line,
        'PAUSED' if p.paused else 'PLAYING',
    ):
        sctx.draw_text(line)


def register_sidebar(add):
    add('Status', _draw_status, priority=100, key='builtin:status',
        draggable=True, default_free_xy=(0.02, 0.28),
        default_size=(210, 130))
