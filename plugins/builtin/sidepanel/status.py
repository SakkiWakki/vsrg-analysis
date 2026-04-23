"""Built-in sidebar section: time / speed / notes / keycount / SV / play state."""
from __future__ import annotations

from analysis.components import Manifest, SURFACE_GUI
from plugins.builtin.sidepanel import SidebarFields


MANIFEST = Manifest(
    key='builtin:status',
    name='Status',
    supported_surfaces={SURFACE_GUI},
    requires_data={
        't_now', 'play_rate', 'paused', 'note_count',
        'keycount', 'sv_enabled', 'sv_suspended', 'sv_sections',
    },
    plugin_fields={
        'sidebar': SidebarFields(
            priority=100,
            draggable=True,
            default_free_xy=(0.02, 0.28),
            default_size=(210, 130),
        ),
    },
)


def _draw(ctx):
    sv_enabled = ctx.data.sv_enabled()
    sv_suspended = ctx.data.sv_suspended()
    sv_sections = ctx.data.sv_sections()

    if not sv_sections or sv_suspended:
        sv_line = 'SV: n/a'
    else:
        sv_line = 'SV: on' if sv_enabled else 'SV: off'

    for line in (
        f't = {ctx.data.t_now():+7.3f}s',
        f'speed = {ctx.data.play_rate():.2f}x',
        f'notes = {ctx.data.note_count()}',
        f'keycount = {ctx.data.keycount()}',
        sv_line,
        'PAUSED' if ctx.data.paused() else 'PLAYING',
    ):
        ctx.draw_text(line)


def register_components(add):
    add(MANIFEST, _draw)
