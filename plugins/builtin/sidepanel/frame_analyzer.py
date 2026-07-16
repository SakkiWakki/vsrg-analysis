"""Built-in sidebar section: render-frame timing analyzer.

Displays render cadence and jitter from ``data.frame_stats()``, which
the renderer ticks once per rendered frame. The section's own draw runs
at the HUD cache cadence, so it must not measure its own call rate.
"""
from __future__ import annotations

from analysis.components import Manifest, SURFACE_GUI
from plugins.builtin.sidepanel import SidebarFields


MANIFEST = Manifest(
    key='builtin:frame_analyzer',
    name='Frame Analyzer',
    supported_surfaces={SURFACE_GUI},
    requires_data={'t_now', 'paused', 'audio_status', 'frame_stats'},
    plugin_fields={
        'sidebar': SidebarFields(
            priority=110,
            draggable=True,
            default_free_xy=(0.02, 0.44),
            default_size=(240, 185),
        ),
    },
)


def _ms(v: float) -> str:
    return f'{v * 1000.0:6.2f}ms'


def _fps(v: float) -> str:
    if v <= 1e-9:
        return '   inf'
    return f'{1.0 / v:6.1f}'


def _audio_line(count: int, last: str) -> str:
    """Format the audio-callback status line.

    `count` is the total of PortAudio status events plus ring-empty
    events seen since the engine started. `last` is a free-form string
    carrying the most recent flag plus a `fill=NN%` gauge of the
    producer ring. The gauge is shown even when count is zero so a
    chronically-near-empty ring (producer barely keeping up) is
    visible before it actually causes an underflow."""
    extra = last.strip()
    if count <= 0:
        return f'audio: ok    {extra}' if extra else 'audio: ok'
    return f'audio:  {count}  {extra}'


def _draw(ctx):
    s = ctx.data.frame_stats()

    state = 'PAUSED' if ctx.data.paused() else 'PLAYING'
    audio_count, audio_last = ctx.data.audio_status()
    audio_line = _audio_line(audio_count, audio_last)
    if s is None:
        ctx.draw_text('Frame Analyzer')
        ctx.draw_text('collecting samples...')
        ctx.draw_text(f'state: {state}')
        ctx.draw_text(audio_line)
        return

    lines = (
        'Frame Analyzer',
        f'state: {state}',
        f'fps  now/avg = {_fps(s["inst_dt"])} / {_fps(s["avg_dt"])}',
        f'dt   now/avg = {_ms(s["inst_dt"])} / {_ms(s["avg_dt"])}',
        f'dt   p95/p99 = {_ms(s["p95_dt"])} / {_ms(s["p99_dt"])}',
        f'dt   min/max = {_ms(s["min_dt"])} / {_ms(s["max_dt"])}',
        f'jitter (std) = {_ms(s["std_dt"])}',
        f'hitches (2x) = {s["hitches"]} / {s["n"]}',
        audio_line,
    )
    for line in lines:
        ctx.draw_text(line)


def register_components(add):
    add(MANIFEST, _draw)
