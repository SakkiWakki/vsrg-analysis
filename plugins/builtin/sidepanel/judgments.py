"""Built-in judgments component.

Shows per-window hit counts for the current judge, with nudge buttons
to shift the judge on games that support it.

Ported to the unified component API: one ``draw(ctx)`` renders on the
sidebar (where ``judge_nudge`` clicks do route back to the player) and
on any future overlay that exposes ``judgment_counts`` in its game
state. On the gamescope overlay today the buttons render as chrome
only since clicks don't route back — that's fine, the live readout is
the useful part there.
"""
from __future__ import annotations

from analysis.components import (
    ComponentManifest,
    DataNotAvailable,
    SURFACE_OVERLAY,
    SURFACE_GUI,
)
from analysis.components.overlay_backend import OverlayFields
from analysis.player.render import theme
from plugins.builtin.sidepanel import SidebarFields


_NUDGE_BTN_W = 28
# Etterna judges are integer steps — sign is all that matters. osu OD is
# float; ±0.1 per click mirrors the rate slider's feel.
_OSU_OD_STEP = 0.1


def _draw(ctx):
    ctx.spacer()
    ctx.draw_heading('Judgments')

    # Switcher row. The data needed for the label (judge_label) is
    # replay-side only — overlay data source raises DataNotAvailable.
    # Skip the row cleanly on surfaces without it.
    try:
        judge_label = ctx.data.judge_label()
        game = ctx.data.game()
        step = 1.0 if game == 'etterna' else _OSU_OD_STEP
        slots = ctx.split_row(n=3)
        left, mid, right = slots
        ctx.button_at((left[0], ctx.y, _NUDGE_BTN_W, theme.ROW_BUTTON_H),
                      '−', 'judge_nudge', -step, center=True)
        ctx.button_at((right[0] + right[1] - _NUDGE_BTN_W, ctx.y,
                       _NUDGE_BTN_W, theme.ROW_BUTTON_H),
                      '+', 'judge_nudge', step, center=True)
        readout_x = _NUDGE_BTN_W
        readout_w = ctx.w - 2 * _NUDGE_BTN_W
        ctx.text(judge_label,
                 readout_x + max(0, (readout_w - len(judge_label) * 6) // 2),
                 ctx.y + theme.TEXT_BASELINE_BUTTON,
                 color=theme.BTN_FG)
        ctx.y += theme.ROW_TALL_H
    except DataNotAvailable:
        # Overlay side: no judge concept, skip the switcher.
        pass

    counts = ctx.data.judgment_counts()
    try:
        colors = ctx.data.judgment_colors()
    except DataNotAvailable:
        # Overlay side: fall back to a single color for all windows.
        colors = {}

    try:
        windows = ctx.data.judgment_windows()
    except DataNotAvailable:
        # Overlay-only path: just show counts in the order the source
        # iterates.
        windows = [(name, 0.0) for name in counts if name != 'miss']

    default_color = theme.BTN_FG
    for name, width_s in windows:
        count = counts.get(name, 0)
        if width_s:
            line = f'{name:<6}  ±{width_s*1000:5.1f}ms  n={count}'
        else:
            line = f'{name:<10}  n={count}'
        ctx.draw_text(line, color=colors.get(name, default_color))

    miss_count = counts.get('miss', 0)
    ctx.draw_text(f'miss             n={miss_count}',
                  color=colors.get('miss', (220, 60, 60)))


MANIFEST = ComponentManifest(
    key='builtin:judgments',
    name='Judgments',
    supported_surfaces={SURFACE_GUI, SURFACE_OVERLAY},
    requires_data={'judgment_counts'},
    optional_data={'judgment_windows', 'judgment_colors', 'judge_label',
                   'game'},
    plugin_fields={
        'sidebar': SidebarFields(
            priority=200,
            draggable=True,
            default_free_xy=(0.02, 0.04),
            default_size=(210, 200),
        ),
        'overlay': OverlayFields(
            default_xy=(0.02, 0.04),
            default_size=(0.18, 0.22),
        ),
    },
)


def register_components(add):
    add(MANIFEST, _draw)
