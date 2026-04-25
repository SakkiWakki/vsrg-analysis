"""Quaver game adapter ; scroll mode only for now.

Everything else is a stub: parse_replay/judgement/library scan raise or
return empty so the game registers, its scroll mode shows up in the HUD
cycle, and cross-game comparisons work. Replay parsing, .qua chart
handling, audio resolution and judgement windows can be filled in
incrementally.
"""
from __future__ import annotations

from analysis.core.game import GameAdapter
from analysis.player import scroll


class QuaverAdapter(GameAdapter):
    name = 'quaver'

    def parse_replay(self, path, chart_path=None):
        raise NotImplementedError('Quaver replay parsing not implemented yet')

    def judgement_windows(self, replay, **_):
        raise NotImplementedError

    def judge_label(self, replay, **_):
        return ''

    def default_scroll_mode(self):
        return 'quaver'

    def viz_windows(self, replay, **_):
        raise NotImplementedError


# --- Quaver scroll mode -----------------------------------------------------
# Ported from Quaver's TimingGroupControllerKeys.ScrollSpeed + TrackRounding.
# `value` is the user-facing scroll speed shown in Quaver's options menu
# (5.0 to 100.0, default 15.0). Quaver stores this internally as an int
# 10x larger (50 to 1000, default 150) and divides by 10 in its formula;
# we skip that round-trip and work in the displayed scale directly.
_QUAVER_SKIN_SCALE = 1920.0 / 1366.0
_QUAVER_BASE_WINDOW_H = 768.0
_MS_PER_S = 1000.0


def _quaver_pxps_at_base_window(value):
    scroll_speed = value / 20.0 * _QUAVER_SKIN_SCALE
    return scroll_speed * _MS_PER_S


def _quaver_to_pxps(value, opts, p):
    window_scale = p.H / _QUAVER_BASE_WINDOW_H
    return _quaver_pxps_at_base_window(float(value)) * window_scale


def _quaver_from_pxps(pxps, opts, p):
    window_scale = p.H / _QUAVER_BASE_WINDOW_H
    return pxps / (_quaver_pxps_at_base_window(1.0) * window_scale)


scroll.register(scroll.ScrollMode(
    key='quaver',
    label='Quaver',
    game='quaver',
    to_pxps=_quaver_to_pxps,
    from_pxps=_quaver_from_pxps,
    default_value=15.0,
    value_bounds=(5.0, 100.0),
    nudge=scroll.integer_step_nudge,
    format_value=lambda v: (f'Q {int(v)}' if abs(v - round(v)) < 1e-4
                            else f'Q {v:.1f}'),
))


ADAPTER = QuaverAdapter()
