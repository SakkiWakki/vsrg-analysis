"""Mines must move exactly like the taps around them.

Chart-stream sprites (mines/lifts/fakes) project to screen y through the
same primitive taps use (`batch_time_to_y`). The regression here: Quaver
mines dropped their TimingGroup on the way to the renderer, so a mine in
a group whose SV opposes the `$Default` stream scrolled in the wrong
direction (down while its column's taps scrolled up).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.games.quaver.parse import _build_mine_arrays, _group_notes_by_col
from analysis.player.init.notes_model import (NotesModel, copy_chart_streams,
                                               stream_groups_or_none)
from analysis.player.render.layers.chart_extras import _chart_stream_ys
from analysis.player.sv.engine import QuaverSVEngine
from analysis.player.sv.render import SvRenderController


NOTE_T = 3.0


def _flip_engine():
    """Default stream scrolls forward; the `flip` group scrolls backward
    (negative SV) the whole chart."""
    return QuaverSVEngine([], groups={
        '$Default': {'sections': [(0.0, 1.0)], 'initial_velocity': 1.0},
        'flip': {'sections': [(0.0, -1.0)], 'initial_velocity': -1.0},
    })


def _player_with_mine(group):
    engine = _flip_engine()
    player = SimpleNamespace(H=800, hit_line_y_frac=0.5, scroll_speed=1.0,
                             judge_y_px=lambda: 400.0,
                             _sv_engine=engine, sv_enabled=True)
    ctrl = SvRenderController(player)
    player.batch_time_to_y = ctrl.batch_time_to_y

    notes = NotesModel()
    copy_chart_streams(notes, {
        'mine_times': [NOTE_T],
        'mine_cols': [0],
        'mine_until': [np.inf],
        'mine_groups': [group],
    })
    player.notes = notes
    ctrl.build_ghost_sv_caches()
    return player, ctrl


def _frame_at(engine, t):
    return SimpleNamespace(use_sv=True, raw_t=t,
                           visual_cum_now=engine.cumulative_at(t),
                           render_multiplier=engine.render_multiplier_at(t))


def _tap_and_mine_y(player, ctrl, group, t):
    frame = _frame_at(player._sv_engine, t)
    tap_y = float(ctrl.batch_time_to_y(
        np.array([NOTE_T]), frame,
        groups=np.array([group], dtype=object))[0])
    ctx = SimpleNamespace(player=player, frame=frame)
    mine_y = float(_chart_stream_ys(
        ctx, player.notes.mine_times, player.notes.mine_sv,
        stream_groups_or_none(player.notes.mine_groups),
        np.array([0], dtype=np.intp))[0])
    return tap_y, mine_y


@pytest.mark.parametrize('group', ['$Default', 'flip'])
def test_mine_y_equals_tap_y_in_same_group(group):
    player, ctrl = _player_with_mine(group)
    for t in (1.0, 2.0):
        tap_y, mine_y = _tap_and_mine_y(player, ctrl, group, t)
        assert mine_y == pytest.approx(tap_y), (
            f'group={group} t={t}: mine at {mine_y}, tap at {tap_y}')


def test_mine_displacement_matches_tap_under_reversed_group():
    """The visible symptom: as the playhead advances, a `flip`-group tap
    approaches the judgment line while the (buggy) mine walked the other
    way. Displacement over the same interval must be identical."""
    player, ctrl = _player_with_mine('flip')
    tap_y1, mine_y1 = _tap_and_mine_y(player, ctrl, 'flip', 1.0)
    tap_y2, mine_y2 = _tap_and_mine_y(player, ctrl, 'flip', 2.0)
    assert mine_y2 - mine_y1 == pytest.approx(tap_y2 - tap_y1)


def test_quaver_parse_carries_mine_timing_group():
    hitobjects = [
        {'time': 1000, 'column': 0, 'end_time': None, 'is_mine': True,
         'is_hold': False, 'group': 'flip'},
        {'time': 2000, 'column': 1, 'end_time': None, 'is_mine': True,
         'is_hold': False, 'group': '$Default'},
    ]
    _by_col, _holds, mines_by_col = _group_notes_by_col(hitobjects, 4)
    arrays = _build_mine_arrays(mines_by_col, [])
    assert list(arrays['mine_groups']) == ['flip', '$Default']

    model = NotesModel()
    copy_chart_streams(model, arrays)
    assert list(model.mine_groups) == ['flip', '$Default']


def test_mine_sv_cache_projects_in_own_group():
    player, ctrl = _player_with_mine('flip')
    expected = player._sv_engine.project_times(
        np.array([NOTE_T]), groups=np.array(['flip'], dtype=object))
    assert player.notes.mine_sv == pytest.approx(expected)
