from __future__ import annotations

from analysis.core import game as game_mod


def etterna_windows_for(judge_name='J4'):
    from analysis.games.etterna.judgment import windows_for
    return windows_for(judge_name)


def osu_mania_windows(od):
    from analysis.games.osu.judgment import windows_for
    return windows_for(od)


def prepare_replay_times(replay, bpms=None, sm_offset=0.0):
    return game_mod.get(replay['game']).prepare_replay_times(
        replay,
        bpms=bpms,
        sm_offset=sm_offset,
    )
