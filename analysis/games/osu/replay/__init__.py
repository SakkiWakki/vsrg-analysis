"""osu!mania replay parsing. Aligns `.osr` key events against `.osu`
chart hitobjects via an osu!mania Classic judgment simulation to
produce the same shape as `etterna_replay.parse_replay`:
`{noterows, offsets, columns, notetypes, misses, holds, keycount, ...}`."""
from analysis.games.osu.replay.chart import parse_osu_file, find_osu_by_hash
from analysis.games.osu.replay.osr import (parse_osr_events, rate_for_mods,
                                            OSU_MOD_DOUBLETIME,
                                            OSU_MOD_HALFTIME,
                                            OSU_MOD_NIGHTCORE,
                                            OSU_MOD_RANDOM)
from analysis.games.osu.replay.judge import (stable_hit_windows,
                                              simulate_mania,
                                              TAIL_RELEASE_LENIENCE)
from analysis.games.osu.replay.parse import parse_replay
from analysis.games.osu.replay.paths import find_osu_dirs, list_osu_profiles

__all__ = [
    'parse_replay',
    'parse_osu_file',
    'parse_osr_events',
    'find_osu_dirs',
    'find_osu_by_hash',
    'list_osu_profiles',
    'rate_for_mods',
    'stable_hit_windows',
    'simulate_mania',
    'TAIL_RELEASE_LENIENCE',
    'OSU_MOD_DOUBLETIME',
    'OSU_MOD_HALFTIME',
    'OSU_MOD_NIGHTCORE',
    'OSU_MOD_RANDOM',
]
