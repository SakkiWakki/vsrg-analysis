"""Unplayed charts: the shared post-pass that emits one chart-only
entry per catalogued chart with no score, each game's catalogue, and
the autoplay synthesis that opens one."""
from pathlib import Path

import numpy as np
import pytest

from analysis.core.unplayed import (has_unplayed, merge_unplayed,
                                    unplayed_entries)
from analysis.games.osu import adapter as osu_adapter
from analysis.games.osu.replay import autoplay_replay as osu_autoplay
from analysis.games.quaver import adapter as quaver_adapter
from analysis.games.quaver.parse import autoplay_replay as quaver_autoplay


_OSU_MANIA = """osu file format v14

[General]
AudioFilename: song.mp3
Mode: 3

[Metadata]
Title:Testsong
Artist:Tester
Creator:Mapper
Version:Insane

[Difficulty]
CircleSize:4
OverallDifficulty:8

[TimingPoints]
0,300,4,1,0,100,1,0

[HitObjects]
64,192,0,1,0,0:0:0:0:
192,192,0,1,0,0:0:0:0:
320,192,500,128,0,1000:0:0:0:0:
448,192,1200,1,0,0:0:0:0:
"""

_OSU_STD = _OSU_MANIA.replace('Mode: 3', 'Mode: 0')

_QUA = """AudioFile: song.mp3
Title: Testsong
Artist: Tester
Creator: Mapper
DifficultyName: Insane
Mode: Keys4
BPMDoesNotAffectScrollVelocity: true
InitialScrollVelocity: 1
TimingPoints:
- StartTime: 0
  Bpm: 200
HitObjects:
- StartTime: 0
  Lane: 1
- StartTime: 200
  Lane: 2
- StartTime: 500
  EndTime: 1000
  Lane: 3
- StartTime: 1200
  Lane: 4
  Type: Mine
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return p


# --- osu! ------------------------------------------------------------------


def test_osu_autoplay_is_perfect(tmp_path):
    osu = _write(tmp_path, 'chart.osu', _OSU_MANIA)
    replay = osu_autoplay(osu)

    assert list(replay['noterows']) == [0, 0, 500, 1200]
    assert list(replay['columns']) == [0, 1, 2, 3]
    assert not replay['offsets'].any()
    assert not replay['misses'].any()
    assert not replay['miss_pressed'].any()
    # The one LN (col 2, 500->1000) shows up in holds + hold_releases.
    assert replay['holds'] == [(500, 2, 1000)]
    assert replay['ghost_taps'] == []
    assert replay['miss_holds'] == []
    assert replay['keycount'] == 4
    assert replay['mods'] == 0
    assert replay['chart_path'] == str(osu)


def test_osu_autoplay_matches_real_parse_shape(tmp_path):
    """Every key a real osu replay dict carries is present with a
    matching dtype, so the pipeline can't tell a synth from a play."""
    osu = _write(tmp_path, 'chart.osu', _OSU_MANIA)
    # A real parse via a hand-built .osr is heavy; instead assert the
    # autoplay dict is a strict superset in the keys the renderer reads.
    replay = osu_autoplay(osu)
    for key in ('noterows', 'offsets', 'columns', 'notetypes', 'misses',
                'miss_pressed', 'hold_releases', 'sv', 'holds', 'keycount',
                'chart_path', 'chart_meta', 'od', 'mods'):
        assert key in replay, key
    for key in ('noterows', 'offsets', 'columns', 'notetypes', 'misses'):
        assert isinstance(replay[key], np.ndarray)
    assert replay['sv'].engine_key == 'osu_time'


def test_osu_adapter_dispatches_osu_path_to_autoplay(tmp_path):
    osu = _write(tmp_path, 'chart.osu', _OSU_MANIA)
    replay = osu_adapter.ADAPTER.parse_replay(str(osu))
    assert replay['chart_path'] == str(osu)
    assert not replay['misses'].any()


_OSU_INDEX = {
    '/songs/played.osu': (1.0, 10, 'HASHPLAYED',
                          {'song': 'A - Played', 'steps': 'Hard',
                           'creator': 'M', 'keycount': 4, 'mode': 3,
                           'od': 7.0}),
    '/songs/unplayed.osu': (2.0, 20, 'HASHUNPLAYED',
                            {'song': 'B - Fresh', 'steps': 'Insane',
                             'creator': 'N', 'keycount': 7, 'mode': 3,
                             'od': 9.0}),
    '/songs/std.osu': (3.0, 30, 'HASHSTD',
                       {'song': 'C - Circles', 'steps': 'Extra',
                        'creator': 'O', 'keycount': 5, 'mode': 0,
                        'od': 5.0}),
}


def test_osu_unplayed_entries_dedup_and_mania_filter(monkeypatch):
    monkeypatch.setattr(osu_adapter, '_chart_index',
                        lambda progress=None: _OSU_INDEX)
    played = [{'game': 'osu', 'beatmap_hash': 'HASHPLAYED'}]

    entries = unplayed_entries(osu_adapter.ADAPTER, played)
    assert len(entries) == 1
    e = entries[0]
    assert e['unplayed'] is True
    assert e['game'] == 'osu'
    assert e['replay_path'] == '/songs/unplayed.osu'
    assert e['chart_path'] == '/songs/unplayed.osu'
    assert e['beatmap_hash'] == 'HASHUNPLAYED'
    assert e['song'] == 'B - Fresh'
    assert e['keycount'] == 7
    assert e['od'] == 9.0
    assert e['wife'] == 0.0
    assert e['grade'] == ''
    assert e['datetime'] == ''


def test_osu_unplayed_empty_index(monkeypatch):
    monkeypatch.setattr(osu_adapter, '_chart_index',
                        lambda progress=None: {})
    assert unplayed_entries(osu_adapter.ADAPTER, []) == []


# --- Quaver ----------------------------------------------------------------


def test_quaver_autoplay_is_perfect(tmp_path):
    qua = _write(tmp_path, 'chart.qua', _QUA)
    replay = quaver_autoplay(qua)

    # Taps at col 0,1,3 and one LN at col 2; the mine (col 3, t=1200)
    # never enters the judgment stream.
    assert list(replay['noterows']) == [0, 200, 500]
    assert list(replay['columns']) == [0, 1, 2]
    assert not replay['offsets'].any()
    assert not replay['misses'].any()
    assert replay['holds'] == [(500, 2, 1000)]
    assert replay['keycount'] == 4
    assert replay['mods'] == 0
    assert replay['chart_path'] == str(qua)
    # Mine is charted but has zero detonations in a flawless play.
    assert list(replay['mine_cols']) == [3]
    assert list(replay['mine_hit_idx']) == []


def test_quaver_autoplay_matches_real_parse_shape(tmp_path):
    qua = _write(tmp_path, 'chart.qua', _QUA)
    replay = quaver_autoplay(qua)
    for key in ('noterows', 'offsets', 'columns', 'notetypes', 'misses',
                'miss_pressed', 'hold_releases', 'sv', 'holds', 'keycount',
                'chart_path', 'chart_meta', 'judge', 'mods'):
        assert key in replay, key
    assert replay['sv'].engine_key == 'quaver_time'
    # Per-note group array parity: one group id per judged note.
    assert replay['sv'].note_groups is not None
    assert len(replay['sv'].note_groups) == len(replay['noterows'])


def test_quaver_adapter_dispatches_qua_path_to_autoplay(tmp_path):
    qua = _write(tmp_path, 'chart.qua', _QUA)
    replay = quaver_adapter.ADAPTER.parse_replay(str(qua))
    assert replay['chart_path'] == str(qua)
    assert not replay['misses'].any()


_QUAVER_INDEX = {
    '/songs/played.qua': (1.0, 10, 'QHASHPLAYED',
                          {'song': 'A - Played', 'steps': 'Hard',
                           'creator': 'M', 'keycount': 4}),
    '/songs/unplayed.qua': (2.0, 20, 'QHASHUNPLAYED',
                            {'song': 'B - Fresh', 'steps': 'Insane',
                             'creator': 'N', 'keycount': 7}),
}


def test_quaver_unplayed_entries_dedup(monkeypatch):
    monkeypatch.setattr(quaver_adapter, '_chart_index',
                        lambda progress=None: _QUAVER_INDEX)
    played = [{'game': 'quaver', 'beatmap_hash': 'QHASHPLAYED'}]

    entries = unplayed_entries(quaver_adapter.ADAPTER, played)
    assert len(entries) == 1
    e = entries[0]
    assert e['unplayed'] is True
    assert e['game'] == 'quaver'
    assert e['replay_path'] == '/songs/unplayed.qua'
    assert e['beatmap_hash'] == 'QHASHUNPLAYED'
    assert e['keycount'] == 7
    assert e['wife'] == 0.0
    assert e['datetime'] == ''


def test_quaver_unplayed_empty_index(monkeypatch):
    monkeypatch.setattr(quaver_adapter, '_chart_index',
                        lambda progress=None: {})
    assert unplayed_entries(quaver_adapter.ADAPTER, []) == []


# --- the shared post-pass --------------------------------------------------


class _FakeAdapter:
    name = 'fake'
    unplayed_key = 'chart_id'

    def __init__(self, catalogue):
        self._catalogue = catalogue

    def chart_catalogue(self, progress=None):
        return [dict(c) for c in self._catalogue]


def test_unplayed_zeroes_the_score_fields():
    adapter = _FakeAdapter([{'chart_id': 'A', 'song': 'x', 'mtime': 1.0}])
    e = unplayed_entries(adapter, [])[0]
    assert (e['game'], e['unplayed']) == ('fake', True)
    assert (e['wife'], e['rate'], e['maxcombo']) == (0.0, 1.0, 0)
    assert (e['grade'], e['datetime']) == ('', '')
    assert (e['judgments'], e['ssrs']) == ({}, {})


def test_catalogue_fields_win_over_the_zeroed_defaults():
    adapter = _FakeAdapter([{'chart_id': 'A', 'rate': 2.0}])
    assert unplayed_entries(adapter, [])[0]['rate'] == 2.0


def test_unplayed_skips_charts_with_a_blank_identity():
    adapter = _FakeAdapter([{'chart_id': ''}, {'song': 'no id'}])
    assert unplayed_entries(adapter, []) == []


def test_no_unplayed_key_means_no_pass():
    adapter = _FakeAdapter([{'chart_id': 'A'}])
    adapter.unplayed_key = None
    assert unplayed_entries(adapter, []) == []


def test_merge_rederives_the_set_so_a_new_play_retires_its_chart():
    adapter = _FakeAdapter([{'chart_id': 'A'}, {'chart_id': 'B'}])
    cached = [{'chart_id': 'A', 'unplayed': True},
              {'chart_id': 'B', 'unplayed': True}]
    merged = merge_unplayed(adapter, cached, [{'chart_id': 'A'}])

    assert not has_unplayed([merged[0]])
    still_unplayed = [e['chart_id'] for e in merged if e.get('unplayed')]
    assert still_unplayed == ['B']


def test_has_unplayed_distinguishes_a_pre_feature_cache():
    assert not has_unplayed([{'chart_id': 'A'}])
    assert has_unplayed([{'chart_id': 'A'}, {'unplayed': True}])


# --- Etterna ---------------------------------------------------------------

_SM = """
#TITLE:Testsong;
#ARTIST:Tester;
#OFFSET:0.000;
#MUSIC:song.ogg;
#BPMS:0.000=120.000;
#NOTES:
     dance-single:
     :
     Challenge:
     10:
     :
0001
2000
3000
0000
;
"""


def _etterna_install(tmp_path):
    """A minimal Etterna layout: Save/ next to Cache/cache.db and
    Songs/<pack>/<song>/, with the one chart indexed under its real
    chartkey."""
    import sqlite3
    from analysis.games.etterna.sm_chart import generate_chartkey, parse_sm

    (tmp_path / 'Save').mkdir()
    song_dir = tmp_path / 'Songs' / 'Some Pack' / 'Testsong'
    song_dir.mkdir(parents=True)
    sm = song_dir / 'testsong.sm'
    sm.write_text(_SM, encoding='utf-8')

    data = parse_sm(sm)
    chartkey = generate_chartkey(data['charts'][0]['notedata'], data['bpms'],
                                 'dance-single')

    (tmp_path / 'Cache').mkdir()
    con = sqlite3.connect(tmp_path / 'Cache' / 'cache.db')
    con.execute('CREATE TABLE songs (ID, TITLE, ARTIST, DIR)')
    con.execute('CREATE TABLE steps (CHARTKEY, STEPSTYPE, DIFFICULTY,'
                ' STEPFILENAME, SONGID)')
    con.execute("INSERT INTO songs VALUES (1, 'Testsong', 'Tester',"
                " '/Songs/Some Pack/Testsong/')")
    con.execute('INSERT INTO steps VALUES (?, ?, ?, ?, 1)',
                (chartkey, 'dance-single', 4,
                 '/Songs/Some Pack/Testsong/testsong.sm'))
    # A second chart whose file was deleted with its pack.
    con.execute("INSERT INTO steps VALUES ('Xgone', 'dance-single', 3,"
                " '/Songs/Deleted Pack/Song/gone.sm', 1)")
    con.commit()
    con.close()
    return {'save_dir': str(tmp_path / 'Save'), 'extra_songs_dirs': []}


def test_etterna_catalogue_reads_the_game_cache(tmp_path):
    from analysis.games.etterna import chart_cache

    catalogue = chart_cache.catalogue(_etterna_install(tmp_path))
    # The chart whose file no longer resolves drops out.
    assert len(catalogue) == 1
    e = catalogue[0]
    assert e['song'] == 'Tester - Testsong'
    assert e['pack'] == 'Some Pack'
    assert e['steps'] == 'Challenge'
    assert e['keycount'] == 4
    assert e['chart_path'].endswith('Testsong/testsong.sm')
    assert e['replay_path'] == f"{e['chart_path']}::{e['chart_key']}"


def test_etterna_catalogue_without_a_cache_db(tmp_path):
    from analysis.games.etterna import chart_cache

    (tmp_path / 'Save').mkdir()
    assert chart_cache.catalogue({'save_dir': str(tmp_path / 'Save')}) == []
    assert chart_cache.catalogue({'save_dir': None}) == []


def test_etterna_opens_a_chart_ref_as_autoplay(tmp_path):
    from analysis.games.etterna import chart_cache
    from analysis.games.etterna.adapter import ADAPTER

    entry = chart_cache.catalogue(_etterna_install(tmp_path))[0]
    replay = ADAPTER.parse_replay(entry['replay_path'])

    # tap @ row 0 (col 3) and a hold head @ row 48; the tail isn't judged.
    assert list(replay['noterows']) == [0, 48]
    assert list(replay['columns']) == [3, 0]
    assert not replay['offsets'].any()
    assert not replay['misses'].any()
    assert replay['keycount'] == 4


def test_etterna_chart_ref_survives_an_extra_chart_in_the_file(tmp_path):
    """The selector is a chartkey, so it still names the right chart
    after an edit shifts the chart's index in the file."""
    from analysis.games.etterna import chart_cache
    from analysis.games.etterna.adapter import ADAPTER

    dirs = _etterna_install(tmp_path)
    entry = chart_cache.catalogue(dirs)[0]
    sm = Path(entry['chart_path'])
    sm.write_text(_SM.replace('#NOTES:', """#NOTES:
     dance-single:
     :
     Easy:
     1:
     :
1000
;
#NOTES:""", 1), encoding='utf-8')

    replay = ADAPTER.parse_replay(entry['replay_path'])
    assert list(replay['noterows']) == [0, 48]


def test_etterna_chart_ref_for_a_chart_that_no_longer_exists(tmp_path):
    from analysis.games.etterna.adapter import ADAPTER

    sm = tmp_path / 'gone.sm'
    sm.write_text(_SM, encoding='utf-8')
    with pytest.raises(LookupError):
        ADAPTER.parse_replay(f'{sm}::Xnosuchkey')


def test_etterna_replay_paths_are_not_chart_refs():
    from analysis.games.etterna.chart_ref import is_chart_ref

    # Etterna replay files are extensionless scorekeys.
    assert not is_chart_ref('/Save/ReplaysV2/Sabc123def456')
    assert is_chart_ref('/Songs/p/s/chart.sm::0')
    assert is_chart_ref('/Songs/p/s/chart.ssc::Xabc')
    # A `::` inside a pack name is part of the path, not a selector.
    assert not is_chart_ref('/Songs/pack::x/score')
