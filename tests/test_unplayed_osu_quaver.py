"""Unplayed-charts feature for osu! + Quaver: autoplay synthesis from a
bare chart, plus the library post-pass that emits one chart-only entry
per mania chart with no score (deduped against played hashes)."""
import numpy as np

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


def test_osu_unplayed_entries_dedup_and_mania_filter(monkeypatch):
    # index: one played mania chart, one unplayed mania chart, one std map.
    index = {
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
    monkeypatch.setattr(osu_adapter._CHART_INDEX_CACHE, 'load',
                        lambda: index)
    played = [{'game': 'osu', 'beatmap_hash': 'HASHPLAYED'}]

    entries = osu_adapter._unplayed_entries(played)
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
    monkeypatch.setattr(osu_adapter._CHART_INDEX_CACHE, 'load', lambda: None)
    assert osu_adapter._unplayed_entries([]) == []


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


def test_quaver_unplayed_entries_dedup(monkeypatch):
    index = {
        '/songs/played.qua': (1.0, 10, 'QHASHPLAYED',
                             {'song': 'A - Played', 'steps': 'Hard',
                              'creator': 'M', 'keycount': 4}),
        '/songs/unplayed.qua': (2.0, 20, 'QHASHUNPLAYED',
                               {'song': 'B - Fresh', 'steps': 'Insane',
                                'creator': 'N', 'keycount': 7}),
    }
    monkeypatch.setattr(quaver_adapter._CHART_INDEX_CACHE, 'load',
                        lambda: index)
    played = [{'game': 'quaver', 'beatmap_hash': 'QHASHPLAYED'}]

    entries = quaver_adapter._unplayed_entries(played)
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
    monkeypatch.setattr(quaver_adapter._CHART_INDEX_CACHE, 'load',
                        lambda: None)
    assert quaver_adapter._unplayed_entries([]) == []
