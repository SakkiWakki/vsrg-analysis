"""NotITG chart-only adapter: scan, chart refs, autoplay synthesis."""
import numpy as np
import pytest

from analysis.games.notitg.adapter import NotitgAdapter
from analysis.games.notitg.library_scan import (chart_ref, scan_songs,
                                                split_chart_ref)

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
,
4000
0000
3000
M000
;
#NOTES:
     dance-double:
     :
     Easy:
     3:
     :
00000000
00000000
00000000
00000000
;
"""


@pytest.fixture
def songs_dir(tmp_path):
    song = tmp_path / 'Some Pack' / 'Testsong'
    song.mkdir(parents=True)
    (song / 'testsong.sm').write_text(_SM, encoding='utf-8')
    return tmp_path


def test_chart_ref_roundtrip(tmp_path):
    ref = chart_ref(tmp_path / 'a.sm', 2)
    path, index = split_chart_ref(ref)
    assert (str(path), index) == (str(tmp_path / 'a.sm'), 2)
    bare, index = split_chart_ref(str(tmp_path / 'a.sm'))
    assert index == 0


def test_scan_produces_unplayed_entries(songs_dir):
    entries = scan_songs(songs_dir)
    # Only the 4k chart: the empty 8k chart has nothing judgeable and
    # is skipped (UKSRT-style decoy difficulties).
    assert len(entries) == 1
    four_k = entries[0]
    assert four_k['game'] == 'notitg'
    assert four_k['unplayed'] is True
    assert four_k['song'] == 'Tester - Testsong'
    assert four_k['pack'] == 'Some Pack'
    assert four_k['steps'] == 'Challenge 10'
    assert four_k['keycount'] == 4
    assert four_k['replay_path'].endswith('::0')


def test_autoplay_replay_synthesis(songs_dir):
    entries = scan_songs(songs_dir)
    adapter = NotitgAdapter()
    replay = adapter.parse_replay(entries[0]['replay_path'])

    # tap @ row 0 (col 3), hold head @ row 48, roll head @ row 192
    # (encoded as hold head); tails and the mine are not judged.
    assert list(replay['noterows']) == [0, 48, 192]
    assert list(replay['columns']) == [3, 0, 0]
    assert list(replay['notetypes']) == [1, 2, 2]
    assert not replay['offsets'].any()
    assert not replay['misses'].any()
    assert replay['holds'] == [(48, 0), (192, 0)]
    assert replay['keycount'] == 4


def test_resolve_all_uses_chart_ref_not_search(songs_dir):
    entries = scan_songs(songs_dir)
    adapter = NotitgAdapter()
    replay = adapter.parse_replay(entries[0]['replay_path'])
    bpms, offset, audio = adapter.resolve_all(replay)
    assert bpms and bpms[0][1] == 120.0
    assert offset == 0.0
    # chart extras attached: the mine came from the chart, and the roll
    # head is flagged for tail recoloring.
    assert replay['chart_mines'] == [(336, 0)]
    assert (192, 0) in replay['roll_heads']
    # hold ends joined: 3-tuple holds after resolve.
    assert all(len(h) == 3 for h in replay['holds'])


def test_itg_judge_windows_fixed():
    adapter = NotitgAdapter()
    windows = adapter.judgement_windows({})
    assert windows[0] == ('fantastic', pytest.approx(0.023))
    assert windows[-1] == ('wayoff', pytest.approx(0.1815))
    assert adapter.nudge_judge('ITG', +1) == 'ITG'
    assert adapter.judge_label({}) == 'ITG'


def test_game_registered():
    from analysis.core import game as game_mod
    assert 'notitg' in game_mod.all_games()
    from analysis.core import manifest as manifest_mod
    assert manifest_mod.get('notitg').name == 'notitg'
