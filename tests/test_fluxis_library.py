"""fluXis library-entry consolidation from a realm dump."""
from pathlib import Path

import pytest

from analysis.games.fluxis.adapter import _score_entry, _rate_from_mods


@pytest.fixture
def dirs(tmp_path):
    maps_dir = tmp_path / 'maps'
    replays_dir = tmp_path / 'replays'
    (maps_dir / 'set-1').mkdir(parents=True)
    (maps_dir / 'set-1' / 'chart.fsc').write_text('{}')
    replays_dir.mkdir()
    (replays_dir / 'score-1.frp').write_text('{}')
    return {'maps_dir': maps_dir, 'replays_dir': replays_dir}


SCORE = {
    'ID': 'score-1', 'MapID': 'map-1', 'Accuracy': 94.19, 'Grade': 'A',
    'Flawless': 559, 'Perfect': 245, 'Great': 66, 'Alright': 15,
    'Okay': 1, 'Miss': 12, 'MaxCombo': 321, 'Mods': '',
    'Date': '2026-07-16T18:00:45.6656870+00:00',
}
MAPS = {
    'map-1': {
        'ID': 'map-1', 'MapSetID': 'set-1', 'FileName': 'chart.fsc',
        'Difficulty': 'Survive = Smile!', 'KeyCount': 4, 'Hash': 'abc123',
        'Metadata.Title': 'Title', 'Metadata.Artist': 'Artist',
        'Metadata.Mapper': 'Mapper',
    },
}


def test_score_entry_consolidates_schema(dirs):
    e = _score_entry(SCORE, MAPS, dirs)
    assert e['game'] == 'fluxis'
    assert e['song'] == 'Artist - Title'
    assert e['steps'] == 'Survive = Smile!'
    assert e['pack'] == 'Mapper'
    assert e['keycount'] == 4
    assert e['grade'] == 'A'
    assert e['wife'] == pytest.approx(0.9419)
    assert e['datetime'] == '2026-07-16 18:00:45'
    assert e['judgments'] == {'flawless': 559, 'perfect': 245, 'great': 66,
                              'alright': 15, 'okay': 1, 'miss': 12}
    assert Path(e['replay_path']).name == 'score-1.frp'
    assert Path(e['chart_path']).name == 'chart.fsc'
    assert e['maxcombo'] == 321


def test_missing_replay_file_drops_entry(dirs):
    score = dict(SCORE, ID='nonexistent')
    assert _score_entry(score, MAPS, dirs) is None


def test_unmatched_map_still_yields_entry(dirs):
    score = dict(SCORE, MapID='unknown-map')
    e = _score_entry(score, MAPS, dirs)
    assert e is not None
    assert e['song'] == '? - ?'
    assert e['chart_path'] is None
    assert e['keycount'] is None


def test_rate_from_mods():
    assert _rate_from_mods('') == 1.0
    assert _rate_from_mods(None) == 1.0
    assert _rate_from_mods('1.2x') == pytest.approx(1.2)
    assert _rate_from_mods('NoFail,0.75x') == pytest.approx(0.75)
