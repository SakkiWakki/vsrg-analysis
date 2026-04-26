"""Tests for the per-adapter library cache refactor: the Cache helper,
incremental-update paths on each adapter, rebuild isolation between
games, and the `enabled_games` config bypass."""
from __future__ import annotations

from pathlib import Path

import pytest

from analysis.core.cache import Cache


# ---------- Cache helper --------------------------------------------------


@pytest.fixture
def iso_cache(tmp_path, monkeypatch):
    """Redirect all Cache paths under tmp_path so tests don't touch
    the real ~/.cache directory."""
    monkeypatch.setattr('analysis.core.cache.cache_dir',
                        lambda: tmp_path / 'cache')
    yield tmp_path / 'cache'


def test_cache_roundtrip(iso_cache):
    c = Cache('roundtrip.pkl')
    c.save({'a': 1, 'b': [2, 3]})
    assert c.load() == {'a': 1, 'b': [2, 3]}


def test_cache_missing_file_returns_none(iso_cache):
    assert Cache('nonexistent.pkl').load() is None
    assert Cache('nonexistent.pkl').fingerprint() is None


def test_cache_corrupt_file_returns_none(iso_cache):
    c = Cache('corrupt.pkl')
    c.path.parent.mkdir(parents=True, exist_ok=True)
    c.path.write_bytes(b'\x00\x01not a pickle')
    assert c.load() is None
    assert c.fingerprint() is None


def test_cache_fingerprint_separate_from_data(iso_cache):
    c = Cache('fp.pkl')
    c.save([1, 2, 3], fingerprint=('songs', 1700000000.0))
    assert c.fingerprint() == ('songs', 1700000000.0)
    assert c.load() == [1, 2, 3]


def test_cache_clear(iso_cache):
    c = Cache('clearme.pkl')
    c.save('x')
    assert c.load() == 'x'
    c.clear()
    assert c.load() is None
    # idempotent ; clearing twice doesn't raise
    c.clear()


# ---------- Etterna ScoreKey-diff incremental -----------------------------


class _FakeScore(dict):
    """XML-parse output shape is a plain dict; use a subclass just for
    readable test construction."""


def _make_etterna_env(tmp_path, scores):
    """Build a temp Etterna layout: a replays_dir with empty files named
    after each score's scorekey, and an xml_path stub. `scores` is a list
    of dicts matching parse_etterna_xml's return shape."""
    replays = tmp_path / 'ReplaysV2'
    replays.mkdir()
    for s in scores:
        (replays / s['scorekey']).write_bytes(b'')
    xml = tmp_path / 'Etterna.xml'
    xml.write_text('<xml/>')
    return {'xml_path': str(xml), 'replays_dir': str(replays)}


def _patch_etterna(monkeypatch, iso_cache_dir, dirs, scores_by_call):
    """Stub out find_etterna_dirs to return `dirs` and parse_etterna_xml
    to yield `scores_by_call` (a list of lists ; one per invocation).
    Also redirect the adapter's library cache into the isolated dir."""
    from analysis.games.etterna import adapter as ea
    monkeypatch.setattr('analysis.core.cache.cache_dir',
                        lambda: iso_cache_dir)
    # Recreate the module-level Cache so it picks up the redirected dir.
    monkeypatch.setattr(ea, '_LIBRARY_CACHE', Cache('etterna_library.pkl'))
    monkeypatch.setattr('analysis.games.etterna.replay.find_etterna_dirs',
                        lambda: dirs)
    calls = iter(scores_by_call)
    monkeypatch.setattr('analysis.games.etterna.replay.parse_etterna_xml',
                        lambda _path: next(calls))


def _base_score(key, **overrides):
    s = {
        'scorekey': key,
        'song': 'Song',
        'pack': 'Pack',
        'steps': 'hard',
        'rate': 1.0,
        'ssrnormpercent': 0.95,
        'grade': 'AA',
        'datetime': '2026-04-20 12:00:00',
        'ssrs': {'Overall': 20.0},
        'maxcombo': 500,
        'chartkey': 'X' + key,
        'stepstype': 'dance-single',
        'judgescale': 1.0,
        'judgments': {'W1': 100},
    }
    s.update(overrides)
    return s


def test_etterna_incremental_one_new_score(tmp_path, monkeypatch):
    from analysis.games.etterna.adapter import EtternaAdapter
    a = EtternaAdapter()

    score1 = _base_score('S1-key')
    score2 = _base_score('S2-key', song='Second Song')

    dirs = _make_etterna_env(tmp_path, [score1, score2])
    _patch_etterna(monkeypatch, tmp_path / 'cache', dirs,
                   scores_by_call=[[score1], [score1, score2]])

    first = a.rebuild()
    assert len(first) == 1
    assert first[0]['scorekey'] == 'S1-key'

    second = a.incremental_update()
    assert len(second) == 2
    assert {e['scorekey'] for e in second} == {'S1-key', 'S2-key'}
    # The existing entry object is preserved verbatim (not re-parsed):
    assert first[0] in second


def test_etterna_incremental_n_new_scores(tmp_path, monkeypatch):
    from analysis.games.etterna.adapter import EtternaAdapter
    a = EtternaAdapter()

    existing = [_base_score(f'E{i}') for i in range(3)]
    newcomers = [_base_score(f'N{i}', song=f'New {i}') for i in range(5)]
    all_scores = existing + newcomers

    dirs = _make_etterna_env(tmp_path, all_scores)
    _patch_etterna(monkeypatch, tmp_path / 'cache', dirs,
                   scores_by_call=[existing, all_scores])

    a.rebuild()
    merged = a.incremental_update()
    new_keys = {e['scorekey'] for e in merged if e['scorekey'].startswith('N')}
    assert new_keys == {'N0', 'N1', 'N2', 'N3', 'N4'}
    assert len(merged) == 8


def test_etterna_incremental_new_song_different_keycount(tmp_path, monkeypatch):
    """New score uses a 6-key stepstype; keycount should reflect that
    rather than inheriting from the existing 4-key entries."""
    from analysis.games.etterna.adapter import EtternaAdapter
    a = EtternaAdapter()

    old = _base_score('OLD', stepstype='dance-single')  # 4k
    new = _base_score('NEW', stepstype='dance-solo',    # 6k
                      song='Six-key Song')

    dirs = _make_etterna_env(tmp_path, [old, new])
    _patch_etterna(monkeypatch, tmp_path / 'cache', dirs,
                   scores_by_call=[[old], [old, new]])

    a.rebuild()
    merged = a.incremental_update()
    by_key = {e['scorekey']: e for e in merged}
    assert by_key['OLD']['keycount'] == 4
    assert by_key['NEW']['keycount'] == 6


def test_etterna_incremental_no_new_scores(tmp_path, monkeypatch):
    from analysis.games.etterna.adapter import EtternaAdapter
    a = EtternaAdapter()
    scores = [_base_score('A'), _base_score('B')]
    dirs = _make_etterna_env(tmp_path, scores)
    _patch_etterna(monkeypatch, tmp_path / 'cache', dirs,
                   scores_by_call=[scores, scores])

    first = a.rebuild()
    second = a.incremental_update()
    assert second == first


def test_etterna_incremental_same_content_mtime_bumped(tmp_path, monkeypatch):
    """Etterna rewrites Etterna.xml on every session even without new
    scores. ScoreKey-diff must treat a content-identical rewrite as
    no-op: no new entries, no churn."""
    from analysis.games.etterna.adapter import EtternaAdapter
    a = EtternaAdapter()

    scores = [_base_score('X'), _base_score('Y')]
    dirs = _make_etterna_env(tmp_path, scores)
    _patch_etterna(monkeypatch, tmp_path / 'cache', dirs,
                   scores_by_call=[scores, list(scores)])

    a.rebuild()
    # Bump the xml mtime to simulate a rewrite.
    Path(dirs['xml_path']).write_text('<xml><!-- bumped --></xml>')
    merged = a.incremental_update()
    assert len(merged) == 2
    assert {e['scorekey'] for e in merged} == {'X', 'Y'}


# ---------- Osu incremental with warm chart index -------------------------


def _patch_osu(monkeypatch, iso_cache_dir):
    from analysis.games.osu import adapter as oa
    monkeypatch.setattr('analysis.core.cache.cache_dir',
                        lambda: iso_cache_dir)
    monkeypatch.setattr(oa, '_LIBRARY_CACHE', Cache('osu_library.pkl'))
    monkeypatch.setattr(oa, '_CHART_INDEX_CACHE', Cache('osu_chart_index.pkl'))


def test_osu_incremental_no_new_replays(tmp_path, monkeypatch):
    from analysis.games.osu import adapter as oa
    _patch_osu(monkeypatch, tmp_path / 'cache')
    monkeypatch.setattr(oa, '_osr_paths', lambda: [])
    monkeypatch.setattr(oa, '_build_chart_hash_lookup',
                        lambda progress=None: {})

    a = oa.OsuAdapter()
    first = a.rebuild()
    assert first == []
    # Second call with no new replays should not fall through to rebuild;
    # it returns the cached empty list.
    second = a.incremental_update()
    assert second == []


def test_osu_incremental_picks_up_new_replay(tmp_path, monkeypatch):
    from analysis.games.osu import adapter as oa
    _patch_osu(monkeypatch, tmp_path / 'cache')

    # Stub replay paths: rebuild sees one, incremental sees two.
    state = {'paths': [Path('/fake/replay1.osr')]}
    monkeypatch.setattr(oa, '_osr_paths', lambda: list(state['paths']))
    parse_calls = []

    def stub_parse(paths, progress=None):
        parse_calls.append(list(paths))
        return [{'game': 'osu',
                 'replay_path': str(p),
                 'beatmap_hash': f'h-{p.name}',
                 'song': f'[placeholder-{p.name}]',
                 'mtime': 0.0}
                for p in paths]

    monkeypatch.setattr(oa, '_parse_osr_batch', stub_parse)

    a = oa.OsuAdapter()
    a.rebuild()
    state['paths'].append(Path('/fake/replay2.osr'))
    merged = a.incremental_update()
    assert len(merged) == 2
    # Parse (which now does inline enrichment via the chart-hash lookup)
    # should have been called on *just* the new entries the second time,
    # not the full library.
    assert len(parse_calls) == 2
    assert len(parse_calls[1]) == 1
    assert str(parse_calls[1][0]).endswith('replay2.osr')


def test_osu_chart_index_reuses_cached_hashes(tmp_path, monkeypatch):
    """Building the chart index twice in a row, with no file changes,
    should not invoke the hasher the second time."""
    from analysis.games.osu import adapter as oa
    _patch_osu(monkeypatch, tmp_path / 'cache')

    songs = tmp_path / 'Songs'
    songs.mkdir()
    c1 = songs / 'a.osu'
    c1.write_text('[Metadata]\nArtist:A\nTitle:T\nVersion:E\nCreator:C\n')

    call_count = {'n': 0}

    def fake_hash(path_str):
        call_count['n'] += 1
        return path_str, 'deadbeef'

    monkeypatch.setattr(oa, '_hash_chart_from_path_str', fake_hash)
    monkeypatch.setattr(oa, '_chart_meta',
                        lambda _p: {'song': 'A - T', 'steps': 'E',
                                    'creator': 'C', 'keycount': 4})

    oa._build_chart_index(str(songs))
    first = call_count['n']
    assert first >= 1
    oa._build_chart_index(str(songs))
    assert call_count['n'] == first, \
        'second build must reuse cached hashes'


# ---------- rebuild isolation ---------------------------------------------


def test_rebuild_one_game_does_not_touch_other_game_cache(tmp_path, monkeypatch):
    from analysis.games.osu import adapter as oa
    from analysis.games.etterna import adapter as ea
    _patch_osu(monkeypatch, tmp_path / 'cache')
    # Etterna cache redirected separately to the same dir.
    monkeypatch.setattr(ea, '_LIBRARY_CACHE', Cache('etterna_library.pkl'))

    ea._LIBRARY_CACHE.save([{'game': 'etterna', 'scorekey': 'E1',
                             'replay_path': '/tmp/e.bin'}])

    monkeypatch.setattr(oa, '_osr_paths', lambda: [])
    monkeypatch.setattr(oa, '_build_chart_hash_lookup',
                        lambda progress=None: {})

    oa.OsuAdapter().rebuild()
    # Etterna's cache file on disk must be untouched.
    assert ea._LIBRARY_CACHE.load() == [
        {'game': 'etterna', 'scorekey': 'E1', 'replay_path': '/tmp/e.bin'}]


# ---------- disabled-game bypass ------------------------------------------


def test_build_library_skips_disabled_games(tmp_path, monkeypatch):
    """When `library.enabled_games` excludes a game, its adapter methods
    must not be called at all."""
    from analysis.core import search, game as game_mod

    calls = []

    class _Stub:
        def __init__(self, name):
            self.name = name

        def incremental_update(self, progress=None):
            calls.append(('incr', self.name))
            return [{'game': self.name, 'replay_path': f'/{self.name}/1'}]

        def rebuild(self, progress=None):
            calls.append(('rebuild', self.name))
            return [{'game': self.name, 'replay_path': f'/{self.name}/1'}]

        def load_cached(self):
            calls.append(('load', self.name))
            return None

    registry = {'etterna': _Stub('etterna'), 'osu': _Stub('osu')}
    monkeypatch.setattr(game_mod, 'all_games', lambda: dict(registry))

    # Only etterna enabled.
    monkeypatch.setattr(search, 'enabled_games', lambda: {'etterna'})

    lib = search.build_library()
    assert [e['game'] for e in lib] == ['etterna']
    assert all(c[1] == 'etterna' for c in calls), \
        f'osu adapter was touched when disabled: {calls}'


def test_enabled_games_defaults_to_all_when_unset(tmp_path, monkeypatch):
    """Missing `library.enabled_games` key = every registered game
    enabled. Conftest redirects the config store to an isolated tmp
    file per test, so no seeded value interferes."""
    from analysis.core import search, game as game_mod

    class _Stub:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(game_mod, 'all_games',
                        lambda: {'etterna': _Stub('etterna'),
                                 'osu': _Stub('osu')})
    assert search.enabled_games() == {'etterna', 'osu'}


def test_enabled_games_ignores_unknown_names(tmp_path, monkeypatch):
    """A game name in config that isn't registered must be silently
    ignored, not crash the library build."""
    from analysis.core import search, game as game_mod
    from analysis.config import store

    monkeypatch.setattr(game_mod, 'all_games', lambda: {'etterna': object()})
    store.get_config().set('library.enabled_games',
                           ['etterna', 'ancient_game_that_no_longer_exists'])
    store.get_config().flush()

    assert search.enabled_games() == {'etterna'}
