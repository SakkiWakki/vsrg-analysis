"""Tests for the tosu overlay integration.

Covers translation.py and discovery.py without importing Qt. bridge.py
and view.py require a QApplication and are integration-tested in
/tmp/test_tosu_headless.py instead.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from analysis.components.api import (
    ChartMetadata,
    ChartPaths,
    ChartStats,
    DataNotAvailable,
)
from plugins.unsafe.tosu_overlay.translation import (
    _acc_percent,
    _flatten_filter_list,
    _hits_dict,
    _mode_for_game,
    build_precise_state,
    build_tosu_state,
    parse_filter_message,
    prune_to_filters,
    unstable_rate_from_errors,
)


# ---------------------------------------------------------------------------
# FakeGameState: implements the minimum GameState surface used by translation
# ---------------------------------------------------------------------------

class FakeGameState:
    """Configurable stub implementing the GameState methods translation
    reads. Any method can be disabled by passing ``None`` -- the stub
    raises DataNotAvailable to exercise the safe-call path."""

    def __init__(self, **overrides):
        defaults = {
            'game': 'osu',
            'keycount': 4,
            'combo': 25,
            'paused': False,
            't_now': 10.0,
            'judgment_counts': {'300': 20, '100': 3, '50': 1, 'miss': 1},
            'hit_errors_ms': (5, -3, 10, -8, 2),
            'unstable_rate': 34.5,
            'max_combo': 50,
            'score': 999999,
            'current_grade': 'A',
            'mods_short': 'HD',
            'mods_raw': {'bitfield': 8, 'rate': 1.0},
            'play_rate_effective': 1.0,
            'player_name': 'TestPlayer',
            'chart_metadata': ChartMetadata(
                artist='Test Artist', title='Test Song',
                creator='Mapper', version='Hard', md5='abc123',
            ),
            'chart_stats': ChartStats(
                mode_name='osu', difficulty=8.0, rating=4.5,
                bpm_common=180.0, bpm_min=180.0, bpm_max=180.0,
                length_ms=120000, first_object_ms=1000,
                last_object_ms=121000, total_objects=500,
                hold_count=50, max_combo=550,
                extra={'od': 8.0, 'cs': 4.0, 'ar': 9.0, 'hp': 7.0,
                       'stars': 4.5},
            ),
            'chart_paths': ChartPaths(
                chart_folder='123 Test Song', audio_filename='audio.mp3',
                background_filename='bg.jpg', skin_folder='My Skin',
                library_root='/osu!/Songs',
            ),
        }
        self._vals = {**defaults, **overrides}

    def _get(self, key):
        if key not in self._vals or self._vals[key] is None:
            raise DataNotAvailable(key)
        return self._vals[key]

    def game(self):            return self._get('game')
    def keycount(self):        return self._get('keycount')
    def combo(self):           return self._get('combo')
    def paused(self):          return self._get('paused')
    def t_now(self):           return self._get('t_now')
    def judgment_counts(self): return self._get('judgment_counts')
    def judgment_windows(self): return [('300', 16.5), ('100', 73.5)]
    def hit_errors_ms(self):   return self._get('hit_errors_ms')
    def unstable_rate(self):   return self._get('unstable_rate')
    def max_combo(self):       return self._get('max_combo')
    def score(self):           return self._get('score')
    def current_grade(self):   return self._get('current_grade')
    def mods_short(self):      return self._get('mods_short')
    def mods_raw(self):        return self._get('mods_raw')
    def play_rate_effective(self): return self._get('play_rate_effective')
    def player_name(self):     return self._get('player_name')
    def chart_metadata(self):  return self._get('chart_metadata')
    def chart_stats(self):     return self._get('chart_stats')
    def chart_paths(self):     return self._get('chart_paths')


# ---------------------------------------------------------------------------
# parse_filter_message
# ---------------------------------------------------------------------------

class TestParseFilterMessage:
    def test_non_filter_message_returns_none(self):
        assert parse_filter_message('{"type":"hello"}') is None

    def test_string_list(self):
        msg = 'applyFilters:["play.hits","beatmap.time.live"]'
        assert parse_filter_message(msg) == frozenset(
            {'play.hits', 'beatmap.time.live'})

    def test_empty_list(self):
        assert parse_filter_message('applyFilters:[]') == frozenset()

    def test_object_list_form(self):
        msg = 'applyFilters:[{"field":"play","keys":["hits","combo"]}]'
        assert parse_filter_message(msg) == frozenset(
            {'play.hits', 'play.combo'})

    def test_nested_object_list_form(self):
        # play -> mods -> name  expands to 'play.mods.name'
        msg = ('applyFilters:[{"field":"play","keys":'
               '[{"field":"mods","keys":["name","rate"]},"score"]}]')
        result = parse_filter_message(msg)
        assert 'play.mods.name' in result
        assert 'play.mods.rate' in result
        assert 'play.score' in result

    def test_invalid_json_returns_none(self):
        assert parse_filter_message('applyFilters:not-json') is None


class TestFlattenFilterList:
    def test_non_list_returns_empty(self):
        assert _flatten_filter_list('oops') == frozenset()
        assert _flatten_filter_list({'k': 'v'}) == frozenset()

    def test_string_items(self):
        assert _flatten_filter_list(['a.b', 'c']) == frozenset({'a.b', 'c'})

    def test_object_items(self):
        data = [{'field': 'play', 'keys': ['score', 'accuracy']}]
        assert _flatten_filter_list(data) == frozenset(
            {'play.score', 'play.accuracy'})

    def test_unknown_items_silently_dropped(self):
        assert _flatten_filter_list([42, None]) == frozenset()


# ---------------------------------------------------------------------------
# prune_to_filters
# ---------------------------------------------------------------------------

class TestPruneToFilters:
    def test_empty_filters_returns_full_state(self):
        state = {'play': {'score': 1}, 'beatmap': {'title': 'T'}}
        assert prune_to_filters(state, frozenset()) is state

    def test_prefix_filter_keeps_subtree(self):
        state = {'play': {'score': 1}, 'beatmap': {'title': 'T'}}
        result = prune_to_filters(state, frozenset({'play'}))
        assert result == {'play': {'score': 1}}

    def test_deep_filter_prunes_siblings(self):
        state = {'play': {'hits': {'300': 5}, 'score': 1}}
        result = prune_to_filters(state, frozenset({'play.hits'}))
        assert result == {'play': {'hits': {'300': 5}}}

    def test_parent_filter_keeps_all_children(self):
        state = {'play': {'hits': {'300': 5}, 'score': 1}, 'state': {}}
        result = prune_to_filters(state, frozenset({'play'}))
        assert 'state' not in result
        assert result['play'] == {'hits': {'300': 5}, 'score': 1}


# ---------------------------------------------------------------------------
# build_tosu_state shape
# ---------------------------------------------------------------------------

class TestBuildTosuStateShape:
    def test_required_top_level_keys(self):
        state = build_tosu_state(FakeGameState())
        for key in ('state', 'beatmap', 'play', 'resultsScreen',
                    'performance', 'profile', 'settings',
                    'directPath', 'folders', 'files',
                    'leaderboard', 'tourney', 'session', 'client'):
            assert key in state

    def test_play_mode_is_mania_for_osu(self):
        state = build_tosu_state(FakeGameState(game='osu'))
        assert state['play']['mode'] == {'number': 3, 'name': 'mania'}

    def test_hp_stubbed_at_1(self):
        state = build_tosu_state(FakeGameState())
        assert state['play']['healthBar'] == {'normal': 1.0, 'smooth': 1.0}

    def test_pp_stubbed_at_0(self):
        state = build_tosu_state(FakeGameState())
        assert state['play']['pp']['current'] == 0
        assert state['play']['pp']['fc'] == 0


class TestBuildTosuStateData:
    def test_chart_metadata_propagates(self):
        state = build_tosu_state(FakeGameState())
        assert state['beatmap']['artist'] == 'Test Artist'
        assert state['beatmap']['title'] == 'Test Song'
        assert state['beatmap']['mapper'] == 'Mapper'
        assert state['beatmap']['version'] == 'Hard'
        assert state['beatmap']['checksum'] == 'abc123'

    def test_beatmap_time_live_in_ms(self):
        state = build_tosu_state(FakeGameState(t_now=5.5))
        assert state['beatmap']['time']['live'] == 5500

    def test_osu_stats_from_extra(self):
        state = build_tosu_state(FakeGameState())
        assert state['beatmap']['stats']['od']['original'] == 8.0
        assert state['beatmap']['stats']['cs']['original'] == 4.0
        assert state['beatmap']['stats']['ar']['original'] == 9.0
        assert state['beatmap']['stats']['hp']['original'] == 7.0
        assert state['beatmap']['stats']['stars']['total'] == 4.5

    def test_mods_propagated(self):
        state = build_tosu_state(FakeGameState())
        assert state['play']['mods']['number'] == 8
        assert state['play']['mods']['name'] == 'HD'
        assert state['play']['mods']['rate'] == 1.0

    def test_mods_default_is_NM(self):
        gs = FakeGameState(mods_short='', mods_raw={'bitfield': 0})
        state = build_tosu_state(gs)
        assert state['play']['mods']['name'] == 'NM'

    def test_state_number_playing_vs_menu(self):
        assert build_tosu_state(FakeGameState(paused=False))['state']['number'] == 2
        assert build_tosu_state(FakeGameState(paused=True))['state']['number'] == 0

    def test_hold_count_in_stats(self):
        state = build_tosu_state(FakeGameState())
        assert state['beatmap']['stats']['objects']['holds'] == 50
        assert state['beatmap']['stats']['objects']['total'] == 500

    def test_player_name(self):
        state = build_tosu_state(FakeGameState())
        assert state['play']['playerName'] == 'TestPlayer'

    def test_hit_error_array(self):
        state = build_tosu_state(FakeGameState(hit_errors_ms=(1, 2, 3)))
        assert state['play']['hitErrorArray'] == [1, 2, 3]

    def test_grade(self):
        state = build_tosu_state(FakeGameState(current_grade='S'))
        assert state['play']['rank']['current'] == 'S'

    def test_combo(self):
        gs = FakeGameState(combo=42, max_combo=99)
        state = build_tosu_state(gs)
        assert state['play']['combo']['current'] == 42
        assert state['play']['combo']['max'] == 99

    def test_paths_propagate(self):
        state = build_tosu_state(FakeGameState())
        assert state['directPath']['beatmapFolder'] == '123 Test Song'
        assert state['directPath']['skinFolder'] == 'My Skin'
        assert state['folders']['songs'] == '/osu!/Songs'

    def test_bpm_common(self):
        state = build_tosu_state(FakeGameState())
        assert state['beatmap']['stats']['bpm']['common'] == 180.0


class TestBuildTosuStateDegrades:
    """When a GameState method raises DataNotAvailable, the output
    should still be well-formed with zeroed fields."""

    def test_missing_chart_metadata(self):
        gs = FakeGameState(chart_metadata=None)
        state = build_tosu_state(gs)
        assert state['beatmap']['artist'] == ''
        assert state['beatmap']['title'] == ''

    def test_missing_all_score_fields(self):
        gs = FakeGameState(
            score=None, combo=None, max_combo=None,
            current_grade=None, hit_errors_ms=None,
            judgment_counts=None,
        )
        state = build_tosu_state(gs)
        assert state['play']['score'] == 0
        assert state['play']['combo'] == {'current': 0, 'max': 0}
        assert state['play']['rank']['current'] == ''
        assert state['play']['hits']['0'] == 0

    def test_missing_mods(self):
        gs = FakeGameState(mods_short=None, mods_raw=None)
        state = build_tosu_state(gs)
        # Fallback: NM for display, 0 bitfield
        assert state['play']['mods']['name'] == 'NM'
        assert state['play']['mods']['number'] == 0


# ---------------------------------------------------------------------------
# build_precise_state
# ---------------------------------------------------------------------------

class TestBuildPreciseState:
    def test_shape(self):
        precise = build_precise_state(FakeGameState())
        assert 'hitErrors' in precise
        assert 'currentTime' in precise
        assert 'keyOverlay' in precise

    def test_current_time_ms(self):
        precise = build_precise_state(FakeGameState(t_now=3.25))
        assert precise['currentTime'] == 3250

    def test_hit_errors_pass_through(self):
        precise = build_precise_state(FakeGameState(hit_errors_ms=(7, -4)))
        assert precise['hitErrors'] == [7, -4]

    def test_missing_hit_errors_empty_list(self):
        precise = build_precise_state(FakeGameState(hit_errors_ms=None))
        assert precise['hitErrors'] == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHitsDict:
    def test_standard_counts(self):
        hits = _hits_dict({'300': 5, '100': 2, 'miss': 1})
        assert hits['300'] == 5
        assert hits['100'] == 2
        assert hits['0'] == 1

    def test_falls_back_to_named_variants(self):
        hits = _hits_dict({'perfect': 3, 'great': 2, 'good': 1, 'bad': 0})
        assert hits['300'] == 3
        assert hits['200'] == 2
        assert hits['100'] == 1
        assert hits['50'] == 0


class TestAccuracyPercent:
    def test_all_best(self):
        assert _acc_percent({'geki': 10}) == 100.0

    def test_all_miss(self):
        assert _acc_percent({'miss': 10}) == 0.0

    def test_no_hits(self):
        assert _acc_percent({}) == 0.0


class TestModeForGame:
    def test_osu(self):
        assert _mode_for_game('osu') == (3, 'mania')

    def test_etterna(self):
        assert _mode_for_game('etterna') == (3, 'mania')

    def test_unknown(self):
        assert _mode_for_game('unknown') == (3, 'mania')


class TestUnstableRateFromErrors:
    def test_empty(self):
        assert unstable_rate_from_errors([]) == 0.0

    def test_single(self):
        assert unstable_rate_from_errors([5]) == 0.0

    def test_known_value(self):
        # stdev of [10, -10] around mean 0 is 10 -> UR = 100
        assert unstable_rate_from_errors([10, -10]) == 100.0


# ---------------------------------------------------------------------------
# discovery (no Qt needed)
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_find_overlays_returns_list(self, tmp_path):
        from plugins.unsafe.tosu_overlay.discovery import find_overlays
        overlay_dir = tmp_path / 'my-overlay'
        overlay_dir.mkdir()
        (overlay_dir / 'index.html').write_text('<html></html>')

        with patch(
            'plugins.unsafe.tosu_overlay.discovery._BUILTIN_OVERLAYS',
            tmp_path,
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._USER_OVERLAYS',
            tmp_path / 'nonexistent',
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._DEFAULT_DEV_OVERLAYS',
            tmp_path / 'nonexistent-default',
        ):
            results = find_overlays()

        assert len(results) == 1
        name, path = results[0]
        assert name == 'my-overlay'
        assert path.name == 'index.html'

    def test_ignores_dirs_without_index(self, tmp_path):
        from plugins.unsafe.tosu_overlay.discovery import find_overlays
        (tmp_path / 'broken-overlay').mkdir()

        with patch(
            'plugins.unsafe.tosu_overlay.discovery._BUILTIN_OVERLAYS',
            tmp_path,
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._USER_OVERLAYS',
            tmp_path / 'nonexistent',
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._DEFAULT_DEV_OVERLAYS',
            tmp_path / 'nonexistent-default',
        ):
            results = find_overlays()

        assert results == []

    def test_sorted_by_name(self, tmp_path):
        from plugins.unsafe.tosu_overlay.discovery import find_overlays
        for name in ('zebra', 'alpha', 'middle'):
            d = tmp_path / name
            d.mkdir()
            (d / 'index.html').write_text('')

        with patch(
            'plugins.unsafe.tosu_overlay.discovery._BUILTIN_OVERLAYS',
            tmp_path,
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._USER_OVERLAYS',
            tmp_path / 'nonexistent',
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._DEFAULT_DEV_OVERLAYS',
            tmp_path / 'nonexistent-default',
        ):
            names = [n for n, _ in find_overlays()]

        assert names == sorted(names)

    def test_discovers_from_tosu_overlays_dirs_env(self, tmp_path, monkeypatch):
        from plugins.unsafe.tosu_overlay.discovery import find_overlays

        extra_root = tmp_path / 'extra-root'
        extra_overlay = extra_root / 'community-overlay'
        extra_overlay.mkdir(parents=True)
        (extra_overlay / 'index.html').write_text('<html></html>')

        # Extra directories may be entered with surrounding whitespace.
        monkeypatch.setenv('TOSU_OVERLAYS_DIRS', f'  {extra_root}  ')

        with patch(
            'plugins.unsafe.tosu_overlay.discovery._BUILTIN_OVERLAYS',
            tmp_path / 'nonexistent-builtin',
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._USER_OVERLAYS',
            tmp_path / 'nonexistent-user',
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._DEFAULT_DEV_OVERLAYS',
            tmp_path / 'nonexistent-default',
        ):
            results = find_overlays()

        assert results == [('community-overlay', extra_overlay / 'index.html')]

    @pytest.mark.skipif(
        sys.platform == 'win32',
        reason="uses ':' path separator and $VAR syntax; POSIX-only",
    )
    def test_discovers_expanduser_and_expandvars(self, tmp_path, monkeypatch):
        from plugins.unsafe.tosu_overlay.discovery import find_overlays

        fake_home = tmp_path / 'fake-home'
        env_root = tmp_path / 'env-root'
        home_overlay = fake_home / 'from-home' / 'home-overlay'
        env_overlay = env_root / 'env-overlay'
        home_overlay.mkdir(parents=True)
        env_overlay.mkdir(parents=True)
        (home_overlay / 'index.html').write_text('<html></html>')
        (env_overlay / 'index.html').write_text('<html></html>')

        monkeypatch.setenv('HOME', str(fake_home))
        monkeypatch.setenv('OVERLAY_ENV_ROOT', str(env_root))
        monkeypatch.setenv(
            'TOSU_OVERLAYS_DIRS',
            '~/from-home:$OVERLAY_ENV_ROOT',
        )

        with patch(
            'plugins.unsafe.tosu_overlay.discovery._BUILTIN_OVERLAYS',
            tmp_path / 'nonexistent-builtin',
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._USER_OVERLAYS',
            tmp_path / 'nonexistent-user',
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._DEFAULT_DEV_OVERLAYS',
            tmp_path / 'nonexistent-default',
        ):
            names = [name for name, _ in find_overlays()]

        assert names == ['env-overlay', 'home-overlay']

    def test_fallback_to_tmp_tosu_counters_when_env_unset(
        self, tmp_path, monkeypatch
    ):
        from plugins.unsafe.tosu_overlay.discovery import find_overlays

        fallback_root = tmp_path / 'tmp-counters'
        overlay_dir = fallback_root / 'fallback-overlay'
        overlay_dir.mkdir(parents=True)
        (overlay_dir / 'index.html').write_text('<html></html>')

        monkeypatch.delenv('TOSU_OVERLAYS_DIRS', raising=False)
        with patch(
            'plugins.unsafe.tosu_overlay.discovery._DEFAULT_DEV_OVERLAYS',
            fallback_root,
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._BUILTIN_OVERLAYS',
            tmp_path / 'nonexistent-builtin',
        ), patch(
            'plugins.unsafe.tosu_overlay.discovery._USER_OVERLAYS',
            tmp_path / 'nonexistent-user',
        ):
            results = find_overlays()

        assert results == [('fallback-overlay', overlay_dir / 'index.html')]


# ---------------------------------------------------------------------------
# v1 schema (legacy menu/gameplay)
# ---------------------------------------------------------------------------

class TestV1Schema:
    def test_has_menu_and_gameplay(self):
        from plugins.unsafe.tosu_overlay import translation_v1
        state = translation_v1.build_state(FakeGameState())
        assert 'menu' in state
        assert 'gameplay' in state

    def test_menu_state_is_integer_enum(self):
        from plugins.unsafe.tosu_overlay import translation_v1
        playing = translation_v1.build_state(FakeGameState(paused=False))
        paused  = translation_v1.build_state(FakeGameState(paused=True))
        # v1 overlays do `data.menu.state == 2` for playing.
        assert playing['menu']['state'] == 2
        assert paused['menu']['state'] == 0

    def test_menu_bm_metadata_populated(self):
        from plugins.unsafe.tosu_overlay import translation_v1
        state = translation_v1.build_state(FakeGameState())
        md = state['menu']['bm']['metadata']
        assert md['artist'] == 'Test Artist'
        assert md['title']  == 'Test Song'
        assert md['mapper'] == 'Mapper'
        assert md['difficulty'] == 'Hard'

    def test_menu_bm_stats_uses_osu_native_letters(self):
        from plugins.unsafe.tosu_overlay import translation_v1
        state = translation_v1.build_state(FakeGameState())
        s = state['menu']['bm']['stats']
        # These are exactly the keys Dartandr/cyperdark overlays read.
        assert s['AR'] == 9.0
        assert s['CS'] == 4.0
        assert s['HP'] == 7.0
        assert s['OD'] == 8.0
        assert s['SR'] == 4.5

    def test_menu_pp_strains_is_array(self):
        """Overlays iterate `menu.pp.strains`; must never be undefined."""
        from plugins.unsafe.tosu_overlay import translation_v1
        state = translation_v1.build_state(FakeGameState())
        assert isinstance(state['menu']['pp']['strains'], list)

    def test_gameplay_hits_has_grade_subtree(self):
        from plugins.unsafe.tosu_overlay import translation_v1
        state = translation_v1.build_state(FakeGameState(current_grade='S'))
        assert state['gameplay']['hits']['grade']['current'] == 'S'

    def test_gameplay_hits_includes_unstable_rate(self):
        from plugins.unsafe.tosu_overlay import translation_v1
        state = translation_v1.build_state(FakeGameState(unstable_rate=42.0))
        assert state['gameplay']['hits']['unstableRate'] == 42.0


# ---------------------------------------------------------------------------
# v1 + v2 merge via translation.build_tosu_state
# ---------------------------------------------------------------------------

class TestMergedState:
    def test_has_both_schema_top_level_keys(self):
        state = build_tosu_state(FakeGameState())
        # v2 keys
        assert 'state' in state
        assert 'beatmap' in state
        assert 'play' in state
        # v1 keys
        assert 'menu' in state
        assert 'gameplay' in state

    def test_menu_mods_num_is_set_from_v2(self):
        """`menu.mods.num` used by older overlays; must carry the v2
        bitfield so they read the correct value."""
        gs = FakeGameState(mods_raw={'bitfield': 8, 'rate': 1.0})
        state = build_tosu_state(gs)
        assert state['menu']['mods']['num'] == 8
        assert state['play']['mods']['number'] == 8
