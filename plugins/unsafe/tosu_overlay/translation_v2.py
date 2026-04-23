"""Build a tosu websocket/v2 state dict from a GameState snapshot.

v2 schema: top-level keys ``state, beatmap, play, resultsScreen,
performance, profile, settings, directPath, folders, files, leaderboard,
tourney, session, client``. This is what modern overlays (``api_v2``,
``api_v2_precise``) request.

See also ``translation_v1`` for the legacy ``menu/gameplay`` schema used
by older overlays built on ``api_v1``.
"""
from __future__ import annotations

from analysis.components.api import ChartMetadata, ChartPaths, ChartStats

from plugins.unsafe.tosu_overlay.translation_common import (
    _safe,
    acc_percent,
    client_string,
    hits_dict,
    mode_for_game,
    v2_state_name,
    v2_state_number,
)


def build_state(game_state) -> dict:
    """Construct a tosu v2 state dict. PP/HP/profile/leaderboard are
    stubbed; everything else is populated from the GameState."""
    gs = game_state

    meta: ChartMetadata = _safe(gs.chart_metadata, ChartMetadata())
    stats: ChartStats   = _safe(gs.chart_stats,    ChartStats())
    paths: ChartPaths   = _safe(gs.chart_paths,    ChartPaths())

    paused = _safe(gs.paused, True)
    t_now = _safe(gs.t_now, 0.0)
    t_ms = int(t_now * 1000)

    counts = _safe(gs.judgment_counts, {}) or {}
    hit_errors = list(_safe(gs.hit_errors_ms, ()))
    ur = _safe(gs.unstable_rate, 0.0)
    combo = _safe(gs.combo, 0)
    max_combo = _safe(gs.max_combo, 0)

    score = _safe(gs.score, 0)
    grade = _safe(gs.current_grade, '')
    mods_raw = _safe(gs.mods_raw, {}) or {}
    mods_str = _safe(gs.mods_short, '') or 'NM'
    mods_num = int(mods_raw.get('bitfield', 0))
    rate = _safe(gs.play_rate_effective, 1.0)
    player_name = _safe(gs.player_name, '')

    game_id = _safe(gs.game, '')
    mode_num, mode_name = mode_for_game(game_id)

    # osu-specific AR/CS/HP/stars live in ChartStats.extra so the
    # core dataclass stays neutral.
    ex = stats.extra or {}
    od_val    = ex.get('od',    stats.difficulty)
    cs_val    = ex.get('cs',    0.0)
    ar_val    = ex.get('ar',    0.0)
    hp_val    = ex.get('hp',    0.0)
    stars_val = ex.get('stars', stats.rating)

    hits = hits_dict(counts)

    return {
        'state': {
            'number': v2_state_number(paused),
            'name':   v2_state_name(paused),
        },
        'session': {'playTime': 0, 'playCount': 0},
        'client': client_string(),
        'beatmap': {
            'time': {
                'live':        t_ms,
                'firstObject': stats.first_object_ms,
                'lastObject':  stats.last_object_ms,
                'mp3Length':   stats.length_ms,
            },
            'id':              meta.beatmap_id,
            'set':             meta.beatmap_set_id,
            'artist':          meta.artist,
            'artistUnicode':   meta.artist_unicode or meta.artist,
            'title':           meta.title,
            'titleUnicode':    meta.title_unicode or meta.title,
            'mapper':          meta.creator,
            'version':         meta.version,
            'checksum':        meta.md5,
            'source':          meta.source,
            'tags':            meta.tags,
            'mode':   {'number': mode_num, 'name': mode_name},
            'status': {'number': 0,        'name': 'unknown'},
            'stats': {
                'od': {'original': od_val, 'converted': od_val},
                'cs': {'original': cs_val, 'converted': cs_val},
                'ar': {'original': ar_val, 'converted': ar_val},
                'hp': {'original': hp_val, 'converted': hp_val},
                'bpm': {
                    'common':   stats.bpm_common,
                    'min':      stats.bpm_min,
                    'max':      stats.bpm_max,
                    'realtime': stats.bpm_common,
                },
                'objects': {
                    'circles':  0,
                    'sliders':  0,
                    'spinners': 0,
                    'holds':    stats.hold_count,
                    'total':    stats.total_objects,
                },
                'maxCombo': stats.max_combo,
                'stars': {
                    'total':        stars_val,
                    'live':         stars_val,
                    'aim':          0.0,
                    'speed':        0.0,
                    'flashlight':   0.0,
                    'sliderFactor': 0.0,
                },
            },
        },
        'play': {
            'playerName':    player_name,
            'mode':          {'number': mode_num, 'name': mode_name},
            'score':         score,
            'accuracy':      acc_percent(counts),
            'healthBar':     {'normal': 1.0, 'smooth': 1.0},
            'hits':          hits,
            'hitErrorArray': hit_errors,
            'unstableRate':  ur,
            'rank':          {'current': grade, 'maxThisPlay': grade},
            'pp':            {'current': 0, 'fc': 0, 'maxAchievedThisPlay': 0},
            'mods':          {'number': mods_num, 'name': mods_str, 'rate': rate},
            'combo':         {'current': combo, 'max': max_combo},
        },
        'resultsScreen': {
            'playerName': player_name,
            'score':      score,
            'accuracy':   acc_percent(counts),
            'combo':      {'current': combo, 'max': max_combo},
            'hits':       hits,
            'pp':         {'current': 0, 'fc': 0},
            'mods':       {'number': mods_num, 'name': mods_str, 'rate': rate},
            'rank':       {'current': grade, 'maxThisPlay': grade},
        },
        'performance': {'accuracy': {}, 'graph': {'series': [], 'xaxis': []}},
        'profile': {
            'userStatus':       {'number': 0, 'name': ''},
            'banchoStatus':     {'number': 0, 'name': ''},
            'id':               0,
            'name':             player_name,
            'mode':             {'number': mode_num, 'name': mode_name},
            'rankedScore':      0,
            'level':            0,
            'accuracy':         0,
            'pp':               0,
            'playCount':        0,
            'globalRank':       0,
            'countryCode':      {'name': '', 'code': 0},
            'backgroundColour': '',
        },
        'directPath': {
            'beatmapFile':       paths.audio_filename,
            'beatmapBackground': paths.background_filename,
            'beatmapFolder':     paths.chart_folder,
            'skinFolder':        paths.skin_folder,
            'songsFolder':       paths.library_root,
        },
        'folders': {
            'game':    paths.library_root,
            'skin':    paths.skin_folder,
            'songs':   paths.library_root,
            'beatmap': paths.chart_folder,
        },
        'files': {
            'beatmap':    paths.audio_filename,
            'background': paths.background_filename,
            'audio':      paths.audio_filename,
        },
        'settings': {
            'interfaceVisible':      True,
            'replayUIVisible':       True,
            'chatVisibilityStatus':  {'number': 0, 'name': ''},
            'leaderboardVisible':    False,
            'leaderboardType':       {'number': 0, 'name': ''},
            'progressBarType':       {'number': 0, 'name': ''},
            'bassDensity':           0,
            'resolution': {'fullscreen': False, 'width': 1920,
                           'height': 1080, 'widthFullscreen': 1920,
                           'heightFullscreen': 1080},
            'client':     {'branch': 0, 'version': client_string()},
            'scoreMeter': {'type': {'number': 0, 'name': 'none'},
                           'size': 1.0},
            'cursor':     {'useSkinCursor': False, 'autoSize': False,
                           'size': 1.0},
            'mouse':      {'rawInput': False, 'disableButtons': False,
                           'disableWheel': False, 'sensitivity': 1.0},
            'mania': {
                'speedBPMScale':           False,
                'usePerBeatmapSpeedScale': False,
                'scrollSpeed': {str(i): 20 for i in (1, 4, 5, 6, 7, 8, 9)},
            },
            'sort':  {'number': 0, 'name': ''},
            'group': {'number': 0, 'name': ''},
            'skin': {
                'name':              paths.skin_folder,
                'useSkinSamples':    False,
                'ignoreBeatmapSkins': False,
                'tintSliderBall':    False,
                'useTaikoSkin':      False,
                'cursor':            {'expand': False},
            },
            'mode':  {'number': mode_num, 'name': mode_name},
            'audio': {
                'volume':              {'master': 100, 'music': 100, 'effect': 100},
                'offset':              {'universal': 0},
                'ignoreBeatmapSounds': False,
                'useSkinSamples':      False,
            },
            'background': {'storyboard': False, 'video': False, 'dim': 0},
            'keybinds': {
                'osu':    {'k1': 'Z', 'k2': 'X', 'smokeKey': 'C'},
                'fruits': {'Dash': 'LShift', 'k1': 'Z', 'k2': 'X'},
                'taiko':  {'innerLeft': 'X', 'innerRight': 'C',
                           'outerLeft': 'Z', 'outerRight': 'V'},
                'quickRetry': '` (Backquote)',
            },
        },
        'leaderboard': [],
        'tourney': {
            'manager': {
                'ipcState': 0, 'bestOF': 0,
                'stars':    {'left': 0, 'right': 0},
                'teamName': {'left': '', 'right': ''},
                'bools': {'scoreVisible': False, 'starsVisible': False,
                          'playersVisible': False},
                'chat':     [],
                'gameplay': {'score': {'left': 0, 'right': 0}},
            },
            'ipcClients': [],
        },
    }


def build_precise(game_state) -> dict:
    """``/websocket/v2/precise`` response dict. Streams hit-error data +
    current time at a higher rate for per-hit overlays."""
    return {
        'hitErrors':   list(_safe(game_state.hit_errors_ms, ())),
        'currentTime': int(_safe(game_state.t_now, 0.0) * 1000),
        'keyOverlay': {
            'k1': {'isPressed': False, 'count': 0},
            'k2': {'isPressed': False, 'count': 0},
            'm1': {'isPressed': False, 'count': 0},
            'm2': {'isPressed': False, 'count': 0},
        },
    }
