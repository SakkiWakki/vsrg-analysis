"""Build a legacy tosu/gosu v1 state dict from a GameState snapshot.

v1 schema: top-level ``menu`` (song metadata + map stats + current
state) and ``gameplay`` (per-play counters). Used by overlays built
against the older ``api_v1`` WebSocketManager endpoint -- ``ws://.../ws``.

Modern overlays use ``translation_v2``; some still use v1 or read
both. Since the shim fan-outs pushes to every live WebSocket regardless
of URL, we merge v1 and v2 into the same payload (v1 keys on the outer
object alongside v2 keys). Overlays that only look at v1 ignore the v2
keys and vice versa.

Fields observed across the working overlay set (see filter-audit dump)::

    data.menu.state
    data.menu.bm.id
    data.menu.bm.path.full
    data.menu.bm.rankedStatus
    data.menu.bm.metadata.{artist, title, mapper, difficulty}
    data.menu.bm.stats.{AR, CS, HP, OD, SR}
    data.menu.bm.time.{current, mp3}
    data.menu.mods.str
    data.menu.pp                     # object: may be {strains, ...}
    data.menu.pp.strains
    data.gameplay.accuracy
    data.gameplay.combo.current
    data.gameplay.hits               # subtree; overlays read hits[N] too
    data.gameplay.hits.grade.current
    data.gameplay.hits.sliderBreaks
    data.gameplay.hits.unstableRate
    data.gameplay.hp.{normal, smooth}
    data.gameplay.name               # player name
    data.gameplay.pp.{current, fc}
    data.gameplay.score
"""
from __future__ import annotations

from analysis.components.api import ChartMetadata, ChartPaths, ChartStats

from plugins.unsafe.tosu_overlay.translation_common import (
    _safe,
    acc_percent,
    hits_dict,
    v1_state_number,
)


def build_state(game_state) -> dict:
    """Construct a v1 ``{menu, gameplay}`` dict. Intended to be merged
    into the v2 payload by the caller so a single WS push serves both
    schemas; v1-only consumers still get everything they need."""
    gs = game_state

    meta: ChartMetadata = _safe(gs.chart_metadata, ChartMetadata())
    stats: ChartStats   = _safe(gs.chart_stats,    ChartStats())
    paths: ChartPaths   = _safe(gs.chart_paths,    ChartPaths())

    paused = _safe(gs.paused, True)
    t_now = _safe(gs.t_now, 0.0)
    t_ms = int(t_now * 1000)

    counts = _safe(gs.judgment_counts, {}) or {}
    ur = _safe(gs.unstable_rate, 0.0)
    combo = _safe(gs.combo, 0)
    score = _safe(gs.score, 0)
    grade = _safe(gs.current_grade, '')
    mods_str = _safe(gs.mods_short, '') or 'NM'
    player_name = _safe(gs.player_name, '')

    ex = stats.extra or {}
    od_val    = ex.get('od',    stats.difficulty)
    cs_val    = ex.get('cs',    0.0)
    ar_val    = ex.get('ar',    0.0)
    hp_val    = ex.get('hp',    0.0)
    stars_val = ex.get('stars', stats.rating)

    # v1 "bm.path.full" is the full path to the .osu file relative to
    # the Songs root -- e.g. "123 Song/Song [Hard].osu". Rebuild from
    # chart_folder + any reasonable filename we have (chart metadata
    # rarely exposes the filename by itself; the concat is enough for
    # path-display overlays).
    bm_path_full = ''
    if paths.chart_folder:
        bm_path_full = f'{paths.chart_folder}/{meta.version}.osu'

    gameplay_hits = {
        **hits_dict(counts),
        'grade': {'current': grade, 'maxThisPlay': grade},
        'unstableRate': ur,
    }

    return {
        'menu': {
            'state':  v1_state_number(paused),
            'isChatEnabled': 0,
            'bm': {
                'id':           meta.beatmap_id,
                'set':          meta.beatmap_set_id,
                'md5':          meta.md5,
                'rankedStatus': 0,
                'metadata': {
                    'artist':           meta.artist,
                    'artistOriginal':   meta.artist_unicode or meta.artist,
                    'title':            meta.title,
                    'titleOriginal':    meta.title_unicode or meta.title,
                    'mapper':           meta.creator,
                    'difficulty':       meta.version,
                },
                'stats': {
                    'AR':        ar_val,
                    'CS':        cs_val,
                    'HP':        hp_val,
                    'OD':        od_val,
                    'SR':        stars_val,
                    # Stable/gosu also expose mod-adjusted variants;
                    # we mirror the originals since we don't compute
                    # the mod-adjusted forms separately.
                    'fullSR':    stars_val,
                    'BPM': {
                        'min':    stats.bpm_min,
                        'max':    stats.bpm_max,
                        'common': stats.bpm_common,
                    },
                    'memoryAR':  ar_val,
                    'memoryCS':  cs_val,
                    'memoryHP':  hp_val,
                    'memoryOD':  od_val,
                    'maxCombo':  stats.max_combo,
                },
                'time': {
                    'firstObj':     stats.first_object_ms,
                    'current':      t_ms,
                    'full':         stats.last_object_ms,
                    'mp3':          stats.length_ms,
                },
                'path': {
                    'full':       bm_path_full,
                    'folder':     paths.chart_folder,
                    'file':       f'{meta.version}.osu' if meta.version else '',
                    'bg':         paths.background_filename,
                    'audio':      paths.audio_filename,
                },
            },
            'mods': {
                'num': 0,            # populated by callers that merge with v2
                'str': mods_str if mods_str != 'NM' else '',
            },
            'pp': {
                # Older "strains" visualizers expect a numeric array;
                # we don't compute strains, so an empty array tells
                # them "no curve available" rather than crashing.
                'strains':   [],
                'strainsAll': {'aim': [], 'speed': [], 'total': []},
                # Older PP counters read a single value at menu.pp.X
                # like menu.pp.100 (PP if you 100%'d the rest). We
                # stub them all to 0.
                **{str(p): 0 for p in (100, 99, 98, 97, 96, 95, 90)},
            },
        },
        'gameplay': {
            'gameMode': 3,           # mania
            'name':     player_name,
            'score':    score,
            'accuracy': acc_percent(counts),
            'combo': {
                'current': combo,
                'max':     _safe(gs.max_combo, 0),
            },
            'hp': {'normal': 1.0, 'smooth': 1.0},
            'hits': gameplay_hits,
            'pp':   {'current': 0, 'fc': 0, 'maxThisPlay': 0},
            'keyOverlay': {
                'k1': {'isPressed': False, 'count': 0},
                'k2': {'isPressed': False, 'count': 0},
                'm1': {'isPressed': False, 'count': 0},
                'm2': {'isPressed': False, 'count': 0},
            },
            'leaderboard': {'hasLeaderboard': False, 'isVisible': False,
                            'ourplayer': {}, 'slots': []},
        },
    }
