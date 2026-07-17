"""NotITG game adapter: a chart-only Etterna variant.

NotITG (StepMania 3.95 lineage) shares the .sm format, warp/stop
timing, and note semantics with Etterna, so this adapter subclasses
EtternaAdapter and inherits the chart/timing/hold machinery; the
registry name is what marks the split (library column, judge system,
and the future home of modfile compilation).

NotITG has no replay system. Library entries are the charts
themselves (see library_scan) and `parse_replay` receives a chart ref
(`<simfile>::<index>`), synthesizing a perfect autoplay replay:
offsets 0, nothing missed. Everything downstream - player, judgments,
SV, effects - runs unchanged on it. Chart lookups short-circuit to
the referenced file, never the chartkey/fingerprint search.

Judgement windows are ITG's fixed set (Fantastic .. Way Off), not
Etterna's Wife judges; the judge nudge is a no-op.
"""
from __future__ import annotations

import numpy as np

from analysis.core.cache import Cache
from analysis.games.etterna.adapter import EtternaAdapter
from analysis.games.etterna.sm_chart import (NT_HOLD_HEAD, NT_TAP, parse_sm,
                                             stepstype_keycount)
from analysis.games.notitg.library_scan import (HEAD_TYPES, judged_notes,
                                                scan_songs, simfile_paths,
                                                split_chart_ref)
from analysis.games.notitg.paths import find_notitg_dirs

_LIBRARY_CACHE = Cache('notitg_library.pkl')

_ITG_WINDOWS_MS = (
    ('fantastic', 23.0),
    ('excellent', 44.5),
    ('great', 103.5),
    ('decent', 136.5),
    ('wayoff', 181.5),
)

def _autoplay_arrays(chart) -> dict:
    judged = judged_notes(chart)
    count = len(judged)
    return {
        'noterows': np.array([row for row, _c, _nt in judged],
                             dtype=np.int64),
        'offsets': np.zeros(count, dtype=np.float64),
        'columns': np.array([col for _r, col, _nt in judged],
                            dtype=np.int32),
        'notetypes': np.array(
            [NT_HOLD_HEAD if nt in HEAD_TYPES else NT_TAP
             for _r, _c, nt in judged], dtype=np.int32),
        'misses': np.zeros(count, dtype=bool),
        'holds': [(row, col) for row, col, nt in judged
                  if nt in HEAD_TYPES],
        'dropped_holds': [],
        'mine_hits': [],
        'replay_version': 2,
    }


class NotitgAdapter(EtternaAdapter):
    name = 'notitg'

    # --- chart-only playback ---------------------------------------------

    def parse_replay(self, path, chart_path=None):
        sm_path, index = split_chart_ref(path)
        data = parse_sm(sm_path)
        chart = data['charts'][index]

        replay = _autoplay_arrays(chart)
        replay['filepath'] = str(path)
        replay['keycount'] = stepstype_keycount(chart.get('stepstype', ''))
        self._remember_song(replay,
                            {'file': str(sm_path), 'data': data,
                             'chart': chart})
        return replay

    def _find_chart(self, replay, entry=None, progress=None):
        """The chart ref IS the chart; never chartkey/fingerprint-search
        the Etterna songs folder."""
        sm_path, index = split_chart_ref(replay.get('filepath', ''))
        try:
            data = parse_sm(sm_path)
            chart = data['charts'][index]
        except (OSError, IndexError):
            return None
        return {'file': str(sm_path), 'data': data, 'chart': chart}

    # --- judge system: fixed ITG windows ----------------------------------

    def judgement_windows(self, replay, judge=None, **_):
        return [(name, ms / 1000.0) for name, ms in _ITG_WINDOWS_MS]

    def judge_label(self, replay, judge=None, **_):
        return 'ITG'

    def nudge_judge(self, current, delta):
        return current

    def player_kwargs(self, replay, judge=None, **_):
        return {'ett_judge': 'ITG'}

    def transparent_field(self) -> bool:
        return True

    def _compiled_modfile(self, replay):
        """compile_modfile for this replay's chart, memoized on the replay
        so note_mods / scroll_multipliers / effects share one harvest."""
        cached = replay.get('_notitg_modfile')
        if cached is not None:
            return cached or None
        from analysis.games.notitg.modfile import compile_modfile
        sm_path, _index = split_chart_ref(replay.get('filepath', ''))
        compiled = compile_modfile(sm_path)
        replay['_notitg_modfile'] = compiled or {}
        return compiled

    def background_path(self, replay) -> str | None:
        """Drop the built-in #BACKGROUND when the modfile draws its own
        background actors (gat's BGCHANGES `bg/` tree renders bg.png):
        the built-in static/dimmed copy would duplicate it and, unlike
        the actor tree, never ride the mirror/AFT transforms - reading as
        'the background copied per mirror'. Falls back to the base
        (Etterna) resolution for charts with no compiled background."""
        compiled = self._compiled_modfile(replay)
        if compiled and compiled.get('has_background'):
            return None
        return super().background_path(replay)

    def note_mods(self, replay):
        """Always present for NotITG, even with no modfile: the consumer
        owns scroll orientation (engine-default upscroll comes from the
        zero-channel reverse baseline), so a chart without mods still
        needs it."""
        from analysis.games.notitg.mod_channels import compile_mod_channels
        from analysis.games.notitg.note_mods import NotitgNoteMods
        compiled = self._compiled_modfile(replay)
        channels = compile_mod_channels((compiled or {}).get('mod_events') or [])
        sm_path, _index = split_chart_ref(replay.get('filepath', ''))
        bpms = parse_sm(sm_path)['bpms']
        return NotitgNoteMods(channels, bpms)

    def scroll_multipliers(self, replay):
        from analysis.games.notitg.mod_channels import compile_scroll_multipliers
        compiled = self._compiled_modfile(replay)
        if not compiled or not compiled.get('mod_events'):
            return None
        events, _skipped_cm = compile_scroll_multipliers(compiled['mod_events'])
        return events or None

    def effects(self, replay):
        from analysis.games.notitg.field_instances import NotitgFieldInstances
        from analysis.games.notitg.shader_bridge import notitg_shader_effects
        compiled = self._compiled_modfile(replay)
        if not compiled:
            return []
        effects = list(notitg_shader_effects(compiled.get('shader_flags')))
        field_copies = compiled.get('field_copies')
        if field_copies:
            effects.append(NotitgFieldInstances(
                field_copies,
                aft_bg_timeline=compiled.get('aft_bg_visible'),
                base_hidden=compiled.get('base_field_hidden')))
        return effects

    def storyboard(self, replay):
        """Modfile actors (prank overlays, quads, text, ActorFrame
        groups) render through the storyboard pipeline in SM's 640x480
        screen space. The hierarchical `tree` (XML nesting = groups whose
        transforms compose onto children) is preferred; the flat
        `elements` list is the fallback for charts with no hierarchy."""
        from analysis.player.render.storyboard import Storyboard
        compiled = self._compiled_modfile(replay) or {}
        elements = compiled.get('tree') or compiled.get('elements')
        if not elements:
            return None
        # NotITG presents a hard-cropped 640x480 box, so letterbox that
        # exact box ('min') centered in the chart region and clip to it -
        # actors that run offscreen crop at the design edges, and the
        # centered box lines up with the notefield center (see
        # field_instances._design_map, kept in lockstep).
        return Storyboard(design_w=640.0, design_h=480.0, fit='min',
                          elements=tuple(elements), clip_design_box=True)

    def judgment_colors(self) -> dict:
        return {
            'fantastic': (90, 220, 255), 'excellent': (255, 220, 90),
            'great': (120, 255, 120), 'decent': (200, 120, 255),
            'wayoff': (230, 150, 90), 'miss': (255, 85, 85),
        }

    def mods_short(self, replay) -> str:
        return ''

    def mods_rate_multiplier(self, replay) -> float:
        return 1.0

    def chart_stats_extra(self, replay):
        return {}

    # --- library ----------------------------------------------------------

    def scan_library(self, progress=None):
        songs = find_notitg_dirs().get('songs_dir')
        if not songs:
            return []
        return scan_songs(songs, progress=progress)

    def load_cached(self):
        return _LIBRARY_CACHE.load()

    def save_cached(self, entries):
        _LIBRARY_CACHE.save(
            [e for e in entries if e.get('game') == 'notitg'])

    def rebuild(self, progress=None):
        _LIBRARY_CACHE.clear()
        entries = self.scan_library(progress=progress)
        if entries:
            _LIBRARY_CACHE.save(entries)
        return entries

    def incremental_update(self, progress=None):
        """Rescan only when the set of simfiles changed; header parsing
        every launch would cost seconds on big song folders."""
        cached = _LIBRARY_CACHE.load()
        if cached is None:
            return self.rebuild(progress=progress)

        songs = find_notitg_dirs().get('songs_dir')
        on_disk = ([str(p) for p in simfile_paths(songs)]
                   if songs else [])
        known = sorted({e['chart_path'] for e in cached})
        if sorted(on_disk) == known:
            return cached
        return self.rebuild(progress=progress)


ADAPTER = NotitgAdapter()
