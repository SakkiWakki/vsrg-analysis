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

# Engine field geometry in 640x480 design px: one arrow column is 64,
# and the receptor rows sit at the player seat (SCREEN_CENTER_Y 240)
# offset by the ReceptorArrowsYStandard/-Reverse metrics (-125/+145,
# openitg Player.cpp:127-128).
_ARROW_PX = 64.0
_DESIGN_CENTER_X = 320.0
_RECEPTOR_Y_STANDARD = 240.0 - 125.0
_RECEPTOR_Y_REVERSE = 240.0 + 145.0

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
        """The compiled modfile for this replay's chart, memoized on the
        replay so note_mods / scroll_multipliers / effects share one
        compile. The engine-loop compiler (DESIGN_engine_loop.md) runs the
        chart against a headless SM simulation and records its per-frame
        behaviour into the compiled document."""
        cached = replay.get('_notitg_modfile')
        if cached is not None:
            return cached or None
        sm_path, _index = split_chart_ref(replay.get('filepath', ''))
        from analysis.games.notitg.sim.producers import compile_via_sim
        compiled = compile_via_sim(sm_path)
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
        needs it. Player 0 (the primary field) is the one the base render
        path applies; a dual-player chart's player-1 consumer rides the
        second field capture (see `_second_field` / field_instances).

        A field-3D producer, when the chart has out-of-plane field-tilt
        pokes, supplies the double-apply guard: while the real 3D tilt
        runs, the consumer defers the 2D confusion-tilt approximation of
        the same axes (see note_mods and field_3d module docs)."""
        return self._note_mods_for(replay, player=0)

    @staticmethod
    def _mod_channels(compiled):
        """The compiled ModChannels: the sim compiler precompiles them
        from the exact per-frame chase (the mirin-dict pattern); harvest
        dicts fall back to window compilation."""
        from analysis.games.notitg.mod_channels import compile_mod_channels
        compiled = compiled or {}
        return compiled.get('mod_channels') \
            or compile_mod_channels(compiled.get('mod_events') or [])

    @staticmethod
    def _field_3d_for(compiled, channels, bpms, player=0):
        """The field-3D effect wired to its tilt producers: the recorded
        actor pokes always, plus the scalar confusion tilt mods (with the
        compiled SetVanishPoint stream) when `channels` is supplied.
        None when neither drives a tilt.

        Pass `channels` ONLY where this effect actually renders the mod
        tilt: the base field (player 0, field not owned by player
        instances). Elsewhere - the second-field consumer, dual charts
        whose instances own the field - nothing projects a mod-driven
        tilt, so the guard must leave it to the 2D kernels; the actor
        source still feeds the guard there because the instance channels
        project the SAME recorded rotations."""
        from analysis.games.notitg.field_3d import notitg_field_3d
        from analysis.games.notitg.note_mods import beat_at, beat_segments
        compiled = compiled or {}
        segments = beat_segments(bpms)
        return notitg_field_3d(
            base_hidden=compiled.get('base_field_hidden'),
            player_keyframes=compiled.get('player_field_keyframes'),
            channels=channels, beat_at=lambda t: beat_at(segments, t),
            field_vanish=compiled.get('field_vanish'), player=player)

    def _field_owned(self, compiled) -> bool:
        """Whether player instances own the field rendering (dual-player
        charts): the base capture then draws only through the instance
        transforms, so the base-field 3D effect must not also warp it."""
        return any(inst['kind'] == 'player'
                   for inst in self._field_instances(compiled))

    def _note_mods_for(self, replay, player):
        from analysis.games.notitg.note_mods import NotitgNoteMods
        compiled = self._compiled_modfile(replay)
        channels = self._mod_channels(compiled)
        sm_path, _index = split_chart_ref(replay.get('filepath', ''))
        bpms = parse_sm(sm_path)['bpms']
        mod_source = player == 0 and not self._field_owned(compiled)
        field_3d = self._field_3d_for(compiled,
                                      channels if mod_source else None,
                                      bpms, player=player)
        tilt_active = field_3d.tilt_active if field_3d is not None else None
        return NotitgNoteMods(channels, bpms, field_tilt_active=tilt_active,
                              player=player)

    def _field_instances(self, compiled) -> list:
        """The compiled generic field-instance list (players + proxy/AFT
        copies, each one composed transform channel). The engine-loop
        compiler emits it directly; harvest dicts are converted through
        the same builder so both paths feed one consumer contract."""
        compiled = compiled or {}
        instances = compiled.get('field_instances')
        if instances is not None:
            return instances
        from analysis.games.notitg.field_compose import harvest_instances
        from analysis.games.notitg.mod_channels import compile_mod_channels
        channels = compile_mod_channels(compiled.get('mod_events') or [])
        player_keyframes = compiled.get('player_field_keyframes') or {}
        dual = 1 in channels.players or bool(player_keyframes.get('P2'))
        return harvest_instances(compiled.get('field_copies'),
                                 player_keyframes,
                                 compiled.get('field_oscillators'),
                                 dual=dual)

    def _player_fields(self, replay, instances):
        """A PlayerFieldsSpec mapping each non-primary player a proxy
        targets (>= 2) to its own mod consumer, so the renderer
        re-renders that player's field into `field{N}` for its copies.
        None when no copy needs a per-player capture.

        NotITG runs up to 8 players, each a field the chart mods
        independently; player 1 is the primary 'field' capture (always
        rendered), so it is not in the map. A proxy of player N
        re-renders player N's note pipeline, not player 1's pixels
        (ENGINE_ORACLE 2b). `_note_mods_for` is 0-based (player=N-1).
        Zero cost when every copy is player 1."""
        from analysis.games.notitg.field_instances import PlayerFieldsSpec
        players = sorted({inst.get('player') for inst in instances
                          if inst['kind'] in ('proxy', 'player')
                          and (inst.get('player') or 1) > 1})
        if not players:
            return None
        return PlayerFieldsSpec(
            {n: self._note_mods_for(replay, player=n - 1) for n in players})

    def scroll_multipliers(self, replay):
        from analysis.games.notitg.mod_channels import compile_scroll_multipliers
        compiled = self._compiled_modfile(replay)
        if not compiled or not compiled.get('mod_events'):
            return None
        events, _skipped_cm = compile_scroll_multipliers(compiled['mod_events'])
        return events or None

    def effects(self, replay):
        from analysis.games.notitg.field_instances import (
            NotitgFieldInstances, NotitgScreenCamera)
        from analysis.games.notitg.shader_bridge import (
            chart_shader_effect, notitg_shader_effects)
        compiled = self._compiled_modfile(replay)
        if not compiled:
            return []
        effects = list(notitg_shader_effects(compiled.get('shader_flags')))
        chart_shaders = chart_shader_effect(compiled.get('chart_shaders'))
        if chart_shaders is not None:
            effects.append(chart_shaders)
        base_hidden = compiled.get('base_field_hidden')
        sm_path, _index = split_chart_ref(replay.get('filepath', ''))
        instances = self._field_instances(compiled)
        field_owned = any(inst['kind'] == 'player' for inst in instances)
        if not field_owned:
            # Single-field charts only: the field-3D transform warps the
            # base playfield capture, with BOTH tilt producers (actor
            # pokes + scalar confusion mods). When the player instances
            # exist they own the whole transform (their channels project
            # the same recorded rotations), so applying the capture warp
            # too would double every spin/tilt/skew - and the mod source
            # stays with the 2D kernels there (nothing else projects
            # it). Its tilt_active guard still feeds note_mods either
            # way (_note_mods_for).
            bpms = parse_sm(sm_path)['bpms']
            field_3d = self._field_3d_for(
                compiled, self._mod_channels(compiled), bpms)
            if field_3d is not None:
                effects.append(field_3d)
        screen_transform = compiled.get('screen_transform')
        if screen_transform:
            effects.append(NotitgScreenCamera(screen_transform))
        if instances:
            effects.append(NotitgFieldInstances(
                instances, base_hidden=base_hidden,
                player_fields=self._player_fields(replay, instances)))
        return effects

    def engine_beat_px(self):
        """One beat = one arrow = 64 design px at 1x (openitg
        ARROW_SIZE); the chart's xmods multiply this absolute rate, so
        the field scrolls engine-true regardless of the user's scroll
        setting."""
        return _ARROW_PX

    def design_space(self):
        """NotITG renders a fixed 640x480 design screen and STRETCHES it
        to the window - widescreen play widens the content rather than
        letterboxing it (reference footage fills 16:9 edge-to-edge), and
        offscreen actors crop at the window edge. Stretch each axis to
        fill the chart region and clip to it (see
        field_instances._design_map, kept in lockstep)."""
        from analysis.player.render.document import DesignSpace, FIT_STRETCH
        return DesignSpace(width=640.0, height=480.0, fit=FIT_STRETCH,
                           clip=True)

    def field_geometry(self, chart_rect, keycount):
        """Engine field geometry mapped through the stretch design map:
        adjacent 64-design-px columns centered on the design centre, the
        judgement line at the engine's reverse-side receptor row and the
        mirror line at its standard-side row (the player seat
        SCREEN_CENTER_Y offset by the ReceptorArrowsYReverse/-Standard
        metrics, openitg Player.cpp:127-128 - the pair is deliberately
        asymmetric about the screen centre). Charts author every mod
        amplitude against this 64px grid, so field proportions, mod
        geometry, and the receptor rows all land reference-exact."""
        from analysis.games.notitg.field_instances import _design_map
        kx, ky, ox, oy = _design_map(chart_rect)
        x0 = ox + (_DESIGN_CENTER_X - _ARROW_PX * keycount / 2.0) * kx
        return (x0, _ARROW_PX * kx,
                oy + _RECEPTOR_Y_REVERSE * ky,
                oy + _RECEPTOR_Y_STANDARD * ky)

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
        ds = self.design_space()
        return Storyboard(design_w=ds.width, design_h=ds.height, fit=ds.fit,
                          elements=tuple(elements), clip_design_box=ds.clip)

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
