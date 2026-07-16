from __future__ import annotations

import numpy as np

from analysis.core import game as game_mod
from analysis.player.init.judgment import judge
from analysis.player.plugin.plugin_loader import PluginManager
from analysis.viz.plots import col_colors


def osu_hold_release_offsets(replay, game):
    if game != 'osu':
        return {}

    offsets = {}
    for entry in replay.get('hold_releases') or []:
        start_t, col, _end_t, off_ms = entry
        if off_ms is not None:
            offsets[(start_t, col)] = off_ms / 1000.0
    return offsets


def normalize_miss_pressed(raw, n_misses):
    if raw is not None and len(raw) == n_misses:
        return np.asarray(raw, dtype=bool)
    return np.zeros(n_misses, dtype=bool)


class PlayerInitState:
    def __init__(self, player):
        self.p = player

    def load_replay_arrays(self, replay, game, *, bpms, sm_offset, keycount=None):
        p = self.p
        p._adapter = game_mod.get(game)
        p.times, p.hold_tails, p.keycount = p._adapter.prepare_replay_times(
            replay,
            bpms=bpms,
            sm_offset=sm_offset,
            keycount=keycount,
        )
        # Active-lane timeline for charts whose lane count changes
        # mid-play (fluXis lane switches); None = static layout. The
        # lane layer reads this per frame.
        p._lane_mask = p._adapter.lane_mask(replay)

        p.columns = replay['columns']
        p.offsets = replay['offsets']
        p.misses = replay['misses']
        p.notetypes = replay['notetypes']
        p.hold_release_offsets = osu_hold_release_offsets(replay, game)
        p.miss_pressed = normalize_miss_pressed(
            replay.get('miss_pressed'),
            len(p.misses),
        )
        # Per-note SV group ids (Quaver TimingGroups, parallel to
        # `p.times`). Other games leave it None ; the SV engine and
        # batch-y pipeline both treat None as "every note uses default".
        from analysis.player.sv.replay_doc import replay_sv
        p._note_sv_groups = replay_sv(replay).note_groups

    def init_judge(self, od, ett_judge):
        p = self.p
        judge_kw = p._adapter.judge_kwarg_name()
        p._active_judge = od if judge_kw == 'od' else ett_judge
        self.apply_judge()
        p.judge_colors = p._adapter.judgment_colors()

    def init_palette(self):
        p = self.p
        p.palette = [
            tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
            for c in col_colors(p.keycount)
        ]

    def init_notes_model(self, replay):
        from analysis.player.init.notes_model import build_notes_model, link_miss_holds

        p = self.p
        p.notes = build_notes_model(replay, p.times, p.hold_tails, p._adapter)
        link_miss_holds(p.notes, p.offsets, p.misses, p.miss_pressed)

    def compute_max_draw_pad(self):
        p = self.p
        off_abs = float(np.max(np.abs(p.offsets))) if p.offsets.size else 0.0
        rel_abs = max(
            (abs(v) for v in p.hold_release_offsets.values()),
            default=0.0,
        )
        return max(off_abs, rel_abs)

    def init_side_systems(self):
        from analysis.player.input.events import EventBus
        from analysis.player.hud.hud_state import HudState
        from analysis.player.render.layers.sprite_cache import NoteSpriteCache

        p = self.p
        p.plugins = PluginManager.discover()
        p.plugins.layers.register_note_types(
            p._adapter.note_types(p.replay)
        )

        p._sprite_cache = NoteSpriteCache()
        p._sprite_cache.bind(p._adapter.note_sprites(p.replay))

        p.hud = HudState()
        p.events = EventBus()
        p._ui_status = {'audio_ready': False, 'pitch_correct': True}

    def nudge_judge(self, delta):
        p = self.p
        new_val = p._adapter.nudge_judge(p._active_judge, delta)
        if new_val == p._active_judge:
            return

        p._active_judge = new_val
        self.apply_judge()

    def apply_judge(self):
        p = self.p
        kw = {p._adapter.judge_kwarg_name(): p._active_judge}
        p.windows = p._adapter.judgement_windows(p.replay, **kw)
        p.judge_label = p._adapter.judge_label(p.replay, **kw)
        p.note_judges = [
            judge(off, p.windows, miss)
            for off, miss in zip(p.offsets, p.misses)
        ]
