from __future__ import annotations

import os

import numpy as np

from analysis.player import scroll as scroll_registry
from analysis.player.sv.debug import LOGGER as _SV_DEBUG_LOGGER


class SvRenderController:
    def __init__(self, player):
        self.p = player

    def init(self, sv_sections, replay):
        from analysis.player.playback.timeline import RenderTimeline
        from analysis.player.sv.engine import IdentitySVEngine, TimeSpaceSVEngine

        p = self.p

        if sv_sections is not None:
            p._sv_engine = (
                TimeSpaceSVEngine(list(sv_sections))
                if sv_sections
                else IdentitySVEngine()
            )
        else:
            engine = p._adapter.build_sv_engine(replay)
            p._sv_engine = engine or IdentitySVEngine()

        p.sv_enabled = bool(getattr(p._sv_engine, 'enabled', False))
        p._clock.set_sv_engine(p._sv_engine)
        p._render_timeline = RenderTimeline(p._sv_engine)

        mode_desc = scroll_registry.get(p.scroll_mode)
        if mode_desc and mode_desc.on_enter:
            mode_desc.on_enter(p, p._mode_state[p.scroll_mode])

    @property
    def sv_sections(self):
        return self.p._sv_engine.as_sections()

    def build_cumulative_sv(self):
        self.p._note_sv_cum = self.p._sv_engine.project_times(self.p.times)

    def times_to_sv(self, times):
        return self.p._sv_engine.project_times(np.asarray(times))

    def build_ghost_sv_caches(self):
        p = self.p
        notes = p.notes

        notes.ghost_sv_times = self.times_to_sv(notes.ghost_times)
        notes.miss_hold_press_sv = self.times_to_sv(notes.miss_hold_press)
        notes.miss_hold_release_sv = self.times_to_sv(notes.miss_hold_release)

        if notes.miss_hold_press.size:
            notes.miss_hold_max_sv_dur = float(
                np.max(notes.miss_hold_release_sv - notes.miss_hold_press_sv)
            )
        else:
            notes.miss_hold_max_sv_dur = 0.0

        notes.mine_sv = self.times_to_sv(notes.mine_times)
        notes.lift_sv = self.times_to_sv(notes.lift_times)
        notes.fake_sv = self.times_to_sv(notes.fake_times)

    def cumulative_sv_at(self, t):
        return self.p._sv_engine.cumulative_at(float(t))

    def render_frame_state(self, raw_t):
        p = self.p
        timeline = getattr(p, '_render_timeline', None)

        if timeline is None:
            from analysis.player.playback.timeline import RenderTimeline
            timeline = p._render_timeline = RenderTimeline(p._sv_engine)

        return timeline.render_at(
            raw_t=float(raw_t),
            scroll_speed=float(p.scroll_speed),
            use_sv=bool(p.sv_enabled and p._sv_engine.enabled),
        )

    def render_at(self, raw_t):
        return self.render_frame_state(raw_t)

    def debug_log_sv_frame(self, ctx) -> None:
        p = self.p

        if not _SV_DEBUG_LOGGER.enabled:
            return

        note_limit = max(1, int(os.environ.get('VSRG_SV_DEBUG_NOTES', '6')))
        note_idxs = list(ctx.candidates[:note_limit])
        frame = ctx.frame
        notes = []

        for i in note_idxs:
            note_t = float(p.times[i])
            cum_to = self.cumulative_sv_at(note_t)
            visual_dist = self.visual_sv_distance_from_frame(frame, note_t)
            notes.append({
                'i': int(i),
                't': note_t,
                'col': int(p.notes.columns_list[i]),
                'cum_to': cum_to,
                'visual_dist': visual_dist,
                'y': self.time_to_y(note_t, ctx.t_now, frame),
                'sv': p._sv_engine.debug_snapshot_at(note_t),
            })

        _SV_DEBUG_LOGGER.log({
            'type': 'frame',
            'game': p.game,
            'scroll_mode': p.scroll_mode,
            'sv_enabled': bool(p.sv_enabled),
            't_now': float(ctx.t_now),
            'scroll_speed': float(p.scroll_speed),
            'frame': {
                'raw_t': float(frame.raw_t),
                'target_cum': float(frame.target_cum),
                'visual_cum_now': float(frame.visual_cum_now),
                'render_multiplier': float(frame.render_multiplier),
                'px_per_cum': float(frame.px_per_cum),
                'use_sv': bool(frame.use_sv),
            },
            'window': {
                'target_lo': float(ctx.target_lo),
                'target_hi': float(ctx.target_hi),
                'candidate_count': int(len(ctx.candidates)),
            },
            'now_sv': p._sv_engine.debug_snapshot_at(float(ctx.t_now)),
            'notes': notes,
        })

    def sv_distance(self, t_from, t_to):
        if not self.p.sv_enabled:
            return t_to - t_from
        return self.p._sv_engine.distance(t_from, t_to)

    def visual_sv_distance_from_frame(self, frame, t_to):
        if not getattr(frame, 'use_sv', False):
            return t_to - frame.raw_t

        cum_to = self.cumulative_sv_at(t_to)
        return (cum_to - frame.visual_cum_now) * frame.render_multiplier

    def time_to_y(self, t, t_now, frame=None):
        p = self.p
        judge_y = p.H * p.hit_line_y_frac

        if frame is None:
            frame = self.render_frame_state(t_now)

        return judge_y - self.visual_sv_distance_from_frame(frame, t) * p.scroll_speed

    def batch_time_to_y(self, times, frame):
        p = self.p
        arr = np.asarray(times, dtype=np.float64)

        if arr.size == 0:
            return np.empty(0, dtype=np.float64)

        judge_y = p.H * p.hit_line_y_frac
        scroll_speed = float(p.scroll_speed)

        if not getattr(frame, 'use_sv', False):
            dist = arr - float(frame.raw_t)
        else:
            cum = p._sv_engine.project_times(arr)
            dist = (
                (cum - float(frame.visual_cum_now))
                * float(frame.render_multiplier)
            )

        return judge_y - dist * scroll_speed

    def reset_render_timeline(self):
        return

    def reset_render_playhead(self, raw_t=None):
        del raw_t
        self.reset_render_timeline()

    def toggle_sv(self):
        p = self.p

        if not p._sv_engine.enabled:
            return False

        state = p._state()
        if 'sv_enabled_saved' in state:
            state['sv_enabled_saved'] = not state['sv_enabled_saved']
            self.reset_render_timeline()
            return False

        p.sv_enabled = not p.sv_enabled
        self.reset_render_timeline()
        return p.sv_enabled

    def sv_suspended(self):
        return 'sv_enabled_saved' in self.p._state()
