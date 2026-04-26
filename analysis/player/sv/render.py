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

        p = self.p

        # Build the per-chart engine registry. Each slot is a lazy factory;
        # the native slot is eagerly instantiated to drive the first frame.
        # Engine swap goes through swap_engine() which invalidates the
        # caches that derived from the prior engine.
        self._registry = self._build_registry(sv_sections, replay)
        p._sv_engine = self._registry.active()

        p.sv_enabled = bool(getattr(p._sv_engine, 'enabled', False))
        p._clock.set_sv_engine(p._sv_engine)
        p._render_timeline = RenderTimeline(p._sv_engine)

        _SV_DEBUG_LOGGER.log({
            'type': 'engine_init',
            'native': self._registry.native_key(),
            'available': self._registry.keys(),
            'engine': type(p._sv_engine).__name__,
        })

        mode_desc = scroll_registry.get(p.scroll_mode)
        if mode_desc and mode_desc.on_enter:
            mode_desc.on_enter(p, p._mode_state[p.scroll_mode])

    def _build_registry(self, sv_sections, replay):
        """Construct the per-chart SVEngineRegistry.

        Native engine is determined by whether the replay carries Etterna
        beat-space inputs or osu-style time-space sections. Identity is
        always available. Cross-game engines are registered when the
        chart's data can be translated to that engine's measure -- e.g.,
        Etterna charts can be approximated under a time-space engine via
        the beat engine's as_sections() projection.
        """
        from analysis.player.sv.engine import (IdentitySVEngine,
                                                QuaverSVEngine)
        from analysis.player.sv.measure_engine import (beat_space_engine,
                                                        time_space_engine)
        from analysis.player.sv.registry import (ENGINE_LABELS,
                                                  KEY_ETTERNA_BEAT,
                                                  KEY_IDENTITY, KEY_OSU_TIME,
                                                  KEY_QUAVER_TIME,
                                                  SVEngineRegistry)

        registry = SVEngineRegistry()

        # --- osu-style replay: detect by the presence of replay['sv_sections']
        # or the BPM-map projection; the `sv_sections` argument is a legacy
        # explicit override and may be None even on real osu charts.
        replay_sections = replay.get('sv_sections')
        replay_osu_bpms = replay.get('_osu_bpms')
        if sv_sections is not None or replay_sections is not None \
                or replay_osu_bpms is not None:
            if sv_sections is not None:
                sections = list(sv_sections)
            else:
                sections = list(replay_sections or [])
            osu_bpms = list(replay_osu_bpms or [])

            def make_osu_time():
                return time_space_engine(sections) if sections \
                    else IdentitySVEngine()

            registry.register(KEY_OSU_TIME, ENGINE_LABELS[KEY_OSU_TIME],
                              make_osu_time, native=True, eager=True)

            # Cross-engine beat-space: feed the chart's BPM map into a
            # beat-space engine with empty SCROLLS, so notes scroll at
            # `bpm/60` cull-units per second -- the Etterna XMOD model
            # applied to an osu chart. Lossy for chart-side SV (osu's
            # inherited timing points are time-space-native and don't
            # round-trip through SCROLLS); the BPM behavior is the part
            # that's meaningful cross-engine.
            if osu_bpms:
                def make_etterna_beat():
                    return beat_space_engine(
                        scrolls=[], speeds=[], bpms=osu_bpms,
                        sm_offset=0.0,
                    )
                registry.register(KEY_ETTERNA_BEAT,
                                  ENGINE_LABELS[KEY_ETTERNA_BEAT],
                                  make_etterna_beat)

            registry.register(KEY_IDENTITY, ENGINE_LABELS[KEY_IDENTITY],
                              IdentitySVEngine)
            return registry

        # --- Quaver-style replay: time-space SV with signed cumulative.
        # Detected by `_quaver_sv_sections` on the replay; `initial_velocity`
        # comes from the chart's `InitialScrollVelocity` field (default 1.0).
        quaver_sections = replay.get('_quaver_sv_sections')
        if quaver_sections is not None:
            sections = list(quaver_sections)
            initial_v = float(replay.get('_quaver_initial_velocity', 1.0))

            def make_quaver_time():
                if not sections and abs(initial_v - 1.0) < 1e-12:
                    return IdentitySVEngine()
                return QuaverSVEngine(sections, initial_velocity=initial_v)

            registry.register(KEY_QUAVER_TIME,
                              ENGINE_LABELS[KEY_QUAVER_TIME],
                              make_quaver_time, native=True, eager=True)

            # Cross-engine osu (time): same time-space integrator without
            # the InitialScrollVelocity / signed-cum semantics. Lossy when
            # the chart uses negative SV but useful as an A/B view.
            def make_osu_time_from_quaver():
                if not sections:
                    return IdentitySVEngine()
                return time_space_engine(sections)

            registry.register(KEY_OSU_TIME, ENGINE_LABELS[KEY_OSU_TIME],
                              make_osu_time_from_quaver)
            registry.register(KEY_IDENTITY, ENGINE_LABELS[KEY_IDENTITY],
                              IdentitySVEngine)
            return registry

        # --- Etterna-style replay: beat-space inputs.
        scrolls = replay.get('_etterna_scrolls')
        bpms = replay.get('_etterna_bpms')
        if scrolls is not None or bpms:
            speeds = replay.get('_etterna_speeds') or []
            stops = replay.get('_etterna_stops') or []
            delays = replay.get('_etterna_delays') or []
            warps = replay.get('_etterna_warps') or []
            scrolls = scrolls or []
            bpms = bpms or []
            sm_offset = replay.get('_etterna_offset') or 0.0

            has_sv = bool(scrolls or speeds or len(bpms) > 1
                          or stops or delays or warps)

            def make_etterna_beat():
                if not has_sv:
                    return IdentitySVEngine()
                return beat_space_engine(scrolls, speeds, bpms, sm_offset,
                                         stops=stops, delays=delays,
                                         warps=warps)

            registry.register(KEY_ETTERNA_BEAT,
                              ENGINE_LABELS[KEY_ETTERNA_BEAT],
                              make_etterna_beat, native=True, eager=True)

            # Cross-game time-space engine: project the beat engine to
            # `as_sections()` and feed that into the time-space integrator.
            # Lossy for SPEEDS / cross-BPM scrolls (DESIGN.tex §4 caveat)
            # but the right thing for "show me this chart under osu's model."
            def make_osu_time():
                if not has_sv:
                    return IdentitySVEngine()
                native = registry.get(KEY_ETTERNA_BEAT)
                sections = list(native.as_sections())
                if not sections:
                    return IdentitySVEngine()
                # Beat-space extrapolates with ratio=1.0 for t<0 (matching
                # Etterna's GetDisplayedBeat fallthrough). Time-space, by
                # contrast, extrapolates with the FIRST section's
                # multiplier -- which can flip the sign of cumulative_at
                # in the lead-in region if the chart starts with negative
                # or non-1 scroll. If the chart's first SCROLLS rate is
                # not 1.0, prepend a synthetic (t=0, 1.0) section so the
                # time-space engine's pre-first-segment extrapolation
                # uses ratio 1.0 too. This keeps cumulative continuous
                # at t=0 across an engine swap.
                first_t, first_m = sections[0]
                if first_t > 0.0 or abs(first_m - 1.0) > 1e-12:
                    sections = [(0.0, 1.0)] + sections
                return time_space_engine(sections)

            registry.register(KEY_OSU_TIME, ENGINE_LABELS[KEY_OSU_TIME],
                              make_osu_time)
            registry.register(KEY_IDENTITY, ENGINE_LABELS[KEY_IDENTITY],
                              IdentitySVEngine)
            return registry

        # --- Unknown replay shape: identity only.
        registry.register(KEY_IDENTITY, ENGINE_LABELS[KEY_IDENTITY],
                          IdentitySVEngine, native=True, eager=True)
        return registry

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

        notes.mine_sv = self._chart_stream_to_sv(notes.mine_times, notes.mine_rows)
        notes.lift_sv = self._chart_stream_to_sv(notes.lift_times, notes.lift_rows)
        notes.fake_sv = self._chart_stream_to_sv(notes.fake_times, notes.fake_rows)

    def _chart_stream_to_sv(self, times, rows):
        engine = self.p._sv_engine
        if rows.size and hasattr(engine, 'project_beats'):
            return engine.project_beats(rows.astype(np.float64) / 48.0)
        return self.times_to_sv(times)

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
            play_rate=float(getattr(p, 'play_rate', 1.0)),
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

    def time_for_screen_height(self, t_now: float) -> float:
        """Chart-time delta from `t_now` to the chart-time currently at
        the top of the field, using the engine directly (no predictor
        smoothing). This is the "ms to judge line" that a perfect note
        spawning at the top of the screen would take to reach the
        judgement line under the current SV.

        Computed by bisecting `cumulative_at` (which is monotonic and
        well-tested) instead of routing through `inverse_cumulative_at`
        -- the latter returns chart-time in some engines and a beat in
        others, and on charts with no `#SCROLLS` it degenerates to the
        identity. Falls back to the flat scroll-speed time when SV is
        off / no engine is loaded."""
        p = self.p
        sps = max(0.001, float(p.scroll_speed))
        flat_dt = (p.H * p.hit_line_y_frac) / sps
        engine = p._sv_engine
        if not p.sv_enabled or engine is None or not engine.enabled:
            return flat_dt
        t_now = float(t_now)
        mult = float(engine.render_multiplier_at(t_now))
        if mult <= 0.0:
            return flat_dt
        target_delta_cum = flat_dt / mult
        cum_now = float(engine.cumulative_at(t_now))

        # Expand the upper bound until cumulative_at exceeds the target.
        # Most charts hit it on the first or second doubling.
        hi = max(flat_dt, 1e-3)
        for _ in range(40):
            if float(engine.cumulative_at(t_now + hi)) - cum_now >= target_delta_cum:
                break
            hi *= 2.0
        else:
            return flat_dt

        lo = 0.0
        # 30 iters resolves dt to ~1e-9 s for any reasonable hi -- well
        # below the ms display rounding.
        for _ in range(30):
            mid = (lo + hi) * 0.5
            if float(engine.cumulative_at(t_now + mid)) - cum_now >= target_delta_cum:
                hi = mid
            else:
                lo = mid
        return hi

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
        """Re-anchor the cull-space predictor at the current playhead.
        Called by the player on seek/pause/rate-change/restart so the
        next cumulative_now() snaps to the engine's exact reading."""
        timeline = getattr(self.p, '_render_timeline', None)
        if timeline is None:
            return
        try:
            raw_t = float(self.p._clock.now())
        except Exception:
            return
        timeline.reset(raw_t)

    def reset_render_playhead(self, raw_t=None):
        if raw_t is not None:
            timeline = getattr(self.p, '_render_timeline', None)
            if timeline is not None:
                timeline.reset(float(raw_t))
            return
        self.reset_render_timeline()

    # -- engine swap ---------------------------------------------------

    @property
    def registry(self):
        return getattr(self, '_registry', None)

    def available_engine_keys(self):
        reg = self.registry
        return reg.keys() if reg else []

    def active_engine_key(self):
        reg = self.registry
        return reg.active_key() if reg else None

    def active_engine_label(self):
        reg = self.registry
        if not reg:
            return ''
        key = reg.active_key()
        return reg.label(key) if key else ''

    def swap_engine(self, key: str) -> bool:
        """Swap the active SV engine to `key`. Rebuilds the cumulative
        cache, the RenderTimeline, and the clock's engine reference; also
        refreshes sv_enabled and any CMOD-suspended state.

        Returns True on success, False if the key is unknown.
        """
        from analysis.player.playback.timeline import RenderTimeline
        reg = self.registry
        if reg is None or key not in reg.keys():
            return False
        if reg.active_key() == key:
            return True
        prev_key = reg.active_key()
        p = self.p

        # Capture the SV-folded ms readout under the old engine. After
        # the swap we ratio-correct the new engine's scroll value so the
        # visible time-to-judge stays continuous; otherwise XMOD->osu and
        # similar cross-engine swaps make the field jump scale.
        prev_ms = None
        scroll_state = getattr(p, 'scroll_state', None)
        if scroll_state is not None:
            try:
                prev_ms = float(scroll_state.effective_scroll_ms)
            except Exception:
                prev_ms = None

        engine = reg.set_active(key)
        p._sv_engine = engine

        # Refresh sv_enabled. CMOD (and any other scroll mode that stashes
        # sv_enabled in p._state()['sv_enabled_saved']) holds the prior
        # engine's enabled flag; update it to the new engine so on_exit
        # restores the right value.
        new_enabled = bool(getattr(engine, 'enabled', False))
        state = p._state()
        if 'sv_enabled_saved' in state:
            # Suspended by a scroll mode (e.g. CMOD): the user-visible
            # sv_enabled stays whatever the mode forced it to. The saved
            # value is what gets restored on mode exit; sync it.
            state['sv_enabled_saved'] = new_enabled
        else:
            p.sv_enabled = new_enabled

        # Rebuild caches that derived from the prior engine.
        p._clock.set_sv_engine(engine)
        p._render_timeline = RenderTimeline(engine)
        if hasattr(p, 'times'):
            self.build_cumulative_sv()
        # Ghost-note caches (miss-hold press/release, mines, lifts, fakes)
        # also live in cumulative space; without rebuild they drift
        # silently against live notes after a swap. build_ghost_sv_caches
        # tolerates `notes` being absent (e.g. early in chart load).
        if hasattr(p, 'notes'):
            try:
                self.build_ghost_sv_caches()
            except AttributeError:
                # Notes container not fully populated yet; live caches are
                # rebuilt anyway when the chart finishes loading.
                pass

        # Restore the visible time-to-judge so the user doesn't see the
        # field rescale on engine swap. Skipped when we couldn't sample
        # the prior ms (e.g. early init before scroll_state exists).
        if prev_ms is not None and scroll_state is not None:
            scroll_state.set_effective_scroll_ms(prev_ms)

        _SV_DEBUG_LOGGER.log({
            'type': 'engine_swap',
            'from': prev_key,
            'to': key,
            'engine': type(engine).__name__,
            'enabled': new_enabled,
        })
        return True

    def cycle_engine(self) -> str | None:
        reg = self.registry
        if reg is None:
            return None
        new_key = reg.next_key()
        if new_key is None or new_key == reg.active_key():
            return reg.active_key()

        # If the new engine has a primary game, route through set_game so
        # the scroll mode (cmod/xmod/osu) follows. set_game internally calls
        # swap_engine for us. Otherwise (identity), just swap directly.
        from analysis.player.sv.registry import ENGINE_PRIMARY_GAME
        primary = ENGINE_PRIMARY_GAME.get(new_key)
        p = self.p
        if primary and primary != getattr(p, 'game', None) \
                and hasattr(p, 'set_game'):
            p.set_game(primary)
            # set_game may have rejected the swap if the primary engine
            # isn't available for the current chart; force the engine swap
            # here regardless so cycle_engine always advances.
            if reg.active_key() != new_key:
                self.swap_engine(new_key)
        else:
            self.swap_engine(new_key)
        return new_key

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
