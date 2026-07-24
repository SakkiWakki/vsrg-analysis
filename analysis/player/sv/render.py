from __future__ import annotations

import os

import numpy as np

from analysis.player import scroll as scroll_registry
from analysis.player.init.notes_model import stream_groups_or_none
from analysis.player.sv.debug import LOGGER as _SV_DEBUG_LOGGER


class SvRenderController:
    def __init__(self, player):
        self.p = player

    def init(self, replay):
        from analysis.player.playback.timeline import RenderTimeline

        p = self.p

        # Build the per-chart engine registry. Each slot is a lazy factory;
        # the native slot is eagerly instantiated to drive the first frame.
        # Engine swap goes through swap_engine() which invalidates the
        # caches that derived from the prior engine.
        self._registry = self._build_registry(replay)
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

    def _build_registry(self, replay):
        """Construct the per-chart SVEngineRegistry from `replay['sv']`.

        Dispatch is purely by the doc's `engine_kind` + `engine_key`.
        Cross-engine slots are derived from capability fields:
          - any chart with a non-empty `bpms` map (and not already the
            beat-space native) -> beat-space slot via `beat_space_engine`.
          - any time-space chart (and not already the osu-time native)
            -> osu-time slot via `time_space_engine` on the same sections.
          - any beat-space chart -> osu-time slot via `project_beat_to_time`
            (lazy, captures the registry so it can fetch the freshly-built
            native engine when activated).
        Identity is always registered last so users can always fall back."""
        from analysis.player.sv.engine import (IdentitySVEngine,
                                                QuaverSVEngine)
        from analysis.player.sv.measure_engine import (beat_space_engine,
                                                        project_beat_to_time,
                                                        time_space_engine)
        from analysis.player.sv.registry import (ENGINE_LABELS,
                                                  KEY_ETTERNA_BEAT,
                                                  KEY_IDENTITY, KEY_OSU_TIME,
                                                  KEY_QUAVER_TIME,
                                                  SVEngineRegistry)
        from analysis.player.sv.replay_doc import (KIND_BEAT_SPACE,
                                                    KIND_TIME_SPACE,
                                                    replay_sv)

        doc = replay_sv(replay)
        registry = SVEngineRegistry()

        # --- native --------------------------------------------------------
        match doc.engine_kind:
            case _kind if _kind == KIND_TIME_SPACE \
                    and doc.engine_key == KEY_QUAVER_TIME:
                sections = list(doc.sections)
                initial_v = float(doc.initial_velocity)
                groups = doc.groups

                def make_quaver_time():
                    if not sections and abs(initial_v - 1.0) < 1e-12 \
                            and not groups:
                        return IdentitySVEngine()
                    kwargs = {'groups': groups} if groups else {}
                    return QuaverSVEngine(sections, initial_velocity=initial_v,
                                           **kwargs)

                registry.register(KEY_QUAVER_TIME,
                                  ENGINE_LABELS[KEY_QUAVER_TIME],
                                  make_quaver_time, native=True, eager=True)

            case _kind if _kind == KIND_TIME_SPACE \
                    and doc.engine_key == KEY_OSU_TIME:
                sections = list(doc.sections)

                def make_osu_time_native():
                    return time_space_engine(sections) if sections \
                        else IdentitySVEngine()

                registry.register(KEY_OSU_TIME, ENGINE_LABELS[KEY_OSU_TIME],
                                  make_osu_time_native,
                                  native=True, eager=True)

            case _kind if _kind == KIND_BEAT_SPACE:
                scrolls = list(doc.scrolls)
                speeds = list(doc.speeds)
                stops = list(doc.stops)
                delays = list(doc.delays)
                warps = list(doc.warps)
                bpms = list(doc.bpms)
                sm_offset = float(doc.sm_offset)

                def make_etterna_beat_native():
                    return beat_space_engine(scrolls, speeds, bpms, sm_offset,
                                             stops=stops, delays=delays,
                                             warps=warps)

                registry.register(KEY_ETTERNA_BEAT,
                                  ENGINE_LABELS[KEY_ETTERNA_BEAT],
                                  make_etterna_beat_native,
                                  native=True, eager=True)

            case _:
                # Unknown / identity replay shape: identity is the only slot.
                registry.register(KEY_IDENTITY, ENGINE_LABELS[KEY_IDENTITY],
                                  IdentitySVEngine, native=True, eager=True)
                return registry

        # --- cross-engine slots derived from capabilities -----------------
        # Time-space charts also expose the *other* time-space engine as
        # an A/B view: a Quaver chart gets osu-time (no signed-cum), and
        # an osu chart gets quaver-time only when the chart's signed-cum
        # semantics actually differ. We expose osu-time only here; quaver-
        # time on an osu chart is meaningless without InitialScrollVelocity.
        if doc.engine_kind == KIND_TIME_SPACE \
                and doc.engine_key != KEY_OSU_TIME and doc.sections:
            cross_sections = list(doc.sections)

            def make_osu_time_cross():
                return time_space_engine(cross_sections)

            registry.register(KEY_OSU_TIME, ENGINE_LABELS[KEY_OSU_TIME],
                              make_osu_time_cross)

        # Any chart with a BPM map can be projected into beat-space (the
        # Etterna XMOD model). Skipped when beat-space is already native.
        if doc.engine_kind != KIND_BEAT_SPACE and doc.bpms:
            cross_bpms = list(doc.bpms)

            def make_etterna_beat_cross():
                return beat_space_engine(
                    scrolls=[], speeds=[], bpms=cross_bpms, sm_offset=0.0,
                )

            registry.register(KEY_ETTERNA_BEAT,
                              ENGINE_LABELS[KEY_ETTERNA_BEAT],
                              make_etterna_beat_cross)

        # Beat-space charts can be projected to time-space via the native
        # engine's `as_sections()`. Lazy: the closure fetches the native
        # engine from the registry when the slot is activated, so the
        # projection sees whatever shape `as_sections()` produces at that
        # point in time.
        if doc.engine_kind == KIND_BEAT_SPACE:
            def make_osu_time_from_beat():
                native = registry.get(KEY_ETTERNA_BEAT)
                return project_beat_to_time(native)

            registry.register(KEY_OSU_TIME, ENGINE_LABELS[KEY_OSU_TIME],
                              make_osu_time_from_beat)

        registry.register(KEY_IDENTITY, ENGINE_LABELS[KEY_IDENTITY],
                          IdentitySVEngine)
        return registry

    @property
    def sv_sections(self):
        return self.p._sv_engine.as_sections()

    def build_cumulative_sv(self):
        p = self.p
        # Cull-space cumulative is built against the default stream
        # (Quaver: `$Default`) rather than each note's own group, so all
        # notes share one comparable cum-axis for the window bisect. Per-
        # note groups still drive the per-frame y-projection in
        # `batch_time_to_y`, so visual positions stay correct ; this just
        # means culling is slightly over-conservative when a group's
        # render multiplier diverges from the default.
        p._note_sv_cum = p._sv_engine.project_times(p.times)
        if hasattr(p._sv_engine, 'cumulative_at_groups'):
            self._build_quaver_ln_caches()

    def _build_quaver_ln_caches(self):
        """Per-LN auxiliary arrays Quaver needs for correct rendering:

        - `_ln_tail_flip[i]` : True when the tail sprite is drawn upside
          down (Quaver's `ShouldFlipLongNoteEnd`). False for taps.
        - `_ln_head_cum[i]`, `_ln_tail_cum[i]` : cumulative positions of
          the LN's head + tail in its own group's stream. NaN for taps.
        - `_ln_change_times[i]`, `_ln_change_cums[i]` : sign-change
          waypoints inside the LN's interval (numpy arrays, one entry
          per direction reversal). Empty for legacy-rendered charts and
          for LNs whose SV doesn't reverse inside the body.

        The renderer reconstructs `EarliestHeldPosition` /
        `LatestHeldPosition` per frame from these by filtering the
        waypoints to the still-future ones, matching Quaver's dynamic
        body-shrink as the playhead crosses each reversal."""
        p = self.p
        n = len(p.times)
        flip = np.zeros(n, dtype=bool)
        head_cum = np.full(n, np.nan, dtype=np.float64)
        tail_cum = np.full(n, np.nan, dtype=np.float64)
        change_times = [None] * n
        change_cums = [None] * n
        from analysis.player.sv.replay_doc import replay_sv
        legacy = bool(replay_sv(p.replay or {}).flags.get('legacy_ln', False))
        groups = getattr(p, '_note_sv_groups', None)
        ln_tail_times = getattr(getattr(p, 'notes', None),
                                 'ln_tail_times', None)
        if ln_tail_times is None:
            p._ln_tail_flip = flip
            p._ln_head_cum = head_cum
            p._ln_tail_cum = tail_cum
            p._ln_change_times = change_times
            p._ln_change_cums = change_cums
            p._ln_legacy_rendering = legacy
            return
        engine = p._sv_engine
        for i in range(n):
            end_t = float(ln_tail_times[i])
            if not np.isfinite(end_t):
                continue
            gid = groups[i] if groups is not None else None
            head_t = float(p.times[i])
            flip[i] = engine.is_sv_negative_at(end_t, group_id=gid)
            h_cum, t_cum, ts, cs = engine.body_waypoints(
                head_t, end_t, group_id=gid)
            head_cum[i] = h_cum
            tail_cum[i] = t_cum
            # Legacy LNs ignore sign changes -- store empty arrays so
            # the renderer's filter loop is a no-op.
            if legacy:
                change_times[i] = np.zeros(0, dtype=np.float64)
                change_cums[i] = np.zeros(0, dtype=np.float64)
            else:
                change_times[i] = ts
                change_cums[i] = cs
        p._ln_tail_flip = flip
        p._ln_head_cum = head_cum
        p._ln_tail_cum = tail_cum
        p._ln_change_times = change_times
        p._ln_change_cums = change_cums
        p._ln_legacy_rendering = legacy

    def times_to_sv(self, times, groups=None):
        engine = self.p._sv_engine
        arr = np.asarray(times)
        if groups is not None and hasattr(engine, 'cumulative_at_groups'):
            return engine.project_times(arr, groups=groups)
        return engine.project_times(arr)

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

        notes.mine_sv = self._chart_stream_to_sv(
            notes.mine_times, notes.mine_rows,
            groups=stream_groups_or_none(notes.mine_groups))
        notes.lift_sv = self._chart_stream_to_sv(notes.lift_times, notes.lift_rows)
        notes.fake_sv = self._chart_stream_to_sv(notes.fake_times, notes.fake_rows)
        # Unified stream table: gather the per-type caches into table
        # order (the mines+lifts+fakes concatenation permuted by the
        # table's sort, mirroring _build_stream_table).
        if notes.stream_order.size:
            notes.stream_sv = np.concatenate(
                [np.asarray(notes.mine_sv, dtype=np.float64),
                 np.asarray(notes.lift_sv, dtype=np.float64),
                 np.asarray(notes.fake_sv, dtype=np.float64)]
            )[notes.stream_order]

    def _chart_stream_to_sv(self, times, rows, groups=None):
        engine = self.p._sv_engine
        if rows.size and hasattr(engine, 'project_beats'):
            return engine.project_beats(rows.astype(np.float64) / 48.0)
        return self.times_to_sv(times, groups=groups)

    def cumulative_sv_at(self, t):
        return self.p._sv_engine.cumulative_at(float(t))

    def effective_scroll_speed(self, t) -> float:
        """Scroll rate in px/s at `t`; every px-per-distance consumer
        must route through here.

        Games with an ENGINE-PRESCRIBED rate (`GameAdapter.
        engine_beat_px`, NotITG's 64px-per-beat arrow grid) derive it
        from the chart: beat_px * bps(t) * design scale - the user's
        scroll setting never enters, because the chart's xmods are
        absolute speed and its appearance windows (hidden/sudden) are
        authored against that exact grid. Everything else: the user
        scroll speed. Both take the chart's eased scroll-multiplier
        (fluXis scroll-multiply / NotITG xmods; 1.0 when absent)."""
        speed = self._engine_rate(t)
        if speed is None:
            speed = float(self.p.scroll_speed)
        timeline = getattr(self.p, '_scroll_mult_timeline', None)
        if timeline is not None:
            (mult,) = timeline.sample(float(t))
            speed *= max(0.0, mult)
        return speed

    def _engine_rate(self, t) -> float | None:
        """The engine-prescribed px/s at `t` (before the chart
        multiplier), or None when this game has none."""
        beat_px = getattr(self.p, '_engine_beat_px', None)
        if not beat_px:
            return None
        segments = self.p._engine_bps_segments
        bps = segments[0][1]
        for t_start, seg_bps in segments:
            if t_start > float(t):
                break
            bps = seg_bps
        space = getattr(self.p._adapter, 'design_space', lambda: None)()
        design_h = getattr(space, 'height', 480.0)
        _x, _y, _w, h = self.p.chart_rect
        return float(beat_px) * bps * (h / design_h)

    def render_frame_state(self, raw_t):
        p = self.p
        timeline = getattr(p, '_render_timeline', None)

        if timeline is None:
            from analysis.player.playback.timeline import RenderTimeline
            timeline = p._render_timeline = RenderTimeline(p._sv_engine)

        return timeline.render_at(
            raw_t=float(raw_t),
            scroll_speed=self.effective_scroll_speed(raw_t),
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
        sps = max(0.001, self.effective_scroll_speed(t_now))
        flat_dt = p.judge_y_px() / sps
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
        judge_y = p.judge_y_px()

        if frame is None:
            frame = self.render_frame_state(t_now)

        return (judge_y - self.visual_sv_distance_from_frame(frame, t)
                * self.effective_scroll_speed(t_now))

    def batch_time_to_y(self, times, frame, groups=None, cum=None):
        """Project an array of chart-times to screen-y at `frame`. The
        optional `groups` array (parallel to `times`) routes each entry
        through its Quaver TimingGroup ; everything else uses the
        engine's default stream.

        Per-note groups need a per-group playhead cum: a note's offset
        from the playhead is `note_pos - playhead_pos_in_same_group`,
        not `note_pos - playhead_pos_in_default_group`. Mixing the two
        produces order-of-magnitude wrong y for any group whose stream
        diverges from the default's (e.g. Quaver charts with negative
        SV in some groups but not others).

        `cum` optionally supplies precomputed cull-space positions
        (parallel to `times`); chart streams pass their cached
        projection so beat-space engines keep row-space anchoring for
        old negative-BPM warp aliases. When omitted, `times` is
        projected through the engine."""
        p = self.p
        arr = np.asarray(times, dtype=np.float64)
        if cum is not None:
            cum = np.asarray(cum, dtype=np.float64)

        if arr.size == 0:
            return np.empty(0, dtype=np.float64)

        judge_y = p.judge_y_px()
        scroll_speed = self.effective_scroll_speed(float(frame.raw_t))

        if not getattr(frame, 'use_sv', False):
            dist = arr - float(frame.raw_t)
        else:
            engine = p._sv_engine
            if groups is not None and hasattr(engine, 'cumulative_at_groups'):
                if cum is None:
                    cum = engine.project_times(arr, groups=groups)
                # Per-note playhead cum, evaluated in each note's own
                # stream -- matches Quaver's `GetSpritePosition`.
                playhead_cum = engine.cumulative_at_groups(
                    float(frame.raw_t), groups)
                # Per-note SSF zoom in the same group's stream
                # (DESIGN.tex z(tau)). Falls back to frame's scalar
                # render_multiplier when the engine doesn't know about
                # per-group SSF.
                if hasattr(engine, 'render_multiplier_at_groups'):
                    mult = engine.render_multiplier_at_groups(
                        float(frame.raw_t), groups)
                else:
                    mult = float(frame.render_multiplier)
                dist = (cum - playhead_cum) * mult
            else:
                if cum is None:
                    cum = engine.project_times(arr)
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
