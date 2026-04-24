"""Scroll-velocity engine abstraction.

Each game adapter builds its own `SVEngine` (see `GameAdapter.build_sv_engine`),
so game-specific positioning math doesn't leak into the Player. The engine
converts between chart-time (what the replay stores) and SV-space (what the
renderer uses for note Y positions and visible-window culling).

Ex:
- `TimeSpaceSVEngine`  — piecewise-constant multiplier in time-space. osu!mania
                         uses this (SV from timing points is a time-space curve).
- `BeatSpaceSVEngine`  — Etterna's #SCROLLS as a piecewise-constant velocity
                         on *beats*, plus #SPEEDS as a uniform-field zoom
                         sampled at the song position. Required for charts
                         where #SCROLLS spans BPM changes — integrating in
                         time-space silently accumulates error.

Games that don't populate SV leave `adapter.build_sv_engine()` at the base
`None` default; the Player then runs with identity SV (distance = t_to - t_from).
"""
from __future__ import annotations

import bisect
from typing import Protocol

import numpy as np


class SVEngine(Protocol):
    """What the Player needs from any SV implementation.

    Two spaces are exposed:

    * **Render-space** (`distance`): used by `_time_to_y`. May depend on the
      current song position — e.g. Etterna #SPEEDS zooms the whole field
      based on where the playhead sits. Called per visible note per frame.

    * **Cull-space** (`project_times` / `cumulative_at`): monotonic, purely
      position-independent integrator. Used to pre-cache note positions for
      off-screen bisect culling. Invariant the Player relies on:

          cumulative_at(b) - cumulative_at(a) == distance(a, b)
          WHEN no position-dependent effect applies (e.g. SPEEDS = 1 flat).

      During brief position-dependent transitions (Etterna SPEEDS change),
      the culling window is approximate — typically one SPEEDS segment's
      multiplier off, which the Player pads for.
    """

    # True when the chart has non-trivial SV. The Player mirrors this into
    # `self.sv_enabled`; scroll modes like CMOD flip it off via on_enter.
    enabled: bool

    def distance(self, t_from: float, t_to: float) -> float:
        """Render-space distance between two chart times. May apply
        position-dependent zoom (e.g. Etterna SPEEDS at t_from)."""

    def cumulative_at(self, t: float) -> float:
        """Cull-space cumulative at chart time t. Monotonic in t."""

    def cumulative_velocity_at(self, t: float) -> float:
        """Local d(cumulative_at)/dt at chart time t. Used by the visual
        playhead predictor to advance in the same space the renderer uses
        without inverting through scroll=0 plateaus."""

    def inverse_cumulative_at(self, sv: float) -> float:
        """Inverse of cumulative_at: chart-time t such that
        cumulative_at(t) == sv. Well-defined wherever cumulative_at is
        strictly increasing; in flat regions (SV=0) we return an arbitrary
        chart-time within the plateau."""

    def project_times(self, times: np.ndarray) -> np.ndarray:
        """Batch `cumulative_at` — the renderer bisects the result."""

    def as_sections(self) -> list[tuple[float, float]]:
        """Legacy `[(time_sec, multiplier)]` projection for components and
        sidebar readers. Engines whose SV can't be faithfully expressed as
        time-space sections return an approximation (good enough for
        sidebar display, not for positioning)."""

    def render_multiplier_at(self, t: float) -> float:
        """Position-dependent multiplier applied to cull-space deltas for
        on-screen drawing. Most engines return 1; Etterna's beat-space
        engine returns the active #SPEEDS zoom at the playhead."""

    def debug_snapshot_at(self, t: float) -> dict:
        """Optional structured snapshot for temporary debugging."""

    def max_visible_t_from(self, song_t: float) -> float:
        """Upper bound on chart-time for the visible window. Defaults to
        infinity (culling is purely SV-space driven). Etterna's engine
        returns a beat-based cap matching ArrowEffects::FindDisplayedBeats'
        binary-search convergence — without it, scroll=0 regions produce
        huge SV-equal runs that all pass cull-space bisection."""
        return float('inf')


def _empty_engine_sections(_engine) -> list[tuple[float, float]]:
    return []


# ----------------------------------------------------------------------
# Time-space engine — osu!mania, and any other game that models SV as a
# piecewise-constant multiplier sampled on wall-clock time.
# ----------------------------------------------------------------------


class TimeSpaceSVEngine:
    """Integrates a piecewise-constant `(time_sec, multiplier)` curve.

    This is the original SV model. Used by osu!mania verbatim — its timing
    points map to time-space SV ratios and nothing in the positioning
    formula depends on the current song position."""

    def __init__(self, sections: list[tuple[float, float]]):
        self._sections = list(sections)
        self.enabled = bool(self._sections)
        self._times = np.array([s[0] for s in self._sections], dtype=np.float64)
        self._values = np.array([s[1] for s in self._sections], dtype=np.float64)
        n = len(self._sections)
        self._cum = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            dt = self._times[i] - self._times[i - 1]
            self._cum[i] = self._cum[i - 1] + dt * self._values[i - 1]

    def cumulative_at(self, t: float) -> float:
        if not self._sections:
            return float(t)
        idx = int(np.searchsorted(self._times, t, side='right')) - 1
        if idx < 0:
            return (t - float(self._times[0])) * float(self._values[0])
        return float(self._cum[idx]) + (t - float(self._times[idx])) * float(self._values[idx])

    def inverse_cumulative_at(self, sv: float) -> float:
        """Chart-time t such that cumulative_at(t) == sv. Used by the
        cull-space clock smoother to project back from a smoothed SV
        value to a chart-time the player can render."""
        if not self._sections:
            return float(sv)
        idx = int(np.searchsorted(self._cum, sv, side='right')) - 1
        if idx < 0:
            v = float(self._values[0])
            return float(self._times[0]) + (sv / v if v else 0.0)
        v = float(self._values[idx])
        return float(self._times[idx]) + (sv - float(self._cum[idx])) / v if v else float(self._times[idx])

    def cumulative_velocity_at(self, t: float) -> float:
        if not self._sections:
            return 1.0
        idx = int(np.searchsorted(self._times, t, side='right')) - 1
        if idx < 0:
            return float(self._values[0])
        return float(self._values[idx])

    def distance(self, t_from: float, t_to: float) -> float:
        return self.cumulative_at(t_to) - self.cumulative_at(t_from)

    def project_times(self, times: np.ndarray) -> np.ndarray:
        """Vectorized `cumulative_at` over an array — one np.searchsorted
        + fused numpy arithmetic replaces the N-iteration Python loop."""
        t = np.asarray(times, dtype=np.float64)
        if not t.size:
            return np.empty(0, dtype=np.float64)
        if not self._sections:
            return t.copy()
        # idx[j] = last i with self._times[i] <= t[j]
        idx = np.searchsorted(self._times, t, side='right') - 1
        # Entries before the first timing point extrapolate with _values[0];
        # use clip so we can index _cum / _times / _values safely, then
        # override those entries with the extrapolation formula.
        safe_idx = np.clip(idx, 0, len(self._times) - 1)
        cum = self._cum[safe_idx]
        base_t = self._times[safe_idx]
        vals = self._values[safe_idx]
        out = cum + (t - base_t) * vals
        pre_mask = idx < 0
        if pre_mask.any():
            out[pre_mask] = (t[pre_mask] - self._times[0]) * self._values[0]
        return out

    def as_sections(self) -> list[tuple[float, float]]:
        return list(self._sections)

    def render_multiplier_at(self, t: float) -> float:
        del t
        return 1.0

    def debug_snapshot_at(self, t: float) -> dict:
        t = float(t)
        return {
            'engine': 'time',
            't': t,
            'cumulative': self.cumulative_at(t),
            'render_multiplier': 1.0,
            'cumulative_velocity': self.cumulative_velocity_at(t),
        }

    def max_visible_t_from(self, song_t: float) -> float:
        return float('inf')


# ----------------------------------------------------------------------
# Beat-space engine — Etterna's XMOD positioning.
# ----------------------------------------------------------------------
#
# Etterna (ArrowEffects.cpp::GetYOffset, XMOD branch):
#
#     YOffset_beats = DisplayedBeat(noteBeat) - DisplayedBeat(songBeat)
#     YOffset      *= GetDisplayedSpeedPercent(songBeat, songSec)
#     YOffset      *= ARROW_SPACING * ScrollSpeed
#
# DisplayedBeat is a piecewise-linear function of real beats, built by
# integrating #SCROLLS ratios (PlayerState.cpp::ResetCacheInfo).
# GetDisplayedSpeedPercent is evaluated at the CURRENT song position (not at
# the note) — it acts as a uniform zoom factor on the whole field. To fit
# into the Player's px/sec contract, we convert beat distance to an
# "effective-seconds" quantity using the chart's base BPM.
#
# SPEEDS is position-dependent: the field zoom at time t depends on t itself.
# That means project_times (which builds a static cache) can only be an
# approximation; we evaluate SPEEDS at each sampled time, which is exact when
# SPEEDS is static (the common case) and off by one segment's worth during
# SPEEDS transitions. The per-frame `distance(song_t, note_t)` uses SPEEDS at
# song_t (matches Etterna exactly).
#
# Not modeled: Lua modscripts (.lua next to the .ssc). Some packs ship a
# "Speedist"-style script that rewrites the player's XMOD at runtime based on
# their CMOD preference (e.g. "Undiscovered Colors"'s script sets XMOD =
# CMOD / (140 * rate)). We don't embed a Lua VM, so charts with these scripts
# render at whatever XMOD the UI supplies. To match Etterna visually on such
# charts, set XMOD manually to whatever the Lua would compute; for the common
# `CMod / (bpm_goal * rate)` pattern that's `your_cmod / (bpm_goal * rate)`.


class _TimingMap:
    """Pre-walked BPMs + STOPS + DELAYS + WARPS event stream, with monotonic
    bisect-indexable checkpoints for O(log n) beat<->time conversion.

    Walking the event stream on every `_beat_to_time` / `_time_to_beat` call
    was the render-hot-path bottleneck on Etterna charts with many BPM/stop
    segments (Undiscovered Colors has 310 BPMs + 591 stops — a linear walk
    costs ~34us per call, and `distance()` does two of those per visible
    note per frame). Walking once into checkpoint arrays drops the per-call
    cost to a bisect + constant-time arithmetic."""

    def __init__(self, bpms, sm_offset, stops, delays, warps):
        bpms = sorted(bpms or [(0.0, 120.0)])
        stops = sorted(stops or [])
        delays = sorted(delays or [])
        warps = sorted(warps or [])
        sm_offset = float(sm_offset)

        # Single unified event stream. Same kind-precedence as
        # sm_chart.beat_to_time so results are bit-identical.
        events = []
        for b, v in delays: events.append((b, 0, v))
        for b, v in bpms:   events.append((b, 1, v))
        for b, v in warps:  events.append((b, 2, v))
        for b, v in stops:  events.append((b, 3, v))
        events.sort(key=lambda e: (e[0], e[1]))

        # Checkpoints: after processing each event, record
        #   beat_enter:   cur_beat at moment this event starts (pre-event)
        #   time_enter:   time at that moment
        #   beat_exit:    cur_beat after applying the event (==beat_enter for
        #                 BPM/DELAY/STOP; beat_enter+warp_len for WARP)
        #   time_exit:    time after event (adds delay/stop duration)
        #   bps_after:    sec-per-beat-inverse that applies from this event
        #                 to the next
        # The segment between events[i] and events[i+1] advances at bps_after
        # from beat_exit to events[i+1]'s beat.
        self._beat_enter: list[float] = []
        self._time_enter: list[float] = []
        self._beat_exit: list[float] = []
        self._time_exit: list[float] = []
        self._bps_after: list[float] = []

        t = -sm_offset
        cur_beat = 0.0
        bps = bpms[0][1] / 60.0
        warp_end = None

        # TODO: Generalize this to non-Etterna
        for (eb, kind, val) in events:
            if eb < 0:
                continue
            # Advance from cur_beat to eb along current bps.
            if warp_end is not None and eb <= warp_end:
                cur_beat = eb
            elif warp_end is not None and cur_beat < warp_end <= eb:
                cur_beat = warp_end
                warp_end = None
                t += (eb - cur_beat) / bps
                cur_beat = eb
            else:
                t += (eb - cur_beat) / bps
                cur_beat = eb

            beat_enter = cur_beat
            time_enter = t

            if kind == 0:       # DELAY: pause before
                t += float(val)
            elif kind == 1:     # BPM change
                bps = float(val) / 60.0
            elif kind == 2:     # WARP — beat teleports forward, time doesn't
                warp_end = cur_beat + float(val)
            elif kind == 3:     # STOP: pause after
                t += float(val)

            # beat_exit reflects the post-event beat cursor for bisect
            # lookups — for WARP, that's the warp landing; for others,
            # unchanged from beat_enter. The event loop's own cur_beat
            # stays at beat_enter for warps so subsequent in-warp events
            # still get skipped correctly by the warp_end state check.
            if kind == 2:
                beat_exit = warp_end
            else:
                beat_exit = cur_beat
            time_exit = t

            self._beat_enter.append(beat_enter)
            self._time_enter.append(time_enter)
            self._beat_exit.append(beat_exit)
            self._time_exit.append(time_exit)
            self._bps_after.append(bps)

        # Initial bps for beat lookups at time < first event.
        self._bps_initial = bpms[0][1] / 60.0
        self._t_at_beat_zero = -sm_offset
        self._trailing_warp_end = warp_end

        # Numpy mirrors for batched lookup paths (project_times). Kept
        # in sync with the list fields; rebuild would require recomputing
        # both if we ever mutated them, but _TimingMap is constructed once
        # per engine and never edited.
        self._beat_enter_np = np.asarray(self._beat_enter, dtype=np.float64)
        self._time_enter_np = np.asarray(self._time_enter, dtype=np.float64)
        self._beat_exit_np = np.asarray(self._beat_exit, dtype=np.float64)
        self._time_exit_np = np.asarray(self._time_exit, dtype=np.float64)
        self._bps_after_np = np.asarray(self._bps_after, dtype=np.float64)

    def time_to_beat_array(self, times: np.ndarray) -> np.ndarray:
        """Vectorized `time_to_beat`. Same rules as the scalar path
        (frozen beats inside STOP/DELAY pauses, extrapolation before the
        first event and past the last), applied over an entire array at
        once."""
        t = np.asarray(times, dtype=np.float64)
        if not t.size:
            return np.empty(0, dtype=np.float64)
        if self._beat_enter_np.size == 0:
            return (t - self._t_at_beat_zero) * self._bps_initial

        out = np.empty_like(t)
        # Pre-first-event branch: extrapolate from beat 0 at initial bps.
        before = t < self._time_enter_np[0]
        if before.any():
            out[before] = (t[before] - self._t_at_beat_zero) * self._bps_initial
        remaining = ~before
        if not remaining.any():
            return out

        tr = t[remaining]
        # Pause window test: t lies between time_enter[idx] and time_exit[idx]
        # at the same idx — beat is frozen to beat_exit[idx].
        pause_idx = np.searchsorted(self._time_enter_np, tr, side='right') - 1
        pause_valid = pause_idx >= 0
        pause_in = np.zeros_like(tr, dtype=bool)
        if pause_valid.any():
            safe_p = np.clip(pause_idx, 0, self._time_exit_np.size - 1)
            pause_in = pause_valid & (tr < self._time_exit_np[safe_p])
        out_r = np.empty_like(tr)
        if pause_in.any():
            out_r[pause_in] = self._beat_exit_np[pause_idx[pause_in]]
        not_pause = ~pause_in
        if not_pause.any():
            tr2 = tr[not_pause]
            idx = np.searchsorted(self._time_exit_np, tr2, side='right') - 1
            idx = np.clip(idx, 0, self._time_exit_np.size - 1)
            out_r[not_pause] = (self._beat_exit_np[idx]
                                + (tr2 - self._time_exit_np[idx])
                                * self._bps_after_np[idx])
        out[remaining] = out_r
        return out

    def beat_to_time(self, beat: float) -> float:
        """Time at `beat`. Bisect the event list, then advance at
        post-event bps from that event to `beat`."""
        import bisect as _b
        if not self._beat_enter:
            return self._t_at_beat_zero + beat / self._bps_initial
        # Find last event whose beat_enter <= target beat. (If target beat
        # lies in the event's own span [beat_enter, beat_exit], time is
        # mid-transition — return time_enter for beats before event
        # completion, time_exit for after. Effectively: if target_beat ==
        # beat_enter, the PRE-event time is correct.)
        idx = _b.bisect_right(self._beat_enter, beat) - 1
        if idx < 0:
            # Before the first event: advance from beat 0 at initial bps.
            return self._t_at_beat_zero + beat / self._bps_initial
        # Is the target beat inside a WARP span that this event opened?
        # A WARP event has beat_exit > beat_enter; beats strictly between
        # land at time_exit (no time passes).
        bx = self._beat_exit[idx]
        if beat < bx:
            return self._time_exit[idx]
        # Past this event — advance at bps_after to `beat`.
        return self._time_exit[idx] + (beat - bx) / self._bps_after[idx]

    def time_to_beat(self, t: float) -> float:
        """Inverse of beat_to_time. Bisect on time_exit (the post-event
        times are monotonically non-decreasing)."""
        import bisect as _b
        if not self._time_exit:
            return (t - self._t_at_beat_zero) * self._bps_initial
        if t < self._time_enter[0]:
            # Before the first event: walk back at initial bps.
            return (t - self._t_at_beat_zero) * self._bps_initial
        # If we're inside a STOP/DELAY window, chart beat is frozen at that
        # event's beat until the pause finishes.
        pause_idx = _b.bisect_right(self._time_enter, t) - 1
        if pause_idx >= 0 and t < self._time_exit[pause_idx]:
            return self._beat_exit[pause_idx]
        # Find last event whose time_exit <= t, then advance from its
        # post-event state.
        idx = _b.bisect_right(self._time_exit, t) - 1
        if idx < 0:
            return self._beat_exit[0]
        # Past all events — use the last event's post-event state.
        return self._beat_exit[idx] + (t - self._time_exit[idx]) * self._bps_after[idx]

    def bps_at_time(self, t: float) -> float:
        """Instantaneous beats-per-second at time t.

        Returns 0 inside STOP/DELAY windows where chart beat is frozen."""
        import bisect as _b
        if not self._time_enter:
            return self._bps_initial
        pause_idx = _b.bisect_right(self._time_enter, t) - 1
        if pause_idx >= 0 and t < self._time_exit[pause_idx]:
            return 0.0
        idx = _b.bisect_right(self._time_exit, t) - 1
        if idx < 0:
            return self._bps_initial
        return self._bps_after[idx]


class BeatSpaceSVEngine:
    """Etterna #SCROLLS + #SPEEDS positioning.

    scrolls: list[(beat, ratio)] in beat-space
    speeds:  list[(beat, ratio)] or list[(beat, ratio, delay, unit)] in
             beat-space, where unit is 0=beats or 1=seconds
    bpms:    list[(beat, bpm)] used to convert beat-space distance to the
             Player's time-space px/sec units
    sm_offset: song OFFSET in seconds (positive = audio starts later)
    stops/delays/warps: optional timing events — passed to beat_to_time so
             beat<->time conversion tracks Etterna's GetElapsedTimeInternal."""

    def __init__(self, scrolls, speeds, bpms, sm_offset,
                 stops=None, delays=None, warps=None):
        self._scrolls = list(scrolls or [])
        self._speeds = [self._normalize_speed_segment(s) for s in (speeds or [])]
        self._speed_beats = [s[0] for s in self._speeds]
        self._bpms = list(bpms or [(0.0, 120.0)])
        self._sm_offset = float(sm_offset)
        self._stops = list(stops or [])
        self._delays = list(delays or [])
        self._delay_at_beats = {float(b): float(v) for b, v in self._delays}
        self._warps = list(warps or [])
        # Base BPM = first BPM segment, matching Etterna's m_fReadBPM seed.
        # ArrowEffects divides by m_fReadBPM under MaxScrollBPM; our conversion
        # from beat distance to seconds uses sec_per_beat at this rate so XMOD
        # 1.0 keeps a constant px/beat regardless of BPM changes elsewhere.
        self._base_bpm = float(self._bpms[0][1]) if self._bpms else 120.0
        self._sec_per_base_beat = 60.0 / self._base_bpm
        # Pre-walked timing map so _beat_to_time / _time_to_beat are O(log n)
        # instead of O(segments) per call — critical on charts with 100s of
        # BPM changes and stops (e.g. Undiscovered Colors).
        self._timing = _TimingMap(self._bpms, self._sm_offset,
                                   self._stops, self._delays, self._warps)

        # DisplayedBeat cache: (beat, displayed_beat, ratio). At any real beat
        # b in [segment_i.beat, segment_{i+1}.beat], displayed_beat =
        #   segment_i.displayed_beat + (b - segment_i.beat) * segment_i.ratio
        # If the first #SCROLLS doesn't start at beat 0, we prepend an
        # implicit ratio=1 segment matching Etterna's ResetCacheInfo.
        self._cache: list[tuple[float, float, float]] = []
        if self._scrolls:
            if self._scrolls[0][0] > 0.0:
                self._cache.append((0.0, 0.0, 1.0))
            displayed = 0.0
            last_beat = 0.0
            last_ratio = 1.0
            for (b, r) in self._scrolls:
                displayed += (b - last_beat) * last_ratio
                self._cache.append((b, displayed, r))
                last_beat = b
                last_ratio = r
        self._cache_beats = [c[0] for c in self._cache]
        # Numpy mirrors for vectorized displayed_beat over arrays — kept in
        # sync with `_cache`; `_cache` stays Python-list for scalar hits.
        self._cache_beats_np = np.asarray(self._cache_beats, dtype=np.float64)
        self._cache_db_np = np.asarray([c[1] for c in self._cache], dtype=np.float64)
        self._cache_ratio_np = np.asarray([c[2] for c in self._cache], dtype=np.float64)
        # Sorted-by-displayed-beat table for inverse_cumulative_at bisects.
        # SCROLLS with negative or zero ratios make displayed_beat
        # non-monotonic, so we skip those segments during inverse lookup.
        self._cache_dbs_monotonic: list[tuple[float, int]] = []
        last_db = -float('inf')
        for i, c in enumerate(self._cache):
            db = c[1]
            if db >= last_db:
                self._cache_dbs_monotonic.append((db, i))
                last_db = db
        self._cache_dbs_only = [x[0] for x in self._cache_dbs_monotonic]

        self.enabled = bool(self._scrolls or self._speeds)

    # --- real-beat ↔ real-time helpers ----------------------------------

    def _beat_to_time(self, beat: float) -> float:
        return self._timing.beat_to_time(beat)

    def _time_to_beat(self, t: float) -> float:
        return self._timing.time_to_beat(t)

    @staticmethod
    def _normalize_speed_segment(seg) -> tuple[float, float, float, int]:
        if len(seg) >= 4:
            beat, ratio, delay, unit = seg[:4]
            return (float(beat), float(ratio), float(delay),
                    0 if int(unit) == 0 else 1)
        if len(seg) == 3:
            beat, ratio, delay = seg
            return float(beat), float(ratio), float(delay), 0
        beat, ratio = seg[:2]
        return float(beat), float(ratio), 0.0, 0

    # --- SCROLLS integral ----------------------------------------------

    def _displayed_beat(self, beat: float) -> float:
        if not self._cache:
            return beat
        idx = bisect.bisect_right(self._cache_beats, beat) - 1
        if idx < 0:
            # Before first segment: ratio=1 extrapolation (matches Etterna's
            # GetDisplayedBeat fallthrough `return beat`).
            return beat
        b, db, r = self._cache[idx]
        return db + (beat - b) * r

    def _scroll_ratio_at_beat(self, beat: float) -> float:
        if not self._cache:
            return 1.0
        idx = bisect.bisect_right(self._cache_beats, beat) - 1
        if idx < 0:
            return 1.0
        return float(self._cache[idx][2])

    # --- SPEEDS evaluator (position-dependent!) -------------------------

    def _delay_at_beat(self, beat: float) -> float:
        return self._delay_at_beats.get(float(beat), 0.0)

    def _speed_percent_at(self, beat: float, music_seconds: float) -> float:
        """Evaluate Etterna's GetDisplayedSpeedPercent(songBeat, songSec)."""
        if not self._speeds:
            return 1.0
        idx = bisect.bisect_right(self._speed_beats, beat) - 1
        if idx < 0:
            return 1.0
        seg_beat, seg_ratio, seg_delay, seg_unit = self._speeds[idx]
        start_time = self._beat_to_time(seg_beat) - self._delay_at_beat(seg_beat)
        if seg_unit == 1:
            end_time = start_time + seg_delay
        else:
            end_beat = seg_beat + seg_delay
            end_time = self._beat_to_time(end_beat) - self._delay_at_beat(end_beat)

        first = self._speeds[0]
        if idx == 0 and first[2] > 0.0 and music_seconds < start_time:
            return 1.0
        if end_time >= music_seconds and (idx > 0 or first[2] > 0.0):
            prior_ratio = 1.0 if idx == 0 else self._speeds[idx - 1][1]
            duration = end_time - start_time
            ratio_used = 1.0 if duration == 0.0 else (music_seconds - start_time) / duration
            return prior_ratio + (seg_ratio - prior_ratio) * ratio_used
        return seg_ratio

    # --- SVEngine interface ---------------------------------------------

    def cumulative_at(self, t: float) -> float:
        """Cull-space cumulative = displayed-beat integral (SCROLLS only),
        converted to base-BPM seconds. Excludes SPEEDS so the cache stays
        consistent as the playhead moves."""
        b = self._time_to_beat(t)
        return self._displayed_beat(b) * self._sec_per_base_beat

    def cumulative_velocity_at(self, t: float) -> float:
        beat = self._time_to_beat(t)
        scroll = self._scroll_ratio_at_beat(beat)
        return scroll * self._timing.bps_at_time(t) * self._sec_per_base_beat

    def inverse_cumulative_at(self, sv: float) -> float:
        """Chart-time t such that cumulative_at(t) == sv. Used by the
        cull-space clock smoother. The inverse is only well-defined where
        displayed_beat is strictly increasing (SCROLLS ratio > 0); regions
        of ratio <= 0 collapse many beats to the same sv-value, so we
        return the earliest matching chart-time there."""
        if not self._cache:
            return float(sv)
        db_target = sv / self._sec_per_base_beat
        arr = self._cache_dbs_only
        if not arr:
            return float(sv)
        idx = bisect.bisect_right(arr, db_target) - 1
        if idx < 0:
            # Before first cache entry: ratio=1 extrapolation from beat 0
            return self._timing.beat_to_time(db_target)
        cache_idx = self._cache_dbs_monotonic[idx][1]
        b, db, r = self._cache[cache_idx]
        if r > 0:
            beat = b + (db_target - db) / r
        else:
            beat = b  # zero/negative ratio segment: collapse to its start
        return self._timing.beat_to_time(beat)

    def distance(self, t_from: float, t_to: float) -> float:
        """Render-space distance. SCROLLS cumulative difference, zoomed
        uniformly by SPEEDS at the playhead (Etterna's ArrowEffects.cpp
        XMOD branch: GetDisplayedSpeedPercent evaluated at the current
        song position, applied to the whole YOffset)."""
        d = self.cumulative_at(t_to) - self.cumulative_at(t_from)
        b_from = self._time_to_beat(t_from)
        return d * self._speed_percent_at(b_from, t_from)

    def project_times(self, times: np.ndarray) -> np.ndarray:
        """Vectorized `cumulative_at` over an array. Calls the _TimingMap
        and displayed-beat lookups in batch (one np.searchsorted each)
        instead of two Python bisects per entry — gives the renderer
        O(log n) per note with a single numpy pass for the whole frame."""
        t = np.asarray(times, dtype=np.float64)
        if not t.size:
            return np.empty(0, dtype=np.float64)
        # beat = time_to_beat_array(t) — STOPS/DELAYS/WARPS aware.
        beats = self._timing.time_to_beat_array(t)
        if self._cache_beats_np.size == 0:
            return beats * self._sec_per_base_beat
        # displayed_beat(beat) vectorized.
        idx = np.searchsorted(self._cache_beats_np, beats, side='right') - 1
        pre_mask = idx < 0
        safe = np.clip(idx, 0, self._cache_beats_np.size - 1)
        db_out = (self._cache_db_np[safe]
                  + (beats - self._cache_beats_np[safe]) * self._cache_ratio_np[safe])
        if pre_mask.any():
            # Etterna GetDisplayedBeat fallthrough: `return beat`.
            db_out[pre_mask] = beats[pre_mask]
        return db_out * self._sec_per_base_beat

    def as_sections(self) -> list[tuple[float, float]]:
        """Back-compat projection for sidebar/components readers. Samples the
        combined scroll*speed curve at every change point and emits time-space
        (time_sec, multiplier) pairs. Not faithful at SPEEDS transitions or
        across BPM changes, but fine for display."""
        if not (self._scrolls or self._speeds):
            return []
        beats = sorted({s[0] for s in self._scrolls} | {s[0] for s in self._speeds})

        def last_value(pairs, target):
            val = 1.0
            for item in pairs:
                b, v = item[0], item[1]
                if b <= target:
                    val = v
                else:
                    break
            return val

        out = []
        for b in beats:
            t = self._beat_to_time(b)
            mult = last_value(self._scrolls, b) * last_value(self._speeds, b)
            out.append((t, mult))
        out.sort(key=lambda x: x[0])
        return out

    def render_multiplier_at(self, t: float) -> float:
        t = float(t)
        return self._speed_percent_at(self._time_to_beat(t), t)

    def debug_snapshot_at(self, t: float) -> dict:
        t = float(t)
        beat = self._time_to_beat(t)
        displayed_beat = self._displayed_beat(beat)
        return {
            'engine': 'beat',
            't': t,
            'beat': beat,
            'displayed_beat': displayed_beat,
            'scroll_ratio': self._scroll_ratio_at_beat(beat),
            'speed_percent': self._speed_percent_at(beat, t),
            'cumulative': displayed_beat * self._sec_per_base_beat,
            'cumulative_velocity': self.cumulative_velocity_at(t),
        }

    # ArrowEffects::FindDisplayedBeats does a binary search for the first
    # off-screen beat. When scroll=0 the search collapses because YOffset
    # stays 0 regardless of how far ahead you look — the 10/2/... halving
    # sum caps out around songBeat + ~20. Without matching that cap, our
    # SV-space bisect keeps every note with the same SV-cum value, so a
    # scroll=0 region lets the entire pile pass culling.
    _MAX_LOOKAHEAD_BEATS = 20.0

    def max_visible_t_from(self, song_t: float) -> float:
        song_beat = self._time_to_beat(song_t)
        return self._beat_to_time(song_beat + self._MAX_LOOKAHEAD_BEATS)


# ----------------------------------------------------------------------
# Identity fallback — used when the adapter returns None or the chart has
# no SV data. Keeps the Player's code path uniform.
# ----------------------------------------------------------------------


class IdentitySVEngine:
    enabled = False

    def distance(self, t_from: float, t_to: float) -> float:
        return t_to - t_from

    def cumulative_at(self, t: float) -> float:
        return float(t)

    def inverse_cumulative_at(self, sv: float) -> float:
        return float(sv)

    def cumulative_velocity_at(self, t: float) -> float:
        del t
        return 1.0

    def project_times(self, times: np.ndarray) -> np.ndarray:
        return np.asarray(times, dtype=np.float64).copy()

    def as_sections(self) -> list[tuple[float, float]]:
        return []

    def render_multiplier_at(self, t: float) -> float:
        del t
        return 1.0

    def debug_snapshot_at(self, t: float) -> dict:
        t = float(t)
        return {
            'engine': 'identity',
            't': t,
            'cumulative': t,
            'render_multiplier': 1.0,
            'cumulative_velocity': 1.0,
        }

    def max_visible_t_from(self, song_t: float) -> float:
        return float('inf')
