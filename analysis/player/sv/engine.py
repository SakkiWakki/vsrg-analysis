"""Scroll-velocity engine abstraction.

Each game adapter builds its own `SVEngine` (see `GameAdapter.build_sv_engine`),
so game-specific positioning math doesn't leak into the Player. The engine
converts between chart-time (what the replay stores) and SV-space (what the
renderer uses for note Y positions and visible-window culling).

Ex:
- `TimeSpaceSVEngine`  ; piecewise-constant multiplier in time-space. osu!mania
                         uses this (SV from timing points is a time-space curve).
- `BeatSpaceSVEngine`  ; Etterna's #SCROLLS as a piecewise-constant velocity
                         on *beats*, plus #SPEEDS as a uniform-field zoom
                         sampled at the song position. Required for charts
                         where #SCROLLS spans BPM changes ; integrating in
                         time-space silently accumulates error.

Games that don't populate SV leave `adapter.build_sv_engine()` at the base
`None` default; the Player then runs with identity SV (distance = t_to - t_from).
"""
from __future__ import annotations

import bisect
from typing import Protocol

import numpy as np

from analysis.player.sv.timing import TimingMap as _TimingMap


class SVEngine(Protocol):
    """What the Player needs from any SV implementation.

    Two spaces are exposed:

    * **Render-space** (`distance`): used by `_time_to_y`. May depend on the
      current song position ; e.g. Etterna #SPEEDS zooms the whole field
      based on where the playhead sits. Called per visible note per frame.

    * **Cull-space** (`project_times` / `cumulative_at`): monotonic, purely
      position-independent integrator. Used to pre-cache note positions for
      off-screen bisect culling. Invariant the Player relies on:

          cumulative_at(b) - cumulative_at(a) == distance(a, b)
          WHEN no position-dependent effect applies (e.g. SPEEDS = 1 flat).

      During brief position-dependent transitions (Etterna SPEEDS change),
      the culling window is approximate ; typically one SPEEDS segment's
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
        """Batch `cumulative_at` ; the renderer bisects the result."""

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
        binary-search convergence ; without it, scroll=0 regions produce
        huge SV-equal runs that all pass cull-space bisection."""
        return float('inf')

    def breakpoints(self) -> np.ndarray:
        """Sorted chart-times where dC/dt is discontinuous. Used by
        CullSpacePredictor to split extrapolation across SV/BPM/warp
        boundaries exactly. Default: empty (= no breakpoints; the
        predictor falls back to single-segment extrapolation)."""


def _empty_engine_sections(_engine) -> list[tuple[float, float]]:
    return []


# ----------------------------------------------------------------------
# Time-space engine ; osu!mania, and any other game that models SV as a
# piecewise-constant multiplier sampled on wall-clock time.
# ----------------------------------------------------------------------


class TimeSpaceSVEngine:
    """Integrates a piecewise-constant `(time_sec, multiplier)` curve.

    This is the original SV model. Used by osu!mania verbatim ; its timing
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
        """Vectorized `cumulative_at` over an array ; one np.searchsorted
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

    def breakpoints(self) -> np.ndarray:
        """Time-space dC/dt is constant on each section; section start
        times are the breakpoints."""
        return self._times.copy()


# ----------------------------------------------------------------------
# Beat-space engine ; Etterna's XMOD positioning.
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
# the note) ; it acts as a uniform zoom factor on the whole field. To fit
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




class BeatSpaceSVEngine:
    """Etterna #SCROLLS + #SPEEDS positioning.

    scrolls: list[(beat, ratio)] in beat-space
    speeds:  list[(beat, ratio)] or list[(beat, ratio, delay, unit)] in
             beat-space, where unit is 0=beats or 1=seconds
    bpms:    list[(beat, bpm)] used to convert beat-space distance to the
             Player's time-space px/sec units
    sm_offset: song OFFSET in seconds (positive = audio starts later)
    stops/delays/warps: optional timing events ; passed to beat_to_time so
             beat<->time conversion tracks Etterna's GetElapsedTimeInternal."""

    def __init__(self, scrolls, speeds, bpms, sm_offset,
                 stops=None, delays=None, warps=None):
        from analysis.player.sv.scrolls import ScrollsCache
        from analysis.player.sv.speeds import (SpeedsEvaluator,
                                                normalize_speed_segment)
        self._scrolls = list(scrolls or [])
        self._speeds = [normalize_speed_segment(s) for s in (speeds or [])]
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
        # instead of O(segments) per call ; critical on charts with 100s of
        # BPM changes and stops (e.g. Undiscovered Colors).
        self._timing = _TimingMap(self._bpms, self._sm_offset,
                                   self._stops, self._delays, self._warps)
        self._speeds_eval = SpeedsEvaluator(speeds, self._timing,
                                             delay_at_beats=self._delay_at_beats)

        # Pre-integrated displayed-beat curve from #SCROLLS, with a sorted
        # inverse table for the cull-space clock smoother. See scrolls.py.
        self._scrolls_cache = ScrollsCache(self._scrolls)

        self.enabled = bool(self._scrolls or self._speeds
                            or len(self._bpms) > 1
                            or self._stops or self._delays or self._warps)

    # --- real-beat ↔ real-time helpers ----------------------------------

    def _beat_to_time(self, beat: float) -> float:
        return self._timing.beat_to_time(beat)

    def _time_to_beat(self, t: float) -> float:
        return self._timing.time_to_beat(t)

    # --- SCROLLS integral ----------------------------------------------

    def _displayed_beat(self, beat: float) -> float:
        return self._scrolls_cache.displayed_beat(beat)

    def _scroll_ratio_at_beat(self, beat: float) -> float:
        return self._scrolls_cache.ratio_at_beat(beat)

    # --- SPEEDS evaluator (position-dependent!) -------------------------

    def _speed_percent_at(self, beat: float, music_seconds: float) -> float:
        return self._speeds_eval.percent_at(beat, music_seconds)

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
        cull-space clock smoother. In scroll<=0 plateaus the inverse is
        not well-defined; ScrollsCache.inverse_displayed_beat returns the
        earliest matching chart-time there."""
        # cumulative_at = displayed_beat(beat(t)) * sec_per_base_beat,
        # so the inverse is beat_to_time(displayed_beat^-1(sv / spb)).
        # With no #SCROLLS, displayed_beat is the identity so the inverse
        # collapses to beat_to_time(sv / spb) -- NOT float(sv), which
        # used to leak the cumulative value back as if it were chart-time.
        db_target = sv / self._sec_per_base_beat
        if not self._scrolls_cache:
            return self._timing.beat_to_time(db_target)
        return self._scrolls_cache.inverse_displayed_beat(db_target, self._timing)

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
        instead of two Python bisects per entry ; gives the renderer
        O(log n) per note with a single numpy pass for the whole frame."""
        t = np.asarray(times, dtype=np.float64)
        if not t.size:
            return np.empty(0, dtype=np.float64)
        # beat = time_to_beat_array(t) ; STOPS/DELAYS/WARPS aware.
        beats = self._timing.time_to_beat_array(t)
        return self.project_beats(beats)

    def project_beats(self, beats: np.ndarray) -> np.ndarray:
        """Project chart beats directly into Etterna displayed-beat space.

        Chart-only sprites inside old negative-BPM warp aliases need this:
        their elapsed time collapses to the warp endpoint, but Etterna still
        positions the sprite from its chart beat until the playhead jumps."""
        return self._scrolls_cache.displayed_beat_array(beats) * self._sec_per_base_beat

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
    # stays 0 regardless of how far ahead you look ; the 10/2/... halving
    # sum caps out around songBeat + ~20. Without matching that cap, our
    # SV-space bisect keeps every note with the same SV-cum value, so a
    # scroll=0 region lets the entire pile pass culling.
    _MAX_LOOKAHEAD_BEATS = 20.0

    def max_visible_t_from(self, song_t: float) -> float:
        song_beat = self._time_to_beat(song_t)
        return self._beat_to_time(song_beat + self._MAX_LOOKAHEAD_BEATS)

    def breakpoints(self) -> np.ndarray:
        """All chart-times where dC/dt is discontinuous: BPM-segment
        boundaries (incl. STOP/DELAY enter/exit), warp atoms, and SCROLLS
        change points. Used by CullSpacePredictor."""
        timing = self._timing
        # Timing-map events: BPM changes + STOP/DELAY entries/exits + warp times.
        bpm_times = list(timing._time_enter) + list(timing._time_exit)
        # SCROLLS change points (in beat-space, convert to chart-time).
        scroll_times = [self._beat_to_time(b)
                        for (b, _) in self._scrolls]
        return np.asarray(sorted(set(bpm_times + scroll_times)),
                          dtype=np.float64)


# ----------------------------------------------------------------------
# Identity fallback ; used when the adapter returns None or the chart has
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

    def breakpoints(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float64)
