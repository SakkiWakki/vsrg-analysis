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

    def project_times(self, times: np.ndarray) -> np.ndarray:
        """Batch `cumulative_at` — the renderer bisects the result."""

    def as_sections(self) -> list[tuple[float, float]]:
        """Legacy `[(time_sec, multiplier)]` projection for components and
        sidebar readers. Engines whose SV can't be faithfully expressed as
        time-space sections return an approximation (good enough for
        sidebar display, not for positioning)."""


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

    def distance(self, t_from: float, t_to: float) -> float:
        return self.cumulative_at(t_to) - self.cumulative_at(t_from)

    def project_times(self, times: np.ndarray) -> np.ndarray:
        if not times.size:
            return np.empty(0, dtype=np.float64)
        return np.array([self.cumulative_at(float(t)) for t in times],
                        dtype=np.float64)

    def as_sections(self) -> list[tuple[float, float]]:
        return list(self._sections)


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


class BeatSpaceSVEngine:
    """Etterna #SCROLLS + #SPEEDS positioning.

    scrolls: list[(beat, ratio)] in beat-space
    speeds:  list[(beat, ratio)] in beat-space
    bpms:    list[(beat, bpm)] used to convert beat-space distance to the
             Player's time-space px/sec units
    sm_offset: song OFFSET in seconds (positive = audio starts later)"""

    def __init__(self, scrolls, speeds, bpms, sm_offset):
        self._scrolls = list(scrolls or [])
        self._speeds = list(speeds or [])
        self._bpms = list(bpms or [(0.0, 120.0)])
        self._sm_offset = float(sm_offset)
        # Base BPM = first BPM segment, matching Etterna's m_fReadBPM seed.
        # ArrowEffects divides by m_fReadBPM under MaxScrollBPM; our conversion
        # from beat distance to seconds uses sec_per_beat at this rate so XMOD
        # 1.0 keeps a constant px/beat regardless of BPM changes elsewhere.
        self._base_bpm = float(self._bpms[0][1]) if self._bpms else 120.0
        self._sec_per_base_beat = 60.0 / self._base_bpm

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

        self.enabled = bool(self._scrolls or self._speeds)

    # --- real-beat ↔ real-time helpers ----------------------------------

    def _beat_to_time(self, beat: float) -> float:
        """Inline BPM-map integration (ported from sm_chart.beat_to_time but
        avoids a module import per call). Returns seconds since audio start."""
        bpms = sorted(self._bpms)
        if not bpms:
            return -self._sm_offset + beat * self._sec_per_base_beat
        if beat <= bpms[0][0]:
            return -self._sm_offset + (beat - bpms[0][0]) * (60.0 / bpms[0][1])
        t = -self._sm_offset
        for i in range(len(bpms)):
            b0, bpm0 = bpms[i]
            b1 = bpms[i + 1][0] if i + 1 < len(bpms) else float('inf')
            if beat <= b1:
                return t + (beat - b0) * (60.0 / bpm0)
            t += (b1 - b0) * (60.0 / bpm0)
        return t

    def _time_to_beat(self, t: float) -> float:
        """Inverse of _beat_to_time. Needed to evaluate SCROLLS/SPEEDS at a
        given chart time (the Player only tracks seconds)."""
        bpms = sorted(self._bpms)
        if not bpms:
            return (t + self._sm_offset) / self._sec_per_base_beat
        # Walk until cumulative time reaches t
        t_target = t + self._sm_offset
        cum_t = 0.0
        for i in range(len(bpms)):
            b0, bpm0 = bpms[i]
            b1 = bpms[i + 1][0] if i + 1 < len(bpms) else float('inf')
            sec_per_beat = 60.0 / bpm0
            if i == 0 and t_target < 0:
                # Before beat 0: extrapolate backward with first BPM
                return b0 + t_target / sec_per_beat
            segment_sec = (b1 - b0) * sec_per_beat if b1 != float('inf') else float('inf')
            if cum_t + segment_sec >= t_target:
                return b0 + (t_target - cum_t) / sec_per_beat
            cum_t += segment_sec
        # Past last segment: extrapolate with last BPM
        last_b, last_bpm = bpms[-1]
        return last_b + (t_target - cum_t) / (60.0 / last_bpm)

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

    # --- SPEEDS evaluator (position-dependent!) -------------------------

    def _speed_percent_at_beat(self, beat: float) -> float:
        """Evaluate #SPEEDS at a real beat. Ignores transition duration —
        treats each SPEEDS change as instantaneous at its declared beat.
        Full transition interpolation would require wall-clock time plus
        UNIT_BEATS / UNIT_SECONDS dispatch; we position notes instead of
        animating so the step function is adequate."""
        if not self._speeds:
            return 1.0
        idx = bisect.bisect_right([s[0] for s in self._speeds], beat) - 1
        if idx < 0:
            return 1.0
        return float(self._speeds[idx][1])

    # --- SVEngine interface ---------------------------------------------

    def cumulative_at(self, t: float) -> float:
        """Cull-space cumulative = displayed-beat integral (SCROLLS only),
        converted to base-BPM seconds. Excludes SPEEDS so the cache stays
        consistent as the playhead moves."""
        b = self._time_to_beat(t)
        return self._displayed_beat(b) * self._sec_per_base_beat

    def distance(self, t_from: float, t_to: float) -> float:
        """Render-space distance. SCROLLS cumulative difference, zoomed
        uniformly by SPEEDS at the playhead (Etterna's ArrowEffects.cpp
        XMOD branch: GetDisplayedSpeedPercent evaluated at the current
        song position, applied to the whole YOffset)."""
        d = self.cumulative_at(t_to) - self.cumulative_at(t_from)
        b_from = self._time_to_beat(t_from)
        return d * self._speed_percent_at_beat(b_from)

    def project_times(self, times: np.ndarray) -> np.ndarray:
        if not times.size:
            return np.empty(0, dtype=np.float64)
        return np.array([self.cumulative_at(float(t)) for t in times],
                        dtype=np.float64)

    def as_sections(self) -> list[tuple[float, float]]:
        """Back-compat projection for sidebar/components readers. Samples the
        combined scroll*speed curve at every change point and emits time-space
        (time_sec, multiplier) pairs. Not faithful at SPEEDS transitions or
        across BPM changes, but fine for display."""
        if not (self._scrolls or self._speeds):
            return []
        beats = sorted({b for b, _ in self._scrolls} | {b for b, _ in self._speeds})

        def last_value(pairs, target):
            val = 1.0
            for b, v in pairs:
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

    def project_times(self, times: np.ndarray) -> np.ndarray:
        return np.asarray(times, dtype=np.float64).copy()

    def as_sections(self) -> list[tuple[float, float]]:
        return []
