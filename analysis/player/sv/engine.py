"""Scroll-velocity engine abstraction.

The Player's renderer routes every note position through an `SVEngine` so
game-specific positioning math doesn't leak in. The engine converts between
chart-time (what the replay stores) and SV-space (what the renderer uses for
note Y positions and visible-window culling). Per-replay engines are built by
`SvRenderController.build_engine_registry`; classes here are the building
blocks.

Implementations:
- `IdentitySVEngine`  ; no-op. Distance(a, b) = b - a. Used for charts with
                        no SV data and as the fallback during init.
- `QuaverSVEngine`    ; piecewise-constant multiplier in time-space, signed
                        cumulative (negative SV allowed). Quaver semantics.
                        See `Quaver/Shared/.../ScrollGroupControllerKeys.cs`.

The two heavy engines (osu time-space, Etterna beat-space) live in
`measure_engine.py`; they're built by `time_space_engine` /
`beat_space_engine` factory functions over a shared integrator.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


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


# ----------------------------------------------------------------------
# Quaver time-space engine ; piecewise-constant multiplier in time-space
# with two Quaver-specific shifts:
#   * pre-first-section uses the chart's `InitialScrollVelocity` (default
#     1.0), not `sections[0].multiplier` (Quaver/Shared/.../
#     ScrollGroupControllerKeys.cs::GetPositionFromTime, index==0 branch).
#   * negative multipliers are valid ; cumulative is signed and not
#     necessarily monotonic in chart-time. Notes can scroll back up the
#     screen. Callers that bisect on cum-space (the renderer's culling)
#     must check `cumulative_monotonic` and fall back to time-domain.
# BPM is intentionally ignored: Quaver's positioning never reads
# TimingPoint.Bpm. NaN multipliers coerce to 0, matching Quaver's
# `if (float.IsNaN(multiplier)) multiplier = 0` guard.
# ----------------------------------------------------------------------


class QuaverSVEngine:
    """Quaver-style time-space SV with signed cumulative.

    sections: list[(start_time_sec, multiplier)] sorted by start_time
    initial_velocity: multiplier active for `t < sections[0].start_time`
                      (Qua.InitialScrollVelocity, default 1.0). For times
                      after the first section start this field plays no
                      role -- matches Quaver, where it only sets the
                      pre-first-section pad."""

    cumulative_monotonic = False

    def __init__(self, sections: list[tuple[float, float]],
                 initial_velocity: float = 1.0):
        self._sections = list(sections)
        self._initial = float(initial_velocity)
        self._times = np.array([s[0] for s in self._sections], dtype=np.float64)
        # NaN multipliers -> 0 (Quaver coerces NaN at draw time; we coerce
        # once here so cumulative integrates cleanly).
        vals = np.array([s[1] for s in self._sections], dtype=np.float64)
        if vals.size:
            vals = np.where(np.isnan(vals), 0.0, vals)
        self._values = vals
        n = len(self._sections)
        # cum[i] = displayed-position at sections[i].start_time, building
        # from cum[0] = first_t * initial_velocity (Quaver's index==0
        # branch evaluated at first_t).
        self._cum = np.zeros(n, dtype=np.float64)
        if n:
            self._cum[0] = float(self._times[0]) * self._initial
            for i in range(1, n):
                dt = self._times[i] - self._times[i - 1]
                self._cum[i] = self._cum[i - 1] + dt * float(self._values[i - 1])
        # The engine is "active" if it changes anything from identity:
        # any sections at all, or a non-1 initial velocity.
        self.enabled = bool(self._sections) or abs(self._initial - 1.0) > 1e-12

    def cumulative_at(self, t: float) -> float:
        if not self._sections:
            return float(t) * self._initial
        idx = int(np.searchsorted(self._times, t, side='right')) - 1
        if idx < 0:
            return t * self._initial
        return float(self._cum[idx]) + (t - float(self._times[idx])) * float(self._values[idx])

    def inverse_cumulative_at(self, sv: float) -> float:
        """Best-effort inverse. With negative multipliers cumulative is
        non-monotonic, so the inverse is multi-valued; we return the
        earliest chart-time matching `sv` by scanning sections in order.
        Production paths don't currently call this on the Quaver engine
        -- the predictor and smoother are off in prod."""
        if not self._sections:
            v = self._initial
            return float(sv) / v if v else 0.0
        # Pre-first-section linear region.
        first_t = float(self._times[0])
        v0 = self._initial
        if v0 != 0.0:
            t_pre = sv / v0
            if t_pre <= first_t:
                return t_pre
        # Walk sections forward; first one whose endpoint brackets `sv`.
        for i in range(len(self._sections)):
            seg_start_cum = float(self._cum[i])
            seg_v = float(self._values[i])
            seg_start_t = float(self._times[i])
            seg_end_t = (float(self._times[i + 1]) if i + 1 < len(self._times)
                         else float('inf'))
            seg_end_cum = (float(self._cum[i + 1]) if i + 1 < len(self._cum)
                           else seg_start_cum + (seg_end_t - seg_start_t) * seg_v)
            lo, hi = sorted((seg_start_cum, seg_end_cum))
            if lo <= sv <= hi and seg_v != 0.0:
                return seg_start_t + (sv - seg_start_cum) / seg_v
        # Fall back to extrapolating the last segment.
        last_v = float(self._values[-1])
        if last_v == 0.0:
            return float(self._times[-1])
        return float(self._times[-1]) + (sv - float(self._cum[-1])) / last_v

    def cumulative_velocity_at(self, t: float) -> float:
        if not self._sections:
            return self._initial
        idx = int(np.searchsorted(self._times, t, side='right')) - 1
        if idx < 0:
            return self._initial
        return float(self._values[idx])

    def distance(self, t_from: float, t_to: float) -> float:
        return self.cumulative_at(t_to) - self.cumulative_at(t_from)

    def project_times(self, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, dtype=np.float64)
        if not t.size:
            return np.empty(0, dtype=np.float64)
        if not self._sections:
            return t * self._initial
        idx = np.searchsorted(self._times, t, side='right') - 1
        safe_idx = np.clip(idx, 0, len(self._times) - 1)
        cum = self._cum[safe_idx]
        base_t = self._times[safe_idx]
        vals = self._values[safe_idx]
        out = cum + (t - base_t) * vals
        pre_mask = idx < 0
        if pre_mask.any():
            out[pre_mask] = t[pre_mask] * self._initial
        return out

    def as_sections(self) -> list[tuple[float, float]]:
        # Surface initial_velocity as a synthetic (0, initial) head when
        # it differs from the implicit ratio=1 expected by sidebar
        # readers, so the readout reflects what's actually applied
        # before the first SV point.
        if not self._sections:
            if abs(self._initial - 1.0) > 1e-12:
                return [(0.0, self._initial)]
            return []
        out = []
        first_t = float(self._times[0])
        if first_t > 0.0 and abs(self._initial - 1.0) > 1e-12:
            out.append((0.0, self._initial))
        out.extend((float(t), float(v))
                   for t, v in zip(self._times, self._values))
        return out

    def render_multiplier_at(self, t: float) -> float:
        del t
        return 1.0

    def debug_snapshot_at(self, t: float) -> dict:
        t = float(t)
        return {
            'engine': 'quaver',
            't': t,
            'cumulative': self.cumulative_at(t),
            'render_multiplier': 1.0,
            'cumulative_velocity': self.cumulative_velocity_at(t),
            'initial_velocity': self._initial,
        }

    def max_visible_t_from(self, song_t: float) -> float:
        return float('inf')

    def breakpoints(self) -> np.ndarray:
        return self._times.copy()


# ----------------------------------------------------------------------
# Identity fallback ; used when the chart has no SV data. Keeps the
# Player's code path uniform.
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
