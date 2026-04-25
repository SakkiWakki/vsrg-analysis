"""Single integrator for `cumulative_at` (per DESIGN.tex §9.1, §10).

The cumulative function is

    C_mu(t) = int_{(t_0, t]} v(tau) d_mu(tau)

with `mu` decomposed as AC density rho + atoms. The integrand `v` may be
piecewise-constant (the historical case, default `step` easing) or
piecewise-linear (`linear` easing -- Quaver SSF, fluXis multiplier events
within a single segment). The integrator stores `v` as a per-cell
(v_start, v_end) pair on the common refinement so the AC contribution of
each cell is a trapezoid integral

    int v * rho dt  =  rho * width * (v_start + v_end) / 2,

exact when `v` is linear on the cell and reducing to `rho * width * v` when
v_start == v_end (step easing). Both behaviors are handled by the same
arithmetic.

This module computes the cumulative on a precomputed grid (the union of all
discontinuity points + easing-window endpoints) and exposes scalar / vector
lookup. It is engine-agnostic: both time-space (rho=1, no atoms) and
beat-space (rho=BPM/60, atoms at warps) go through the same code.
"""
from __future__ import annotations

import numpy as np

from analysis.player.sv.document import SVData, SVDocument, TimingMeasure


class CumulativeIntegrator:
    """Precomputed C_mu(t) over the discontinuity grid.

    Construction:
      1. Build the common refinement of the AC-segment boundaries and the
         SV record times. Atoms are inserted as zero-width events.
      2. On each refinement interval [t_k, t_{k+1}), v and rho are constant;
         the AC contribution is v_k * rho_k * (t_{k+1} - t_k).
      3. Atoms at tau_w in (t_k, t_{k+1}] add v(tau_w^+) * mass_w.
      4. Cumulative samples C[i] = C_mu(grid[i]) are the running sum.

    Lookup:
      cumulative_at(t) bisects `grid` and adds the partial AC contribution
      from the segment containing t, plus atoms in (grid[idx], t].
    """

    def __init__(self, doc: SVDocument):
        mu = doc.measure
        v = doc.data
        # Build the common refinement of boundaries U sv_times U atom_times.
        # For eased segments, also include the easing-window endpoints
        # (record.time + record.duration) so each easing window is its
        # own grid cell and trapezoid integration is exact.
        parts = [mu.boundaries]
        if not v.empty:
            parts.append(v.times)
            if v.has_eased_segments():
                # Easing-window ends: only meaningful for non-step segments.
                ends = []
                for i, easing in enumerate(v.easings):
                    if easing == 'step':
                        continue
                    dur = float(v.durations[i])
                    if dur > 0.0:
                        ends.append(float(v.times[i]) + dur)
                if ends:
                    parts.append(np.asarray(ends, dtype=np.float64))
        if mu.atom_times.size:
            parts.append(mu.atom_times)
        grid = np.unique(np.concatenate(parts))
        self._grid = grid

        # AC density on each refinement interval [grid[i], grid[i+1]).
        # densities[k] applies on [boundaries[k], boundaries[k+1]); for any
        # grid interval contained in segment k, the density is densities[k].
        if grid.size >= 2:
            mids = 0.5 * (grid[:-1] + grid[1:])
            seg_idx = np.searchsorted(mu.boundaries, mids, side='right') - 1
            seg_idx = np.clip(seg_idx, 0, mu.densities.size - 1)
            rho_per_interval = mu.densities[seg_idx]
        else:
            rho_per_interval = np.zeros(0, dtype=np.float64)
        self._rho_interval = rho_per_interval

        # v at the START and END of each refinement interval. For step
        # segments these are equal (v_start == v_end == multiplier). For
        # linear segments, v ramps from start_multiplier to end_multiplier
        # over `duration`; the integrator evaluates this ramp at each
        # cell's start and end times.
        if grid.size >= 2:
            v_start, v_end = self._evaluate_v_at_cell_endpoints(v, grid)
        else:
            v_start = np.zeros(0, dtype=np.float64)
            v_end = np.zeros(0, dtype=np.float64)
        self._v_interval = v_start          # back-compat alias for tests
        self._v_interval_start = v_start
        self._v_interval_end = v_end

        # AC contributions per cell: trapezoid rule, exact for linear v.
        # Reduces to rho * width * v for step segments (v_start == v_end).
        if grid.size >= 2:
            widths = np.diff(grid)
            ac_contrib = (0.5 * (v_start + v_end)
                          * rho_per_interval * widths)
        else:
            ac_contrib = np.zeros(0, dtype=np.float64)

        # Atoms: each atom at tau_w contributes mass_w * v(tau_w^-) and lands
        # at the grid index where grid == tau_w. Since tau_w was unioned into
        # the grid, this is exact.
        #
        # Why pre-warp v(tau_w^-): in Etterna semantics the SCROLLS ratio
        # that applies to the warped beats is the one ACTIVE during the warp
        # span, not the one at the landing. A SCROLLS row exactly on the
        # landing beat does not retroactively reweight the warped beats.
        # See DESIGN.tex §5.2 -- the atom encodes the integral of v over
        # the warp's beat extent, evaluated at the warp's start.
        atom_grid_contrib = np.zeros(grid.size, dtype=np.float64)
        for tau_w, mass_w in zip(mu.atom_times, mu.atom_masses):
            j = int(np.searchsorted(grid, tau_w))
            if j >= grid.size or grid[j] != tau_w:
                continue
            v_before = self._v_at_pointwise(tau_w, side='left', v=v)
            atom_grid_contrib[j] += v_before * mass_w
        self._atom_grid_contrib = atom_grid_contrib

        # Cumulative samples at every grid point.
        # C[0] = 0, C[i] = sum_{k<i} ac_contrib[k] + sum_{j<=i} atom_grid_contrib[j]
        # Atoms at grid[j] are counted as included once we've reached grid[j]
        # (half-open (t_0, t]).
        cum = np.empty(grid.size, dtype=np.float64)
        cum[0] = atom_grid_contrib[0] if grid.size else 0.0
        if grid.size >= 2:
            running_ac = np.concatenate(([0.0], np.cumsum(ac_contrib)))
            running_atoms = np.cumsum(atom_grid_contrib)
            cum = running_ac + running_atoms
        self._cum = cum

    # -------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------

    def cumulative_at(self, t: float) -> float:
        """Scalar C_mu(t).

        # PORT BOUNDARY (hot path).
        # Native port:
        #     idx = bsearch(grid, t)              # ~20 ns branchless
        #     dt  = t - grid[idx]
        #     return cum[idx] + (v_s + (v_e - v_s) * dt/width) * rho * dt * 0.5
        # ~30-40 ns total. The Python branchy version is ~1 us.
        """
        if self._grid.size == 0:
            return 0.0
        if t <= self._grid[0]:
            # Extrapolate before grid[0] using the first cell's (v_start, rho).
            # Pre-grid v is constant at v_start[0] (no easing applies).
            if self._v_interval_start.size:
                return float(self._cum[0]
                             + (t - self._grid[0])
                             * self._v_interval_start[0]
                             * self._rho_interval[0])
            return 0.0
        idx = int(np.searchsorted(self._grid, t, side='right')) - 1
        idx = max(0, min(idx, self._grid.size - 1))
        base = self._cum[idx]
        if idx + 1 < self._grid.size:
            return float(base + self._cell_partial_contrib(idx, t))
        # Past last grid point: extrapolate with last cell's (v_end, rho).
        if self._v_interval_end.size:
            v_k = self._v_interval_end[-1]
            rho_k = self._rho_interval[-1]
            return float(base + (t - self._grid[-1]) * v_k * rho_k)
        return float(base)

    def project_times(self, times: np.ndarray) -> np.ndarray:
        """Vectorized C_mu over an array."""
        t = np.asarray(times, dtype=np.float64)
        if not t.size:
            return np.empty(0, dtype=np.float64)
        if self._grid.size == 0:
            return np.zeros_like(t)
        idx = np.searchsorted(self._grid, t, side='right') - 1
        pre_mask = idx < 0
        idx_safe = np.clip(idx, 0, self._grid.size - 1)
        base = self._cum[idx_safe]
        if self._v_interval_start.size:
            # Two cases:
            #   (a) idx in [0, grid.size - 2]: t lies in cell `idx`.
            #       Partial = trapezoid from grid[idx] to t.
            #   (b) idx == grid.size - 1: t is past the last grid point.
            #       Extrapolate: partial = (t - grid[-1]) * v_e[-1] * rho[-1].
            past_end = idx_safe >= self._grid.size - 1
            interval_idx = np.clip(idx_safe, 0, self._v_interval_start.size - 1)
            cell_start = self._grid[interval_idx]
            v_s = self._v_interval_start[interval_idx]
            v_e = self._v_interval_end[interval_idx]
            rho = self._rho_interval[interval_idx]
            cell_widths_full = np.diff(self._grid)
            cell_widths = cell_widths_full[interval_idx]
            dt = t - cell_start
            with np.errstate(divide='ignore', invalid='ignore'):
                frac = np.where(cell_widths > 0, dt / cell_widths, 0.0)
            v_at_t = v_s + (v_e - v_s) * frac
            partial = rho * dt * 0.5 * (v_s + v_at_t)
            # Past-end override: dt is from grid[-1], not from cell_start.
            if past_end.any():
                dt_past = t - self._grid[-1]
                partial = np.where(past_end, dt_past * v_e * rho, partial)
            out = base + partial
            if pre_mask.any():
                out[pre_mask] = (self._cum[0]
                                 + (t[pre_mask] - self._grid[0])
                                 * self._v_interval_start[0]
                                 * self._rho_interval[0])
            return out
        return np.zeros_like(t)

    def velocity_at(self, t: float) -> float:
        """Local d(C_mu)/dt at t = v(t) * rho(t). For linear-easing cells,
        v(t) varies within the cell; returns the linearly interpolated
        value. Atoms are not visible to a derivative."""
        if self._v_interval_start.size == 0:
            return 0.0
        if t <= self._grid[0]:
            return float(self._v_interval_start[0] * self._rho_interval[0])
        idx = int(np.searchsorted(self._grid, t, side='right')) - 1
        idx = max(0, min(idx, self._v_interval_start.size - 1))
        v_t = self._v_at_t_in_cell(idx, t)
        return float(v_t * self._rho_interval[idx])

    def inverse_at(self, sv: float) -> float:
        """Inverse C_mu: smallest t such that C_mu(t) >= sv. Atoms produce
        flat regions in t (a jump in C); we return the atom's tau there.

        For step cells (v_start == v_end) the inverse is linear: solve
        rho * dt * v = sv - cum_lo for dt. For linear cells, the cumulative
        is quadratic in dt, and we'd need the quadratic formula -- not
        implemented; cross-engine consumers don't currently inverse over
        eased segments, so we approximate with the average v of the cell.
        """
        if self._cum.size == 0:
            return float(sv)
        if sv <= self._cum[0]:
            v0 = (self._v_interval_start[0]
                  if self._v_interval_start.size else 1.0)
            rho0 = self._rho_interval[0] if self._rho_interval.size else 1.0
            denom = v0 * rho0
            if denom == 0.0:
                return float(self._grid[0])
            return float(self._grid[0] + (sv - self._cum[0]) / denom)
        idx = int(np.searchsorted(self._cum, sv, side='right')) - 1
        idx = max(0, min(idx, self._cum.size - 1))
        if idx + 1 >= self._cum.size:
            v_k = (self._v_interval_end[-1]
                   if self._v_interval_end.size else 1.0)
            rho_k = self._rho_interval[-1] if self._rho_interval.size else 1.0
            denom = v_k * rho_k
            if denom == 0.0:
                return float(self._grid[-1])
            return float(self._grid[idx] + (sv - self._cum[idx]) / denom)
        # Average v over the cell (exact for step, average-trapezoid for
        # linear). Good enough for the cull-space inverse use case.
        v_avg = 0.5 * (self._v_interval_start[idx]
                       + self._v_interval_end[idx])
        rho_k = self._rho_interval[idx]
        denom = v_avg * rho_k
        if denom == 0.0:
            return float(self._grid[idx])
        return float(self._grid[idx] + (sv - self._cum[idx]) / denom)

    # -------------------------------------------------------------------
    # Per-cell evaluation helpers
    # -------------------------------------------------------------------

    def _cell_partial_contrib(self, idx: int, t: float) -> float:
        """AC contribution from `grid[idx]` to `t`, where `t` lies in the
        cell starting at `grid[idx]`. Trapezoid rule with linear
        interpolation of v."""
        cell_start = self._grid[idx]
        cell_end = self._grid[idx + 1]
        v_s = self._v_interval_start[idx]
        v_e = self._v_interval_end[idx]
        rho = self._rho_interval[idx]
        dt = t - cell_start
        width = cell_end - cell_start
        if width <= 0:
            return 0.0
        v_at_t = v_s + (v_e - v_s) * (dt / width)
        return rho * dt * 0.5 * (v_s + v_at_t)

    def _v_at_t_in_cell(self, idx: int, t: float) -> float:
        cell_start = self._grid[idx]
        if idx + 1 >= self._grid.size:
            return float(self._v_interval_end[idx])
        cell_end = self._grid[idx + 1]
        width = cell_end - cell_start
        if width <= 0:
            return float(self._v_interval_start[idx])
        v_s = self._v_interval_start[idx]
        v_e = self._v_interval_end[idx]
        return float(v_s + (v_e - v_s) * ((t - cell_start) / width))

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _evaluate_v_at_cell_endpoints(v: SVData, grid: np.ndarray):
        """Return (v_start, v_end) per grid cell.

        For step segments these are equal: the segment's multiplier.
        For linear segments, v ramps from `multiplier` at `time` to
        `end_multiplier` at `time + duration`, then holds at
        `end_multiplier` until the next segment starts. The returned
        arrays are sampled at `grid[:-1]` (cell starts, side='right' for
        ties so a segment-start tick gives the segment's value, not the
        previous segment's) and `grid[1:]` (cell ends, side='left').
        """
        n_cells = grid.size - 1
        if v.empty:
            return (np.ones(n_cells, dtype=np.float64),
                    np.ones(n_cells, dtype=np.float64))

        cell_starts = grid[:-1]
        cell_ends = grid[1:]
        # Cell-start: value at tau^+ -- the segment whose [time, ...) covers
        # tau. searchsorted(side='right') - 1 gives the index of the latest
        # segment with time <= tau. Pre-first-segment extrapolates as the
        # first segment's start multiplier (matches legacy behavior).
        v_start = CumulativeIntegrator._eval_v_array(v, cell_starts, side='right')
        # Cell-end: value at tau^- -- the segment that *ends* at tau is the
        # one whose [time, next_time) ends at this cell-end. side='left' - 1
        # gives the segment whose time < tau (excluding ties).
        v_end = CumulativeIntegrator._eval_v_array(v, cell_ends, side='left')
        return v_start, v_end

    @staticmethod
    def _eval_v_array(v: SVData, taus: np.ndarray, side: str) -> np.ndarray:
        """Vectorized v(tau) at each tau, honoring per-segment easing.

        `side='right'` attributes ties to the next segment (a tau equal to
        a segment-start lands in that segment); `side='left'` to the
        previous (a tau equal to a segment-start lands in the segment
        that ends there). This matches numpy.searchsorted semantics.
        """
        taus = np.asarray(taus, dtype=np.float64)
        if not taus.size:
            return np.empty(0, dtype=np.float64)
        if v.empty:
            return np.ones_like(taus)

        idx = np.searchsorted(v.times, taus, side=side) - 1
        out = np.empty_like(taus)
        pre_mask = idx < 0
        if pre_mask.any():
            # Pre-first-segment: extrapolate as multiplier[0].
            out[pre_mask] = v.multipliers[0]
        post_mask = ~pre_mask
        if post_mask.any():
            idx_p = idx[post_mask]
            taus_p = taus[post_mask]
            seg_time = v.times[idx_p]
            seg_mult = v.multipliers[idx_p]
            seg_end_mult = v.end_multipliers[idx_p]
            seg_duration = v.durations[idx_p]
            # Within the easing window: lerp from seg_mult to seg_end_mult.
            # After: hold at seg_end_mult. (For step segments, seg_end_mult
            # == seg_mult so this collapses to a constant.)
            tau_in_seg = taus_p - seg_time
            with np.errstate(divide='ignore', invalid='ignore'):
                ramp_frac = np.where(seg_duration > 0,
                                     np.clip(tau_in_seg / seg_duration, 0.0, 1.0),
                                     1.0)   # zero duration: skip directly to end
            out[post_mask] = seg_mult + (seg_end_mult - seg_mult) * ramp_frac
        return out

    @staticmethod
    def _v_at_pointwise(tau, side, v: SVData) -> float:
        """v(tau^+) when side='right', v(tau^-) when side='left'.
        Used to attribute atom mass. For step segments this returns the
        segment multiplier; for linear segments, the lerped value at tau.
        """
        out = CumulativeIntegrator._eval_v_array(
            v, np.array([tau], dtype=np.float64), side=side)
        return float(out[0])
