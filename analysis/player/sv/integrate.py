"""Single integrator for `cumulative_at` (per DESIGN.tex §9.1).

The cumulative function is

    C_mu(t) = int_{(t_0, t]} v(tau) d_mu(tau)

with `mu` decomposed as AC density rho + atoms. For piecewise-constant `v`
and piecewise-constant rho on a common refinement, C_mu is piecewise-linear-
with-jumps; evaluation reduces to a bisection plus constant-time arithmetic.

This module computes the cumulative on a precomputed grid (the union of all
discontinuity points) and exposes scalar / vector lookup. It is engine-agnostic:
both time-space (rho=1, no atoms) and beat-space (rho=BPM/60, atoms at warps)
go through the same code.
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
        # Atoms appear in the grid at their tau so the running sum picks them
        # up exactly once.
        parts = [mu.boundaries]
        if not v.empty:
            parts.append(v.times)
        if mu.atom_times.size:
            parts.append(mu.atom_times)
        grid = np.unique(np.concatenate(parts))
        self._grid = grid

        # AC density on each refinement interval [grid[i], grid[i+1]).
        # densities[k] applies on [boundaries[k], boundaries[k+1]); for any
        # grid interval contained in segment k, the density is densities[k].
        # Mid-point of the grid interval is a robust segment selector even
        # when the interval lies on a boundary edge.
        if grid.size >= 2:
            mids = 0.5 * (grid[:-1] + grid[1:])
            seg_idx = np.searchsorted(mu.boundaries, mids, side='right') - 1
            seg_idx = np.clip(seg_idx, 0, mu.densities.size - 1)
            rho_per_interval = mu.densities[seg_idx]
        else:
            rho_per_interval = np.zeros(0, dtype=np.float64)
        self._rho_interval = rho_per_interval

        # v on each refinement interval. For grid[i] before v.times[0],
        # extrapolate using multipliers[0] (matches existing engines).
        if grid.size >= 2:
            if v.empty:
                v_per_interval = np.ones_like(rho_per_interval)
            else:
                mids = 0.5 * (grid[:-1] + grid[1:])
                vidx = np.searchsorted(v.times, mids, side='right') - 1
                vidx_safe = np.clip(vidx, 0, v.multipliers.size - 1)
                v_per_interval = v.multipliers[vidx_safe]
                pre_mask = vidx < 0
                if pre_mask.any():
                    v_per_interval = v_per_interval.copy()
                    v_per_interval[pre_mask] = v.multipliers[0]
        else:
            v_per_interval = np.zeros(0, dtype=np.float64)
        self._v_interval = v_per_interval

        # AC contributions per refinement interval.
        if grid.size >= 2:
            widths = np.diff(grid)
            ac_contrib = v_per_interval * rho_per_interval * widths
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
        """Scalar C_mu(t)."""
        if self._grid.size == 0:
            return 0.0
        if t <= self._grid[0]:
            # Extrapolate before grid[0] using the first interval's (v, rho).
            if self._v_interval.size:
                return float(self._cum[0]
                             + (t - self._grid[0])
                             * self._v_interval[0]
                             * self._rho_interval[0])
            return 0.0
        idx = int(np.searchsorted(self._grid, t, side='right')) - 1
        idx = max(0, min(idx, self._grid.size - 1))
        base = self._cum[idx]
        if idx + 1 < self._grid.size:
            v_k = self._v_interval[idx]
            rho_k = self._rho_interval[idx]
            return float(base + (t - self._grid[idx]) * v_k * rho_k)
        # Past last grid point: extrapolate with last interval's (v, rho).
        if self._v_interval.size:
            v_k = self._v_interval[-1]
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
        # Pre-grid samples: extrapolate from grid[0] with first (v,rho).
        pre_mask = idx < 0
        idx_safe = np.clip(idx, 0, self._grid.size - 1)
        base = self._cum[idx_safe]
        # For idx == grid.size - 1 (past end), use last interval's (v,rho).
        # For interior, use _v_interval[idx] / _rho_interval[idx]. Build a
        # per-sample (v, rho) by indexing into _v_interval / _rho_interval
        # with idx clipped to interval count - 1.
        if self._v_interval.size:
            interval_idx = np.clip(idx_safe, 0, self._v_interval.size - 1)
            v_k = self._v_interval[interval_idx]
            rho_k = self._rho_interval[interval_idx]
            base_t = self._grid[idx_safe]
            out = base + (t - base_t) * v_k * rho_k
            if pre_mask.any():
                out[pre_mask] = (self._cum[0]
                                 + (t[pre_mask] - self._grid[0])
                                 * self._v_interval[0]
                                 * self._rho_interval[0])
            return out
        return np.zeros_like(t)

    def velocity_at(self, t: float) -> float:
        """Local d(C_mu)/dt at t = v(t) * rho(t) (the AC density times the
        integrand). Atoms are not visible to a derivative; the velocity here
        is the smooth part. Used by the visual-playhead predictor."""
        if self._v_interval.size == 0:
            return 0.0
        if t <= self._grid[0]:
            return float(self._v_interval[0] * self._rho_interval[0])
        idx = int(np.searchsorted(self._grid, t, side='right')) - 1
        idx = max(0, min(idx, self._v_interval.size - 1))
        return float(self._v_interval[idx] * self._rho_interval[idx])

    def inverse_at(self, sv: float) -> float:
        """Inverse C_mu: smallest t such that C_mu(t) >= sv. Used by the
        cull-space clock smoother. Atoms produce flat regions in t (a jump
        in C); we return the atom's tau in that case."""
        if self._cum.size == 0:
            return float(sv)
        if sv <= self._cum[0]:
            v0 = self._v_interval[0] if self._v_interval.size else 1.0
            rho0 = self._rho_interval[0] if self._rho_interval.size else 1.0
            denom = v0 * rho0
            if denom == 0.0:
                return float(self._grid[0])
            return float(self._grid[0] + (sv - self._cum[0]) / denom)
        idx = int(np.searchsorted(self._cum, sv, side='right')) - 1
        idx = max(0, min(idx, self._cum.size - 1))
        if idx + 1 >= self._cum.size:
            v_k = self._v_interval[-1] if self._v_interval.size else 1.0
            rho_k = self._rho_interval[-1] if self._rho_interval.size else 1.0
            denom = v_k * rho_k
            if denom == 0.0:
                return float(self._grid[-1])
            return float(self._grid[idx] + (sv - self._cum[idx]) / denom)
        v_k = self._v_interval[idx]
        rho_k = self._rho_interval[idx]
        denom = v_k * rho_k
        if denom == 0.0:
            return float(self._grid[idx])
        return float(self._grid[idx] + (sv - self._cum[idx]) / denom)

    # -------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------

    @staticmethod
    def _v_at_pointwise(tau, side, v: SVData) -> float:
        """v(tau^+) when side='right', v(tau^-) when side='left'.
        Used to attribute atom mass; piecewise-constant v means the value
        right after tau is the multiplier of the segment starting at or
        before tau."""
        if v.empty:
            return 1.0
        if side == 'right':
            idx = int(np.searchsorted(v.times, tau, side='right')) - 1
        else:
            idx = int(np.searchsorted(v.times, tau, side='left')) - 1
        if idx < 0:
            return float(v.multipliers[0])
        return float(v.multipliers[idx])
