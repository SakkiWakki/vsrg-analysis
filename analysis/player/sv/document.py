"""Canonical SV document.

Per DESIGN.tex, an SV document is a measure-theoretic object: render distance
is the integral of an integrand `v` against a Radon measure `mu` on chart-time.
The document carries `mu` as a Lebesgue decomposition (AC density + atoms) and
`v` as piecewise-constant records.

Two parts:

* `TimingMeasure`  -- the measure mu. AC part stored as
  `(boundary_times, ac_density)` where ac_density is piecewise-constant on
  segments [boundary_times[i], boundary_times[i+1]]. Singular part stored as
  `(atom_times, atom_masses)`. Lebesgue measure (time-space) is encoded as
  `ac_density = ones`, no atoms. Stieltjes dB (beat-space) is encoded as
  `ac_density = BPM(tau)/60` with atoms at warps.

* `SVData`         -- the integrand `v`. Piecewise-constant records on
  chart-time, each carrying (time, multiplier, group). For the canonical form
  we store BOTH chart-time and beat (when available) so engines integrating
  against either measure don't need to reconvert. Per-column selectors are
  forward-compatible (selector `None` means "all columns") but only the
  all-columns path is exercised by current engines.

The integrator (`integrate.py`) consumes (TimingMeasure, SVData) and produces
the cumulative function C_mu(t). Engines pick which measure to request from
the document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# TimingMeasure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingMeasure:
    """A positive Radon measure on chart-time, stored as
    Lebesgue decomposition.

    AC part: piecewise-constant density `rho(tau)` on segments delimited by
    `boundaries`. `boundaries` is length n+1 (or 2 for single-segment), and
    `densities` is length n. Density is right-continuous: on
    [boundaries[i], boundaries[i+1]) the density is densities[i].

    Atoms: list of (tau_w, mass_w) pairs. Each atom contributes mass_w to
    any half-open interval (a, b] containing tau_w.

    Conventions:
    - `boundaries` must be sorted, strictly increasing where possible.
      Stops -- where two consecutive entries have the same tau and density
      collapses -- are NOT part of the AC representation; stops appear as
      regions where rho = 0 in the AC density.
    - Beat-space: `boundaries` are BPM-segment edges,
      `densities[i] = bpm_i / 60`, atoms at warp tau values with mass
      Delta_b * (sec_per_base_beat factor) -- the per-engine factor is
      applied at integration, the document stores raw beat-mass.
    """

    boundaries: np.ndarray   # float64, shape (n+1,)
    densities: np.ndarray    # float64, shape (n,)
    atom_times: np.ndarray   # float64, shape (k,) -- sorted
    atom_masses: np.ndarray  # float64, shape (k,)

    @staticmethod
    def lebesgue(t_start: float = 0.0, t_end: float = 1.0) -> 'TimingMeasure':
        """The plain Lebesgue measure d_tau on [t_start, t_end]."""
        return TimingMeasure(
            boundaries=np.array([t_start, t_end], dtype=np.float64),
            densities=np.array([1.0], dtype=np.float64),
            atom_times=np.zeros(0, dtype=np.float64),
            atom_masses=np.zeros(0, dtype=np.float64),
        )

    @staticmethod
    def from_timing_map(boundaries, densities, atoms=()) -> 'TimingMeasure':
        bnd = np.asarray(boundaries, dtype=np.float64)
        den = np.asarray(densities, dtype=np.float64)
        atom_t = np.asarray([a[0] for a in atoms], dtype=np.float64)
        atom_m = np.asarray([a[1] for a in atoms], dtype=np.float64)
        order = np.argsort(atom_t)
        return TimingMeasure(bnd, den, atom_t[order], atom_m[order])


# ---------------------------------------------------------------------------
# SVData (the integrand)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SVRecord:
    """Single piecewise-constant SV segment.

    `time` is chart-time; `multiplier` is the integrand value v(tau) on
    [time, next_record.time). `selector` reserved for per-column / per-note;
    `None` means "all columns".

    `group` reserved for Quaver / fluXis scroll groups; `"main"` for everything
    we currently support.
    """

    time: float
    multiplier: float
    selector: Optional[frozenset[int]] = None
    group: str = 'main'


@dataclass(frozen=True)
class SVData:
    """The integrand v of an SV document.

    Stored as parallel numpy arrays for the all-columns / single-group case
    (the only case currently exercised); selectors and groups are kept on the
    record list for forward-compat introspection.

    `times`, `multipliers` have shape (n,). The integrand is piecewise-
    constant: v(tau) = multipliers[k] for tau in [times[k], times[k+1]).
    Before times[0], v extrapolates using multipliers[0] (matches existing
    engine behavior).
    """

    times: np.ndarray
    multipliers: np.ndarray
    records: tuple[SVRecord, ...] = field(default_factory=tuple)

    @staticmethod
    def from_sections(sections) -> 'SVData':
        """Build from the legacy `[(time, multiplier)]` shape used by the
        time-space engine and as_sections() on beat-space."""
        if not sections:
            return SVData(
                times=np.zeros(0, dtype=np.float64),
                multipliers=np.zeros(0, dtype=np.float64),
                records=(),
            )
        times = np.asarray([s[0] for s in sections], dtype=np.float64)
        mults = np.asarray([s[1] for s in sections], dtype=np.float64)
        recs = tuple(SVRecord(time=float(t), multiplier=float(m))
                     for t, m in zip(times, mults))
        return SVData(times=times, multipliers=mults, records=recs)

    @property
    def empty(self) -> bool:
        return self.times.size == 0


# ---------------------------------------------------------------------------
# SVDocument -- the bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SVDocument:
    """Canonical SV document: (measure, integrand, zoom-hook).

    `measure` is `mu`. `data` is `v`. `zoom_fn` is an optional callable
    `t -> float` returning the position-dependent multiplier z(t); if None,
    z is identically 1.

    `enabled` mirrors the existing engine flag -- True iff the document
    encodes any non-trivial scroll behavior.
    """

    measure: TimingMeasure
    data: SVData
    zoom_fn: Optional[object] = None    # callable(t) -> float
    enabled: bool = False
