"""Canonical SV document.

Per DESIGN.tex, an SV document is a measure-theoretic object: render distance
is the integral of an integrand `v` against a Radon measure `mu` on chart-time.
The document carries `mu` as a Lebesgue decomposition (AC density + atoms) and
`v` as a sequence of records that may be piecewise-constant (the historical
case) OR piecewise-smooth via per-segment easing (Quaver SSF, fluXis multiplier
events).

Two parts:

* `TimingMeasure`  -- the measure mu. AC part stored as
  `(boundary_times, ac_density)` where ac_density is piecewise-constant on
  segments [boundary_times[i], boundary_times[i+1]]. Singular part stored as
  `(atom_times, atom_masses)`. Lebesgue measure (time-space) is encoded as
  `ac_density = ones`, no atoms. Stieltjes dB (beat-space) is encoded as
  `ac_density = BPM(tau)/60` with atoms at warps.

* `SVData`         -- the integrand `v`. Piecewise-smooth records on
  chart-time, each carrying (time, multiplier, duration, easing, ...). The
  default easing is `step` (piecewise-constant), reproducing the original
  behavior. `linear` and other easings are evaluated by the integrator's
  per-curve closed-form table (see integrate.py). Per-column selectors are
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


# Easing names recognized by the integrator. See _EASING_TABLE in
# integrate.py for the actual closed-form integrators.
EASING_STEP = 'step'         # piecewise-constant (legacy default)
EASING_LINEAR = 'linear'     # lerp from this segment's m to next segment's m
                              # over `duration`; constant `m` afterwards.

VALID_EASINGS = frozenset({EASING_STEP, EASING_LINEAR})


@dataclass(frozen=True)
class SVRecord:
    """Single SV segment.

    `time` is chart-time; the segment runs from [time, next_record.time).

    Easing semantics:
      step    (default): integrand is constant `multiplier` for the whole
              segment. Reproduces the original piecewise-constant behavior.
      linear: integrand lerps from `multiplier` (at `time`) to `end_multiplier`
              (at `time + duration`), then holds at `end_multiplier` until the
              next record's `time`. If `duration` is 0 or unset, behaves like
              `step`. `end_multiplier` defaults to the next record's
              `multiplier` when None and the integrator can see the next
              record; otherwise it's the same as `multiplier` (degenerate
              linear = step).

    `duration` is in chart-time seconds. 0 means "no smooth ramp; emit the
    multiplier as a step." `selector` is reserved for per-column / per-note
    (None = all columns). `group` is reserved for Quaver / fluXis scroll
    groups (default `"main"`).
    """

    time: float
    multiplier: float
    duration: float = 0.0
    easing: str = EASING_STEP
    end_multiplier: Optional[float] = None
    selector: Optional[frozenset[int]] = None
    group: str = 'main'


@dataclass(frozen=True)
class SVData:
    """The integrand v of an SV document.

    Stored as parallel numpy arrays for fast per-segment lookup, plus the
    full record list for the integrator's easing dispatch. Most callers
    (`from_sections`) populate only `times`/`multipliers` with `step`
    easing -- the historical piecewise-constant case.

    `times`, `multipliers` have shape (n,). Easing-aware fields
    (`durations`, `end_multipliers`, `easings`) have the same shape;
    they're populated to defaults when missing so the integrator can read
    them uniformly.
    """

    times: np.ndarray
    multipliers: np.ndarray
    durations: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64))
    end_multipliers: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64))
    easings: tuple[str, ...] = field(default_factory=tuple)
    records: tuple[SVRecord, ...] = field(default_factory=tuple)

    @staticmethod
    def from_sections(sections) -> 'SVData':
        """Build from the legacy `[(time, multiplier)]` shape used by the
        time-space engine and as_sections() on beat-space. All segments
        emit step easing -- no smooth ramps -- so behavior matches the
        original piecewise-constant integrator exactly."""
        if not sections:
            return SVData(
                times=np.zeros(0, dtype=np.float64),
                multipliers=np.zeros(0, dtype=np.float64),
                durations=np.zeros(0, dtype=np.float64),
                end_multipliers=np.zeros(0, dtype=np.float64),
                easings=(),
                records=(),
            )
        times = np.asarray([s[0] for s in sections], dtype=np.float64)
        mults = np.asarray([s[1] for s in sections], dtype=np.float64)
        durs = np.zeros_like(mults)
        ends = mults.copy()        # step: end == start
        easings = tuple([EASING_STEP] * len(sections))
        recs = tuple(SVRecord(time=float(t), multiplier=float(m))
                     for t, m in zip(times, mults))
        return SVData(times=times, multipliers=mults,
                      durations=durs, end_multipliers=ends,
                      easings=easings, records=recs)

    @staticmethod
    def from_records(records: tuple[SVRecord, ...]) -> 'SVData':
        """Build from a fully-specified record list. Each record's easing
        and duration is preserved.

        end_multiplier resolution:
          step easing: forced to == multiplier (the segment is constant,
                       and the integrator's trapezoid integration with
                       v_start == v_end collapses to v * width).
          linear easing: missing end_multiplier defaults to the NEXT
                       segment's multiplier, so a chain of linear records
                       forms a continuous polyline. Last segment with
                       missing end_multiplier holds at its multiplier.
        """
        if not records:
            return SVData.from_sections([])
        n = len(records)
        times = np.fromiter((r.time for r in records), dtype=np.float64,
                            count=n)
        mults = np.fromiter((r.multiplier for r in records), dtype=np.float64,
                            count=n)
        durs = np.fromiter((r.duration for r in records), dtype=np.float64,
                            count=n)
        ends = np.empty(n, dtype=np.float64)
        for i, r in enumerate(records):
            if r.easing == EASING_STEP:
                ends[i] = float(r.multiplier)
            elif r.end_multiplier is not None:
                ends[i] = float(r.end_multiplier)
            elif i + 1 < n:
                ends[i] = float(records[i + 1].multiplier)
            else:
                ends[i] = float(r.multiplier)
        easings = tuple(r.easing for r in records)
        return SVData(times=times, multipliers=mults, durations=durs,
                      end_multipliers=ends, easings=easings,
                      records=tuple(records))

    @property
    def empty(self) -> bool:
        return self.times.size == 0

    def has_eased_segments(self) -> bool:
        """True if any segment uses non-step easing. Lets the integrator
        skip the per-segment easing dispatch in the common case."""
        return any(e != EASING_STEP for e in self.easings)


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
