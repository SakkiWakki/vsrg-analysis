"""Append-only piecewise-closed-form timeline: the shared value substrate.

One scalar channel. A single writer appends time-ordered segments; any
number of readers sample any published time. Three properties carry the
whole design:

- Closed form: every segment evaluates at any t inside it from its own
  stored fields (no neighbor reads, no accumulated state), so a seek is
  one directory bisect, never a re-simulation.
- Always canonical: instant pokes pass through the same slope-corridor
  collapse `effects.timeline.simplify_instants` applies in batch, but
  maintained incrementally as writer state, so there is no post-pass and
  no cache to invalidate.
- Frontier publication: the writer bumps a single float after appending;
  readers clamp queries to it. Sealed rows are immutable, so concurrent
  reads during a live sweep need no locks.

Storage is columnar per kind (each kind keeps exactly its natural
fields) plus one merged time directory, so adding a kind later touches
nothing existing. Sampling holds a segment's end value until the next
segment starts, matching `EventTimeline`'s instant-hold semantics.

Costs: O(1) amortized per poke (no bake step), O(1) amortized per
monotone `sample` through a `Cursor`, O(log n) on a discontinuous jump.
"""
from __future__ import annotations

import math
from bisect import bisect_right

from analysis.player.render.effects.easing import ease
from analysis.player.render.effects.timeline import SIMPLIFY_EPS

KIND_HOLD = 0
KIND_RAMP = 1
KIND_OSC = 2
KIND_SLAB = 3

_EASE_LINEAR = 0


def _sin_cycles(u: float) -> float:
    return math.sin(u * 2.0 * math.pi)


def _cos_cycles(u: float) -> float:
    return math.cos(u * 2.0 * math.pi)


def _triangle_cycles(u: float) -> float:
    """Sine-aligned: 0 at whole cycles, +1 at quarter, -1 at three-quarter."""
    u = u - math.floor(u)
    if u < 0.25:
        return 4.0 * u
    if u < 0.75:
        return 2.0 - 4.0 * u
    return 4.0 * u - 4.0


def _square_cycles(u: float) -> float:
    return 1.0 if (u - math.floor(u)) < 0.5 else -1.0


OSC_SIN = 0
OSC_COS = 1
OSC_TRIANGLE = 2
OSC_SQUARE = 3

OSC_SHAPES = {
    OSC_SIN: _sin_cycles,
    OSC_COS: _cos_cycles,
    OSC_TRIANGLE: _triangle_cycles,
    OSC_SQUARE: _square_cycles,
}


class Cursor:
    """A reader's finger: the directory index of its last-hit segment.
    Each independent reader owns one; sharing a cursor across readers
    with unrelated access patterns forfeits the amortized bound but
    never affects correctness."""

    __slots__ = ('index',)

    def __init__(self):
        self.index = -1


class SegmentTimeline:
    """See module docstring. Writer methods (`poke`, `add_ramp`,
    `add_osc`, `add_slab`, `publish`, `finish`) belong to the single
    writer; `sample` belongs to readers."""

    def __init__(self, rest: float = 0.0, eps: float = SIMPLIFY_EPS):
        self._rest = float(rest)
        self._eps = float(eps)

        # Merged directory: segment starts in append order (monotone).
        self._dir_t0: list[float] = []
        self._dir_kind: list[int] = []
        self._dir_row: list[int] = []

        self._hold_v: list[float] = []

        self._ramp_t1: list[float] = []
        self._ramp_v0: list[float] = []
        self._ramp_v1: list[float] = []
        self._ramp_ease: list[int] = []

        self._osc_t1: list[float] = []
        self._osc_base: list[float] = []
        self._osc_mag: list[float] = []
        self._osc_period: list[float] = []
        self._osc_phase: list[float] = []
        self._osc_shape: list[int] = []

        self._slab_t1: list[float] = []
        self._slab_hz: list[float] = []
        self._slab_off: list[int] = []
        self._slab_n: list[int] = []
        self._slab_pool: list[float] = []

        # Open poke run (the incremental slope corridor): head point,
        # last point, accepted-point count, feasible slope interval.
        self._run_th = 0.0
        self._run_vh = 0.0
        self._run_tl = 0.0
        self._run_vl = 0.0
        self._run_n = 0
        self._run_lo = 0.0
        self._run_hi = 0.0

        self.frontier = float('-inf')

    # -- writer ----------------------------------------------------------

    def poke(self, t: float, v: float) -> None:
        """Record an instant value. Runs of pokes whose chord from the
        run head reproduces every interior point within `eps` collapse
        to one segment, exactly as `simplify_instants` collapses them
        in batch; the corridor is maintained incrementally so each poke
        is O(1) with no rescan."""
        if self._run_n == 0:
            self._start_run(t, v)
            return

        dt = t - self._run_th
        if dt <= 0.0:
            self._seal_run()
            self._start_run(t, v)
            return

        slope = (v - self._run_vh) / dt
        if not self._run_lo <= slope <= self._run_hi:
            self._seal_run()
            self._start_run(t, v)
            return

        tol = self._eps / dt
        self._run_lo = max(self._run_lo, slope - tol)
        self._run_hi = min(self._run_hi, slope + tol)
        self._run_tl = t
        self._run_vl = v
        self._run_n += 1

    def add_ramp(self, t0: float, t1: float, v0: float, v1: float,
                 ease_id: int = _EASE_LINEAR) -> None:
        """A structural tween: eases v0 -> v1 over [t0, t1], holds v1
        after. Kept verbatim, never merged."""
        self._seal_run()
        self._push_dir(t0, KIND_RAMP, len(self._ramp_t1))
        self._ramp_t1.append(float(t1))
        self._ramp_v0.append(float(v0))
        self._ramp_v1.append(float(v1))
        self._ramp_ease.append(int(ease_id))

    def add_hold(self, t: float, v: float) -> None:
        """A structural instant: recorded verbatim, never merged into a
        poke run (the batch pipeline's non-plain keyframes - ease-from
        pins, multi-component values - keep this semantics)."""
        self._seal_run()
        self._push_hold(float(t), float(v))

    def add_osc(self, t0: float, t1: float, base: float, mag: float,
                period: float, phase: float = 0.0,
                shape_id: int = OSC_SIN) -> None:
        """An analytic oscillator span: base + mag * shape(cycles) with
        cycles = (t - t0) / period + phase. Holds `base` after t1."""
        self._seal_run()
        self._push_dir(t0, KIND_OSC, len(self._osc_t1))
        self._osc_t1.append(float(t1))
        self._osc_base.append(float(base))
        self._osc_mag.append(float(mag))
        self._osc_period.append(float(period))
        self._osc_phase.append(float(phase))
        self._osc_shape.append(int(shape_id))

    def add_slab(self, t0: float, hz: float, samples) -> None:
        """A dense uniformly-sampled block (the residue floor for
        content no analytic kind models). Linearly interpolates between
        samples, holds the last sample after the block."""
        if not samples:
            return
        self._seal_run()
        self._push_dir(t0, KIND_SLAB, len(self._slab_t1))
        self._slab_t1.append(t0 + (len(samples) - 1) / float(hz))
        self._slab_hz.append(float(hz))
        self._slab_off.append(len(self._slab_pool))
        self._slab_n.append(len(samples))
        self._slab_pool.extend(float(s) for s in samples)

    def publish(self, t: float) -> None:
        """Make everything up to `t` readable. The open poke run stays
        open (a later poke may extend it); readers inside it get the
        run's chord, which the corridor invariant bounds within eps of
        any value the sealed segment will later produce there."""
        self.frontier = float(t)

    def finish(self) -> None:
        """Seal the open run and lift the frontier: the timeline is
        complete and fully readable."""
        self._seal_run()
        self.frontier = float('inf')

    def _start_run(self, t: float, v: float) -> None:
        self._run_th = self._run_tl = float(t)
        self._run_vh = self._run_vl = float(v)
        self._run_n = 1
        self._run_lo = float('-inf')
        self._run_hi = float('inf')

    def _seal_run(self) -> None:
        n = self._run_n
        if n == 0:
            return
        self._run_n = 0

        # Mirrors `_collapse_run`: short runs stay instants (holds), a
        # near-constant run keeps its head alone (the transition stays
        # at the run START), a sloped run becomes one linear ramp.
        if n <= 2 or abs(self._run_vl - self._run_vh) <= self._eps:
            self._push_hold(self._run_th, self._run_vh)
            if n == 2:
                self._push_hold(self._run_tl, self._run_vl)
            return
        self._push_dir(self._run_th, KIND_RAMP, len(self._ramp_t1))
        self._ramp_t1.append(self._run_tl)
        self._ramp_v0.append(self._run_vh)
        self._ramp_v1.append(self._run_vl)
        self._ramp_ease.append(_EASE_LINEAR)

    def _push_hold(self, t: float, v: float) -> None:
        self._push_dir(t, KIND_HOLD, len(self._hold_v))
        self._hold_v.append(v)

    def _push_dir(self, t0: float, kind: int, row: int) -> None:
        t0 = float(t0)
        assert not self._dir_t0 or t0 >= self._dir_t0[-1], \
            'segment starts must be appended in time order'
        self._dir_t0.append(t0)
        self._dir_kind.append(kind)
        self._dir_row.append(row)

    # -- readers ---------------------------------------------------------

    def sample(self, t: float, cursor: Cursor | None = None) -> float:
        """Value at `t` (clamped to the frontier): before any content
        returns `rest`; inside a segment evaluates it; between segments
        holds the previous segment's end value."""
        t = float(t)
        if t > self.frontier:
            t = self.frontier

        n = len(self._dir_t0)
        if self._run_n and t >= self._run_th:
            return self._run_value(t)
        idx = self._locate(t, n, cursor)
        if idx < 0:
            return self._rest
        return self._segment_value(idx, t)

    def _locate(self, t: float, n: int, cursor: Cursor | None) -> int:
        starts = self._dir_t0
        if cursor is None:
            return bisect_right(starts, t, 0, n) - 1

        idx = cursor.index
        if idx < 0 or idx >= n or starts[idx] > t:
            idx = bisect_right(starts, t, 0, n) - 1
        else:
            while idx + 1 < n and starts[idx + 1] <= t:
                idx += 1
        cursor.index = idx
        return idx

    def _run_value(self, t: float) -> float:
        # A 2-point run has no corridor guarantee (the second point is
        # accepted unconditionally), and it seals as HOLDs, not a ramp:
        # provisional reads must step, never bridge the gap. From the
        # third accepted point on, the chord IS the eventual sealed
        # segment, so interpolating is exact.
        if self._run_n <= 2:
            return self._run_vl if t >= self._run_tl else self._run_vh
        if t >= self._run_tl:
            return self._run_vl
        f = (t - self._run_th) / (self._run_tl - self._run_th)
        return self._run_vh + (self._run_vl - self._run_vh) * f

    def _segment_value(self, idx: int, t: float) -> float:
        t0 = self._dir_t0[idx]
        row = self._dir_row[idx]
        match self._dir_kind[idx]:
            case 0:
                return self._hold_v[row]
            case 1:
                return self._ramp_value(row, t0, t)
            case 2:
                return self._osc_value(row, t0, t)
            case _:
                return self._slab_value(row, t0, t)

    def _ramp_value(self, row: int, t0: float, t: float) -> float:
        t1 = self._ramp_t1[row]
        v1 = self._ramp_v1[row]
        if t >= t1 or t1 <= t0:
            return v1
        v0 = self._ramp_v0[row]
        f = ease(self._ramp_ease[row], (t - t0) / (t1 - t0))
        return v0 + (v1 - v0) * f

    def _osc_value(self, row: int, t0: float, t: float) -> float:
        base = self._osc_base[row]
        if t >= self._osc_t1[row]:
            return base
        cycles = (t - t0) / self._osc_period[row] + self._osc_phase[row]
        return base + self._osc_mag[row] * OSC_SHAPES[self._osc_shape[row]](cycles)

    def _slab_value(self, row: int, t0: float, t: float) -> float:
        off = self._slab_off[row]
        last = self._slab_n[row] - 1
        if t >= self._slab_t1[row]:
            return self._slab_pool[off + last]
        u = (t - t0) * self._slab_hz[row]
        i = min(int(u), last - 1) if last > 0 else 0
        f = u - i
        lo = self._slab_pool[off + i]
        hi = self._slab_pool[off + min(i + 1, last)]
        return lo + (hi - lo) * f

    def __len__(self) -> int:
        return len(self._dir_t0)
