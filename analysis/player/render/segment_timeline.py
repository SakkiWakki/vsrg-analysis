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

# Step for tracing an interrupted CURVED ease, whose duration can no longer
# stand in for its parameter (`_trace`). One display frame: finer than the
# thing being reconstructed is sampled at.
_EASE_TRACE_DT = 1.0 / 60.0


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

    def __init__(self, rest: float = 0.0, eps: float = SIMPLIFY_EPS,
                 step: bool = False):
        self._rest = float(rest)
        self._eps = float(eps)
        # A STEP lane carries a discrete value (a visibility bit, a sprite
        # frame index) that only ever jumps. Corridor collapse approximates a
        # run of instants with a linear ramp, which is sound for a continuous
        # signal and destroys a discrete one: a 0-then-1 pair becomes a ramp
        # bridging the gap, so every sample between reads FRACTIONAL. A step
        # lane holds each poke instead, so its value is always one it was
        # actually written with.
        self._step = bool(step)

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
        if self._step:
            self.add_hold(t, v)
            return

        if self._run_n == 0:
            self._start_run(t, v)
            return

        # A poke not strictly after the LAST accepted point is a
        # zero-tween chain step (structural): seal and restart, exactly
        # as `simplify_instants` breaks its run there. Guarding only
        # against the run head would let a duplicate write at the tail
        # count as a third sample and upgrade a two-point step pair into
        # a ramp bridging the whole gap.
        if t <= self._run_tl:
            self._seal_run()
            self._start_run(t, v)
            return

        dt = t - self._run_th
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

    # An OSC span has no exact breakpoint form (the consumer ramps
    # LINEARLY between breakpoints), so it is the one kind the export
    # approximates - at a cadence fine enough that a design-space
    # oscillation reads smooth.
    OSC_EXPORT_DT = 1.0 / 60.0

    def breakpoints(self, t1: float, osc_dt: float = OSC_EXPORT_DT):
        """This timeline as `(ts, vals, durs, eases)` for a piecewise
        linear-ramp consumer, covering every segment starting at or before
        `t1`, or None when the export cannot be trusted (see below).

        The consumer model (`storyboard_native` channels, `SegmentTimeline`'s
        own lowered form): before `ts[0]` it serves the channel's rest; at
        breakpoint `i` it holds `vals[i]` when `durs[i] <= 0`, else eases
        `vals[i] -> vals[i+1]` over `durs[i]` under ease id `eases[i]`.
        Every kind but OSC maps onto that exactly - HOLD is one breakpoint,
        RAMP is its two endpoints carrying its own ease, SLAB is its sample
        grid (whose linear interpolation IS the consumer's) - so an exported
        channel replays `sample` rather than approximating it, at the
        timeline's own resolution instead of a fixed sampling cadence.

        An OPEN poke run is the last segment: `sample` gives it priority from
        its head time, so the sealed directory is cut off there and the run's
        own breakpoints follow. This used to return None instead - a flat
        time-ordered list could not express the priority while segments
        exported their full spans - and the caller fell back to DENSE
        SAMPLING, which cannot follow a lane the chart repokes every 20ms:
        Corrupted's x came out 176px off that way.
        """
        ts: list[float] = []
        vals: list[float] = []
        durs: list[float] = []
        eases: list[int] = []
        # The open run shadows the directory from its head time on.
        sealed_end = self._run_th if self._run_n else math.inf

        def emit(t, v, dur=0.0, ease_id=_EASE_LINEAR):
            ts.append(float(t))
            vals.append(float(v))
            durs.append(float(dur))
            eases.append(int(ease_id))

        for idx, t0 in enumerate(self._dir_t0):
            if t0 > t1 or t0 >= sealed_end:
                break
            # A segment ends where the NEXT one begins, whatever span its own
            # row claims: `sample` picks by bisect over `_dir_t0`, so a later
            # segment takes over from its own start and truncates whatever was
            # still ramping. Exporting the full span instead pushed
            # breakpoints PAST the interrupting segment and the array came out
            # non-monotonic - 1979 of Bonfire's 5447 alpha breakpoints went
            # backwards, and the consumer binary-searches, so it read
            # arbitrary values (0.0009 where the curve says 0.9927).
            cutoff = min(self._dir_t0[idx + 1]
                         if idx + 1 < len(self._dir_t0) else math.inf,
                         sealed_end)
            clipped = self._truncating(idx, cutoff, emit)
            self._segment_breakpoints(idx, t0, clipped, osc_dt)
            clipped.finish()

        if self._run_n:
            self._run_breakpoints(emit)
        return ts, vals, durs, eases

    def _truncating(self, idx: int, cutoff: float, emit):
        """`emit` clipped to `[.., cutoff)` for segment `idx`.

        A breakpoint at or past `cutoff` belongs to a span the next segment
        already took over, so it is dropped; the one before it is shortened to
        end exactly at `cutoff` and followed by the segment's own value THERE,
        read from `_segment_value` rather than re-derived per kind. That last
        breakpoint shares its time with the next segment's first, and the
        consumer's bisect takes the later one - so the interrupted span
        interpolates correctly and the handover still lands on the new
        segment."""
        state = {'last': None, 'cut': False}

        def clipped(t, v, dur=0.0, ease_id=_EASE_LINEAR):
            t = float(t)
            if t >= cutoff:
                state['cut'] = True
                return
            if dur and t + dur > cutoff:
                state['cut'] = True
                if ease_id != _EASE_LINEAR:
                    # Shortening the DURATION of an eased span would replay
                    # the whole ease curve over the surviving part, so the
                    # value is only right at the two ends and wrong through
                    # the middle - 13% low across Corrupted's alpha. Trace the
                    # survivor instead, at the resolution a curved ease needs.
                    state['last'] = self._trace(idx, t, cutoff, emit)
                    return
                dur = cutoff - t
            state['last'] = t
            emit(t, v, dur, ease_id)

        def finish():
            if state['cut'] and state['last'] is not None:
                emit(cutoff, self._segment_value(idx, cutoff))

        clipped.finish = finish
        return clipped

    def _trace(self, idx: int, a: float, b: float, emit) -> float:
        """Emit `[a, b)` of segment `idx` as linear steps following its own
        value, and return the last breakpoint's time.

        For a span whose shape the consumer's single linear ramp cannot
        carry - a curved ease that got interrupted, so its duration can no
        longer stand in for its parameter."""
        steps = max(1, int(math.ceil((b - a) / _EASE_TRACE_DT)))
        step = (b - a) / steps
        last = a
        for k in range(steps):
            last = a + k * step
            emit(last, self._segment_value(idx, last), step)
        return last

    def _segment_breakpoints(self, idx: int, t0: float, emit, osc_dt) -> None:
        """Append one segment's breakpoints through `emit` (see
        `breakpoints`); mirrors `_segment_value` kind for kind."""
        row = self._dir_row[idx]
        match self._dir_kind[idx]:
            case 0:
                emit(t0, self._hold_v[row])
            case 1:
                self._ramp_breakpoints(row, t0, emit)
            case 2:
                self._osc_breakpoints(row, t0, emit, osc_dt)
            case _:
                self._slab_breakpoints(row, t0, emit)

    def _ramp_breakpoints(self, row: int, t0: float, emit) -> None:
        t1, v0, v1 = self._ramp_t1[row], self._ramp_v0[row], self._ramp_v1[row]
        if t1 <= t0:
            emit(t0, v1)
            return
        emit(t0, v0, t1 - t0, self._ramp_ease[row])
        emit(t1, v1)

    def _osc_breakpoints(self, row: int, t0: float, emit, osc_dt) -> None:
        """The one approximated kind: trace the oscillator across its span
        as linear ramps, then hold its base (what `_osc_value` serves past
        `t1`)."""
        t1 = self._osc_t1[row]
        steps = max(1, int(math.ceil((t1 - t0) / osc_dt)))
        step = (t1 - t0) / steps
        for k in range(steps):
            emit(t0 + k * step, self._osc_value(row, t0, t0 + k * step), step)
        emit(t1, self._osc_base[row])

    def _slab_breakpoints(self, row: int, t0: float, emit) -> None:
        off, n = self._slab_off[row], self._slab_n[row]
        step = 1.0 / self._slab_hz[row]
        for i in range(n - 1):
            emit(t0 + i * step, self._slab_pool[off + i], step)
        emit(self._slab_t1[row], self._slab_pool[off + n - 1])

    def _run_breakpoints(self, emit) -> None:
        """The open poke run's breakpoints, mirroring `_run_value`: a run of
        one or two points steps, a longer one is the chord it will seal as."""
        if self._run_n <= 2:
            emit(self._run_th, self._run_vh)
            if self._run_tl > self._run_th:
                emit(self._run_tl, self._run_vl)
            return
        emit(self._run_th, self._run_vh, self._run_tl - self._run_th)
        emit(self._run_tl, self._run_vl)

    def __len__(self) -> int:
        return len(self._dir_t0)
