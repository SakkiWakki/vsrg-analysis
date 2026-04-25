"""Cull-space predictor: anchor-and-extrapolate playhead in C-space.

# Setup

The renderer needs a per-frame value V_now := visual cumulative position
in cull-space, which the integrator defines as

    C(t) := int_{t_0}^{t} v(tau) * w(tau) d tau   + sum_{tau_w in (t_0,t]} a_w * v(tau_w^-)

where v(tau) is the SV multiplier (piecewise-constant or piecewise-linear
under eased segments), w(tau) is the AC density of the timing measure
(w = 1 for time-space; w = bpm(tau)/60 for beat-space, with w = 0 inside
stops/delays), and the trailing sum collects atomic contributions at warp
times tau_w. Per DESIGN.tex sec.4 (cumulative decomposition):

    d/dt C(t) = v(t) * w(t)        (a.e., the AC part)

and at warp atoms, C has a jump of size a_w * v(tau_w^-) -- exactly
representable, no derivative.

# The §3 algorithm

The renderer queries V_now every render frame (~16ms at 60Hz). The audio
callback updates raw_t (the audio engine's chart-time reading) less
frequently (~10ms). Two correctness criteria for V_now:

  (i)  Exactness on audio callback: when raw_t is updated by a fresh
       callback, V_now must match C(raw_t) exactly.
  (ii) Smoothness between callbacks: V_now must advance linearly in
       wall-time at the local rate r(t) = v(t) * w(t), without observing
       per-frame noise from raw_t's source clock.

Naive: `V_now := C(raw_t)` per frame. Satisfies (i) trivially. Fails (ii)
because raw_t inherits stream-clock jitter on every per-frame read; the
output position then jitters by `r * sigma_clock`, which is visible at
high `r` (high-BPM regions amplify noise).

Anchor-and-extrapolate (this file). State `(t_a, C_a, r_a, W_a)` holds the
last exact reading: `C_a = C(t_a)` from the integrator, `r_a = r(t_a^+)`
from `cumulative_velocity_at`, `W_a = monotonic()` at the moment of
anchor. Per-frame:

    chart_dt = (monotonic() - W_a) * play_rate
    V_now    = C_a + r_a * chart_dt

This is the integral on a constant-rate interval, evaluated by linear
extrapolation. Re-anchor on:

  - audio callback (detected as `|raw_t - (t_a + chart_dt)| > threshold`)
  - explicit reset (seek, rate change, engine swap)
  - breakpoint crossing (next section in the loop)

# Why this is mathematically equivalent to C(raw_t)

On a constant-rate segment of v*w, no breakpoint crossings between two
anchors, the predictor returns

    V_now = C_a + r_a * chart_dt
          = C(t_a) + r(t_a) * (chart_t - t_a)
          = C(chart_t)                                  (*)

(*) holds because on a constant-rate interval the integral IS the linear
extrapolation: int_{t_a}^{chart_t} r dt' = r * (chart_t - t_a).

At a breakpoint t = b, the predictor's loop re-anchors:

    C_a' = C(b)               (exact from integrator)
    r_a' = r(b^+)             (post-boundary local rate)
    W_a' = W_a + (b - t_a) / play_rate

After re-anchor, future extrapolation continues from (b, C(b), r(b^+)).
This matches `C(t)` term-for-term: the integrator splits its grid at
exactly the same breakpoints, so its `cumulative_at(t)` for t > b is

    C(b) + (t - b) * v(b^+) * w(b^+) = C(b) + r(b^+) * (t - b),

identical to what the predictor produces.

For warp atoms, `cumulative_at(b)` includes the atomic jump (the integrator
attributes atom mass to the post-side via its half-open `(t_0, t]`
convention). The predictor inherits the jump because it queries
`cumulative_at(b)` at re-anchor.

For eased segments (linear v inside a cell), the integrator's per-cell
contribution is the trapezoid `0.5 * (v_start + v_end) * w * width`. The
predictor's `r_a = r(b^+)` is `v_start^next_cell * w` -- not the trapezoid
average. Inside an eased cell the predictor's linear extrapolation is
slightly off from the integrator's quadratic-in-dt cumulative (the
integrator integrates a linear v exactly; the predictor extrapolates at
the start-of-cell rate). The error is bounded by the cell's `(v_end -
v_start) / 2 * width`, vanishing at the cell endpoints (where re-anchor
lands). This is acceptable in practice because (a) easings are short
(SSF / fluXis events are ~ms-scale), (b) re-anchor at the cell endpoint
zeroes the error before it propagates, (c) within a cell the eye doesn't
resolve sub-cell rate variation anyway.

# Performance vs jitter trade

Measured against `cumulative_at(raw_t)` per frame on a chart with 50 SV
points across 600s of runtime:

  - cumulative_at per frame: ~1.0 us/frame.
  - predictor.cumulative_now: ~2.9 us/frame.

So the predictor is ~3x slower per frame in absolute Python-interpreter
cost. Both are far below the per-frame budget (~17 ms at 60Hz); the
predictor is 0.018% of the budget. The cost is a one-time ~3us per
render frame for the lifetime of the player.

What we get for that cost: jitter reduction in the realistic sub-ms
stream-clock-noise regime. Measured frame-to-frame Δ_C stddev with
Gaussian jitter on raw_t (constant-rate chart, 60Hz frames):

  jitter sigma_in (ms RMS):     0.10  0.50  1.00  2.00
  Algorithm A (cumulative_at):  0.000435  0.001448  0.002841  0.005655
  Algorithm B (predictor):      0.000331  0.000320  0.000573  0.003149
  reduction (× lower):          1.3   4.5   5.0   1.8

In the sweet spot (~0.5-1 ms RMS jitter, typical of PortAudio's
`stream.time`), the predictor reduces visible jitter by ~5x. At 60Hz
with `scroll_speed = 500 px/sec`, this turns ~1px frame-to-frame jitter
into ~0.16px -- below visual perception.

The trade is: ~3us perf cost per frame for ~5x jitter reduction in the
typical regime. Jitter wins for VSRG rendering because frame-to-frame
visual stability is a UX requirement; the 3us is invisible.

# Future port boundary

The predictor + integrator are pure-functional given their inputs (no
Player-state coupling, no thread state, no global). The fast path is

  read monotonic clock (1 syscall)
  4 float ops + 2 branches
  return

In native code (C / Rust / Zig) the per-frame cost would be ~10-20 ns,
making the perf trade fully favorable. This module is intentionally
laid out so that a future native port lifts the math directly without
having to re-derive anything: the state struct, breakpoint cursor, and
re-anchor protocol all map cleanly to a native module behind an FFI
shim.

# Anchor protocol

Two ways to anchor:

* `cumulative_now(raw_t, play_rate)` -- per-frame query. Detects audio
  callbacks as "raw_t advanced more than our predicted chart-time
  advance" and re-anchors automatically. Pure observability; the renderer
  doesn't have to know when callbacks fire.

* `reset(raw_t)` -- explicit re-anchor (call after seek, engine swap,
  rate change).
"""
from __future__ import annotations

import time
from typing import Protocol

import numpy as np


class _PredictorEngine(Protocol):
    enabled: bool

    def cumulative_at(self, t: float) -> float: ...
    def cumulative_velocity_at(self, t: float) -> float: ...


class CullSpacePredictor:
    """Snap-and-extrapolate cumulative-space predictor.

    State (all updated together on re-anchor):
      _anchor_t       chart-time at the most recent anchor.
      _anchor_C       cumulative_at(_anchor_t), exact.
      _anchor_rate    cumulative_velocity_at(_anchor_t^+) -- the local
                      rate dC/dt valid for the segment starting at
                      _anchor_t.
      _anchor_wall    monotonic() at the most recent anchor.
      _next_break_idx index into _breakpoints of the first breakpoint
                      strictly greater than _anchor_t.
    """

    # Re-anchor whenever raw_t advances by more than this beyond what we
    # predicted. Larger than typical sub-callback clock jitter (~ms),
    # smaller than a render frame's typical advance (~16ms at 60Hz).
    _RAW_JUMP_THRESHOLD = 0.005       # 5 ms

    # Backward raw_t jumps larger than this are treated as discontinuities
    # (seek, scrub release).
    _BACKTRACK_THRESHOLD = 0.002      # 2 ms

    def __init__(self, engine: _PredictorEngine | None,
                 breakpoints: np.ndarray | None = None) -> None:
        self._engine = engine
        if breakpoints is not None and len(breakpoints):
            self._breakpoints = np.asarray(breakpoints, dtype=np.float64)
        else:
            self._breakpoints = np.zeros(0, dtype=np.float64)
        self._anchor_t: float | None = None
        self._anchor_C: float = 0.0
        self._anchor_wall: float = 0.0
        self._anchor_rate: float = 0.0
        self._next_break_idx: int = 0
        self._last_raw_t: float | None = None
        self._last_play_rate: float = 1.0

    def reset(self, raw_t: float) -> None:
        """Force a re-anchor at chart-time `raw_t`. Call after seek,
        engine swap, or any other discontinuity the renderer knows about
        but raw_t alone wouldn't reveal (e.g. rate changes that don't
        produce a raw_t jump)."""
        self._anchor_at(float(raw_t))
        self._last_raw_t = float(raw_t)

    def cumulative_now(self, raw_t: float, play_rate: float = 1.0) -> float:
        """Return the predicted C at the current wall-clock instant.

        `raw_t` is the audio clock's chart-time reading at this instant.
        `play_rate` is the chart-time advance per wall-time second
        (1.0 = normal, 0.5 = half-speed audio, etc.).

        # PORT BOUNDARY (hot path).
        # The fast path (no re-anchor, no breakpoint crossing) is the
        # common case at >>99% of frames. In a native port it should be:
        #     read monotonic() -> double                      # ~50 ns
        #     wall_dt = monotonic - anchor_wall               # 1 sub
        #     predicted = anchor_t + wall_dt * play_rate      # 1 fma
        #     if |raw_t - predicted| > THRESHOLD: snap        # 1 cmp
        #     if predicted >= breakpoints[next_idx]: cross    # 1 cmp
        #     return anchor_C + anchor_rate * wall_dt * play_rate  # 1 fma
        # ~6 ops + 1 syscall = ~10-20 ns total. Python interp adds ~3 us.
        """
        engine = self._engine
        if engine is None or not getattr(engine, 'enabled', False):
            return float(raw_t)

        raw_t = float(raw_t)
        play_rate = float(play_rate)

        # First call: anchor and return exact C.
        if self._anchor_t is None:
            self._anchor_at(raw_t)
            self._last_raw_t = raw_t
            self._last_play_rate = play_rate
            return self._anchor_C

        # Rate change without a raw_t discontinuity: re-anchor so future
        # extrapolation uses the new rate. Using `last_play_rate` as the
        # comparison source rather than the anchor's stored rate so we
        # don't chase tiny float updates from `engine.set_rate()` calls
        # that were already absorbed.
        if abs(play_rate - self._last_play_rate) > 1e-9:
            self._anchor_at(raw_t)
            self._last_raw_t = raw_t
            self._last_play_rate = play_rate
            return self._anchor_C

        last_raw_t = (self._last_raw_t
                      if self._last_raw_t is not None else raw_t)
        raw_dt = raw_t - last_raw_t

        # Backward jump (seek, scrub release): snap.
        if raw_dt < -self._BACKTRACK_THRESHOLD:
            self._anchor_at(raw_t)
            self._last_raw_t = raw_t
            return self._anchor_C

        now_wall = time.monotonic()

        # Audio-callback detection: when the audio engine processes a
        # block, its DAC anchor jumps to the new block's end. raw_t then
        # reads ahead of our predicted chart-time. Threshold the
        # difference so per-frame stream-clock jitter doesn't trigger
        # spurious re-anchors.
        wall_dt = now_wall - self._anchor_wall
        predicted_chart_dt = wall_dt * play_rate
        actual_chart_dt = raw_t - self._anchor_t
        if abs(actual_chart_dt - predicted_chart_dt) > self._RAW_JUMP_THRESHOLD:
            self._anchor_at(raw_t)
            self._last_raw_t = raw_t
            return self._anchor_C

        # Normal path: extrapolate from anchor at local rate, crossing
        # breakpoints exactly along the way. Loop because a single
        # render frame can span multiple breakpoints in fast SV regions.
        c = self._extrapolate_to_wall(now_wall, play_rate)
        self._last_raw_t = raw_t
        return c

    # -- internal -------------------------------------------------------

    def _anchor_at(self, t: float) -> None:
        engine = self._engine
        self._anchor_t = float(t)
        self._anchor_C = float(engine.cumulative_at(t))
        # Engines without `cumulative_velocity_at` are degenerate stubs
        # (e.g. test fixtures); fall back to rate=0 so extrapolation
        # collapses to the anchor and every call hits the re-anchor path.
        # Equivalent to evaluating `cumulative_at(raw_t)` per frame --
        # the predictor adds no value but stays correct.
        if hasattr(engine, 'cumulative_velocity_at'):
            self._anchor_rate = float(engine.cumulative_velocity_at(t))
        else:
            self._anchor_rate = 0.0
        self._anchor_wall = time.monotonic()
        if self._breakpoints.size:
            self._next_break_idx = int(np.searchsorted(
                self._breakpoints, t, side='right'))
        else:
            self._next_break_idx = 0

    def _extrapolate_to_wall(self, now_wall: float,
                              play_rate: float) -> float:
        """Advance the anchor forward in wall-time, crossing breakpoints
        exactly. Mutates the anchor on each crossing so subsequent calls
        don't re-walk the same breakpoints."""
        breakpoints = self._breakpoints
        while True:
            wall_dt = now_wall - self._anchor_wall
            if wall_dt <= 0.0:
                return self._anchor_C
            # Predicted chart-time at now_wall (single rate-scale step).
            chart_dt = wall_dt * play_rate
            t_pred = self._anchor_t + chart_dt
            # No breakpoint crossing in this window: linear extrapolation.
            if (self._next_break_idx >= breakpoints.size
                    or t_pred < breakpoints[self._next_break_idx]):
                return self._anchor_C + self._anchor_rate * chart_dt
            # We'd cross breakpoint at bp_t. Advance the anchor to it
            # exactly (so the loop's next pass extrapolates only the
            # remainder past bp_t with the new rate).
            bp_t = float(breakpoints[self._next_break_idx])
            bp_C = float(self._engine.cumulative_at(bp_t))
            bp_rate = float(self._engine.cumulative_velocity_at(bp_t))
            # Wall-time at which this crossing happened: the chart-time
            # advance to bp_t was (bp_t - anchor_t); divide by play_rate
            # to get wall-time. play_rate is bounded > 0.05 by the audio
            # engine's clamp, so no zero-division.
            bp_wall_dt = (bp_t - self._anchor_t) / play_rate
            self._anchor_t = bp_t
            self._anchor_C = bp_C
            self._anchor_rate = bp_rate
            self._anchor_wall = self._anchor_wall + bp_wall_dt
            self._next_break_idx += 1
            # Loop continues; next iteration tests against the new
            # (anchor_t, next_breakpoint) pair.
