"""Spatial curve algebra for note-position mods (prototype).

A mod's per-note geometric contribution is a CURVE: a vectorized
function of the note's signed scroll offset (`y_offset`, in the same
pixel domain the scroll model already produces), returning a value array
aligned with the notes. Per-frame scalars a curve needs -- song time,
beat, the note columns, the mod percent -- ride in a `Ctx` bundle so the
curve stays a pure `f(y_offset, ctx) -> ndarray`.

This mirrors `scheduler.Channel` (`curve(coord) -> value`, a `rest` for
the unset case): a Channel is a curve over a time-Clock coordinate; a mod
curve here is its spatial sibling over the y-offset coordinate. One curve
drives ONE scalar axis (x / y / z / rot_x / rot_y / rot_z / zoom); a
note's transform is assembled from the named axis-curves, matching our
engine's per-axis position/rotation surface.

Why closures over numpy rather than an object hierarchy or Python
generators: composition (`add`, `scale`, `chain`, `shift`) happens ONCE
at modstring-parse time, folding a stack of primitives into a single
closure; evaluation is then one vectorized numpy pass over every visible
note per frame. No per-note Python, no generator step cost.

Shape observed across the position mods (drunk / tornado / tipsy / bumpy
/ digital ... and the confusion tilts): an AFFINE PHASE in y_offset (plus
per-frame scalar terms), through a PERIODIC KERNEL (sin / cos / tan /
cosecant), scaled by an amplitude. A y-independent tilt (confusion) is
the degenerate zero-slope, identity-kernel case of the same shape. So a
mod is ported by naming its shape + parameters, not transcribing math.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Ctx:
    """Per-frame scalars a curve may read, shared across all notes in the
    batch. `cols` is the per-note column index array (aligned with the
    y_offset the curve is evaluated on); `note_beats` the per-note beat
    (aligned the same way, for beat-keyed mods like dizzy); `t` song time
    (s), `beat` the current song beat, `arrow_size` the column pitch.
    Extend as new mod families need more -- a curve reads only what it uses."""
    t: float = 0.0
    beat: float = 0.0
    cols: np.ndarray | None = None
    note_beats: np.ndarray | None = None
    arrow_size: float = 64.0


# A curve maps the note y_offset array (+ frame ctx) to a value array.
Curve = Callable[[np.ndarray, Ctx], np.ndarray]


# ---------------------------------------------------------------------------
# Primitives (each returns a Curve closure)
# ---------------------------------------------------------------------------

def const(value) -> Curve:
    """A curve flat in y_offset. `value` may be a scalar, a per-note array
    (a mod's per-column percents / a per-column shift, one per note), or a
    function of ctx (`lambda c: ...`) for a per-frame-but-y-independent
    term -- the confusion tilts are this: a whole-field angle broadcast to
    every note. Each broadcasts against the batch's y_offset shape."""
    if callable(value):
        return lambda y, c: np.broadcast_to(value(c), np.shape(y)).astype(np.float64)
    return lambda y, c: np.broadcast_to(value, np.shape(y)).astype(np.float64)


def affine_phase(y_coeff, terms=()) -> Curve:
    """The affine phase `y_coeff*y_offset + sum(terms)` used inside the
    periodic kernels. `y_coeff` is a scalar or `lambda c: ...`; each entry
    of `terms` is a scalar or a `lambda(y, c) -> array` (e.g. the per-
    column `col*freq` term, or a `t*(1+speed)` time term). Kept separate
    from the kernel so drunk/bumpy/tornado share one phase builder."""
    def curve(y, c):
        k = y_coeff(c) if callable(y_coeff) else y_coeff
        out = k * np.asarray(y, dtype=np.float64)
        for term in terms:
            out = out + (term(y, c) if callable(term) else term)
        return out
    return curve


def sine(phase: Curve) -> Curve:
    return lambda y, c: np.sin(phase(y, c))


def cosine(phase: Curve) -> Curve:
    return lambda y, c: np.cos(phase(y, c))


def tangent(phase: Curve, cosecant: bool = False) -> Curve:
    """SelectTanType: tan(phase), or 1/sin(phase) under the Cosecant flag.
    The sharp-kernel swap the tan* companions use in place of sin/cos."""
    if cosecant:
        return lambda y, c: 1.0 / np.sin(phase(y, c))
    return lambda y, c: np.tan(phase(y, c))


def power(exponent, phase: Curve) -> Curve:
    """Raise a phase to a fixed exponent -- the polynomial sibling of the
    trig kernels (a power of an affine phase instead of a trig function).
    parabola / attenuate are `power(2, affine_phase(...))`."""
    return lambda y, c: np.power(phase(y, c), exponent)


# ---------------------------------------------------------------------------
# Discontinuous kernels
#
# Non-smooth waveforms of a phase (the digital / zigzag / sawtooth / square
# family). Each swaps in for sin/cos the same way, but its output jumps.
# A jump means a phase error of even 1 ULP at an input landing ON a break
# flips the branch and shifts the output a WHOLE step -- so a discontinuous
# kernel must be fed `grouped_phase` (the engine's exact operation order),
# never the distributive `affine_phase` that is only sub-ULP-safe under a
# continuous kernel. `grouped_phase` drives the break-branch selection to
# agree bit-for-bit, taking the error to exactly zero.
# ---------------------------------------------------------------------------

def grouped_phase(outer, shift, div) -> Curve:
    """The affine phase `outer * (y_offset + shift) / div` kept in the
    engine's exact left-to-right multiply/add/divide grouping (NOT
    distributed like `affine_phase`). Required for the discontinuous
    kernels; see the section note. `outer`/`shift`/`div` are scalars."""
    return lambda y, c: outer * (np.asarray(y, dtype=np.float64) + shift) / div


def triangle(phase: Curve) -> Curve:
    """RageTriangle: a triangle wave in [-1, 1], period 2*PI over the
    phase. Rises 0->1 on u in [0, 0.5), falls 1->-1 on [0.5, 1.5), rises
    -1->0 on [1.5, 2), where u = (phase mod 2PI)/PI."""
    def curve(y, c):
        u = np.mod(phase(y, c), 2.0 * np.pi) / np.pi
        return np.where(u < 0.5, u * 2.0,
                        np.where(u < 1.5, 1.0 - (u - 0.5) * 2.0,
                                 -4.0 + u * 2.0))
    return curve


def square_wave(phase: Curve) -> Curve:
    """RageSquare: +1 over the first half of each 2*PI period, -1 over the
    second. The engine's `< 0.01` guard nudges a near-zero wrapped angle up
    by 2*PI (a hold-flicker hack flipping a small positive phase to -1),
    reproduced exactly."""
    def curve(y, c):
        a = np.mod(phase(y, c), 2.0 * np.pi)
        a = np.where(a < 0.01, a + 2.0 * np.pi, a)
        return np.where(a >= np.pi, -1.0, 1.0)
    return curve


def sawtooth_wave(phase: Curve) -> Curve:
    """Rising sawtooth in [0, 1): the fractional part `p - floor(p)` of the
    phase (a linear ramp for sawtooth, not an angle)."""
    return lambda y, c: (lambda p: p - np.floor(p))(phase(y, c))


def quantize(kernel: Curve, levels) -> Curve:
    """Snap a kernel's output to `levels` discrete steps:
    `round(levels * kernel) / levels`. `levels` is a scalar or
    `lambda c: ...`. The digital staircase over a sine kernel."""
    if callable(levels):
        return lambda y, c: np.round(levels(c) * kernel(y, c)) / levels(c)
    lv = float(levels)
    return lambda y, c: np.round(lv * kernel(y, c)) / lv


# ---------------------------------------------------------------------------
# Combinators (compose curves at build time)
# ---------------------------------------------------------------------------

def scale(k, curve: Curve) -> Curve:
    """Multiply a curve by `k`: a scalar, a per-note array (a mod's
    numbered per-column percents), or `lambda c: ...` (an amplitude read
    from ctx). numpy broadcasts each against the curve's value array."""
    if callable(k):
        return lambda y, c: k(c) * curve(y, c)
    return lambda y, c: k * curve(y, c)


def add(*curves: Curve) -> Curve:
    """Sum of curves on the same axis (how a note accumulates every mod
    that writes that axis)."""
    def curve(y, c):
        total = np.zeros(np.shape(y), dtype=np.float64)
        for f in curves:
            total = total + f(y, c)
        return total
    return curve


def chain(outer, inner: Curve) -> Curve:
    """Feed a curve's output as the input to `outer` (a plain numpy ufunc
    like np.abs, or another single-arg transform). `abs(cos(phase))` is
    `chain(np.abs, cosine(phase))`."""
    return lambda y, c: outer(inner(y, c))
