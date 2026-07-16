"""osu.Framework easing curves, indexed by that framework's `Easing`
enum ordering (fluXis serializes easing ids from it, e.g. 13 =
OutQuint, the lane-switch default).

Only the polynomial / sine / expo / circ families are implemented;
the exotic tail (elastic, back, bounce, pow10) falls back to OutQuint,
which is visually close for the sub-second transforms they decorate.
"""
from __future__ import annotations

import math


def _in_pow(p):
    return lambda u: u ** p


def _out_pow(p):
    return lambda u: 1.0 - (1.0 - u) ** p


def _in_out_pow(p):
    def f(u):
        if u < 0.5:
            return (2.0 * u) ** p / 2.0
        return 1.0 - (2.0 * (1.0 - u)) ** p / 2.0
    return f


def _in_sine(u):
    return 1.0 - math.cos(u * math.pi / 2.0)


def _out_sine(u):
    return math.sin(u * math.pi / 2.0)


def _in_out_sine(u):
    return (1.0 - math.cos(u * math.pi)) / 2.0


def _in_expo(u):
    return 0.0 if u <= 0.0 else 2.0 ** (10.0 * (u - 1.0))


def _out_expo(u):
    return 1.0 if u >= 1.0 else 1.0 - 2.0 ** (-10.0 * u)


def _in_out_expo(u):
    if u < 0.5:
        return _in_expo(2.0 * u) / 2.0
    return 0.5 + _out_expo(2.0 * u - 1.0) / 2.0


def _in_circ(u):
    return 1.0 - math.sqrt(1.0 - u * u)


def _out_circ(u):
    return math.sqrt(1.0 - (1.0 - u) ** 2)


def _in_out_circ(u):
    if u < 0.5:
        return _in_circ(2.0 * u) / 2.0
    return 0.5 + _out_circ(2.0 * u - 1.0) / 2.0


# Index-aligned with osu.Framework's Easing enum.
_CURVES = (
    lambda u: u,                                   # 0 None
    _out_pow(2), _in_pow(2),                       # 1 Out, 2 In (quad)
    _in_pow(2), _out_pow(2), _in_out_pow(2),       # 3..5 Quad
    _in_pow(3), _out_pow(3), _in_out_pow(3),       # 6..8 Cubic
    _in_pow(4), _out_pow(4), _in_out_pow(4),       # 9..11 Quart
    _in_pow(5), _out_pow(5), _in_out_pow(5),       # 12..14 Quint
    _in_sine, _out_sine, _in_out_sine,             # 15..17 Sine
    _in_expo, _out_expo, _in_out_expo,             # 18..20 Expo
    _in_circ, _out_circ, _in_out_circ,             # 21..23 Circ
)

_FALLBACK = _out_pow(5)   # OutQuint


def ease(kind: int, u: float) -> float:
    """Eased progress for raw progress `u` in [0, 1]."""
    u = max(0.0, min(1.0, float(u)))
    if 0 <= kind < len(_CURVES):
        return _CURVES[kind](u)
    return _FALLBACK(u)
