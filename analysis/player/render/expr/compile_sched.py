"""Compile backend: lower a guard AST to a scheduler `Channel`.

The tree-walk evaluator re-queries the surface per node - fine for tests,
too slow for a per-tick integration loop. This backend lowers the AST ONCE
into a nested closure that closes over the surface's clock readers, so
evaluating the guard at a song time is a plain Python call: no AST walk, no
per-node surface dispatch, no interpreter re-entry.

A leaf `Sym('beat')` binds at compile time to the surface's
`clock_reader('beat')` (the beat IntegralClock's reader); a literal folds to
a constant; a compiled constant (`v[i]` off a constant table) folds too.
Any leaf that resolves to neither a clock reader nor a constant makes the
whole guard UNCOMPILABLE and `compile_guard` returns None - the caller
falls back (skips the window / re-runs the interpreter), never guesses.

The composed curve is authored directly in song-time seconds (each leaf
reader applies its own clock internally), so the returned `Channel` uses the
identity `SongTimeClock` at the top and `guard.at(t_seconds) -> bool`.
"""
from __future__ import annotations

import math
import operator
from typing import Callable

from analysis.player.render.expr import ast
from analysis.player.render.expr.surface import UNRESOLVED, Surface
from analysis.player.render.scheduler import Channel, SongTimeClock

_SONG_TIME = SongTimeClock()

_CMP = {
    '<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge,
    '==': operator.eq, '~=': operator.ne,
}
_ARITH = {
    '+': operator.add, '-': operator.sub, '*': operator.mul,
    '/': operator.truediv, '%': operator.mod, '^': operator.pow,
}

# `math.*` functions that inline into an analytic curve as NATIVE ops (the
# contract's "native math -> native, inlined into a curve"). Deterministic,
# pure functions only: `math.random` is excluded (non-deterministic -> not a
# curve). Lua `math.log(x)` is natural log; `math.pow`/`math.fmod` mirror the
# operators. Degree variants and `atan2` cover the oscillator idioms.
_MATH = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
    'atan2': math.atan2, 'sinh': math.sinh, 'cosh': math.cosh,
    'tanh': math.tanh, 'exp': math.exp, 'log': math.log,
    'log10': math.log10, 'sqrt': math.sqrt, 'abs': abs,
    'floor': lambda x: float(math.floor(x)),
    'ceil': lambda x: float(math.ceil(x)),
    'fmod': math.fmod, 'pow': math.pow,
    'deg': math.degrees, 'rad': math.radians,
    'min': min, 'max': max,
}
# `math` constants that fold to a literal.
_MATH_CONST = {'pi': math.pi, 'huge': math.inf}

# A lowered node is a `seconds -> value` callable, or None when the subtree
# cannot be compiled (an unresolved leaf).
_Reader = Callable[[float], object]


def compile_guard(node: ast.Node, surface: Surface) -> Channel | None:
    """A `Channel` whose curve returns the guard's value at a song time, or
    None when any leaf is uncompilable. `guard.at(t) -> bool` for a boolean
    guard; the same shape yields a numeric bound curve for a bare bound."""
    reader = _lower(node, surface)
    if reader is None:
        return None
    return Channel(curve=lambda coord: reader(coord), clock=_SONG_TIME,
                   rest=False)


def _lower(node: ast.Node, surface: Surface) -> _Reader | None:
    match node:
        case ast.Num(value=v):
            return lambda t: v
        case ast.Bool(value=v):
            return lambda t: v
        case ast.Sym(name=name):
            return _lower_symbol(name, surface)
        case ast.Field(base=ast.Sym(name='math'), name=const) \
                if const in _MATH_CONST:
            value = _MATH_CONST[const]
            return lambda t: value
        case ast.Index(base=ast.Sym(name=table), key=key):
            return _lower_index(table, key, surface)
        case ast.Call(fn=ast.Field(base=ast.Sym(name='math'), name=fn),
                      args=args) if fn in _MATH:
            return _lower_math(_MATH[fn], args, surface)
        case ast.Unary(op=op, operand=operand):
            return _lower_unary(op, _lower(operand, surface))
        case ast.Binary(op='and', left=left, right=right):
            return _lower_and(_lower(left, surface), _lower(right, surface))
        case ast.Binary(op='or', left=left, right=right):
            return _lower_or(_lower(left, surface), _lower(right, surface))
        case ast.Binary(op=op, left=left, right=right):
            return _lower_binary(op, _lower(left, surface),
                                 _lower(right, surface))
    return None


def _lower_math(fn, args: tuple, surface: Surface) -> _Reader | None:
    """A `math.f(a, b, ...)` call -> a reader applying the native `fn` to its
    lowered args. Any uncompilable arg makes the whole call uncompilable (the
    curve falls to residue), never a guessed value."""
    readers = [_lower(arg, surface) for arg in args]
    if any(r is None for r in readers):
        return None
    return lambda t: fn(*[r(t) for r in readers])


def _lower_symbol(name: str, surface: Surface) -> _Reader | None:
    reader = surface.clock_reader(name)
    if reader is not None:
        return reader
    value = surface.symbol(name)
    if value is UNRESOLVED:
        return None
    return lambda t: value


def _lower_index(table: str, key: ast.Node, surface: Surface) -> _Reader | None:
    # A table index only compiles when the base and key fold to constants
    # (a compiled v[]/e[] entry); a live-indexed table is uncompilable.
    base = surface.symbol(table)
    key_reader = _lower(key, surface)
    if base is UNRESOLVED or key_reader is None:
        return None
    value = surface.index(base, key_reader(0.0))
    if value is UNRESOLVED:
        return None
    return lambda t: value


def _lower_unary(op: str, operand: _Reader | None) -> _Reader | None:
    if operand is None:
        return None
    if op == '-':
        return lambda t: -operand(t)
    if op == 'not':
        return lambda t: not operand(t)
    return None


def _lower_binary(op: str, left: _Reader | None,
                  right: _Reader | None) -> _Reader | None:
    if left is None or right is None:
        return None
    fn = _CMP.get(op) or _ARITH.get(op)
    if fn is None:
        return None
    return lambda t: fn(left(t), right(t))


def _lower_and(left: _Reader | None, right: _Reader | None) -> _Reader | None:
    # Both sides must compile: a guard whose conjunct we cannot resolve is
    # not compilable (the window-extraction fold that drops a proven-true
    # conjunct happens at extraction time, not here).
    if left is None or right is None:
        return None
    return lambda t: bool(left(t)) and bool(right(t))


def _lower_or(left: _Reader | None, right: _Reader | None) -> _Reader | None:
    if left is None or right is None:
        return None
    return lambda t: bool(left(t)) or bool(right(t))
