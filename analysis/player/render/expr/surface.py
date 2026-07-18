"""The value surface an evaluator resolves operands against.

A guard reads names (`beat`, `mod_time`, `fgcurcommand`), table elements
(`v[3]`, `e[1]`), and calls (`perframe(a, b)`). The `Surface` is where those
resolve to values - the SAME surface the plugin ecosystem exposes, so a
guard and a plugin read one source of truth. The evaluator never names game
vocabulary; a game supplies a `Surface` (NotITG's lives in
`analysis/games/notitg/guard_surface.py`).

Anything the surface cannot resolve returns `UNRESOLVED` - a sentinel, not a
value. It propagates like a poison value through arithmetic and comparison
(any UNRESOLVED operand -> UNRESOLVED result) with the short-circuit
exceptions for `and`/`or`, so a guard that cannot be proven collapses to
UNRESOLVED and its window is skipped (never guessed) - the discipline the
old `_beat_arg is None` had, now general.

Two resolution paths:
- `symbol` / `index` / `call` are EAGER: return a concrete value or
  UNRESOLVED at a fixed instant. The tree-walk backend uses these.
- `clock_reader(name)` returns a `seconds -> value` callable (or None) for
  the COMPILE backend: a driver symbol like `beat` binds to its clock once
  at compile time so the compiled guard is a plain call per tick.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


class _Unresolved:
    """Singleton sentinel for an operand off the value surface."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return 'UNRESOLVED'

    def __bool__(self) -> bool:
        # Guard against accidental truthiness in a `if resolved:` check -
        # callers must compare `is UNRESOLVED` explicitly.
        raise TypeError('UNRESOLVED has no boolean value; compare identity')


UNRESOLVED = _Unresolved()

# A resolution is a concrete Python value (number/bool/str) or UNRESOLVED.
Resolution = object


@runtime_checkable
class Surface(Protocol):
    """What an evaluator queries. A game implements this over its live host;
    the plugin surface implements it over the exposed channel values."""

    def symbol(self, name: str) -> Resolution:
        """`beat`/`mod_time`/`measure`/a state var -> value, else UNRESOLVED."""
        ...

    def index(self, base: Resolution, key: Resolution) -> Resolution:
        """`table[key]` -> element when `base` is a resolved indexable and
        `key` a resolved key, else UNRESOLVED."""
        ...

    def call(self, name: str, args: list) -> Resolution:
        """`name(args)` (e.g. `perframe`) -> value, else UNRESOLVED. `args`
        are already-resolved (a caller passes UNRESOLVED through)."""
        ...

    def clock_reader(self, name: str) -> Callable[[float], float] | None:
        """A `seconds -> value` reader for driver symbol `name` (compile
        path), or None when `name` is not a clock-backed driver."""
        ...


class ConstSurface:
    """A surface that resolves ONLY literals and a fixed constant table -
    every driver symbol (`beat`, `mod_time`) is UNRESOLVED. Used by window
    extraction, which wants a guard's constant BOUND expression evaluated
    while treating the driver structurally. `constants` maps a name to a
    Python value (a number, or a list/dict for a compiled `v`/`e` table)."""

    def __init__(self, constants: dict | None = None):
        self._constants = constants or {}

    def symbol(self, name: str) -> Resolution:
        return self._constants.get(name, UNRESOLVED)

    def index(self, base: Resolution, key: Resolution) -> Resolution:
        if base is UNRESOLVED or key is UNRESOLVED:
            return UNRESOLVED
        try:
            if isinstance(base, (list, tuple)):
                return base[int(key) - 1]      # Lua tables are 1-indexed
            if isinstance(base, dict):
                return base.get(key, UNRESOLVED)
        except (IndexError, ValueError, TypeError):
            return UNRESOLVED
        return UNRESOLVED

    def call(self, name: str, args: list) -> Resolution:
        return UNRESOLVED

    def clock_reader(self, name: str) -> Callable[[float], float] | None:
        return None
