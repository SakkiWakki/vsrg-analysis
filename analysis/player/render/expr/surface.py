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

    def set_index(self, base: Resolution, key: Resolution, value) -> bool:
        """`base[key] = value` on a HOST table (a lupa table a load pass built,
        used by a body as scratch/accumulator state). Returns True when the
        write landed on a host table the surface owns, False when `base` is not
        such a table (the interpreter's own `LuaTable` writes itself). A surface
        with no live host to mutate (window extraction) is a no-op returning
        False."""
        ...

    def is_host_table(self, value) -> bool:
        """True when `value` is a host table (a lupa table a load pass built).
        Lets the value-model tell a host TABLE from a host FUNCTION when both are
        opaque, callable host objects - so `type(t)` reports `'table'`, not
        `'function'`. A surface with no host world returns False."""
        ...

    def call(self, name: str, args: list) -> Resolution:
        """`name(args)` (e.g. `perframe`) -> value, else UNRESOLVED. `args`
        are already-resolved (a caller passes UNRESOLVED through)."""
        ...

    def method(self, recv: Resolution, name: str, args: list) -> Resolution:
        """`recv:name(args)` in VALUE position -> the getter's value, else
        UNRESOLVED. A read (`self:GetX()`), never an effect: a surface must
        not mutate here. `recv`/`args` are already-resolved."""
        ...

    def poke(self, recv: Resolution, name: str, args: list) -> None:
        """`recv:name(args)` in EFFECT position (statement) - apply the setter
        to `recv` (`self:zoom(x)`). Returns nothing; a surface with no live
        world to mutate (window extraction) is a no-op. `recv`/`args` are
        already-resolved; an UNRESOLVED recv is dropped."""
        ...

    def iter_table(self, table: Resolution) -> list | None:
        """`(key, value)` pairs for a HOST table (a lupa table a load pass
        built), so a generic-for can iterate a table the interpreter did not
        create. None when `table` is not a host table the surface owns (the
        interpreter's own tables iterate themselves)."""
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

    def set_index(self, base: Resolution, key: Resolution, value) -> bool:
        # No live host to mutate: window extraction reads constants only.
        return False

    def is_host_table(self, value) -> bool:
        return False

    def call(self, name: str, args: list) -> Resolution:
        return UNRESOLVED

    def method(self, recv: Resolution, name: str, args: list) -> Resolution:
        # No live world: a getter has no value and an effect no target. Window
        # extraction reads only constants, so both are inert here.
        return UNRESOLVED

    def poke(self, recv: Resolution, name: str, args: list) -> None:
        return None

    def iter_table(self, table: Resolution) -> list | None:
        return None

    def clock_reader(self, name: str) -> Callable[[float], float] | None:
        return None
