"""Shared `Surface` base for a lupa-backed Lua host.

The guard evaluator/compiler and the AST interpreter (`analysis/player/render/
expr/`) read names, table elements, and calls off a `Surface`. Any game that
scripts with lupa needs the same handling of raw lupa tables: read/write an
element with Lua 1-based keys, distinguish a host TABLE from a host FUNCTION for
`type()`, iterate a table for a generic-for, and invoke a colon-method on a raw
table. `LuaHostSurface` collects that game-neutral machinery so each game's
concrete surface overrides only its own vocabulary (clock symbols, engine
verbs, actor routing).

It is deliberately game-agnostic: no actor/recorder/clock concepts appear here.
A subclass supplies `_global(name)` (how this host reads a bare global) and may
override `iter_table`/`is_host_table` to exclude host objects that happen to be
lupa tables but are not data containers (e.g. actor recorders).
"""
from __future__ import annotations

from analysis.player.render.expr.surface import UNRESOLVED, Resolution


# The concrete lupa table type, learned on first sight and cached: an exact
# `type(x) is _LUA_TABLE_TYPE` compare is far cheaper than the duck-check in the
# per-tick index hot path (millions of calls on a real chart). lupa hands out
# one table class, so one cache entry covers every table.
_LUA_TABLE_TYPE = None


def _iterable(value) -> bool:
    try:
        iter(value)
        return True
    except TypeError:
        return False


def _is_lua_table(value) -> bool:
    """Duck-typed lupa-table check: a Lua table supports integer indexing but
    is not a Python string/bytes. Fast path: a value of the cached lupa table
    type is a table outright; the duck-check runs only until the type is
    learned."""
    global _LUA_TABLE_TYPE
    if type(value) is _LUA_TABLE_TYPE:
        return True
    if hasattr(value, '__getitem__') and not isinstance(value, (str, bytes)):
        # A lupa FUNCTION is also `__getitem__`-able, so seed the cache only
        # from an actually-iterable lupa object (a table). Learning the type
        # from a function would invert every later `type(x) is _LUA_TABLE_TYPE`
        # check - reporting functions as tables and tables as non-tables.
        if _LUA_TABLE_TYPE is None and type(value).__module__.startswith(
                ('lupa', '_lupa')) and _iterable(value):
            _LUA_TABLE_TYPE = type(value)
        return True
    return False


def _is_iterable_lua_table(value) -> bool:
    """A lupa TABLE (iterable) vs a lupa FUNCTION (not) - both indexable, so the
    coarse `_is_lua_table` cannot tell them apart. Only a lupa host object
    qualifies; a table also seeds `_LUA_TABLE_TYPE` for the fast path."""
    global _LUA_TABLE_TYPE
    if not type(value).__module__.startswith(('lupa', '_lupa')):
        return False
    if not _iterable(value):
        return False
    if _LUA_TABLE_TYPE is None:
        _LUA_TABLE_TYPE = type(value)
    return True


def _lua_index(table, key) -> Resolution:
    """Read `table[key]` from a raw lupa table, returning the RAW Lua value
    (a nested table stays a raw table, a function stays callable) so it round-
    trips back into Lua calls without becoming `userdata`. nil/missing ->
    UNRESOLVED."""
    try:
        value = table[key]
    except (KeyError, TypeError):
        return UNRESOLVED
    return UNRESOLVED if value is None else value


class LuaHostSurface:
    """Game-neutral `Surface` machinery for a lupa-backed host: raw-table
    read/write/iterate/classify with Lua key rules and nil<->UNRESOLVED
    marshalling, plus colon-method dispatch on a raw table. A subclass supplies
    `_global(name)` and the game-specific `symbol`/`call`/`method`/`poke`/
    `clock_reader`."""

    def actor_id(self, value) -> int | None:
        """No actor concept at this level - a lupa host is just tables. A game
        subclass that models actors overrides both this and `actor_value`."""
        return None

    def actor_value(self, actor_id: int):
        return None

    def _global(self, name: str):
        """Read a bare global by name from this host's namespace, returning the
        RAW host value (None for absent). Subclass hook: the base is namespace-
        agnostic."""
        raise NotImplementedError

    def _read_global(self, name: str) -> Resolution:
        value = self._global(name)
        if value is None:
            return UNRESOLVED
        if isinstance(value, bool) or isinstance(value, (int, float)):
            return value
        if _is_lua_table(value):
            # Return the RAW lupa table, not a wrapper: `index` handles a raw
            # table, and - crucially - a raw table round-trips back into a Lua
            # call (`table.insert(t, x)`) as a real table, where a Python
            # wrapper would arrive as `userdata` and fault the builtin.
            return value
        return UNRESOLVED

    def index(self, base: Resolution, key: Resolution) -> Resolution:
        if base is UNRESOLVED or key is UNRESOLVED:
            return UNRESOLVED
        # Hot path inlined: a lupa table of the learned type reads directly
        # (the per-tick index dominates; avoid the _is_lua_table + _lua_index
        # call hops). String key = field; numeric = element (key unchanged -
        # lupa keeps its own 1-based array keys).
        if type(base) is _LUA_TABLE_TYPE:
            try:
                value = base[key if isinstance(key, str) else int(key)]
            except (KeyError, TypeError, ValueError):
                return UNRESOLVED
            return UNRESOLVED if value is None else value
        try:
            if _is_lua_table(base):
                return _lua_index(base, key if isinstance(key, str)
                                  else int(key))
            if isinstance(base, (list, tuple)):
                return base[int(key) - 1]      # Lua tables are 1-indexed
            if isinstance(base, dict):
                return base.get(key, UNRESOLVED)
        except (IndexError, ValueError, TypeError):
            return UNRESOLVED
        return UNRESOLVED

    def set_index(self, base: Resolution, key: Resolution, value) -> bool:
        """`base[key] = value` on a lupa host table (a body writing scratch
        state - `pc_strinkku[i] = string.sub(...)`). Mirrors `index`'s key rule
        (string key = field, numeric = 1-based element). UNRESOLVED marshals to
        nil so a host table never stores the sentinel. Returns True on a landed
        write, False when `base` is not a host table (the interpreter's own
        LuaTable handles itself upstream)."""
        if base is UNRESOLVED or key is UNRESOLVED:
            return False
        if type(base) is not _LUA_TABLE_TYPE and not _is_lua_table(base):
            return False
        try:
            base[key if isinstance(key, str) else int(key)] = \
                None if value is UNRESOLVED else value
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def is_host_table(self, value) -> bool:
        # Strict: a lupa TABLE, not a lupa FUNCTION. Both are callable host
        # objects that duck-type as indexable (`_is_lua_table` says yes to
        # both), so `type()` needs the finer split - a table iterates, a
        # function does not. Prefer the exact learned type; fall back to the
        # iterability probe (which also learns the type on first table sight).
        if _LUA_TABLE_TYPE is not None:
            return type(value) is _LUA_TABLE_TYPE
        return _is_iterable_lua_table(value)

    def _lua_method(self, recv, name, args) -> Resolution:
        """Invoke `recv:name(args)` when `recv` is a live Lua TABLE (colon-call:
        the receiver is the implicit first arg). A missing method, a nil result,
        or any fault is UNRESOLVED - never a hard fault, so the interpreter
        degrades per-call the way the Lua path swallows."""
        if not _is_lua_table(recv):
            return UNRESOLVED
        try:
            fn = recv[name]
        except (KeyError, TypeError):
            return UNRESOLVED
        if not callable(fn):
            return UNRESOLVED
        try:
            result = fn(recv, *args)
        except Exception:
            return UNRESOLVED
        return UNRESOLVED if result is None else result

    def iter_table(self, table: Resolution) -> list | None:
        """`(key, value)` pairs for a raw lupa table (a load pass built `local
        t = {}` then `table.insert`ed into it, iterated by a generic-for). None
        for a non-lupa value (the interpreter's own LuaTable iterates itself). A
        subclass may override to exclude host tables that are not data
        containers."""
        if not _is_lua_table(table):
            return None
        try:
            return list(table.items())
        except (AttributeError, TypeError):
            return None
