"""The Python bridge for the native (Rust) frame interpreter's migration
frontier. The Rust core calls this object's protocol methods for every
live-engine crossing; it marshals handle-ids <-> live Python objects and
delegates to the existing `NotitgGuardSurface` + `SimEnvironment`, so the native
tick loop drives the SAME sim the Python interpreter does.

This is the "thin abstraction layer" the port design names: it owns NO
semantics, only the handle bookkeeping + a pass-through to the surface. As the
surface itself ports to Rust, this bridge shrinks to nothing (the frontier
becomes native), and the core never changes.

The protocol (called from Rust `PyFrontier`, values are primitives or a tagged
`{"__handle__": id}` / `{"__host_fn__": id}` dict for an opaque host object):

    symbol(name) call(name, args) method(recv, name, args) poke(recv, name, args)
    index(handle, key) set_index(handle, key, value) iter_table(handle)
    call_host(id, args)
"""
from __future__ import annotations

from analysis.player.render.expr.surface import UNRESOLVED

# A live Python object crossing to the core is registered under an int id; the
# core holds only the id (an opaque handle) and hands it back for the next
# crossing. `_HANDLE`/`_HOST_FN` tag which registry an id addresses.
_HANDLE = '__handle__'
_HOST_FN = '__host_fn__'


class NativeFrontier:
    """Bridges the Rust core's frontier calls onto a `NotitgGuardSurface`. One
    per interpreter run; the handle registries live for that run."""

    def __init__(self, surface, host_env):
        self._surface = surface
        # Globals live in the SAME lupa env the load pass populated, so a
        # body's accumulator and the guards/load globals share one namespace
        # (the _LuaEnvStore contract, now across the frontier).
        self._env = host_env
        # id -> live object (actor recorder, lupa table); and the reverse so the
        # same object always gets the same id (identity-stable handles).
        self._objs: dict[int, object] = {}
        self._ids: dict[int, int] = {}
        self._next = 1

    # -- handle marshalling -------------------------------------------------

    def _to_handle(self, obj):
        """A live Python object -> its tagged handle dict; a primitive passes
        through unchanged; UNRESOLVED stays the sentinel (the core compares it by
        identity). A host FUNCTION becomes a host-fn handle (the core calls it
        back via `call_host`); a host TABLE becomes a table handle.

        A lupa TABLE is itself `callable`, so `callable(obj)` alone mis-tags a
        table as a function - the surface's `is_host_table` tells them apart (the
        same distinction `type()` needs)."""
        if obj is UNRESOLVED or obj is None:
            return obj
        if isinstance(obj, (bool, int, float, str)):
            return obj
        key = id(obj)
        hid = self._ids.get(key)
        if hid is None:
            hid = self._next
            self._next += 1
            self._objs[hid] = obj
            self._ids[key] = hid
        is_fn = callable(obj) and not self._surface.is_host_table(obj)
        return {_HOST_FN: hid} if is_fn else {_HANDLE: hid}

    def _from_handle(self, value):
        """A value FROM the core -> a live Python object. A tagged handle dict
        resolves to its registered object; a primitive/None passes through."""
        if isinstance(value, dict):
            if _HANDLE in value:
                return self._objs.get(int(value[_HANDLE]))
            if _HOST_FN in value:
                return self._objs.get(int(value[_HOST_FN]))
        return value

    def _args(self, args):
        return [self._from_handle(a) for a in args]

    # -- the frontier protocol ---------------------------------------------

    def global_get(self, name):
        """Read a global from the host env (frontier-backed globals). An absent
        name is UNRESOLVED (the core then treats it as unbound and asks
        `symbol`)."""
        value = self._env[name]
        return UNRESOLVED if value is None else self._to_handle(value)

    def global_set(self, name, value):
        """Write a global to the host env - so an accumulator persists to the
        shared namespace exactly as the Lua path's globals do."""
        self._env[name] = self._from_handle(value)

    def set_self(self, recv_obj):
        """Bind `self` (the owning actor's recorder) for the coming tick, as a
        handle the body's `self:...` pokes resolve against."""
        self._env['self'] = recv_obj

    def symbol(self, name):
        return self._to_handle(self._surface.symbol(name))

    def call(self, name, args):
        return self._to_handle(self._surface.call(name, self._args(args)))

    def method(self, recv, name, args):
        recv_obj = self._from_handle(recv)
        return self._to_handle(
            self._surface.method(recv_obj, name, self._args(args)))

    def poke(self, recv, name, args):
        self._surface.poke(self._from_handle(recv), name, self._args(args))

    def index(self, handle, key):
        base = self._objs.get(int(handle))
        return self._to_handle(self._surface.index(base, self._from_handle(key)))

    def set_index(self, handle, key, value):
        base = self._objs.get(int(handle))
        return self._surface.set_index(
            base, self._from_handle(key), self._from_handle(value))

    def iter_table(self, handle):
        base = self._objs.get(int(handle))
        pairs = self._surface.iter_table(base)
        if pairs is None:
            return None
        return [[self._to_handle(k), self._to_handle(v)] for k, v in pairs]

    def call_host(self, id_, args):
        fn = self._objs.get(int(id_))
        if not callable(fn):
            return UNRESOLVED
        result = fn(*self._args(args))
        return self._to_handle(result)

    def snapshot_global(self, name):
        """Deep-copy a load-populated host DATA TABLE into a plain Python tree
        (lists for array runs, dicts for keyed parts, primitives at the leaves),
        which the core marshals into native tables. Lets the core read `v[i][j]`
        with ZERO frontier crossings after the one snapshot. Returns UNRESOLVED
        when `name` is not a host table. The caller (core) only snapshots tables
        the body never writes, so the copy staying frozen is correct."""
        obj = self._env[name]
        if obj is None or not self._surface.is_host_table(obj):
            return UNRESOLVED
        snap = self._deep_copy(obj)
        # None means the copy aborted (a non-array/non-primitive inside); the
        # table then stays on the crossing path.
        return UNRESOLVED if snap is None else snap

    def _deep_copy(self, obj, depth=0):
        """A lupa ARRAY table -> a native Python list (nested arrays recurse);
        primitives pass through; a leaf that is not a primitive or a pure array
        table aborts the snapshot (-> UNRESOLVED), so the table stays on the
        crossing path rather than snapshotting something the list model cannot
        hold. The corpus data tables (v/mods/e) are pure nested arrays, which is
        the case this covers; a keyed/mixed table is rare and simply not
        snapshotted. Bounded depth guards a self-referential table."""
        if depth > 16:
            return None
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if not self._surface.is_host_table(obj):
            return None          # a function/actor inside -> not snapshottable
        items = list(obj.items())
        n = len(items)
        is_pure_array = all(
            isinstance(k, (int, float)) and float(k).is_integer()
            and 1 <= int(k) <= n for k, _ in items)
        if not is_pure_array:
            return None          # keyed/mixed -> leave on the crossing path
        ordered = sorted(items, key=lambda kv: int(kv[0]))
        out = []
        for _, v in ordered:
            cv = self._deep_copy(v, depth + 1)
            if cv is None and v is not None:
                return None      # an unsnapshottable element aborts the whole
            out.append(cv)
        return out
