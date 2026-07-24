"""Python driver + frontier for the C op-stream executor (cbody_abi.c).

Builds a `CBody` from an `opstream.OpProgram`, wires the CFrontier callbacks to
the live `NotitgGuardSurface` / interpreter, and runs one tick. CValues cross
the ctypes boundary as uint64 (the NaN-box); scalars (num/bool/nil/str) marshal
directly, and any non-scalar Python value (an actor recorder table, a host
object) is registered in a handle table and passed as a HANDLE CValue - the
frontier maps it back on the way in.

Semantics parity target: the compiled body must produce the same SimActor pokes
the Lua/interpreter path does (gated by keyframe_diff.diff_runs).
"""
from __future__ import annotations

import ctypes
import os

from analysis.player.render.expr import ast as _ast
from analysis.player.render.expr.frame_eval import (
    UNRESOLVED, _NO_BUILTIN, _builtin_call)

_LIB = None

# Cache sentinel: a snapshot candidate not yet resolved. Distinct from a cached
# None (resolved, not snapshottable -> use the crossing path) so each name is
# probed at most once.
_SNAP_MISS = object()


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    so = os.path.join(os.path.dirname(__file__), 'libcbody.so')
    # PyDLL, not CDLL: every call keeps the GIL. A CDLL released it per
    # call, so each of the ~20 frontier crossings + ~8 marshalling calls
    # per tick paid a release/reacquire; against a busy render thread
    # each reacquire waits on the switch interval (a GIL convoy measured
    # at ~1700x sweep slowdown, frozen-visuals-with-live-audio). Holding
    # the GIL through cbody_run gives the same scheduling profile as the
    # lupa path: short C stretches inside normal thread quanta.
    lib = ctypes.PyDLL(so)
    u64 = ctypes.c_uint64
    lib.cbody_new.restype = ctypes.c_void_p
    lib.cbody_new.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.cbody_free.argtypes = [ctypes.c_void_p]
    lib.cbody_set_ops.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.cbody_set_consts.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.cbody_set_const_str.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.cbody_set_names.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.cbody_set_name.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.cbody_set_frontier.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 16
    lib.cbody_run.restype = ctypes.c_int
    lib.cbody_run.argtypes = [ctypes.c_void_p, u64]
    lib.cbody_str.restype = ctypes.c_char_p
    lib.cbody_str.argtypes = [ctypes.c_void_p, u64, ctypes.POINTER(ctypes.c_int)]
    lib.cbody_intern.restype = u64
    lib.cbody_intern.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.cbody_mark_stable.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.cbody_set_clock_ids.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.cbody_set_clock.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    lib.cbody_frame_get.restype = u64
    lib.cbody_frame_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.cbody_frame_set.argtypes = [ctypes.c_void_p, ctypes.c_int, u64]
    lib.cbody_table_new_array.restype = u64
    lib.cbody_table_new_array.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.cbody_table_seti.argtypes = [ctypes.c_void_p, u64, ctypes.c_int64, u64]
    lib.cbody_table_geti.restype = u64
    lib.cbody_table_geti.argtypes = [ctypes.c_void_p, u64, ctypes.c_int64]
    lib.cbody_table_len.restype = ctypes.c_int64
    lib.cbody_table_len.argtypes = [ctypes.c_void_p, u64]
    for name, ret, args in [
        ('cbody_num', u64, [ctypes.c_double]), ('cbody_nil', u64, []),
        ('cbody_true', u64, []), ('cbody_false', u64, []),
        ('cbody_unresolved', u64, []), ('cbody_handle', u64, [u64]),
        ('cbody_is_num', ctypes.c_int, [u64]), ('cbody_is_str', ctypes.c_int, [u64]),
        ('cbody_is_nil', ctypes.c_int, [u64]), ('cbody_is_bool', ctypes.c_int, [u64]),
        ('cbody_is_handle', ctypes.c_int, [u64]), ('cbody_is_unres', ctypes.c_int, [u64]),
        ('cbody_is_table', ctypes.c_int, [u64]),
        ('cbody_as_num', ctypes.c_double, [u64]), ('cbody_bool_val', ctypes.c_int, [u64]),
        ('cbody_handle_id', u64, [u64]),
    ]:
        fn = getattr(lib, name); fn.restype = ret; fn.argtypes = args
    _LIB = lib
    return lib


# CFrontier callback signatures (must match exec.h). CValue = uint64.
U64 = ctypes.c_uint64
_SYMBOL   = ctypes.CFUNCTYPE(U64, ctypes.c_void_p, ctypes.c_char_p)
_GGET     = ctypes.CFUNCTYPE(U64, ctypes.c_void_p, ctypes.c_char_p)
_GSET     = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p, U64)
_GETTER   = ctypes.CFUNCTYPE(U64, ctypes.c_void_p, U64, ctypes.c_char_p, ctypes.POINTER(U64), ctypes.c_int)
_POKE     = ctypes.CFUNCTYPE(None, ctypes.c_void_p, U64, ctypes.c_char_p, ctypes.POINTER(U64), ctypes.c_int)
_CALL     = ctypes.CFUNCTYPE(U64, ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(U64), ctypes.c_int)
_CALLVAL  = ctypes.CFUNCTYPE(U64, ctypes.c_void_p, U64, ctypes.POINTER(U64), ctypes.c_int)
_INDEX    = ctypes.CFUNCTYPE(U64, ctypes.c_void_p, U64, U64)
_SETINDEX = ctypes.CFUNCTYPE(None, ctypes.c_void_p, U64, U64, U64)
_LENGTH   = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_void_p, U64)
_TINSERT  = ctypes.CFUNCTYPE(None, ctypes.c_void_p, U64, U64)
_ITERSET  = ctypes.CFUNCTYPE(U64, ctypes.c_void_p, ctypes.POINTER(U64), ctypes.c_int)
_ITERNEXT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, U64, ctypes.POINTER(U64), ctypes.c_int)
_FALLBACK = ctypes.CFUNCTYPE(U64, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(U64), ctypes.c_int)
_ABORTED  = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class CompiledBodyC:
    """A ctypes-driven op-stream body over the live surface. `run(self_table)`
    executes one tick. Holds the callback closures alive (ctypes requires it)."""

    def __init__(self, program, surface, store, fallback_nodes, fallback_run,
                 interp=None):
        self._lib = _load_lib()
        self._surface = surface
        self._store = store            # accumulator globals (dict-like get/set)
        self._nodes = fallback_nodes   # node-id -> AST node for FALLBACK
        self._fallback_run = fallback_run  # (node, env_read, env_write) -> value
        self._interp = interp          # for invoking host-fn globals (call_sym)
        # Load-set snapshot: bare-symbol reads the body never writes, resolved
        # once to an arena TABLE so nested v[i][j] indexing stays in C. `symbol`
        # consults these before crossing; None caches a non-snapshottable result.
        self._snap_names = set(getattr(program, 'symbol_reads', ()) or ())
        self._snap_cache: dict[str, int] = {}
        # arena TABLE id -> the original lupa host table it snapshots. INDEX
        # resolves in C, but the rare crossing of a snapshotted table back to
        # Python (type()/#/passed to a host fn) returns the real host object so
        # its semantics match the non-snapshot path exactly.
        self._table_origin: dict[int, object] = {}

        ser = program.serialize()
        self._b = self._lib.cbody_new(ser['nslots'], 4096)
        # ops
        self._lib.cbody_set_ops(self._b, ser['ops'], ser['nops'])
        # consts (kinds/nums) then string consts
        self._lib.cbody_set_consts(self._b, ser['const_kinds'], ser['const_nums'], ser['nconsts'])
        si = 0
        for i, k in enumerate(ser['const_kinds']):
            if k == 1:  # str const: nums[i] held the index into const_strs
                s = ser['const_strs'][si].encode('utf-8', 'surrogatepass'); si += 1
                self._lib.cbody_set_const_str(self._b, i, s, len(s))
        # names
        self._lib.cbody_set_names(self._b, len(ser['names']))
        for i, nm in enumerate(ser['names']):
            e = nm.encode('utf-8', 'surrogatepass')
            self._lib.cbody_set_name(self._b, i, e, len(e))
        self._name_id = {nm: i for i, nm in enumerate(ser['names'])}
        self._clock_wired = False

        # handle registry: id -> python object (non-scalar values crossing out)
        self._handles: dict[int, object] = {}
        self._handle_next = 1
        self._obj_to_handle: dict[int, int] = {}

        self._install_frontier()

    # -- marshalling --------------------------------------------------------
    def _to_cv(self, v):
        """Python value -> CValue uint64."""
        lib = self._lib
        if v is UNRESOLVED:
            return lib.cbody_unresolved()
        if v is None:
            return lib.cbody_nil()
        if isinstance(v, bool):
            return lib.cbody_true() if v else lib.cbody_false()
        if isinstance(v, (int, float)):
            return lib.cbody_num(float(v))
        if isinstance(v, str):
            e = v.encode('utf-8', 'surrogatepass')
            return lib.cbody_intern(self._b, e, len(e))
        # non-scalar: register a handle
        key = id(v)
        h = self._obj_to_handle.get(key)
        if h is None:
            h = self._handle_next; self._handle_next += 1
            self._handles[h] = v
            self._obj_to_handle[key] = h
        return lib.cbody_handle(h)

    def _from_cv(self, cv):
        """CValue uint64 -> Python value."""
        lib = self._lib
        if lib.cbody_is_num(cv):
            return lib.cbody_as_num(cv)
        if lib.cbody_is_nil(cv):
            return None
        if lib.cbody_is_unres(cv):
            return UNRESOLVED
        if lib.cbody_is_bool(cv):
            return bool(lib.cbody_bool_val(cv))
        if lib.cbody_is_str(cv):
            ln = ctypes.c_int()
            p = lib.cbody_str(self._b, cv, ctypes.byref(ln))
            return p[:ln.value].decode('utf-8', 'surrogatepass')
        if lib.cbody_is_handle(cv):
            hid = lib.cbody_handle_id(cv)
            return self._handles.get(hid)
        if lib.cbody_is_table(cv):
            # A snapshotted load-set table crossing back to Python (type()/#/
            # passed to a host fn): return the ORIGINAL host object so its
            # semantics match the non-snapshot path. INDEX never reaches here -
            # it resolves in C against the arena.
            return self._table_origin.get(lib.cbody_handle_id(cv))
        return None

    def _args(self, arr, argc):
        return [self._from_cv(arr[i]) for i in range(argc)]

    def _snapshot_symbol(self, name):
        """Resolve a never-written symbol to an arena TABLE CValue, or None if it
        is not a snapshottable host DATA table (then `symbol` falls through to
        the crossing path). Mirrors native_frontier._deep_copy: only a PURE
        nested-array lupa table (v/mods/e...) snapshots; a keyed/mixed table or a
        table holding a function/actor stays on the frontier."""
        r = self._surface.symbol(name)
        obj = r
        if obj is UNRESOLVED or obj is None or not self._surface.is_host_table(obj):
            return None
        cv = self._copy_into_arena(obj)
        if cv is not None:
            self._table_origin[self._lib.cbody_handle_id(cv)] = obj
        return cv

    def _copy_into_arena(self, obj, depth=0):
        """A lupa ARRAY table -> an arena TABLE CValue (nested arrays recurse);
        primitives box directly; a non-primitive, non-pure-array leaf aborts the
        whole snapshot (returns None) so the table stays crossing. Bounded depth
        guards a self-referential table."""
        if depth > 16:
            return None
        if obj is None or isinstance(obj, bool):
            return self._to_cv(obj)
        if isinstance(obj, (int, float)):
            return self._lib.cbody_num(float(obj))
        if isinstance(obj, str):
            e = obj.encode('utf-8', 'surrogatepass')
            return self._lib.cbody_intern(self._b, e, len(e))
        if not self._surface.is_host_table(obj):
            return None
        items_fn = getattr(obj, 'items', None)
        if items_fn is None:
            # A host table without dict iteration (a time-varying
            # surface view like a membership table) can never land as
            # a frozen snapshot; it stays on the crossing path.
            return None
        items = list(items_fn())
        n = len(items)
        pure_array = all(
            isinstance(k, (int, float)) and float(k).is_integer()
            and 1 <= int(k) <= n for k, _ in items)
        if not pure_array:
            return None
        ordered = sorted(items, key=lambda kv: int(kv[0]))
        tbl = self._lib.cbody_table_new_array(self._b, n)
        for idx, (_, v) in enumerate(ordered, start=1):
            cv = self._copy_into_arena(v, depth + 1)
            if cv is None and v is not None:
                return None
            self._lib.cbody_table_seti(self._b, tbl, idx,
                                       self._lib.cbody_nil() if cv is None else cv)
        return tbl

    def _call_host(self, fn, args):
        """Invoke a host (lupa) callable resolved from a symbol - the call_sym
        step 3 path (perframe/mpf/...). Uses the interp's closure caller so
        UNRESOLVED->nil marshalling matches the Lua path exactly."""
        if self._interp is not None:
            return self._interp._call_closure(fn, args, 0)
        try:
            return fn(*[None if a is UNRESOLVED else a for a in args])
        except Exception:
            return UNRESOLVED

    # -- frontier callbacks (kept as attributes so ctypes holds them alive) --
    def _install_frontier(self):
        surf = self._surface
        store = self._store
        lib = self._lib
        nil_cv = lib.cbody_nil()

        def symbol(ctx, name):
            nm = name.decode()
            if nm in self._snap_names:
                cv = self._snap_cache.get(nm, _SNAP_MISS)
                if cv is _SNAP_MISS:
                    cv = self._snapshot_symbol(nm)
                    self._snap_cache[nm] = cv
                    if cv is not None:
                        # A LANDED arena snapshot is self-contained, so
                        # the C side may cache this symbol forever. A
                        # driver symbol (beat) or non-table global never
                        # reaches here - those stay per-tick.
                        name_id = self._name_id.get(nm)
                        if name_id is not None:
                            self._lib.cbody_mark_stable(self._b, name_id)
                if cv is not None:
                    return cv
            r = surf.symbol(nm)
            return self._to_cv(_res(r))
        def gget(ctx, name):
            v = store.get(name.decode()) if store is not None else UNRESOLVED
            return self._to_cv(v)
        def gset(ctx, name, cv):
            if store is not None:
                store.set(name.decode(), self._from_cv(cv))
        def getter(ctx, recv, verb, args, argc):
            r = surf.method(self._from_cv(recv), verb.decode(), self._args(args, argc))
            return self._to_cv(_res(r))
        def poke(ctx, recv, verb, args, argc):
            surf.poke(self._from_cv(recv), verb.decode(), self._args(args, argc))
        def call(ctx, name, args, argc):
            # Free call name(args). Resolution order MUST match
            # frame_compile_exec.call_sym: (0) a stdlib BUILTIN (type/tonumber/
            # tostring/table.*) serviced by the interpreter itself - NOT the
            # surface (surface.call returns UNRESOLVED for these, which silently
            # broke `type(x)=='function'` gates); (1) a host-fn global -
            # symbol(name) resolving to a CALLABLE is invoked (perframe/mpf);
            # (2) UNRESOLVED arg -> UNRESOLVED; (3) surface.call fallback.
            nm = name.decode()
            arg_vs = self._args(args, argc)
            b = _builtin_call(_ast.Sym(name=nm), arg_vs, surf)
            if b is not _NO_BUILTIN:
                return self._to_cv(_res(b))
            gfn = surf.symbol(nm)
            if gfn is not UNRESOLVED and callable(gfn):
                r = self._call_host(gfn, arg_vs)
                return self._to_cv(_res(r))
            if UNRESOLVED in arg_vs:
                return self._to_cv(UNRESOLVED)
            r = surf.call(nm, arg_vs)
            return self._to_cv(_res(r))
        def call_value(ctx, fn, args, argc):
            # a[3](x) / a local closure: fn is a resolved callable value (a host
            # lupa closure via a handle, or an interp closure). Invoke it.
            f = self._from_cv(fn)
            arg_vs = self._args(args, argc)
            if f is None or not callable(f):
                return self._to_cv(UNRESOLVED)
            r = self._call_host(f, arg_vs)
            return self._to_cv(_res(r))
        def index(ctx, base, key):
            r = surf.index(self._from_cv(base), self._from_cv(key))
            return self._to_cv(_res(r))
        def set_index(ctx, base, key, value):
            surf.set_index(self._from_cv(base), self._from_cv(key), self._from_cv(value))
        def length(ctx, base):
            # #host_table / table.getn: the host table is a _LuaTable; use Lua #.
            t = self._from_cv(base)
            try:
                return len(t) if t is not None else -1
            except Exception:
                return -1
        def table_insert(ctx, base, v):
            t = self._from_cv(base)
            if t is not None:
                try:
                    # Lua table.insert(t, v): append at #t+1
                    t[len(t) + 1] = self._from_cv(v)
                except Exception:
                    pass
        def iter_setup(ctx, exprs, n):
            # exprs = [mode, table]: mode 0=ipairs, 1=pairs. A SNAPSHOTTED table
            # (arena TABLE CValue) iterates directly over the arena, yielding
            # (index, arena-row) pairs already as CValues - so v[j] inside the
            # loop hits the arena INDEX fast path instead of a lupa row crossing.
            # Otherwise fall back to the host iteration (surface.iter_table),
            # matching frame_eval._iter_pairs.
            table_cv = exprs[1]
            if lib.cbody_is_table(table_cv):
                it = self._arena_rows(table_cv)
            else:
                mode = int(self._from_cv(exprs[0]))
                rows = _iter_rows(surf, mode, self._from_cv(table_cv))
                it = ([self._to_cv(c) for c in row] for row in rows)
            h = self._handle_next; self._handle_next += 1
            self._handles[h] = it
            return h
        def iter_next(ctx, it_id, vars_out, nvars):
            it = self._handles.get(it_id)
            try:
                row = next(it)
            except StopIteration:
                return 0
            for i in range(nvars):
                vars_out[i] = row[i] if i < len(row) else nil_cv
            return 1
        def fallback(ctx, node_id, frame, nslots):
            node = self._nodes[node_id]
            v = self._fallback_run(node)
            return self._to_cv(v)
        def aborted(ctx):
            if hasattr(surf, 'aborted'):
                return 1 if surf.aborted() else 0
            return 0

        # keep alive
        self._cb = (
            _SYMBOL(symbol), _GGET(gget), _GSET(gset), _GETTER(getter),
            _POKE(poke), _CALL(call), _CALLVAL(call_value),
            _INDEX(index), _SETINDEX(set_index),
            _LENGTH(length), _TINSERT(table_insert),
            _ITERSET(iter_setup), _ITERNEXT(iter_next), _FALLBACK(fallback),
            _ABORTED(aborted),
        )
        ptrs = [ctypes.cast(c, ctypes.c_void_p) for c in self._cb]
        self._lib.cbody_set_frontier(self._b, None, *ptrs)

    def _arena_rows(self, table_cv):
        """Iterate a snapshotted (arena) table as (index_cv, element_cv) pairs,
        both CValues, so iter_next writes them without marshalling and any v[j]
        on an element row stays in the C INDEX fast path. A snapshotted table is
        a dense pure array (no holes), so ipairs and pairs agree on the 1..n
        run - hence one iterator regardless of mode."""
        lib = self._lib
        n = lib.cbody_table_len(self._b, table_cv)
        for i in range(1, n + 1):
            yield (lib.cbody_num(float(i)), lib.cbody_table_geti(self._b, table_cv, i))

    def run(self, self_table, beat=None, t=None):
        if beat is not None and t is not None:
            if not self._clock_wired:
                self._clock_wired = True
                self._lib.cbody_set_clock_ids(
                    self._b, self._name_id.get('GetSongBeat', -1),
                    self._name_id.get('GetSongTime', -1))
            self._lib.cbody_set_clock(self._b, float(beat), float(t))
        self._handles.clear(); self._obj_to_handle.clear()
        rc = self._lib.cbody_run(self._b, self._to_cv(self_table))
        return rc

    def __del__(self):
        try:
            self._lib.cbody_free(self._b)
        except Exception:
            pass


def _res(r):
    """Normalize a Surface Resolution to a plain Python value for marshalling.
    A Resolution may be a raw scalar or a wrapper; UNRESOLVED passes through."""
    return r


def _iter_rows(surface, mode, table):
    """The (key, value) rows a generic-for iterates - matches frame_eval.
    _iter_pairs. `mode` 0=ipairs, 1=pairs. A host lupa table iterates via
    surface.iter_table; a py list/dict (arena-marshalled or native) directly.
    ipairs stops at the first nil in the 1..n run; pairs yields every key."""
    if table is None or table is UNRESOLVED:
        return []
    # host lupa table (the load-set mod tables): surface.iter_table gives (k,v)
    rows = surface.iter_table(table)
    if rows is not None:
        if mode == 0:  # ipairs: contiguous 1..n prefix only
            out = []
            i = 1
            d = {k: v for k, v in rows}
            while i in d:
                out.append((i, d[i])); i += 1
            return out
        return list(rows)
    # py-native container fallback
    if isinstance(table, (list, tuple)):
        return list(enumerate(table, start=1))
    if isinstance(table, dict):
        if mode == 0:
            out = []; i = 1
            while i in table:
                out.append((i, table[i])); i += 1
            return out
        return list(table.items())
    return []
