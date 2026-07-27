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
import struct
from enum import IntEnum

from analysis.player.render.expr import ast as _ast
from analysis.player.render.expr.frame_eval import (
    UNRESOLVED, _NO_BUILTIN, _builtin_call)

_LIB = None

# The NaN-box layout, mirroring cvalue.h: a boxed non-double carries the
# sign+exponent+quiet signature in the top 13 bits, then a 3-bit tag and a
# 48-bit payload; anything else IS an IEEE-754 double. The tags must stay in
# the enum's order.
_CV_SIG = 0xFFF8000000000000
_CV_SIG_MASK = 0xFFF8000000000000
_CV_TAG_SHIFT = 48
_CV_TAG_MASK = 0x7 << _CV_TAG_SHIFT
_CV_PAYLOAD_MASK = 0x0000FFFFFFFFFFFF


class _Tag(IntEnum):
    """The 3-bit box tags. Values ARE the C enum's - an IntEnum rather than
    loose constants so `_from_cv` can match on them (a bare name in a `case`
    is a capture pattern, not a value one)."""
    NIL = 0
    FALSE = 1
    TRUE = 2
    UNRESOLVED = 3
    STR = 4
    TABLE = 5
    ACTOR = 6
    HANDLE = 7


def _boxed(tag: _Tag, payload: int = 0) -> int:
    return _CV_SIG | (tag << _CV_TAG_SHIFT) | (payload & _CV_PAYLOAD_MASK)


# Plain-int tag values for the `_from_cv` dispatch chain. Derived from `_Tag`
# so they cannot drift from it, but bound as module globals because the hot
# path compares millions of times per chart and an enum member costs an extra
# attribute load per comparison.
_TAG_NIL = int(_Tag.NIL)
_TAG_FALSE = int(_Tag.FALSE)
_TAG_TRUE = int(_Tag.TRUE)
_TAG_UNRESOLVED = int(_Tag.UNRESOLVED)
_TAG_STR = int(_Tag.STR)
_TAG_TABLE = int(_Tag.TABLE)
_TAG_HANDLE = int(_Tag.HANDLE)
_TAG_ACTOR = int(_Tag.ACTOR)

_CV_NIL = _boxed(_Tag.NIL)
_CV_FALSE = _boxed(_Tag.FALSE)
_CV_TRUE = _boxed(_Tag.TRUE)
_CV_UNRESOLVED = _boxed(_Tag.UNRESOLVED)
_CV_HANDLE_BASE = _boxed(_Tag.HANDLE)
_CV_ACTOR_BASE = _boxed(_Tag.ACTOR)

_f64_to_bits = struct.Struct('<d').pack
_bits_to_f64 = struct.Struct('<d').unpack
_u64_to_bits = struct.Struct('<Q').pack
_bits_to_u64 = struct.Struct('<Q').unpack


def _pack_f64(v: float) -> int:
    """A Python number as its IEEE-754 bit pattern - the Num CValue itself."""
    return _bits_to_u64(_f64_to_bits(v))[0]


def _unpack_f64(cv: int) -> float:
    """A Num CValue's bits back as a Python float."""
    return _bits_to_f64(_u64_to_bits(cv))[0]


def _no_actor(_actor_id):
    """`actor_value` for a surface with no actor concept - nothing can have
    produced an ACTOR CValue, so nothing can ask to resolve one."""
    return None


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
    lib.cbody_set_frontier.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 18
    lib.cbody_run.restype = ctypes.c_int
    lib.cbody_run.argtypes = [ctypes.c_void_p, u64]
    lib.cbody_str.restype = ctypes.c_char_p
    lib.cbody_str.argtypes = [ctypes.c_void_p, u64, ctypes.POINTER(ctypes.c_int)]
    lib.cbody_intern.restype = u64
    lib.cbody_intern.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.cbody_mark_stable.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.cbody_clear_stable.argtypes = [ctypes.c_void_p]
    lib.cbody_set_stable.argtypes = [ctypes.c_void_p, ctypes.c_int, u64]
    lib.cbody_actor_capacity.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.cbody_set_actor_prop.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.cbody_set_actor_tweening.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
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
_GETPROP  = ctypes.CFUNCTYPE(U64, ctypes.c_void_p, U64, ctypes.c_int)
_SETPROP  = ctypes.CFUNCTYPE(None, ctypes.c_void_p, U64, ctypes.c_int, U64)


class CompiledBodyC:
    """A ctypes-driven op-stream body over the live surface. `run(self_table)`
    executes one tick. Holds the callback closures alive (ctypes requires it)."""

    def __init__(self, program, surface, store, fallback_nodes, fallback_run,
                 interp=None, stable_names=(), prop_slots=()):
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

        # Abort flag the executor LOADS (never calls). Cleared per run; set by
        # a raising crossing when the surface models aborting, and by any host
        # crossing that RAISES - ctypes cannot carry an exception out of a
        # callback, so `_pending_exc` holds it until `run` can re-raise it.
        self._abort = ctypes.c_int(0)
        self._pending_exc: BaseException | None = None

        # handle registry: id -> python object (non-scalar values crossing out)
        self._handles: dict[int, object] = {}
        self._handle_next = 1
        self._obj_to_handle: dict[int, int] = {}
        # Stable boxes for NAMED host objects, and the entries that therefore
        # survive the per-tick registry reset. See `_out_named`.
        self._name_handles: dict[str, int] = {}
        self._pinned: dict[int, object] = {}

        # Actor identity, when the surface models it. `actor_id` classifies a
        # value crossing OUT; `actor_value` resolves an id crossing back IN.
        # A surface without them (any non-game host) leaves every site's gate
        # to switch off after warmup, costing exactly what it costs today.
        self._actor_id = getattr(surface, 'actor_id', None)
        self._actor_value = getattr(surface, 'actor_value', _no_actor)
        # Which sources the COMPILER proved can reach a method receiver. Only
        # those tag actors; every other crossing keeps emitting handles, which
        # is the existing path and always correct - just not fast. Deciding
        # this statically rather than by sampling actor density is what makes
        # probes ~= uses instead of ~2:1 against.
        self._receivers = getattr(program, 'receivers', None)
        # GET_PROP id -> the verb it was lowered from (the non-actor fallback
        # still has to answer as that verb).
        self._prop_verbs = getattr(program, 'prop_verbs', ())
        self._set_verbs = getattr(program, 'set_verbs', ())
        # Names safe to cache ACROSS ticks: the caller proved nothing rebinds
        # them mid-run. Empty means "cache nothing", which is what every
        # surface without the analysis gets.
        # Names cached ACROSS ticks. Sound because the surface reports every
        # global write by name (see `_on_global_write`), so a stale entry is
        # refreshed the moment the host changes it - including a write through
        # a computed name, which no static analysis could have predicted.
        self._stable_names = frozenset(stable_names or ())
        self._stable_ids = tuple(
            (nm, self._name_id[nm]) for nm in sorted(self._stable_names)
            if nm in self._name_id)
        self._global_gen_fn = getattr(surface, 'global_generation', None)
        self._global_gen = None

        # Seed the cross-tick cache and keep it honest: the surface tells us
        # by name whenever the host rebinds one of these.
        # Settled-actor mirror: property reads answer from the executor with
        # no crossing while the actor's tween queue is idle. `prop_slots` maps
        # a property NAME to the GET_PROP id space, so only properties the body
        # can actually read are mirrored.
        self._prop_slots = dict(prop_slots)
        install = getattr(surface, 'install_prop_mirror', None)
        if install is not None and self._prop_slots:
            self._actor_cap = 0
            install(self._on_prop_write, self._on_tween_change)

        observe = getattr(surface, 'observe_global_writes', None)
        if observe is not None and self._stable_names:
            observe(self._on_global_write)

        self._install_frontier()

    def _actor_marshal(self):
        """A crossing-OUT marshal that tags a live actor with its id.

        Identical to `_to_cv` except that a non-scalar the surface recognises as
        an actor becomes an ACTOR CValue instead of a registry handle. The probe
        sits AFTER the scalar prefix, never wrapped around `_to_cv`: most values
        crossing out are numbers, and asking the host whether a float is an
        actor is pure loss.

        Only the sites the compiler proved can reach a receiver get this - see
        `_install_frontier` and `opstream.ReceiverSources`."""
        probe = self._actor_id

        def marshal(v):
            if v is UNRESOLVED:
                return _CV_UNRESOLVED
            if v is None:
                return _CV_NIL
            if v is True:
                return _CV_TRUE
            if v is False:
                return _CV_FALSE
            if isinstance(v, (int, float)):
                return _pack_f64(v)
            if isinstance(v, str):
                e = v.encode('utf-8', 'surrogatepass')
                return self._lib.cbody_intern(self._b, e, len(e))
            rec_id = probe(v)
            if rec_id is not None:
                # int(): LuaJIT numbers are doubles, so the host gives a float.
                return _CV_ACTOR_BASE | int(rec_id)
            key = id(v)
            h = self._obj_to_handle.get(key)
            if h is None:
                h = self._handle_next; self._handle_next += 1
                self._handles[h] = v
                self._obj_to_handle[key] = h
            return _CV_HANDLE_BASE | h

        return marshal

    # -- marshalling --------------------------------------------------------
    #
    # Boxing and unboxing happen HERE, in Python integer arithmetic, not
    # through the ABI's cv_* helpers. The NaN-box layout is a documented,
    # frozen bit contract (cvalue.h, mirrored by the _CV_* constants above),
    # and asking C about it cost a ctypes round-trip PER PREDICATE - up to
    # seven of them to unbox one value, on a path that runs ~170x per tick.
    # The helpers stay exported for the ABI tests; nothing hot calls them.

    def _to_cv(self, v):
        """Python value -> CValue uint64."""
        if v is UNRESOLVED:
            return _CV_UNRESOLVED
        if v is None:
            return _CV_NIL
        if v is True:
            return _CV_TRUE
        if v is False:
            return _CV_FALSE
        if isinstance(v, (int, float)):
            return _pack_f64(v)
        if isinstance(v, str):
            e = v.encode('utf-8', 'surrogatepass')
            return self._lib.cbody_intern(self._b, e, len(e))
        # non-scalar: register a handle
        key = id(v)
        h = self._obj_to_handle.get(key)
        if h is None:
            h = self._handle_next; self._handle_next += 1
            self._handles[h] = v
            self._obj_to_handle[key] = h
        return _CV_HANDLE_BASE | h

    def _grow_actors(self, rec_id: int) -> None:
        if rec_id < self._actor_cap:
            return
        self._actor_cap = max(64, rec_id + 1, self._actor_cap * 2)
        self._lib.cbody_actor_capacity(
            self._b, self._actor_cap, len(self._prop_slots))

    def _on_prop_write(self, rec_id, prop, value) -> None:
        """A SETTLED property write. Only numbers and only mirrored properties
        reach the executor; everything else keeps crossing, which is the
        existing path."""
        slots = self._prop_slots.get(prop)
        if slots is None or value is None or isinstance(value, (str, tuple)):
            return
        self._grow_actors(rec_id)
        # One property can back SEVERAL ids (GetZoom and GetZoomX both read
        # scale_x); the executor indexes by id, so every one of them updates.
        for slot in slots:
            self._lib.cbody_set_actor_prop(self._b, rec_id, slot, float(value))

    def _on_tween_change(self, rec_id, busy) -> None:
        """The actor's tween queue became occupied or empty. While occupied,
        `get` may be interpolating rather than reading the settled value, so
        the executor must cross for that actor."""
        self._grow_actors(rec_id)
        self._lib.cbody_set_actor_tweening(self._b, rec_id, 1 if busy else 0)

    def _on_global_write(self, name) -> None:
        """The host rebound `name`: refresh its cached value immediately.

        Immediately, not at the next tick, because the write can land MID-tick
        - a command body fired from a poke's queue drain runs arbitrary chart
        Lua. Re-resolving here is a store read, not Lua execution, so it cannot
        re-enter. Writes are far rarer than the reads this saves."""
        name_id = self._name_id.get(name)
        if name_id is None or name not in self._stable_names:
            return
        surf = self._surface
        recv = self._receivers.symbols if self._receivers else ()
        cv = self._out_named(name, _res(surf.symbol(name)), name in recv)
        self._lib.cbody_set_stable(self._b, name_id, cv)

    def _out_named(self, nm, v, tag_actor):
        """Marshal a value OUT from a site that knows the NAME it came from.

        Identical to `_to_cv` for scalars. For a host OBJECT it reuses one box
        per name instead of minting a fresh handle, because lupa hands back a
        NEW wrapper on every global read - so `_obj_to_handle`'s id()-keyed memo
        never hit, and every read of GAMESTATE produced a different CValue
        (38,792 distinct ones over 2,988 ticks). Nothing downstream could then
        tell it was the same object: the executor's clock fast path arms only
        when the GetSongBeat receiver CValue REPEATS, so it had never once
        fired, and `#`/equality on a host global were likewise identity-blind.

        Keying the box on the NAME fixes that while caching NOTHING. The name is
        still resolved on every crossing and the slot is refreshed to whatever
        it resolved to, so a rebind is picked up exactly as before - only the
        box is reused, never the value. That is what makes this need no
        invalidation: a stable BOX is sound on its own, a stable VALUE is what
        would require knowing when the host rebinds."""
        if v is UNRESOLVED:
            return _CV_UNRESOLVED
        if v is None:
            return _CV_NIL
        if v is True:
            return _CV_TRUE
        if v is False:
            return _CV_FALSE
        if isinstance(v, (int, float)):
            return _pack_f64(v)
        if isinstance(v, str):
            e = v.encode('utf-8', 'surrogatepass')
            return self._lib.cbody_intern(self._b, e, len(e))
        if tag_actor and self._actor_id is not None:
            rec_id = self._actor_id(v)
            if rec_id is not None:
                return _CV_ACTOR_BASE | int(rec_id)
        h = self._name_handles.get(nm)
        if h is None:
            h = self._handle_next
            self._handle_next += 1
            self._name_handles[nm] = h
        self._handles[h] = v
        self._pinned[h] = v
        return _CV_HANDLE_BASE | h

    def _from_cv(self, cv):
        """CValue uint64 -> Python value.

        Tested in MEASURED frequency order. `match` on an int subject lowers to
        sequential comparisons, so tag-declaration order put HANDLE - the most
        common non-number tag by two orders of magnitude - sixth in the chain.
        Share of `_from_cv` calls:

            chart          NUM     HANDLE     STR    all others
            gat          59.8%      29.4%   10.3%          0.6%
            do back burn 47.1%      52.8%    0.1%         0.02%

        `do back burn` alone makes 46.5M of these calls in 25 chart-seconds, so
        where the HANDLE test sits outweighs everything below it combined. The
        tags are compared against module-level ints rather than `_Tag` members:
        an enum member costs an attribute load on top of the global load, on a
        path this hot.
        """
        if (cv & _CV_SIG_MASK) != _CV_SIG:
            return _unpack_f64(cv)
        tag = (cv & _CV_TAG_MASK) >> _CV_TAG_SHIFT
        if tag == _TAG_HANDLE:
            return self._handles.get(cv & _CV_PAYLOAD_MASK)
        if tag == _TAG_ACTOR:
            # The HOST OBJECT, never the bare id: `_actor_poke`'s SetTarget
            # branch resolves its argument with `_table_rec_id(args[0])`, the
            # surfaces route on the recorder, and `type(actor)` must stay
            # 'table'. An id leaking out here would silently unbind proxies.
            return self._actor_value(cv & _CV_PAYLOAD_MASK)
        if tag == _TAG_STR:
            ln = ctypes.c_int()
            p = self._lib.cbody_str(self._b, cv, ctypes.byref(ln))
            return p[:ln.value].decode('utf-8', 'surrogatepass')
        if tag == _TAG_NIL:
            return None
        if tag == _TAG_UNRESOLVED:
            return UNRESOLVED
        if tag == _TAG_FALSE:
            return False
        if tag == _TAG_TRUE:
            return True
        if tag == _TAG_TABLE:
            # A snapshotted load-set table crossing back to Python (type()/
            # #/ passed to a host fn): return the ORIGINAL host object so
            # its semantics match the non-snapshot path. INDEX never
            # reaches here - it resolves in C against the arena.
            return self._table_origin.get(cv & _CV_PAYLOAD_MASK)
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
        if n == 0 and depth == 0:
            # A top-level table that iterates EMPTY has nothing to freeze, and
            # may not be empty at all: a host that proxies its namespace (the
            # sandbox's `_G`) keeps the values in a backing store, so raw
            # iteration yields nothing while indexing works fine. Snapshotting
            # that produced an empty arena table, marked the name stable, and
            # every later `_G[...]` read resolved to nil - silently, since a
            # missing global is a legal nil.
            return None
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
        # The verb each GET_PROP id came from, for the non-actor fallback.
        _PROP_VERBS = self._prop_verbs
        _SET_VERBS = self._set_verbs
        stable_names = self._stable_names
        # Static receiver analysis decides which crossings tag actors. A name
        # not in these sets emits a handle exactly as before.
        recv = self._receivers
        to_cv = self._to_cv
        tag = self._actor_marshal() if self._actor_id is not None else to_cv
        recv_symbols = recv.symbols if recv is not None else frozenset()
        recv_calls = recv.calls if recv is not None else frozenset()
        recv_getters = recv.getters if recv is not None else frozenset()
        # Only the NAMED sites tag. `index`, `field`, `call_value`, `iter` and
        # `fallback` carry no name for the analysis to key on, so gating them
        # is all-or-nothing per site - and `index` is the highest-volume
        # crossing in the system (863K in 25 chart-seconds on `do back burn`,
        # containing 90 actors). Turning that on because one `v[i]:method()`
        # appears somewhere in the body is exactly the unselective coupling the
        # static analysis exists to remove; it measured +2.9% there. A value
        # from these sites crosses as a handle and takes the slow path, which
        # is correct - just not fast.
        out_index = to_cv
        out_call_value = to_cv
        out_iter = to_cv
        out_fallback = to_cv
        # The id-entry hooks the fast paths use. A surface without them never
        # produces an ACTOR CValue either, so the paths are unreachable.
        actor_method = getattr(surf, 'actor_method', None)
        actor_poke = getattr(surf, 'actor_poke', None)
        actor_prop = getattr(surf, 'actor_prop', None)
        actor_set_prop = getattr(surf, 'actor_set_prop', None)
        mirror_actor = getattr(surf, 'mirror_actor', None) \
            if self._prop_slots else None
        mirrored: set = set()

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
            cv = self._out_named(nm, _res(r), nm in recv_symbols)
            if nm in stable_names:
                # Cache it in the executor: from here the body reads it without
                # crossing, and `_on_global_write` refreshes it the moment the
                # host rebinds. First read pays, the rest are free.
                name_id = self._name_id.get(nm)
                if name_id is not None:
                    self._lib.cbody_set_stable(self._b, name_id, cv)
            return cv
        def gget(ctx, name):
            nm = name.decode()
            v = store.get(nm) if store is not None else UNRESOLVED
            return self._out_named(nm, v, nm in recv_symbols)
        def gset(ctx, name, cv):
            if store is not None:
                store.set(name.decode(), self._from_cv(cv))
        def getter(ctx, recv, verb, args, argc):
            nm = verb.decode()
            # An ACTOR recv skips _from_cv, surf.method's routing, and the lupa
            # __recorder_id index - env._actor_get takes the id directly. NOT
            # taken for GetChild: both of guard_surface.method's GetChild
            # branches mint and REGISTER a persistent child (and seed P1/P2 at
            # their engine start positions on first call), so it has to run the
            # real thing. No _sync here either - a getter deliberately reads the
            # actor at ITS last-drained time, not env._now.
            if actor_method is not None and (recv & _CV_SIG_MASK) == _CV_SIG \
                    and (recv & _CV_TAG_MASK) >> _CV_TAG_SHIFT == _TAG_ACTOR:
                if nm == 'GetShader':
                    return recv          # chains straight back, no re-marshal
                r = actor_method(recv & _CV_PAYLOAD_MASK, nm,
                                 self._args(args, argc))
                return (tag if nm in recv_getters else to_cv)(_res(r))
            r = surf.method(self._from_cv(recv), nm, self._args(args, argc))
            return (tag if nm in recv_getters else to_cv)(_res(r))
        def get_prop(ctx, recv, prop_id):
            # The compiler already resolved the verb to a property, so there is
            # no name to decode and no verb table to walk - just the read.
            if actor_prop is not None and (recv & _CV_SIG_MASK) == _CV_SIG \
                    and (recv & _CV_TAG_MASK) >> _CV_TAG_SHIFT == _TAG_ACTOR:
                rec_id = recv & _CV_PAYLOAD_MASK
                if mirror_actor is not None and rec_id not in mirrored:
                    # First hard read of this actor: from here its settled
                    # state is mirrored and the executor answers without us.
                    mirrored.add(rec_id)
                    mirror_actor(rec_id)
                v = actor_prop(rec_id, prop_id)
                return to_cv(UNRESOLVED if v is None else v)
            # Not a live actor (a singleton, a nil): answer exactly as the
            # generic getter would, through the verb this id came from.
            r = surf.method(self._from_cv(recv), _PROP_VERBS[prop_id], [])
            return to_cv(_res(r))
        def set_prop(ctx, recv, set_id, value):
            # The compiler resolved the verb to a property; nothing to decode.
            if actor_set_prop is not None and (recv & _CV_SIG_MASK) == _CV_SIG \
                    and (recv & _CV_TAG_MASK) >> _CV_TAG_SHIFT == _TAG_ACTOR:
                actor_set_prop(recv & _CV_PAYLOAD_MASK, set_id,
                               self._from_cv(value))
                return
            # Not a live actor: answer as the generic poke would, through the
            # verb this id came from.
            surf.poke(self._from_cv(recv), _SET_VERBS[set_id],
                      [self._from_cv(value)])
        def poke(ctx, recv, verb, args, argc):
            nm = verb.decode()
            if actor_poke is not None and (recv & _CV_SIG_MASK) == _CV_SIG \
                    and (recv & _CV_TAG_MASK) >> _CV_TAG_SHIFT == _TAG_ACTOR:
                actor_poke(recv & _CV_PAYLOAD_MASK, nm, self._args(args, argc))
                return
            surf.poke(self._from_cv(recv), nm, self._args(args, argc))
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
                return (tag if nm in recv_calls else to_cv)(_res(b))
            gfn = surf.symbol(nm)
            if gfn is not UNRESOLVED and callable(gfn):
                r = self._call_host(gfn, arg_vs)
                return (tag if nm in recv_calls else to_cv)(_res(r))
            if UNRESOLVED in arg_vs:
                return self._to_cv(UNRESOLVED)
            r = surf.call(nm, arg_vs)
            return (tag if nm in recv_calls else to_cv)(_res(r))
        def call_value(ctx, fn, args, argc):
            # a[3](x) / a local closure: fn is a resolved callable value (a host
            # lupa closure via a handle, or an interp closure). Invoke it.
            f = self._from_cv(fn)
            arg_vs = self._args(args, argc)
            if f is None or not callable(f):
                return self._to_cv(UNRESOLVED)
            r = self._call_host(f, arg_vs)
            return out_call_value(_res(r))
        def index(ctx, base, key):
            # base/key unbox unchanged: surf.index must keep firing the
            # recorder metatable (`P1.zoom` resolves to a closure).
            r = surf.index(self._from_cv(base), self._from_cv(key))
            return out_index(_res(r))
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
                it = ([out_iter(c) for c in row] for row in rows)
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
            return out_fallback(v)
        # Abort reporting is a FLAG the executor loads (CFrontier.abort_flag),
        # not a callback it polls. Only the four crossings that can raise -
        # getter, poke, call, call_value - have to refresh it, and only when
        # the surface models aborting at all. No surface in the tree does, so
        # the usual case installs the bare callbacks and the flag stays 0 for
        # the whole run.
        surf_aborted = getattr(surf, 'aborted', None)

        def reporting(fn):
            """Wrap a raising crossing so the host flag reflects the surface."""
            def wrapped(*args):
                result = fn(*args)
                if surf_aborted():
                    self._abort.value = 1
                return result
            return wrapped

        if surf_aborted is not None:
            getter, poke = reporting(getter), reporting(poke)
            call, call_value = reporting(call), reporting(call_value)

        def catching(fn):
            """Wrap a raising crossing so a host exception ABORTS THE TICK
            rather than escaping into ctypes.

            An exception raised inside a ctypes callback does not propagate to
            whoever called `run` - ctypes prints "Exception ignored" and hands
            C a default return, so the body kept executing on a value the Lua
            path never produced, and `_record_fault` never saw it. The Lua
            path abandons the whole body when a host call raises; the Rust
            core learned the same lesson as its `aborted` flag. Holding the
            exception and re-raising after `cbody_run` returns puts it back on
            the caller's `except`, and the abort flag stops the tick where Lua
            stops it."""
            def wrapped(*args):
                try:
                    return fn(*args)
                except BaseException as exc:      # noqa: BLE001 - re-raised
                    if self._pending_exc is None:
                        self._pending_exc = exc
                    self._abort.value = 1
                    return nil_cv
            return wrapped

        getter, poke = catching(getter), catching(poke)
        call, call_value = catching(call), catching(call_value)

        # keep alive
        self._cb = (
            _SYMBOL(symbol), _GGET(gget), _GSET(gset), _GETTER(getter),
            _POKE(poke), _CALL(call), _CALLVAL(call_value),
            _INDEX(index), _SETINDEX(set_index),
            _LENGTH(length), _TINSERT(table_insert),
            _ITERSET(iter_setup), _ITERNEXT(iter_next), _FALLBACK(fallback),
            _GETPROP(get_prop), _SETPROP(set_prop),
        )
        ptrs = [ctypes.cast(c, ctypes.c_void_p) for c in self._cb]
        self._lib.cbody_set_frontier(
            self._b, None, *ptrs,
            ctypes.cast(ctypes.byref(self._abort), ctypes.c_void_p))

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

    def run(self, self_table, beat=None, t=None, self_rec_id=None):
        if beat is not None and t is not None:
            if not self._clock_wired:
                self._clock_wired = True
                self._lib.cbody_set_clock_ids(
                    self._b, self._name_id.get('GetSongBeat', -1),
                    self._name_id.get('GetSongTime', -1))
            self._lib.cbody_set_clock(self._b, float(beat), float(t))
        # Pinned name boxes survive the reset - that is the point of them.
        self._handles = dict(self._pinned)
        self._obj_to_handle.clear()
        self._abort.value = 0
        self._pending_exc = None
        # `self` is ALWAYS an actor and the caller already knows which, so it
        # never needs the classifier - it boxes straight from the id.
        me = (self._to_cv(self_table) if self_rec_id is None
              else _CV_ACTOR_BASE | self_rec_id)
        result = self._lib.cbody_run(self._b, me)
        pending, self._pending_exc = self._pending_exc, None
        if pending is not None:
            raise pending
        return result

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
