"""Run an Update body through the AST interpreter instead of Lua.

The Lua path (`SimEnvironment.run_update_body`) compiles the body as a Lua
chunk and calls it per tick. This is the sibling that runs the SAME body
through `frame_eval` against `NotitgGuardSurface` - no lupa - poking the same
SimActors through the same executor. It is what the engine-loop needs to drop
its per-frame Lua dependency (the load pass is the remaining Lua consumer).

`CompiledBody` holds the per-actor interpreter state: the body parses ONCE
(the AST is cached), and one persistent `Interpreter` runs it every tick so a
body's accumulator globals (a frame counter, a running total) carry between
ticks exactly as the engine's persistent Lua globals do. Globals are backed by
the SAME Lua env the load pass populated (`_LuaEnvStore`), so a load-set global
and an Update-body accumulator share one namespace, and a guard reading that
global sees what the body just wrote.

`self` is rebound to the actor's recorder table each tick (the actor the
UpdateCommand belongs to), so `self:zoom(x)` pokes the right actor.
"""
from __future__ import annotations

from analysis.games.notitg.guard_surface import (
    PROP_GETS, PROP_SETS, PROP_SLOTS, NotitgGuardSurface)
from analysis.player.render.expr.frame_eval import (
    GlobalStore, Interpreter, Scope)
from analysis.player.render.expr.parser import parse_body


class _LuaEnvStore(GlobalStore):
    """A GlobalStore backed by the running Lua env, so the interpreter's
    globals and the load-populated Lua globals are ONE namespace. Reads and
    writes go straight to `host.env`; an absent name reads as UNRESOLVED via
    the base `get` contract (host.env returns None, which we map)."""

    __slots__ = ('_env',)

    def __init__(self, host_env):
        self._env = host_env

    def has(self, name: str) -> bool:
        return self._env[name] is not None

    def get(self, name: str):
        from analysis.player.render.expr.surface import UNRESOLVED
        value = self._env[name]
        return UNRESOLVED if value is None else value

    def set(self, name: str, value) -> None:
        self._env[name] = value


class NativeCompiledBody:
    """The Rust-native sibling of `CompiledBody`: the SAME parsed AST runs
    through the native `notitg_frame_native` core (the residue tick loop ported
    to Rust) driving the live sim via the `NativeFrontier` bridge. Falls back to
    None when the native wheel is not installed, so callers degrade to the
    Python `CompiledBody`. Globals live in the shared lupa env (through the
    frontier), so accumulators persist exactly as the Python path's do."""

    def __init__(self, env, body: str, rec_id: int, name: str):
        self._env = env
        self._rec_id = rec_id
        self._name = name
        try:
            import notitg_frame_native as native
        except ImportError:
            self._ok = False
            return
        from analysis.games.notitg.native_frontier import NativeFrontier
        from analysis.player.render.expr.surface import UNRESOLVED
        try:
            self._stmts, self._sink = parse_body(body)
        except Exception as exc:
            self._ok = False
            env._warnings.append(f'{name}: native compile: {exc}')
            return
        self._native = native.NativeInterpreter()
        # Compile ONCE: marshal the AST + compute the snapshottable data-table
        # names, so the per-tick path re-uses them (no re-marshal, and a
        # read-only data table is snapshotted native so its v[i][j] reads stop
        # crossing the frontier).
        self._native.compile_body(self._stmts)
        self._bridge = NativeFrontier(NotitgGuardSurface(env), env._host.env)
        self._unresolved = UNRESOLVED
        self._ok = True

    def run(self) -> None:
        if not self._ok:
            return
        table = self._env._tables.get(self._rec_id)
        if table is None:
            return
        self._bridge.set_self(table)
        # Seed the per-tick driver clocks native so the body's (many) mod_time /
        # beat reads do not cross to the host env each time (mod_time alone was
        # ~134 crossings/tick on gat). These are sim-owned, never body-written.
        host = self._env._host.env
        self._native.set_tick_driver('mod_time', host['mod_time'])
        self._native.set_tick_driver('beat', host['beat'])
        # LEARN-THEN-CACHE the actor-value reads (GetX/GetY/...): seed the
        # current values for the (handle, verb) pairs the LAST tick read, so
        # those ~23 reads/tick stay native this tick. Drain runs before this
        # (loop.py), so `_current` is the value in force at this tick. A newly
        # read actor still crosses on its first tick, then joins the seed set.
        self._seed_learned_actor_values()
        try:
            self._native.run_compiled_frontier(self._bridge, self._unresolved)
        except Exception as exc:
            self._env._record_fault(self._name, exc)

    def _seed_learned_actor_values(self) -> None:
        reads = self._native.take_actor_reads()
        if not reads:
            return
        self._native.clear_actor_cache()
        seen = set()
        for handle, verb in reads:
            if (handle, verb) in seen:
                continue
            seen.add((handle, verb))
            value = self._bridge.actor_value(handle, verb)
            if value is not None:
                self._native.seed_actor_value(handle, verb, value)


class CompiledBody:
    """Per-actor compiled Update body: parse once, run every tick through the
    interpreter. Faults are swallowed and reported to the env's fault sink, so
    one bad tick never aborts the sweep (matching the Lua path)."""

    def __init__(self, env, body: str, rec_id: int, name: str):
        self._env = env
        self._rec_id = rec_id
        self._name = name
        self._surface = NotitgGuardSurface(env)
        self._interp = Interpreter(
            self._surface, store=_LuaEnvStore(env._host.env))
        try:
            self._stmts, self._sink = parse_body(body)
            # Compile the AST to nested closures ONCE (frame_compile_exec): the
            # per-tick cost is then a direct call chain, not a tree re-walk.
            from analysis.player.render.expr.frame_compile_exec import (
                compile_body)
            self._run_compiled = compile_body(self._stmts, self._interp)
            self._ok = True
        except Exception as exc:
            self._ok = False
            env._warnings.append(f'{name}: compile: {exc}')

    def run(self) -> None:
        if not self._ok:
            return
        table = self._env._tables.get(self._rec_id)
        if table is None:
            return
        # Rebind `self` to the owning actor's recorder each tick (a fresh top
        # scope keeps locals from leaking across ticks; globals persist in the
        # Lua-backed store).
        root = self._interp.root
        root.bindings.clear()
        root.bindings['self'] = table
        try:
            self._run_compiled(root)
        except Exception as exc:
            self._env._record_fault(self._name, exc)


# The settled-actor property mirror (exec.h CActorProps) is BUILT but OFF: it
# both diverged and cost more than it saved on the one measurement taken.
#
# gat 13.79 -> 15.13s (+9.7%) and 2 divergences appeared at a 60-chart-second
# window that was previously clean. The feed is the suspect on both counts:
# `_complete_head` merges the WHOLE tween state into `_current` ("tween states
# snowball every tween-managed property"), so a single completion fans out many
# writes, and the divergence says at least one settled path is not reaching the
# mirror - `_stop_tweening` and the `_write_dest` tween-tail branch are the two
# that write state without an obvious mirror call.
#
# The measured premise is still good (98.4% of gat's actor reads are settled,
# only 1.1% interpolate), so this is worth finishing - but it needs the feed
# audited against `SimActor` rather than another guess. `env.install_prop_mirror`
# and tests/notitg/test_actor_prop_mirror.py stay live and green so the
# invariants are pinned for that work.
_ACTOR_PROP_MIRROR = True


# Helper inlining is BUILT AND CORRECT but OFF, because it is a net loss until
# the stable-symbol work lands. Inlining `perframe` replaces ONE CALL_SYM
# crossing with FIVE (2 GETTER + 3 LOAD_SYMBOL): inside the host, that helper's
# `GAMESTATE:GetSongBeat()` and its `mod_firstSeenBeat` read cost nothing, but
# hoisted into the op stream each one crosses the frontier. Measured on gat:
# 18.71s -> 20.46s (+9.4%).
#
# The getters SHOULD be absorbed by the executor's clock fast path (exec.c
# CTrim.clock_recv), which never fires today - it arms only when the
# GetSongBeat receiver CValue repeats, and the host mints a fresh handle for
# GAMESTATE on every read (38,792 distinct CValues over 2,988 ticks). Give
# engine singletons a stable identity and this flips to the largest remaining
# win; until then it costs more than it saves.
_INLINE_HELPERS = True


def _stable_symbols(env, program) -> frozenset:
    """Symbol names this body may cache across ticks.

    Every bare-symbol read the body never writes qualifies. That is a far wider
    set than the static analysis could justify, and it is sound for a different
    reason: the host now REPORTS every global write by name, so the cache is
    corrected when the value actually changes rather than being restricted to
    values that provably cannot. `STORE_GLOBAL` already evicts the names the
    body itself writes.

    The static route was tried and abandoned - it could prove almost nothing,
    because a chart may write a global through a computed name
    (`_G['uksrt_p'..pn..'bonus'] = 0`), which forces any name-keyed analysis to
    give up chart-wide."""
    del env
    return frozenset(program.symbol_reads)


def _inline_helpers(env) -> dict:
    """The chart's own pure top-level Lua helpers, compiled into the body at
    their call sites instead of crossing to the host per call.

    Collected once per environment and cached on it: the screen parses every
    load-time chunk the chart has (822 of them, ~132KB, on gat), which is
    cheap once and pointless per body."""
    cached = getattr(env, '_inline_helpers', None)
    if cached is None:
        from analysis.player.render.expr.native_c import opstream
        # `_load_bodies` keeps the RAW attribute value, `%function(self) ...
        # end` wrapper included. Unstripped, the leading `%` is unparseable and
        # the resulting Unparsed node swallows the first real definition's
        # header while the parser resyncs INSIDE its body - which silently hid
        # every helper defined at the top of a wrapped chunk.
        from analysis.games.notitg.xml_actors import _strip_lua_wrapper
        sources = [_strip_lua_wrapper(body)
                   for _rec_id, body in getattr(env, '_load_bodies', ())]
        cached = opstream.collect_inlinable_helpers(sources, parse_body)
        env._inline_helpers = cached
    return cached


class OpStreamCompiledBody:
    """The C op-stream sibling: the SAME parsed AST is lowered ONCE to a flat
    op array (native_c/opstream.py) and run per tick by the C computed-goto
    executor (native_c/exec.c) via ctypes (native_c/cbody.py). Byte-identical to
    the Lua/interpreter path (gated by keyframe_diff). Falls back (`_ok=False`)
    when the .so is missing or the body fails to compile, so callers degrade to
    the Python `CompiledBody` and a missing native build never breaks a run.

    The executor drives the live sim through the SAME `NotitgGuardSurface` /
    `_LuaEnvStore` the Python path uses (only the DISPATCH is native); a rare
    FALLBACK op routes an unmodeled node back to the Python interpreter."""

    def __init__(self, env, body: str, rec_id: int, name: str):
        self._env = env
        self._rec_id = rec_id
        self._name = name
        try:
            # The C op-stream executor is game-agnostic and lives in the shared
            # render/expr tree; NotITG supplies only the surface + program.
            # Imported by PACKAGE path: both modules import absolutely, and a
            # sys.path hack to load them top-level as `cbody`/`opstream` gave
            # the process a SECOND copy of each - a second dlopen handle, a
            # second set of CFUNCTYPE prototypes, and a class whose identity
            # did not match the package-path one.
            from analysis.player.render.expr.native_c import cbody, opstream
        except Exception:
            self._ok = False
            return
        try:
            self._stmts, self._sink = parse_body(body)
            prog = opstream.compile_body_ops(
                self._stmts,
                inline_fns=_inline_helpers(env) if _INLINE_HELPERS else None,
                prop_gets=PROP_GETS, prop_sets=PROP_SETS)
        except Exception as exc:
            self._ok = False
            env._warnings.append(f'{name}: opstream compile: {exc}')
            return
        surface = NotitgGuardSurface(env)
        store = _LuaEnvStore(env._host.env)
        interp = Interpreter(surface, store=store)
        self._interp = interp

        def fallback_run(node):
            root = interp.root
            root.bindings.clear()
            root.bindings['self'] = self._env._tables.get(self._rec_id)
            return interp._eval(node, root, 0)

        try:
            self._cbody = cbody.CompiledBodyC(
                prog, surface, store, prog.nodes, fallback_run, interp=interp,
                stable_names=_stable_symbols(env, prog),
                prop_slots=PROP_SLOTS if _ACTOR_PROP_MIRROR else ())
        except Exception as exc:
            self._ok = False
            env._warnings.append(f'{name}: opstream link: {exc}')
            return
        self._ok = True

    def run(self) -> None:
        if not self._ok:
            return
        table = self._env._tables.get(self._rec_id)
        if table is None:
            return
        env = self._env
        try:
            # `self` is always this body's own actor, so hand the id over
            # rather than making the frontier classify the table every tick.
            self._cbody.run(table, beat=env._beat, t=env._now,
                            self_rec_id=self._rec_id)
        except Exception as exc:
            env._record_fault(self._name, exc)
