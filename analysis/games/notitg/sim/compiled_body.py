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

from analysis.games.notitg.guard_surface import NotitgGuardSurface
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
        self._interp.root.bindings.clear()
        self._interp.root.set_local('self', table)
        try:
            self._interp.run(self._stmts)
        except Exception as exc:
            self._env._record_fault(self._name, exc)
