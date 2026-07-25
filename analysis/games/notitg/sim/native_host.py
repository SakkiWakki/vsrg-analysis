"""Native (lupa-free) execution host for the NotITG load pass + Update bodies.

This runs the chart's Lua chunks - Conditions, @expr attributes, InitCommand/
OnCommand bodies, classic-command arg exprs, %expr closures, and the per-frame
Update body - through the in-house parser + AST interpreter against
`NotitgGuardSurface`, instead of compiling them as lupa chunks.

Transition strategy: during the migration the globals namespace stays the SHARED
lupa env (`_LuaEnvStore` bridges the interpreter's GlobalStore to `host.env`),
so a chunk that has moved to native execution and one that still runs as lupa
read and write ONE namespace - no divergence, no sync. When every consumer is
native (M2) the store swaps to a plain native GlobalStore and the lupa env (and
LuaHost) is dropped.

One persistent Interpreter is held so a closure captured at load (`%expr`
command payloads) can be re-invoked later, and so parsing is done once per body.
"""
from __future__ import annotations

from analysis.games.notitg.guard_surface import NotitgGuardSurface
from analysis.games.notitg.sim.compiled_body import _LuaEnvStore
from analysis.player.render.expr import ast as _ast
from analysis.player.render.expr.frame_eval import (
    Interpreter, Scope, _Return)
from analysis.player.render.expr.parser import parse_body


class NativeHost:
    """Owns the interpreter + surface + shared-namespace store for one
    SimEnvironment. `eval_expr` returns a Lua expression's value; `run_chunk`
    executes statements for effect; both bind `self` to the acting actor's
    recorder and swallow faults into a returned (value, error) the caller logs
    like the lupa path did."""

    def __init__(self, env):
        self._env = env
        self._surface = NotitgGuardSurface(env)
        self._store = _LuaEnvStore(env._host.env)
        self._interp = Interpreter(self._surface, store=self._store)
        # Parsed-AST cache keyed by source text - a body/expr parses once even
        # when re-run (per-tick Update, re-expanded includes).
        self._cache: dict = {}

    @property
    def interp(self) -> Interpreter:
        return self._interp

    @property
    def store(self):
        return self._store

    def _parse(self, source: str):
        stmts = self._cache.get(source)
        if stmts is None:
            stmts, _sink = parse_body(source)
            self._cache[source] = stmts
        return stmts

    def _root(self, self_recorder) -> Scope:
        root = Scope(store=self._store)
        root.bindings['self'] = self_recorder
        return root

    def eval_expr(self, expr: str, self_recorder=None):
        """Evaluate `expr` (a Lua expression) and return its value, or a
        fault. Returns (value, error): error is None on success, else the
        exception string; value is None on fault.

        A nil result is None; a result the surface could not answer is
        UNRESOLVED, and callers must NOT read that as nil. The interpreter
        returns UNRESOLVED where lupa raised (calling a name the surface does
        not model used to be "attempt to call a nil value"), so an unresolved
        value belongs on the caller's FAULT path, not its falsy path - see
        `SimEnvironment._condition_falsy`, where reading it as nil silently
        dropped the actor and its whole subtree."""
        try:
            stmts = self._parse(f'return ({expr})')
        except Exception as exc:
            return None, str(exc)
        root = self._root(self_recorder)
        try:
            self._interp.run(stmts, root)
        except _Return as ret:
            return (ret.args[0][0] if ret.args and ret.args[0] else None), None
        except Exception as exc:
            return None, str(exc)
        return None, None

    def run_chunk(self, source: str, self_recorder=None):
        """Execute `source` (Lua statements) for effect. Returns (ok, error):
        ok True on clean run, else False with the exception string. A bare
        `return` inside is honoured (ignored value)."""
        try:
            stmts = self._parse(source)
        except Exception as exc:
            return False, str(exc)
        root = self._root(self_recorder)
        try:
            self._interp.run(stmts, root)
        except _Return:
            pass
        except Exception as exc:
            return False, str(exc)
        return True, None
