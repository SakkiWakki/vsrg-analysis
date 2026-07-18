"""Game-agnostic sandboxed Lua execution.

Each host owns one Lua runtime and one sandbox environment. The
dialect is chosen per personality from lupa's bundled runtimes:
fluXis scripts run under NLua (Lua 5.4 semantics), NotITG/Mirin under
LuaJIT (5.1). `run` loads chunks with the sandbox as their
environment on either family (setfenv vs load-with-env).

The sandbox exposes: the safe scalar builtins, copies of the math/
string/table modules, and anything the personality passes to
`expose`. Deliberately absent: io, os, require, package, dofile,
load/loadstring, debug, collectgarbage, and lupa's python bridge
(register_eval/register_builtins are off, and only plain callables
and converted tables ever cross into Lua).
"""
from __future__ import annotations

import lupa

_SAFE_BUILTINS = (
    'assert', 'error', 'ipairs', 'next', 'pairs', 'pcall', 'select',
    'tonumber', 'tostring', 'type', 'unpack', 'xpcall',
    'setmetatable', 'getmetatable', 'rawget', 'rawset', 'rawequal',
    # 5.1 only (absent under 5.4): SM-family templates sandbox their own
    # helper functions with it (XGML's `prefix(func)` env wrapper). A
    # sandboxed chunk can only retarget environments it can reach, all
    # inside the sandbox.
    'setfenv',
)
_SAFE_MODULES = ('math', 'string', 'table')

_LOADER_FACTORY = """
function(env)
    if setfenv then
        return function(code, name)
            local f, err = loadstring(code, name)
            if not f then error(err, 0) end
            setfenv(f, env)
            return f()
        end
    end
    return function(code, name)
        local f, err = load(code, name, 't', env)
        if not f then error(err, 0) end
        return f()
    end
end
"""

# Like the loader but returns the compiled function WITHOUT running it,
# so a chunk executed many times (a per-frame driver body ticked over
# its windows) compiles once instead of per call.
_COMPILER_FACTORY = """
function(env)
    if setfenv then
        return function(code, name)
            local f, err = loadstring(code, name)
            if not f then error(err, 0) end
            setfenv(f, env)
            return f
        end
    end
    return function(code, name)
        local f, err = load(code, name, 't', env)
        if not f then error(err, 0) end
        return f
    end
end
"""


class LuaScriptError(Exception):
    pass


class LuaHost:
    def __init__(self, dialect: str = 'lua54'):
        runtime = getattr(lupa, dialect)
        # Each bundled dialect module carries its own LuaError class.
        self._lua_error = runtime.LuaError
        self._lua = runtime.LuaRuntime(
            unpack_returned_tuples=True,
            register_eval=False,
            register_builtins=False,
        )
        self._env = self._build_env()
        self._load_in_env = self._lua.eval(_LOADER_FACTORY)(self._env)
        self._compile_in_env = self._lua.eval(_COMPILER_FACTORY)(self._env)

    def _build_env(self):
        source = self._lua.globals()
        env = self._lua.table()
        for name in _SAFE_BUILTINS:
            if source[name] is not None:
                env[name] = source[name]
        for name in _SAFE_MODULES:
            env[name] = source[name]
        # 5.2+ moved unpack into table; scripts written for either
        # family expect the global.
        if env['unpack'] is None:
            env['unpack'] = source['table']['unpack']
        env['_G'] = env
        return env

    def expose(self, name: str, value) -> None:
        """Publish `value` under `name` in the sandbox. Python values
        are converted (dicts/lists become tables recursively); plain
        callables cross as-is."""
        self._env[name] = self.to_lua(value)

    def to_lua(self, value):
        if isinstance(value, dict):
            table = self._lua.table()
            for key, item in value.items():
                table[key] = self.to_lua(item)
            return table
        if isinstance(value, (list, tuple)):
            table = self._lua.table()
            for i, item in enumerate(value, start=1):
                table[i] = self.to_lua(item)
            return table
        return value

    def run(self, source: str, name: str = 'script'):
        """Execute a chunk inside the sandbox; returns its result.
        Raises LuaScriptError on compile or runtime errors (untrusted
        input is an expected failure mode, not a crash)."""
        try:
            return self._load_in_env(source, '@' + name)
        except self._lua_error as exc:
            raise LuaScriptError(str(exc)) from exc

    def compile(self, source: str, name: str = 'script'):
        """Compile a chunk in the sandbox and return the callable WITHOUT
        running it - for chunks executed many times (a per-frame driver
        body), where per-call recompilation would dominate."""
        try:
            return self._compile_in_env(source, '@' + name)
        except self._lua_error as exc:
            raise LuaScriptError(str(exc)) from exc

    def call(self, name: str, *args):
        """Call a sandbox global function; None if it isn't defined."""
        fn = self._env[name]
        if fn is None:
            return None
        try:
            return fn(*args)
        except self._lua_error as exc:
            raise LuaScriptError(str(exc)) from exc

    @property
    def env(self):
        return self._env
