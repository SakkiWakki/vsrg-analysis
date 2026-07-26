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

# The sandbox proxy: reads and writes forward to `store`, and every write is
# reported. Values never live in the proxy itself, so an overwrite fires
# __newindex exactly as a first write does.
_PROXY_FACTORY = """
function(store, notify)
    local env = {}
    setmetatable(env, {
        __index = function(_, key) return store[key] end,
        __newindex = function(_, key, value)
            store[key] = value
            notify(key, value)
        end,
    })
    store['_G'] = env
    return env
end
"""

# `rawset(_G, k, v)` skips __newindex; this replacement keeps the raw semantics
# for any other table while still reporting a write to the sandbox.
_RAWSET_FACTORY = """
function(store, notify, env)
    return function(target, key, value)
        if target == env then
            store[key] = value
            notify(key, value)
            return target
        end
        rawset(target, key, value)
        return target
    end
end
"""

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
    def __init__(self, dialect: str = 'lua54', observe_globals: bool = False):
        runtime = getattr(lupa, dialect)
        # Each bundled dialect module carries its own LuaError class.
        self._lua_error = runtime.LuaError
        self._lua = runtime.LuaRuntime(
            unpack_returned_tuples=True,
            register_eval=False,
            register_builtins=False,
        )
        # Global-write observation (opt-in). The sandbox becomes a PROXY whose
        # values live in a separate store, so `__newindex` fires on every write
        # rather than only the first - an existing key writes straight through
        # a metatable, which is the usual way this technique is got wrong.
        # Costs ~2ns per Lua global access, measured.
        #
        # Opt-in because it changes what RAW table operations see: `#env`,
        # `next(env)` and Python-side `in`/`.items()` all read the proxy, which
        # is empty. Use `has_global`/`global_items` instead of touching `env`
        # directly. LuaHost is shared with games that do neither.
        self._observed = bool(observe_globals)
        self._store = None
        self._mirror: dict = {}
        self._on_global_write = None
        self._env = self._build_env()
        self._load_in_env = self._lua.eval(_LOADER_FACTORY)(self._env)
        self._compile_in_env = self._lua.eval(_COMPILER_FACTORY)(self._env)

    def _build_env(self):
        if self._observed:
            return self._build_observed_env()
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

    def _build_observed_env(self):
        """The sandbox as a store-backed proxy - see `__init__`."""
        source = self._lua.globals()
        store = self._lua.table()
        for name in _SAFE_BUILTINS:
            if source[name] is not None:
                store[name] = source[name]
        for name in _SAFE_MODULES:
            store[name] = source[name]
        if store['unpack'] is None:
            store['unpack'] = source['table']['unpack']
        env = self._lua.eval(_PROXY_FACTORY)(store, self._notify_global_write)
        # `rawset` would bypass __newindex entirely, so the sandbox publishes a
        # notifying one instead of the real builtin. The sandbox decides what
        # its builtins ARE, which is what makes the observation complete rather
        # than merely probable - chart Lua is arbitrary.
        store['rawset'] = self._lua.eval(_RAWSET_FACTORY)(
            store, self._notify_global_write, env)
        self._store = store
        # Seed from the store: the builtins/modules were written raw above, so
        # they never fired __newindex.
        self._mirror = {k: v for k, v in store.items() if isinstance(k, str)}
        self._mirror['_G'] = env
        return env

    def _notify_global_write(self, name, value=None) -> None:
        if not isinstance(name, str):
            return
        # The mirror is why the proxy earns its keep: a global read becomes a
        # Python dict lookup instead of a metamethod call into Lua, and the
        # write that would invalidate it hands us the new value directly.
        self._mirror[name] = value
        if self._on_global_write is not None:
            self._on_global_write(name)

    def observe_global_writes(self, callback) -> None:
        """Call `callback(name)` on every global write. Requires the host to
        have been built with `observe_globals=True`; otherwise there is nothing
        watching and this raises rather than silently never firing."""
        if not self._observed:
            raise LuaScriptError(
                'observe_global_writes needs LuaHost(observe_globals=True)')
        self._on_global_write = callback

    def global_value(self, name: str):
        """A global's current value - a dict lookup when observed, since every
        write reports itself. Falls back to indexing the env otherwise. Returns
        None for an absent name, matching Lua's nil."""
        if self._observed:
            return self._mirror.get(name)
        return self._env[name]

    def has_global(self, name: str) -> bool:
        """`name in env`, honouring the proxy. A raw `in` against an observed
        env is always False - the values live in the store."""
        if self._observed:
            return self._mirror.get(name) is not None
        return self._env[name] is not None

    def global_items(self):
        """(name, value) pairs, honouring the proxy. A raw `.items()` against
        an observed env yields nothing."""
        table = self._store if self._observed else self._env
        return table.items()

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
