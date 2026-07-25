"""Native actor recorder objects - the lupa-free replacement for the recorder
tables the permissive bootstrap built with a metatable (`lua_api.py`
`__make_recorder`/`__make_screen_recorder`).

An actor, to chart Lua, is a table `{__recorder_id = id}` whose metatable
`__index` turns any verb key into a closure: a GETTER verb reads the live actor
(`__actor_get`), a COMMAND verb schedules/runs (`__actor_command`), anything
else pokes (`__actor_poke`) and returns the recorder for chaining
(`a:linear(0.4):x(200)`). That dispatch is fixed and keyed on the
GETTER_NAMES/COMMAND_NAMES registries - not arbitrary metatable behaviour - so
it is expressed directly here as a Python object, no lupa involved.

Two call paths reach a recorder and both work:
- lupa-executed chart code (load InitCommands, until they migrate) does
  `actor:verb(args)`: lupa passes the recorder through as an opaque object and
  colon-indexes it, hitting `__getitem__(verb)` -> the closure below. The
  closure receives the recorder as its implicit first arg (`recv, *args`).
- our AST interpreter routes `recv:verb(args)` through
  `NotitgGuardSurface.method`/`poke`, which call `env._actor_get`/`_actor_poke`
  directly and never touch `__getitem__`.

`__recorder_id` is a real stored key (not dispatched), so `_table_rec_id`'s
`recorder['__recorder_id']` keeps working across both paths.
"""
from __future__ import annotations

from analysis.games.notitg.lua_api import COMMAND_NAMES, SIM_GETTER_NAMES

# The SIM recorder recognizes the broader SIM_GETTER_NAMES (GetSecsIntoEffect,
# GetText, ... - the verbs a chart reads a real value from), matching the
# `_SIM_BOOTSTRAP` __make_recorder that this replaces. The narrower GETTER_NAMES
# is the permissive (non-sim) set and would mis-route sim getters to a poke.
_GETTERS = frozenset(SIM_GETTER_NAMES)
_COMMANDS = frozenset(COMMAND_NAMES)


class Recorder:
    """A live actor as chart Lua sees it. `env` supplies the dispatch callbacks
    (`_actor_get`/`_actor_command`/`_actor_poke`); `rec_id` identifies the
    SimActor. Verb access returns a bound closure mirroring the old metatable."""

    __slots__ = ('_env', '_rec_id')

    # A recorder has __getitem__ (verb dispatch) but is NOT a data table; the
    # shared LuaHostSurface base checks this marker so index/set_index/iter
    # never treat it as one. (Class attr - compatible with __slots__.)
    _vsrg_not_a_table = True

    def __init__(self, env, rec_id: int):
        self._env = env
        self._rec_id = rec_id

    def __getitem__(self, key):
        if key == '__recorder_id':
            return self._rec_id
        if key == 'GetChild':
            return self._get_child
        if key in _GETTERS:
            return self._getter(key)
        if key in _COMMANDS:
            return self._command(key)
        return self._poke(key)

    def _get_child(self, _recv, name=None):
        # The sim recorder resolves GetChild through Python to the XML/synthetic
        # child (the _SIM_BOOTSTRAP recorder does the same), NOT the permissive
        # poke-returns-parent path.
        return self._env._actor_get_child(self._rec_id, name)

    def _getter(self, verb):
        env, rid = self._env, self._rec_id
        def getter(_recv, *args):
            return env._actor_get(rid, verb)
        return getter

    def _command(self, verb):
        env, rid = self._env, self._rec_id
        def command(recv, name=None, *args):
            env._actor_command(rid, verb, name)
            return recv
        return command

    def _poke(self, verb):
        env, rid = self._env, self._rec_id
        def poke(recv, *args):
            env._actor_poke(rid, verb, *args)
            return recv
        return poke


class ScreenRecorder(Recorder):
    """The top screen: a recorder that ALSO answers GetChild/GetTopScreen with
    real recorders (a player, or itself) instead of poking - mirroring
    `__make_screen_recorder`. Every other verb falls through to the base
    getter/command/poke dispatch."""

    def __getitem__(self, key):
        if key == 'GetChild':
            env = self._env
            def get_child(_recv, name=None):
                return env._screen_get_child(name)
            return get_child
        if key == 'GetTopScreen':
            def get_top_screen(_recv):
                return self
            return get_top_screen
        return super().__getitem__(key)
