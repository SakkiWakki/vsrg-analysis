"""Stub engine environment for running NotITG modfile CODE chunks.

The classic template's InitCommand Lua expects the StepMania actor
runtime: `GAMESTATE`, `PREFSMAN`, `SCREENMAN` singletons plus screen
metrics and a handful of message helpers. At LOAD time (the moment we
record) the chart is at beat 0 and no frames have run, so most of this
surface only needs to exist and return benign values; the real work
those chunks do is filling data tables (`mods`, `mods2`, `mod_actions`)
and poking shader flags, which we intercept.

Every singleton is one permissive table: unknown methods resolve to a
dummy that swallows any call and returns another permissive value, so a
chunk touching an engine method we did not anticipate degrades to a
no-op instead of aborting the harvest. The methods that carry semantic
weight at load (GetSongBeat -> 0, SetShaderFlag* -> record) are named
explicitly; everything else is caught by the metatable fallback.
"""
from __future__ import annotations

from analysis.player.render.lua import LuaHost

# A metatable that makes any missing key return a callable/indexable
# dummy. Colon-calls on our python tables pass the table as arg 1, so
# the dummy ignores every argument. Chained access (A:B():C()) and field
# reads (A.x) both land back on a permissive value.
_PERMISSIVE_BOOTSTRAP = """
local function permissive()
    local t = {}
    local mt = {}
    mt.__index = function(_, _key) return permissive() end
    mt.__call = function(_, ...) return permissive() end
    setmetatable(t, mt)
    return t
end
_G.__permissive = permissive

function __make_singleton(overrides)
    local t = overrides or {}
    setmetatable(t, {__index = function(_, _key)
        return function(...) return permissive() end
    end})
    return t
end

-- NotITG embeds Lua 5.0; these live under the LuaJIT (5.1) runtime as
-- their renamed forms. The template calls the 5.0 names.
if math.mod == nil then math.mod = math.fmod end
if table.getn == nil then table.getn = function(t) return #t end end
"""


class StubEnvironment:
    """One LuaJIT host with the classic-template engine stubs installed,
    plus the harvested tables after chunks run."""

    def __init__(self, start_beat: float = 0.0):
        self._start_beat = float(start_beat)
        self._clock_beat = float(start_beat)
        self._shader_flags: list = []
        self._applied_mods: list = []
        self._swallowed = 0
        self._host = LuaHost(dialect='luajit21')
        self._host.run(_PERMISSIVE_BOOTSTRAP, name='permissive-bootstrap')
        self._install()

    def run(self, source: str, name: str) -> None:
        self._host.run(source, name=name)

    # -- harvest ----------------------------------------------------------

    @property
    def mods(self) -> list:
        return self._read_table('mods')

    @property
    def mods2(self) -> list:
        return self._read_table('mods2')

    @property
    def mod_actions(self) -> list:
        return self._read_table('mod_actions')

    @property
    def shader_flags(self) -> list:
        return list(self._shader_flags)

    @property
    def applied_mods(self) -> list:
        """`ApplyGameCommand('mod,<string>')` recordings as
        (beat, modstring, player) tuples; player is None when the call
        omitted the pn argument (applies to everyone)."""
        return list(self._applied_mods)

    @property
    def swallowed(self) -> int:
        """Count of unstubbed engine calls a replayed closure made that
        the permissive fallback absorbed (best-effort; only calls routed
        through the counting proxy are tallied)."""
        return self._swallowed

    def replay_mod_actions(self):
        """Fire every `mod_actions` closure ONCE, in beat order, exactly
        as the template's per-frame reader does (curaction advances
        monotonically). Each closure runs with the engine clock set to
        its own beat so any `GetSongBeat`-driven branch sees the fire
        moment. String payloads are the template's MESSAGEMAN broadcasts
        (named-command triggers we do not model) and are skipped; every
        closure runs under its own try/except so one faulting closure
        cannot abort the replay.

        Returns (fired, failed): counts of closures executed and of
        closures that raised."""
        rows = self.mod_actions
        ordered = sorted(
            enumerate(rows),
            key=lambda pair: (_beat_of(pair[1]), pair[0]))
        fired = 0
        failed = 0
        for _order, row in ordered:
            payload = row.get(2) if isinstance(row, dict) else None
            if not callable(payload):
                continue
            self._clock_beat = _beat_of(row)
            fired += 1
            try:
                payload()
            except Exception:
                failed += 1
        self._clock_beat = self._start_beat
        return fired, failed

    def _read_table(self, name: str) -> list:
        table = self._host.env[name]
        if table is None:
            return []
        rows = []
        index = 1
        while True:
            row = table[index]
            if row is None:
                break
            rows.append(self._row_to_dict(row))
            index += 1
        return rows

    def _row_to_dict(self, row):
        if not self._is_lua_table(row):
            return row
        out = {}
        for key in (1, 2, 3, 4, 5):
            value = row[key]
            if value is not None:
                out[key] = value
        return out

    def _is_lua_table(self, value) -> bool:
        return hasattr(value, '__getitem__') and not isinstance(
            value, (str, bytes))

    # -- engine surface ---------------------------------------------------

    def _install(self) -> None:
        host = self._host
        singleton = host.env['__make_singleton']

        gamestate = singleton(host.to_lua({
            'GetSongBeat': lambda _self: self._clock_beat,
            'GetSongBeatNoOffset': lambda _self: self._clock_beat,
            'SetShaderFlag': self._set_shader_flag,
            'SetShaderFlagNum': self._set_shader_flag_num,
            'ApplyGameCommand': self._apply_game_command,
            'ApplyModifiers': self._apply_modifiers,
        }))
        host.expose('GAMESTATE', gamestate)
        host.expose('PREFSMAN', singleton(host.to_lua({
            'GetPreference': lambda _self, _key=None: '',
        })))
        host.expose('SCREENMAN', singleton(host.to_lua({
            'SystemMessage': lambda _self, *_a: None,
        })))
        host.expose('DISPLAY', singleton(host.to_lua({
            'GetDisplayWidth': lambda _self: 640.0,
            'GetDisplayHeight': lambda _self: 480.0,
            'GetVendor': lambda _self: '',
        })))
        for name in ('MESSAGEMAN', 'STATSMAN', 'SONGMAN', 'THEME',
                     'GAMEMAN', 'NOTESKIN', 'INPUTFILTER', 'PROFILEMAN'):
            host.expose(name, singleton(None))

        host.expose('SCREEN_WIDTH', 640.0)
        host.expose('SCREEN_HEIGHT', 480.0)
        host.expose('SCREEN_CENTER_X', 320.0)
        host.expose('SCREEN_CENTER_Y', 240.0)
        host.expose('SCREEN_LEFT', 0.0)
        host.expose('SCREEN_RIGHT', 640.0)
        host.expose('SCREEN_TOP', 0.0)
        host.expose('SCREEN_BOTTOM', 480.0)
        host.expose('FUCK_EXE', True)

        # The InitCommand/OnCommand wrapper's `self` (the actor) becomes a
        # free global once we strip `function(self)`. A permissive actor
        # lets creation-time self:method() pokes no-op instead of faulting.
        host.run('_G.self = __permissive()', name='self-stub')

        host.expose('Trace', lambda *_a: None)
        host.expose('print', lambda *_a: None)

    def _set_shader_flag(self, _self, key=None, *_a) -> None:
        self._shader_flags.append((self._clock_beat, key, None))

    def _set_shader_flag_num(self, _self, key=None, which=None, *_a) -> None:
        self._shader_flags.append((self._clock_beat, key, which))

    def _apply_game_command(self, _self, command=None, pn=None) -> None:
        """`GAMESTATE:ApplyGameCommand('mod,<modstring>', pn?)`. Only the
        'mod,' family carries a modstring we compile; other game commands
        (screen/preference pokes) are swallowed."""
        if not isinstance(command, str):
            self._swallowed += 1
            return
        head, _sep, rest = command.partition(',')
        if head.strip().lower() != 'mod':
            self._swallowed += 1
            return
        self._applied_mods.append((self._clock_beat, rest.strip(),
                                   _to_int(pn)))

    def _apply_modifiers(self, _self, modstring=None, *_a) -> None:
        """`GAMESTATE:ApplyModifiers('<modstring>')` (the raw form). Same
        one-shot recording as the 'mod,' game command."""
        if isinstance(modstring, str):
            self._applied_mods.append((self._clock_beat, modstring.strip(),
                                       None))
        else:
            self._swallowed += 1


def _beat_of(row) -> float:
    value = row.get(1) if isinstance(row, dict) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
