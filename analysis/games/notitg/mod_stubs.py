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

from analysis.games.notitg.recording_actor import RecordingActor
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

-- A recording actor: every method call (`a:x(100)`, `a:linear(1)`) is
-- routed to Python via __actor_poke and returns the table so SM's
-- chained `a:linear(1):x(0)` keeps working. `id` ties it to a Python
-- RecordingActor; the table is what an InitCommand self-assigns to a
-- global, so later closures poking that global hit the same recorder.
function __make_recorder(id)
    local t = {__recorder_id = id}
    setmetatable(t, {__index = function(_, key)
        return function(_self, ...)
            __actor_poke(id, key, ...)
            return t
        end
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

    def __init__(self, start_beat: float = 0.0, to_seconds=None):
        self._start_beat = float(start_beat)
        self._clock_beat = float(start_beat)
        self._to_seconds = to_seconds or (lambda beat: float(beat))
        self._shader_flags: list = []
        self._applied_mods: list = []
        self._swallowed = 0
        self._recorders: dict = {}
        self._next_recorder_id = 0
        self._host = LuaHost(dialect='luajit21')
        self._host.run(_PERMISSIVE_BOOTSTRAP, name='permissive-bootstrap')
        self._install()

    def run(self, source: str, name: str) -> None:
        self._host.run(source, name=name)

    def run_actor_chunk(self, source: str, name: str) -> None:
        """Run one actor command chunk with a FRESH recording `self`
        bound in the sandbox, so `self:x(100)` records and a trailing
        `NAME = self` binds a global to that recorder. The recorder's
        local clock starts at the chart's load moment (start beat).

        `self` is restored to permissive afterwards so it never lingers
        as a stray recorder global: only the names the chunk assigned
        (`gat_g_rot_intro = self`) keep the recorder alive."""
        recorder_table = self._new_recorder(self._to_seconds(self._start_beat))
        self._host.env['self'] = recorder_table
        self._host.run(source, name=name)
        self._host.env['self'] = self._host.env['__permissive']()

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

    # -- recording actors -------------------------------------------------

    def _new_recorder(self, clock_seconds: float):
        """Create a Python RecordingActor plus its Lua-side table, wired
        by id; returns the Lua table (to bind as `self` / a global)."""
        rec_id = self._next_recorder_id
        self._next_recorder_id += 1
        self._recorders[rec_id] = RecordingActor(clock=clock_seconds)
        return self._host.env['__make_recorder'](rec_id)

    def _actor_poke(self, rec_id, verb=None, *args) -> None:
        recorder = self._recorders.get(_to_int(rec_id))
        if recorder is not None and isinstance(verb, str):
            recorder.poke(verb, list(args))

    def named_actor_keyframes(self) -> dict:
        """global name -> {property: [Keyframe]} for every actor a chunk
        self-assigned to a Lua global (the closures' poke targets). Only
        globals bound to one of our recorder tables are reported; an
        actor with no recorded pokes is dropped."""
        out = {}
        for name in self._recorder_global_names():
            table = self._host.env[name]
            recorder = self._recorder_for_table(table)
            if recorder is None:
                continue
            keyframes = recorder.keyframes()
            if keyframes:
                out[name] = keyframes
        return out

    def _recorder_global_names(self):
        """Sandbox globals whose value is one of our recorder tables."""
        names = []
        for key, value in self._host.env.items():
            if isinstance(key, str) and self._recorder_for_table(value) \
                    is not None:
                names.append(key)
        return names

    def _recorder_for_table(self, table):
        if not hasattr(table, '__getitem__'):
            return None
        try:
            rec_id = table['__recorder_id']
        except (KeyError, TypeError):
            return None
        return self._recorders.get(_to_int(rec_id)) if rec_id is not None \
            else None

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
            beat = _beat_of(row)
            self._clock_beat = beat
            self._reset_recorder_clocks(self._to_seconds(beat))
            fired += 1
            try:
                payload()
            except Exception:
                failed += 1
        self._clock_beat = self._start_beat
        return fired, failed

    def _reset_recorder_clocks(self, seconds: float) -> None:
        """Point every recorder's local command clock at a fire time.
        A closure's pokes chain forward from here (each closure is a
        freshly scheduled command stream)."""
        for recorder in self._recorders.values():
            recorder.reset_clock(seconds)

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
        # lets creation-time self:method() pokes no-op instead of faulting;
        # run_actor_chunk swaps in a recording self per actor.
        host.run('_G.self = __permissive()', name='self-stub')

        host.expose('__actor_poke', self._actor_poke)
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
