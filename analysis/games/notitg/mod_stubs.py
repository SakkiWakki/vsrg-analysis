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
from analysis.games.notitg.xml_actors import (
    _strip_lua_wrapper, parse_command_string)
from analysis.player.render.lua import LuaHost

# Recursion guards for message dispatch: a broadcast may play commands
# that broadcast again (gat's SetupEnding -> ShowScores -> ...). DEPTH
# caps a single nested chain; TOTAL caps the whole load/replay so a
# pathological self-broadcasting handler cannot spin forever.
_MAX_DISPATCH_DEPTH = 24
_MAX_DISPATCH_TOTAL = 100000

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
--
-- Getters (`a:GetX()`, `a:getrotation()`, `AFT:GetTexture()`) route to
-- __actor_get, which returns the recorder's current value(s) so driver
-- closures can read one actor to drive another (`b:zoomx(a:GetX())`)
-- without faulting on a table. __actor_get hands back the permissive
-- sentinel for getters we do not model (`GetChild()` etc.), so those
-- chains keep working as before.
local __GETTER = {
    GetX=true, GetY=true, GetZ=true, GetZoom=true, GetZoomX=true,
    GetZoomY=true, GetRotationX=true, GetRotationY=true,
    GetRotationZ=true, GetTexture=true, getrotation=true,
}
-- Command-dispatch verbs an actor exposes to the message system:
-- `a:playcommand('Name')` runs the actor's <Name>Command now;
-- `a:queuecommand('Name')` runs it after the actor's pending tween
-- time (an approximation of SM's next-frame queue). Both route to
-- Python (__actor_command) with the recorder id so the dispatched
-- body records onto the same recorder as `self`.
local __COMMAND = {playcommand=true, queuecommand=true}
function __make_recorder(id)
    local t = {__recorder_id = id}
    setmetatable(t, {__index = function(_, key)
        if __GETTER[key] then
            return function(_self, ...) return __actor_get(id, key) end
        end
        if __COMMAND[key] then
            return function(_self, name, ...)
                __actor_command(id, key, name)
                return t
            end
        end
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


# How far before beat 0 the actor load pass anchors when the FGCHANGES
# start sits after beat 0, so load-time keyframes strictly precede a
# beat-0 mod_action. Small: load values just hold from a hair earlier.
_LOAD_LEAD_S = 0.01


class StubEnvironment:
    """One LuaJIT host with the classic-template engine stubs installed,
    plus the harvested tables after chunks run."""

    def __init__(self, start_beat: float = 0.0, to_seconds=None):
        self._start_beat = float(start_beat)
        self._clock_beat = float(start_beat)
        self._to_seconds = to_seconds or (lambda beat: float(beat))
        # Actors load BEFORE the song's first beat, so their InitCommand /
        # OnCommand keyframes must sort before any `mod_actions` closure
        # (which can fire at beat 0 - e.g. gat's `char_shame:Spawn`). The
        # FGCHANGES start beat can map to a LATER time than beat 0 (gat's
        # FGCHANGES is beat 0.5), which would invert the two and let a
        # beat-0 Spawn's zoom(1) be masked by the later-recorded
        # InitCommand zoom(0). When the start beat sits after beat 0 we
        # anchor load a hair before beat 0's time; otherwise load stays at
        # the start-beat time (no shift for charts that already load
        # first).
        start_s = self._to_seconds(self._start_beat)
        beat0_s = self._to_seconds(0.0)
        self._load_seconds = (min(start_s, beat0_s - _LOAD_LEAD_S)
                              if start_s > beat0_s else start_s)
        self._shader_flags: list = []
        self._applied_mods: list = []
        self._swallowed = 0
        self._recorders: dict = {}
        self._recorder_tables: dict = {}
        self._next_recorder_id = 0
        self._message_commands: dict = {}
        self._named_commands: dict = {}
        self._dispatch_depth = 0
        self._dispatch_total = 0
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
        _rec_id, recorder_table = self._new_recorder(self._load_seconds)
        self._host.env['self'] = recorder_table
        self._host.run(source, name=name)
        self._host.env['self'] = self._host.env['__permissive']()

    def load_actors(self, root) -> list:
        """Actor-driven load pass: one persistent recorder per actor.

        For every actor in the tree we create a recorder, run its
        load-time InitCommand/OnCommand with `self` bound to it (so
        `self:x(..)` pokes record and `NAME = self` binds the global),
        and register its `<Name>MessageCommand` / `<Name>Command` bodies
        so a later broadcast or play/queuecommand can run them ON THE
        SAME recorder. Registering before running means a broadcast fired
        from an InitCommand already sees every actor's message handlers.

        Returns a list of `name: message` warnings from faulting chunks.
        Each actor's recorder is the SAME object its message commands
        poke, so `SetupFUCK` inserting `self` into a pool and a later
        `fuck_get()` poke land on one timeline."""
        warnings: list = []
        self._register_all_commands(root)
        self._run_load_commands(root, warnings)
        return warnings

    def _register_all_commands(self, actor) -> None:
        """Give every actor a recorder and index its message / named
        command bodies, before any load command runs."""
        rec_id = self._recorder_id_for(actor)
        for message, body in actor.message_commands().items():
            self._message_commands.setdefault(message, []).append(
                (rec_id, body))
        named = actor.named_commands()
        if named:
            self._named_commands[rec_id] = named
        for child in actor.children:
            self._register_all_commands(child)

    def _run_load_commands(self, actor, warnings) -> None:
        rec_id = self._recorder_id_for(actor)
        for attr in ('InitCommand', 'OnCommand'):
            value = actor.attrs.get(attr, '')
            if value.startswith('%'):
                body = _strip_lua_wrapper(value)
                self._run_body_on(rec_id, body, f'{actor.kind}.{attr}',
                                  warnings)
        for child in actor.children:
            self._run_load_commands(child, warnings)

    def _recorder_id_for(self, actor):
        """The recorder id bound to `actor`, created on first ask so the
        whole tree shares one recorder per actor (pokes from Init, On and
        every message command land together)."""
        existing = getattr(actor, '_recorder_id', None)
        if existing is not None:
            return existing
        rec_id, _table = self._new_recorder(self._load_seconds)
        actor._recorder_id = rec_id
        return rec_id

    def _run_body_on(self, rec_id, body, name, warnings) -> None:
        """Run one Lua command body with `self` = the actor's recorder.
        Faults are recorded as warnings, never raised: a community file
        must load even when one handler poked something we do not model."""
        table = self._recorder_tables.get(rec_id)
        if table is None:
            return
        saved = self._host.env['self']
        self._host.env['self'] = table
        try:
            self._host.run(body, name=name)
        except Exception as exc:
            warnings.append(f'{name}: {exc}')
        finally:
            self._host.env['self'] = saved

    # -- message dispatch -------------------------------------------------

    def _run_command_body(self, rec_id, body, name) -> None:
        """Run one registered command body on `rec_id`'s recorder. A
        `%`-prefixed body is a Lua chunk run with `self` bound to the
        recorder; a plain classic string (`hidden,1;zoom,1`) is parsed
        and poked directly. Depth/total capped so a broadcast cycle stops
        cleanly. Faults are swallowed (a handler may poke an unmodeled
        actor); the load/replay warning path already reports load chunks."""
        if self._dispatch_total >= _MAX_DISPATCH_TOTAL \
                or self._dispatch_depth >= _MAX_DISPATCH_DEPTH:
            return
        self._dispatch_total += 1
        self._dispatch_depth += 1
        try:
            self._run_command_body_inner(rec_id, body, name)
        finally:
            self._dispatch_depth -= 1

    def _run_command_body_inner(self, rec_id, body, name) -> None:
        recorder = self._recorders.get(rec_id)
        if not isinstance(body, str) or recorder is None:
            return
        if body.startswith('%'):
            warnings: list = []
            self._run_body_on(rec_id, _strip_lua_wrapper(body), name, warnings)
        else:
            for verb, args in parse_command_string(body):
                self._dispatch_command_verb(recorder, rec_id, verb, args)

    def _dispatch_command_verb(self, recorder, rec_id, verb, args) -> None:
        """A classic command verb inside a running command body. Most
        pokes go to the recorder; a nested `queuecommand,Name` /
        `playcommand,Name` re-enters dispatch (SM chains commands this
        way, e.g. shame's Hurt -> StopVib)."""
        if verb in ('queuecommand', 'playcommand') and args:
            self._actor_command(rec_id, verb, args[0])
        else:
            recorder.poke(verb, args)

    def _dispatch_message(self, name) -> None:
        """`MESSAGEMAN:Broadcast('Name')`: run `<Name>MessageCommand` on
        every actor that defines it, each on its own recorder at the
        current fire clock. Actors register their handlers before any
        command runs, so a broadcast fired during load already reaches
        the whole tree (the pool builders bind their globals here)."""
        if not isinstance(name, str):
            return
        for rec_id, body in self._message_commands.get(name, ()):
            self._sync_recorder_clock(rec_id)
            self._run_command_body(rec_id, body, f'msg:{name}')

    def _actor_command(self, rec_id, verb, name=None) -> None:
        """`actor:playcommand('Name')` / `actor:queuecommand('Name')`:
        run that actor's `<Name>Command`. playcommand runs at the actor's
        current clock; queuecommand queues at clock + pending tween time
        (SM runs a queued command on the next frame, after the in-flight
        tween; advancing the recorder's clock by the pending tween is a
        close approximation and keeps chained commands ordered)."""
        rec_id = _to_int(rec_id)
        commands = self._named_commands.get(rec_id)
        if not isinstance(name, str) or commands is None \
                or name not in commands:
            return
        recorder = self._recorders.get(rec_id)
        if recorder is not None and verb == 'queuecommand':
            recorder.advance_clock_by_pending()
        self._run_command_body(rec_id, commands[name], f'cmd:{name}')

    def _sync_recorder_clock(self, rec_id) -> None:
        """Point a recorder at the current fire clock before a dispatched
        body records, so a message fired mid-replay lands at the right
        time (its keyframes start at the broadcast beat, not load)."""
        recorder = self._recorders.get(rec_id)
        if recorder is not None:
            recorder.reset_clock(self._to_seconds(self._clock_beat))

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
        by id; returns (rec_id, lua_table). The table is kept so a later
        message dispatch can rebind it as `self`."""
        rec_id = self._next_recorder_id
        self._next_recorder_id += 1
        self._recorders[rec_id] = RecordingActor(clock=clock_seconds)
        table = self._host.env['__make_recorder'](rec_id)
        self._recorder_tables[rec_id] = table
        return rec_id, table

    def _actor_poke(self, rec_id, verb=None, *args) -> None:
        recorder = self._recorders.get(_to_int(rec_id))
        if recorder is not None and isinstance(verb, str):
            recorder.poke(verb, list(args))

    def _actor_get(self, rec_id, verb=None):
        """Return a recorder getter's value for a driver closure. Falls
        back to the permissive sentinel for getters the recorder does not
        model (e.g. `GetChild`), so those chained reads keep degrading to
        no-ops instead of returning a stray number."""
        recorder = self._recorders.get(_to_int(rec_id))
        if recorder is not None and isinstance(verb, str):
            if verb == 'getrotation':
                return recorder.getrotation()
            value = recorder.read(verb)
            if value is not None:
                return value
        return self._host.env['__permissive']()

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

    def named_actor_meta(self) -> dict:
        """global name -> {'aft_source': str|None, 'is_aft': bool} for
        every recorder-bound global, whether or not it recorded any
        pokes. The field producer reads `aft_source` to pick out the
        AFT-screen-copy sprites (they draw the captured playfield) from
        ordinary named actors."""
        out = {}
        for name in self._recorder_global_names():
            recorder = self._recorder_for_table(self._host.env[name])
            if recorder is not None:
                out[name] = {'aft_source': recorder.aft_source,
                             'is_aft': recorder.is_aft}
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
        """Fire every `mod_actions` entry ONCE, in beat order, exactly as
        the template's per-frame reader does (curaction advances
        monotonically). Each entry runs with the engine clock set to its
        own beat so any `GetSongBeat`-driven branch sees the fire moment.

        Function payloads are the scheduled closures (actor pokes,
        `fuck_get()` spawns, ...); string payloads are MESSAGEMAN
        broadcasts (`{beat,'SetupFUCK'}`) - the reader does
        `MESSAGEMAN:Broadcast(str)`, so we dispatch the matching message
        commands, which is how the pool builders (`SetupFUCK`) bind their
        globals mid-song. Every entry runs under its own try/except so
        one faulting entry cannot abort the replay.

        Returns (fired, failed): counts of entries executed and of entries
        that raised."""
        rows = self.mod_actions
        ordered = sorted(
            enumerate(rows),
            key=lambda pair: (_beat_of(pair[1]), pair[0]))
        fired = 0
        failed = 0
        for _order, row in ordered:
            payload = row.get(2) if isinstance(row, dict) else None
            if not (callable(payload) or isinstance(payload, str)):
                continue
            beat = _beat_of(row)
            self._clock_beat = beat
            self._reset_recorder_clocks(self._to_seconds(beat))
            fired += 1
            try:
                if callable(payload):
                    payload()
                else:
                    self._dispatch_message(payload)
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
        host.expose('MESSAGEMAN', singleton(host.to_lua({
            'Broadcast': self._broadcast,
        })))
        host.expose('SCREENMAN', singleton(host.to_lua({
            'SystemMessage': lambda _self, *_a: None,
            'PostScreenMessage': lambda _self, *_a: None,
            'GetTopScreen': lambda _self, *_a: None,
        })))
        host.expose('DISPLAY', singleton(host.to_lua({
            'GetDisplayWidth': lambda _self: 640.0,
            'GetDisplayHeight': lambda _self: 480.0,
            'GetVendor': lambda _self: '',
        })))
        for name in ('STATSMAN', 'SONGMAN', 'THEME',
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
        host.expose('__actor_get', self._actor_get)
        host.expose('__actor_command', self._actor_command)
        host.expose('Trace', lambda *_a: None)
        host.expose('print', lambda *_a: None)

    def _broadcast(self, _self, name=None, *_a) -> None:
        """`MESSAGEMAN:Broadcast('Name')`. Runs the matching message
        commands; unknown message names (no actor defines them) no-op."""
        self._dispatch_message(name)

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
