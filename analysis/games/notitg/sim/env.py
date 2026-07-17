"""SimEnvironment: the engine surface the simulated chart runs against.

One LuaHost carrying the same permissive bootstrap and recorder-table
bridge as the harvest path (imported from mod_stubs - the bridge
survives cutover), but routed onto SimActors and ONE timeline:

- `queuecommand`/`queuemessage` append real zero-tweens to each subtree
  actor's queue (Actor::QueueCommand); they fire when the loop's drain
  reaches them. No clock approximations, no separate replay or
  integration passes.
- The chart's own per-frame rig therefore drives itself: gat's
  `InitCommand` ends with `queuecommand('Update')` and its
  `UpdateCommand` re-arms with `sleep(0.02); queuecommand('Update')`
  (default.xml:3838/4695). The loop only advances time.
- `GAMESTATE:ApplyModifiers` / `ApplyGameCommand('mod,..')` land on one
  flat (t, beat, modstring, player) stream; record.py coalesces it into
  windows afterward. Same for shader flags.
- Cross-actor getter reads (a driver reading a data-holder quad's
  `GetX()`) see the read actor's state at its last drain - at most one
  tick stale, exactly the frame quantization the engine itself has.
"""
from __future__ import annotations

from analysis.games.notitg.lua_api import (
    GETTER_NAMES, SIM_GETTER_NAMES, _as_int)
from analysis.games.notitg.mod_stubs import (
    _PERMISSIVE_BOOTSTRAP, _lua_name_set)
from analysis.games.notitg.sim.actor import SimActor
from analysis.games.notitg.xml_actors import (
    _strip_lua_wrapper, parse_command_string)
from analysis.player.render.lua import LuaHost

# Broadcast/command recursion guard: a handler may broadcast a message
# whose handlers broadcast again. Depth-only, deliberately: the engine
# has no TOTAL dispatch budget, and a whole-chart sim legitimately runs
# hundreds of thousands of self-scheduled command fires (every chara /
# pool actor loops via sleep+queuecommand). A global total cap starves
# every chain mid-song at once - it silently no-ops the body whose tail
# re-arms the loop. Runaway protection is the depth cap here plus the
# per-update drain cap and tween-overflow guard on the actor.
_MAX_DISPATCH_DEPTH = 24

# The sim bridge routes two extra getters (GetSecsIntoEffect/GetText)
# the harvest path leaves unrouted; swap the generated __GETTER set
# literal inside the shared bootstrap. The guard catches literal drift -
# a silently failed replace would break the round-trip clocks charts
# build on GetText.
_SIM_BOOTSTRAP = _PERMISSIVE_BOOTSTRAP.replace(
    _lua_name_set(GETTER_NAMES), _lua_name_set(SIM_GETTER_NAMES))
if _SIM_BOOTSTRAP == _PERMISSIVE_BOOTSTRAP:
    raise RuntimeError('sim getter-set substitution found no match')


class SimEnvironment:
    """The singletons, actor registry, and dispatch for one sim run.
    The loop owns time: it calls `set_time(t, beat)` then `drain(t)`
    each tick; everything else happens through the chart's own Lua."""

    def __init__(self, load_seconds: float, rng_seed: int = 0):
        self._host = LuaHost(dialect='luajit21')
        self._host.run(_SIM_BOOTSTRAP, name='bootstrap')
        self._load_seconds = float(load_seconds)
        self._now = float(load_seconds)
        self._beat = 0.0
        self._actors: dict = {}
        self._tables: dict = {}
        self._named_commands: dict = {}
        self._message_commands: dict = {}
        self._children: dict = {}
        self._next_id = 0
        self._active: list = []
        self._applied_mods: list = []
        self._shader_flags: list = []
        self._warnings: list = []
        self._faults = 0
        self._fault_messages: list = []
        self._dispatch_depth = 0
        self._rng_seed = int(rng_seed)
        self._screen_id: int | None = None
        self._screen_children: dict = {}
        self._install()

    # -- results the loop harvests ----------------------------------------

    @property
    def actors(self) -> dict:
        """recorder id -> SimActor."""
        return self._actors

    @property
    def applied_mods(self) -> list:
        """(t, beat, modstring, player) in call order."""
        return self._applied_mods

    @property
    def shader_flags(self) -> list:
        """(t, beat, key, which) in call order."""
        return self._shader_flags

    @property
    def warnings(self) -> list:
        return self._warnings

    @property
    def faults(self) -> int:
        return self._faults

    @property
    def fault_messages(self) -> list:
        """First distinct runtime fault texts (diagnostics; capped)."""
        return self._fault_messages

    def _record_fault(self, name, exc) -> None:
        self._faults += 1
        text = f'{name}: {exc}'
        if len(self._fault_messages) < 50 \
                and text not in self._fault_messages:
            self._fault_messages.append(text)

    def actor_id(self, actor) -> int | None:
        return getattr(actor, '_sim_id', None)

    def player_actor(self, name: str) -> SimActor | None:
        """The screen child recorder for 'PlayerP1'/'PlayerP2', when the
        chart fetched it (base-field hidden / player poke streams)."""
        rec_id = self._screen_children.get(name)
        return self._actors.get(rec_id) if rec_id is not None else None

    def screen_actor(self) -> SimActor | None:
        return (self._actors.get(self._screen_id)
                if self._screen_id is not None else None)

    # -- harvest surface (mirrors the mod_stubs shapes, so the modfile
    # element/screen/field producers consume a sim env directly) ---------

    def named_actor_keyframes(self) -> dict:
        """global name -> {property: [Keyframe]} for chart-bound actor
        globals (`NAME = self`) that recorded pokes."""
        out = {}
        for name, actor in self._named_actors().items():
            keyframes = actor.keyframes()
            if keyframes:
                out[name] = keyframes
        return out

    def named_actor_meta(self) -> dict:
        """global name -> {'aft_source', 'is_aft'} for every bound
        global, poked or not (the field producer keys copies off it)."""
        return {name: {'aft_source': actor.aft_source,
                       'is_aft': actor.is_aft}
                for name, actor in self._named_actors().items()}

    def actor_keyframes(self) -> dict:
        """recorder id -> {property: [Keyframe]}, the complete stream."""
        return {rec_id: actor.keyframes()
                for rec_id, actor in self._actors.items()
                if actor.keyframes()}

    def actor_oscillator_spans(self) -> dict:
        return {rec_id: actor.oscillator_spans()
                for rec_id, actor in self._actors.items()
                if actor.oscillator_spans()}

    def screen_keyframes(self) -> dict:
        actor = self.screen_actor()
        return actor.keyframes() if actor is not None else {}

    def screen_oscillator_spans(self) -> tuple:
        actor = self.screen_actor()
        return actor.oscillator_spans() if actor is not None else ()

    def player_keyframes(self, name: str) -> dict:
        actor = self.player_actor(name)
        return actor.keyframes() if actor is not None else {}

    def player_oscillator_spans(self, name: str) -> tuple:
        actor = self.player_actor(name)
        return actor.oscillator_spans() if actor is not None else ()

    def _named_actors(self) -> dict:
        """Sandbox globals bound to one of our recorder tables."""
        out = {}
        for key, value in self._host.env.items():
            if not isinstance(key, str):
                continue
            actor = self._actor_for_table(value)
            if actor is not None:
                out[key] = actor
        return out

    def _actor_for_table(self, table) -> SimActor | None:
        if not hasattr(table, '__getitem__'):
            return None
        try:
            rec_id = table['__recorder_id']
        except (KeyError, TypeError):
            return None
        return self._actors.get(_as_int(rec_id)) if rec_id is not None \
            else None

    # -- load pass ---------------------------------------------------------

    def load_actors(self, root) -> list:
        """Register every actor's commands, then run load commands in
        TWO tree-order phases, as the engine does: every InitCommand
        (actor creation), then every OnCommand (screen start). An
        OnCommand may therefore read a global that a LATER actor's
        InitCommand bound - the AFT rigs depend on this. The chart's
        self-scheduling Update chain arms itself here (its queuecommand
        lands on the real queue)."""
        self._register(root)
        self._run_load(root, 'InitCommand')
        self._run_load(root, 'OnCommand')
        return self._warnings

    def _register(self, actor) -> None:
        rec_id = self._id_for(actor)
        for message, body in actor.message_commands().items():
            self._message_commands.setdefault(message, []).append(
                (rec_id, body))
        named = actor.named_commands()
        if named:
            self._named_commands[rec_id] = named
        self._children[rec_id] = [self._id_for(c) for c in actor.children]
        for child in actor.children:
            self._register(child)

    def _run_load(self, actor, attr) -> None:
        rec_id = self._id_for(actor)
        value = actor.attrs.get(attr, '')
        if value.startswith('%'):
            self._run_lua_body(rec_id, _strip_lua_wrapper(value),
                               f'{actor.kind}.{attr}', load=True)
        elif value:
            self._run_classic_body(rec_id, value)
        for child in actor.children:
            self._run_load(child, attr)

    def _id_for(self, actor) -> int:
        existing = getattr(actor, '_sim_id', None)
        if existing is not None:
            return existing
        rec_id = self._new_actor()
        actor._sim_id = rec_id
        # The element-tree compiler picks up an actor's complete poke
        # stream by its `_recorder_id` tag; the sim ids serve both.
        actor._recorder_id = rec_id
        return rec_id

    def _new_actor(self) -> int:
        rec_id = self._next_id
        self._next_id += 1
        actor = SimActor(self._now)
        actor.beat_fn = lambda: self._beat
        self._actors[rec_id] = actor
        self._tables[rec_id] = self._host.env['__make_recorder'](rec_id)
        self._active.append(rec_id)
        return rec_id

    # -- time --------------------------------------------------------------

    def set_time(self, t: float, beat: float) -> None:
        self._now = float(t)
        self._beat = float(beat)

    def drain(self, t: float) -> None:
        """Advance every actor's tween queue to `t`, firing queue-borne
        commands as the drains reach them. Idle actors (empty queue) are
        skipped; a dispatch onto one syncs it first (`_sync`)."""
        for rec_id in self._active:
            actor = self._actors[rec_id]
            if actor._tweens:
                self._drain_actor(rec_id, actor, t)

    def _drain_actor(self, rec_id, actor, t) -> None:
        actor.update_to(t, lambda name: self._fire_queued(rec_id, name))

    def _sync(self, rec_id) -> None:
        """Bring an actor to the current time before a dispatched body
        pokes it, so its emissions timestamp at the fire moment."""
        actor = self._actors.get(rec_id)
        if actor is not None and actor.now < self._now:
            self._drain_actor(rec_id, actor, self._now)

    def _fire_queued(self, rec_id, name) -> None:
        """A queue-carried command reached its zero-tween: '!' means
        broadcast (Actor.cpp:1082), otherwise the actor's own <name>
        command runs. Faults are swallowed and counted - one bad frame
        must not kill the sim."""
        try:
            if name.startswith('!'):
                self._broadcast(None, name[1:])
            else:
                body = self._named_commands.get(rec_id, {}).get(name)
                if body is not None:
                    self._run_command_body(rec_id, body, f'cmd:{name}')
        except Exception as exc:
            self._record_fault(f'fire:{name}', exc)

    # -- dispatch ----------------------------------------------------------

    def _run_command_body(self, rec_id, body, name) -> None:
        if self._dispatch_depth >= _MAX_DISPATCH_DEPTH:
            return
        self._dispatch_depth += 1
        try:
            if body.startswith('%'):
                self._run_lua_body(rec_id, _strip_lua_wrapper(body), name)
            else:
                self._run_classic_body(rec_id, body)
        finally:
            self._dispatch_depth -= 1

    def _run_lua_body(self, rec_id, body, name, load=False) -> None:
        self._sync(rec_id)
        saved = self._host.env['self']
        self._host.env['self'] = self._tables[rec_id]
        try:
            self._host.run(body, name=name)
        except Exception as exc:
            if load:
                self._warnings.append(f'{name}: {exc}')
            else:
                self._record_fault(name, exc)
        finally:
            self._host.env['self'] = saved

    def _run_classic_body(self, rec_id, value) -> None:
        self._sync(rec_id)
        actor = self._actors[rec_id]
        for verb, args in parse_command_string(value):
            if verb in ('queuecommand', 'playcommand') and args:
                self._actor_command(rec_id, verb, args[0])
            else:
                actor.poke(verb, args)

    def _broadcast(self, _self, name=None, *_a) -> None:
        """MESSAGEMAN:Broadcast - run <name>MessageCommand on every
        registering actor, at the current time."""
        if not isinstance(name, str):
            return
        for rec_id, body in self._message_commands.get(name, ()):
            self._run_command_body(rec_id, body, f'msg:{name}')

    def _actor_command(self, rec_id, verb, name=None) -> None:
        """playcommand runs <name>Command NOW on the actor and its whole
        subtree (SM RunCommandsOnChildren); queuecommand appends the
        zero-tween to each subtree actor that defines the command, to
        fire when that actor's queue reaches it."""
        if not isinstance(name, str):
            return
        self._command_subtree(_as_int(rec_id), verb, name)

    def _command_subtree(self, rec_id, verb, name) -> None:
        commands = self._named_commands.get(rec_id)
        if commands is not None and name in commands:
            if verb == 'queuecommand':
                self._sync(rec_id)
                self._actors[rec_id].queue_command(name)
            else:
                self._run_command_body(rec_id, commands[name], f'cmd:{name}')
        for child_id in self._children.get(rec_id, ()):
            self._command_subtree(child_id, verb, name)

    # -- Lua bridge callbacks ----------------------------------------------

    def _actor_poke(self, rec_id, verb, *args) -> None:
        rec_id = _as_int(rec_id)
        actor = self._actors.get(rec_id)
        if actor is None:
            return
        self._sync(rec_id)
        actor.poke(verb, list(args))

    def _actor_get(self, rec_id, verb):
        actor = self._actors.get(_as_int(rec_id))
        if actor is None:
            return self._host.env['__permissive']()
        value = actor.read(verb)
        if value is None:
            return self._host.env['__permissive']()
        return value

    def _screen_get_child(self, name):
        """SCREENMAN:GetTopScreen():GetChild(name) - a persistent
        per-name recorder. The players' pokes record the base-field
        visibility stream (P1:hidden(1)); any other child (NoteField,
        Judgment) is a harmless poke-able target for proxy SetTarget."""
        if not isinstance(name, str):
            return self._host.env['__permissive']()
        rec_id = self._screen_children.get(name)
        if rec_id is None:
            rec_id = self._new_actor()
            self._screen_children[name] = rec_id
        return self._tables[rec_id]

    # -- engine singletons -------------------------------------------------

    def _install(self) -> None:
        host = self._host
        singleton = host.env['__make_singleton']

        host.expose('__actor_poke', self._actor_poke)
        host.expose('__actor_get', self._actor_get)
        host.expose('__actor_command', self._actor_command)
        host.expose('__screen_get_child', self._screen_get_child)

        host.expose('GAMESTATE', singleton(host.to_lua({
            'GetSongBeat': lambda _self: self._beat,
            'GetSongBeatNoOffset': lambda _self: self._beat,
            'SetShaderFlag': self._set_shader_flag,
            'SetShaderFlagNum': self._set_shader_flag_num,
            'ApplyGameCommand': self._apply_game_command,
            'ApplyModifiers': self._apply_modifiers,
        })))
        host.expose('PREFSMAN', singleton(host.to_lua({
            'GetPreference': lambda _self, _key=None: '',
        })))
        host.expose('MESSAGEMAN', singleton(host.to_lua({
            'Broadcast': self._broadcast,
        })))
        self._screen_id = self._new_actor()
        top_screen = self._host.env['__make_screen_recorder'](self._screen_id)
        self._tables[self._screen_id] = top_screen
        host.expose('SCREENMAN', singleton(host.to_lua({
            'SystemMessage': lambda _self, *_a: None,
            'PostScreenMessage': lambda _self, *_a: None,
            'GetTopScreen': lambda _self, *_a: top_screen,
        })))
        host.expose('DISPLAY', singleton(host.to_lua({
            'GetDisplayWidth': lambda _self: 640.0,
            'GetDisplayHeight': lambda _self: 480.0,
            'GetVendor': lambda _self: '',
        })))
        for name in ('STATSMAN', 'SONGMAN', 'THEME', 'GAMEMAN',
                     'NOTESKIN', 'INPUTFILTER', 'PROFILEMAN'):
            host.expose(name, singleton(None))

        for name, value in (('SCREEN_WIDTH', 640.0), ('SCREEN_HEIGHT', 480.0),
                            ('SCREEN_CENTER_X', 320.0),
                            ('SCREEN_CENTER_Y', 240.0),
                            ('SCREEN_LEFT', 0.0), ('SCREEN_RIGHT', 640.0),
                            ('SCREEN_TOP', 0.0), ('SCREEN_BOTTOM', 480.0),
                            ('FUCK_EXE', True)):
            host.expose(name, value)

        host.run('_G.self = __permissive()', name='self-stub')
        host.expose('Trace', lambda *_a: None)
        host.expose('print', lambda *_a: None)
        host.expose('__rng_seed', float(self._rng_seed))
        host.run('math.randomseed(__rng_seed)', name='rng-seed')

    def _set_shader_flag(self, _self, key=None, *_a) -> None:
        self._shader_flags.append((self._now, self._beat, key, None))

    def _set_shader_flag_num(self, _self, key=None, which=None, *_a) -> None:
        self._shader_flags.append((self._now, self._beat, key, which))

    def _apply_game_command(self, _self, command=None, pn=None) -> None:
        if not isinstance(command, str):
            return
        head, _sep, rest = command.partition(',')
        if head.strip().lower() == 'mod':
            self._applied_mods.append(
                (self._now, self._beat, rest.strip(), _as_int(pn)))

    def _apply_modifiers(self, _self, modstring=None, *_a) -> None:
        if isinstance(modstring, str):
            self._applied_mods.append(
                (self._now, self._beat, modstring.strip(), None))
