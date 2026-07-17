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

from analysis.games.notitg.lua_api import COMMAND_NAMES, GETTER_NAMES
from analysis.games.notitg.recording_actor import RecordingActor
from analysis.games.notitg.xml_actors import (
    _strip_lua_wrapper, parse_command_string)
from analysis.player.render.lua import LuaHost


def _lua_name_set(names) -> str:
    """A Lua set literal (`{GetX=true, GetY=true}`) from registry name
    lists, so the bridge's `__GETTER` / `__COMMAND` sets are generated
    from the one source of truth (lua_api) instead of hand-kept here."""
    return '{' + ', '.join(f'{name}=true' for name in names) + '}'

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
local __GETTER = __GETTER_SET
-- Command-dispatch verbs an actor exposes to the message system:
-- `a:playcommand('Name')` runs the actor's <Name>Command now;
-- `a:queuecommand('Name')` runs it after the actor's pending tween
-- time (an approximation of SM's next-frame queue). Both route to
-- Python (__actor_command) with the recorder id so the dispatched
-- body records onto the same recorder as `self`.
local __COMMAND = __COMMAND_SET
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

-- The top screen is a recorder (the chart pokes it directly - the
-- `screen:effectmagnitude(..)` camera vibe and `GetTopScreen():zoom(..)`
-- per-frame zoom in gat_updateproxies), so it records like any actor,
-- but it ALSO answers GetChild/GetTopScreen. Those return real recorder
-- tables (a player, or the screen itself) instead of a poke, so player
-- fetches and chained `GetTopScreen():GetChild(..)` keep working.
function __make_screen_recorder(id)
    local t = __make_recorder(id)
    local mt = getmetatable(t)
    local poke_index = mt.__index
    mt.__index = function(tbl, key)
        if key == 'GetChild' then
            return function(_self, name) return __screen_get_child(name) end
        end
        if key == 'GetTopScreen' then
            return function(_self) return t end
        end
        return poke_index(tbl, key)
    end
    return t
end

-- NotITG embeds Lua 5.0; these live under the LuaJIT (5.1) runtime as
-- their renamed forms. The template calls the 5.0 names.
if math.mod == nil then math.mod = math.fmod end
if table.getn == nil then table.getn = function(t) return #t end end
"""

# The `__GETTER` / `__COMMAND` routing sets are the registry's name lists,
# rendered as Lua set literals so the bridge and the recorder cannot drift
# on which calls return a value vs run a command.
_PERMISSIVE_BOOTSTRAP = _PERMISSIVE_BOOTSTRAP.replace(
    '__GETTER_SET', _lua_name_set(GETTER_NAMES)).replace(
    '__COMMAND_SET', _lua_name_set(COMMAND_NAMES))


# How far before beat 0 the actor load pass anchors when the FGCHANGES
# start sits after beat 0, so load-time keyframes strictly precede a
# beat-0 mod_action. Small: load values just hold from a hair earlier.
_LOAD_LEAD_S = 0.01

# Determinism contract for chart RNG. Charts spawn actors at positions
# picked by `math.random` (gat's FUCK datamosh scatters 512 pooled bars
# with random x/y tween targets); those targets are recorded ONCE at
# compile, so an unseeded PRNG would place the bars differently every
# compile and break the compiled-document cache and golden tests. We
# seed the sandbox PRNG deterministically from the chart's own content
# (the same faithful-once-recorded contract as fluXis RandomRange, which
# seeds `random.Random(f'fluxis-script:{seed}')` per script path): the
# same chart always compiles the same scatter, different charts differ.
# One seed per compile (not per invocation) is faithful to the engine,
# which shares one PRNG stream across every spawner call in a run.
_DEFAULT_RNG_SEED = 0


class StubEnvironment:
    """One LuaJIT host with the classic-template engine stubs installed,
    plus the harvested tables after chunks run."""

    def __init__(self, start_beat: float = 0.0, to_seconds=None,
                 rng_seed: int = _DEFAULT_RNG_SEED):
        self._start_beat = float(start_beat)
        self._clock_beat = float(start_beat)
        self._to_seconds = to_seconds or (lambda beat: float(beat))
        self._rng_seed = int(rng_seed)
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
        self._child_recorders: dict = {}
        self._screen_children: dict = {}
        self._screen_recorder_id: int | None = None
        self._dispatch_depth = 0
        self._dispatch_total = 0
        # Per-frame update integration state (None outside a pass).
        self._integration_clock: list | None = None
        self._integration_applied: dict | None = None
        self._integration_beat: float = 0.0
        self._integration_faults = 0
        # When set, actor pokes are dropped (the mod_actions closures a
        # tick re-fires only to evolve time-varying globals - their actor
        # pokes were already captured by the one-shot replay).
        self._recording_frozen = False
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
        command bodies, before any load command runs. The actor tree's
        parent->child recorder links are recorded too, so a play/queue
        command on an ActorFrame propagates to its subtree (SM's
        RunCommandsOnChildren)."""
        rec_id = self._recorder_id_for(actor)
        for message, body in actor.message_commands().items():
            self._message_commands.setdefault(message, []).append(
                (rec_id, body))
        named = actor.named_commands()
        if named:
            self._named_commands[rec_id] = named
        self._child_recorders[rec_id] = [
            self._recorder_id_for(child) for child in actor.children]
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
            elif value:
                self._poke_classic_body(rec_id, value)
        for child in actor.children:
            self._run_load_commands(child, warnings)

    def _poke_classic_body(self, rec_id, value) -> None:
        """Poke a classic-string InitCommand/OnCommand (`diffusealpha,0`)
        onto the actor's load-pass recorder, so its base state is part of
        the same complete poke stream the tree compiler reads (the bg
        Layers' `OnCommand=diffusealpha,0` rest alpha lives here)."""
        recorder = self._recorders.get(rec_id)
        if recorder is None:
            return
        for verb, args in parse_command_string(value):
            self._dispatch_command_verb(recorder, rec_id, verb, args)

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
        elif not self._recording_frozen:
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
        run that actor's `<Name>Command`, then propagate to its whole
        subtree - SM's ActorFrame runs a play/queuecommand on itself AND
        recursively on every child, so `gat_bg:queuecommand('BG2')` fires
        each bg Layer's `BG2Command` crossfade even though gat_bg defines
        no `BG2Command` itself.

        playcommand runs at the actor's current clock; queuecommand queues
        at clock + pending tween time (SM runs a queued command on the next
        frame, after the in-flight tween; advancing the recorder's clock by
        the pending tween is a close approximation and keeps chained
        commands ordered)."""
        if not isinstance(name, str):
            return
        self._run_actor_command_subtree(_to_int(rec_id), verb, name)

    def _run_actor_command_subtree(self, rec_id, verb, name) -> None:
        commands = self._named_commands.get(rec_id)
        if commands is not None and name in commands:
            recorder = self._recorders.get(rec_id)
            if recorder is not None and verb == 'queuecommand':
                recorder.advance_clock_by_pending()
            self._run_command_body(rec_id, commands[name], f'cmd:{name}')
        for child_id in self._child_recorders.get(rec_id, ()):
            self._run_actor_command_subtree(child_id, verb, name)

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
        if recorder is None or not isinstance(verb, str):
            return
        if self._recording_frozen:
            # A mod_actions closure re-fired inside the update integration:
            # keyframe recording (and oscillator state) is frozen - those
            # pokes were already captured by the one-shot replay. Only an
            # accumulator reset a per-frame driver reads back (gat's Toss
            # quad re-anchor) updates the live mirror. The per-frame effect
            # pokes (`screen:effectmagnitude` / `Proxy:vibrate`) come from
            # the Update BODY tick, which runs unfrozen, so they record via
            # the normal path below.
            if self._integration_clock is not None:
                recorder.live_poke(verb, list(args))
            return
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

    def _make_top_screen(self):
        """A recording mirror standing in for `SCREENMAN:GetTopScreen()`.

        The chart pokes the top screen directly (gat_updateproxies zooms
        and offsets it as a whole-scene camera, and `screen:effectmagnitude`
        is a scene vibrate), so it must record like an actor while still
        answering GetChild/GetTopScreen. One persistent recorder id backs
        it, exposed as the screen-transform stream to the field producer."""
        rec_id, _table = self._new_recorder(self._load_seconds)
        self._screen_recorder_id = rec_id
        return self._host.env['__make_screen_recorder'](rec_id)

    def screen_keyframes(self) -> dict:
        """Recorded pokes on the top screen (the whole-scene camera zoom /
        offset / vibrate the per-frame update drives), or {} when nothing
        poked it."""
        recorder = (self._recorders.get(self._screen_recorder_id)
                    if self._screen_recorder_id is not None else None)
        return recorder.keyframes() if recorder is not None else {}

    def _screen_get_child_by_name(self, name=None):
        """`__screen_get_child(name)` from the Lua screen recorder (no
        colon self); delegates to the player-child recorder path."""
        return self._screen_get_child(None, name)

    def _screen_get_child(self, _self, name=None):
        """`SCREENMAN:GetTopScreen():GetChild('PlayerP1')`. The real
        players are engine actors, not chart-declared, but the chart pokes
        them (`P1:hidden(1)` to hide the base NoteField while proxies stand
        in). We hand back a persistent recorder per player name so those
        pokes record onto a timeline the compiler can read as the
        base-field visibility signal. Non-player children return a fresh
        recorder too, so `P1:GetChild('NoteField')` is a usable target for
        a proxy's SetTarget (its pokes are harmless)."""
        if not isinstance(name, str):
            return self._host.env['__permissive']()
        table = self._screen_children.get(name)
        if table is None:
            # Anchor at the current fire clock, not load: the player is
            # first fetched inside a mod_actions broadcast (Hide/Show),
            # and _reset_recorder_clocks (which precedes each fire) has
            # already run, so a recorder born here misses it.
            _rec_id, table = self._new_recorder(
                self._to_seconds(self._clock_beat))
            self._screen_children[name] = table
        return table

    def player_keyframes(self, name: str) -> dict:
        """Recorded pokes for an engine player actor
        (`PlayerP1`/`PlayerP2`), or {} when the chart never poked it. Used
        for the base-field visibility (`hidden`) timeline."""
        table = self._screen_children.get(name)
        recorder = self._recorder_for_table(table) if table is not None \
            else None
        return recorder.keyframes() if recorder is not None else {}

    def proxy_grid(self) -> dict:
        """The `gat_proxies` / `gat_proxiesc` proxy-grid frames as
        composition inputs for the field producer, or {} when the chart
        has no such table.

        gat's t=42 scatter is a 3x3 grid of notefield proxies per player:
        `gat_proxies[pn][i]` frames (their per-frame `hidden`/`rotation`
        and StartShit2 grid `x`/`y`) nested under `gat_allproxies` (the
        stateful accumulator the update integrator drives) and a static
        `gat_allproxiesc` centering offset. The frames self-assign no
        global, so they are invisible to `named_actor_keyframes`; this
        hands the field producer each frame's recorder keyframes plus the
        parent offsets so it can compose the world transform SM applies.

        Returns {'parent': {prop: [kf]}, 'parent_offset': {prop: [kf]},
        'frames': [{'player', 'frame': {prop: [kf]},
        'content': {prop: [kf]}}, ...]}. `content` is the matching
        `gat_proxiesc` proxy child (its zoom), `player` sources the copy to
        that side's notefield."""
        table = self._host.env['gat_proxies']
        content = self._host.env['gat_proxiesc']
        if not self._is_lua_table(table):
            return {}
        parent = self._recorder_for_table(self._host.env['gat_allproxies'])
        offset = self._recorder_for_table(self._host.env['gat_allproxiesc'])
        frames = []
        for player in (1, 2):
            frames.extend(self._proxy_grid_frames(table, content, player))
        if not frames:
            return {}
        return {
            'parent': parent.keyframes() if parent is not None else {},
            'parent_offset': offset.keyframes() if offset is not None else {},
            'frames': frames,
            # The accumulator is poked every tick its driver runs, so its
            # driven spans are the envelope in which the grid exists at all.
            'spans': parent.driven_spans() if parent is not None else (),
        }

    def _proxy_grid_frames(self, table, content, player) -> list:
        """The `gat_proxies[player]` frames paired with their
        `gat_proxiesc[player]` content proxies, as recorder keyframe
        dicts."""
        row = table[player]
        content_row = content[player] if self._is_lua_table(content) else None
        out = []
        index = 1
        while self._is_lua_table(row) and row[index] is not None:
            frame = self._recorder_for_table(row[index])
            if frame is not None and frame.keyframes():
                proxy = (self._recorder_for_table(content_row[index])
                         if content_row is not None
                         and content_row[index] is not None else None)
                out.append({
                    'player': player,
                    'frame': frame.keyframes(),
                    'content': proxy.keyframes() if proxy is not None else {},
                })
            index += 1
        return out

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

    def actor_keyframes(self) -> dict:
        """recorder-id -> {property: [Keyframe]} for every actor that
        recorded pokes during load + replay. This is the COMPLETE poke
        stream per actor: its InitCommand/OnCommand, plus every message /
        play / queuecommand body dispatched onto it (bg crossfades reach
        the anonymous bg Layer children this way - they self-assign no
        global, so `named_actor_keyframes` misses them). The tree compiler
        keys by the actor's `_recorder_id` to pick up its full stream."""
        out = {}
        for rec_id, recorder in self._recorders.items():
            keyframes = recorder.keyframes()
            if keyframes:
                out[rec_id] = keyframes
        return out

    def actor_oscillator_spans(self) -> dict:
        """recorder-id -> tuple of `_OscSpan` for every actor that ran an
        effect oscillator (vibrate/wag/bob/bounce/spin). Keyed like
        `actor_keyframes` so the tree compiler can synthesise each actor's
        oscillator motion into keyframes on its own timeline. Empty for the
        common no-oscillator chart."""
        out = {}
        for rec_id, recorder in self._recorders.items():
            spans = recorder.oscillator_spans()
            if spans:
                out[rec_id] = spans
        return out

    def screen_oscillator_spans(self) -> tuple:
        """Effect-oscillator spans on the top screen (`screen:vibrate()` +
        the per-frame `screen:effectmagnitude(gat_vib:GetX()..)` scene
        shake, gat's t~312-382 datamosh), or () when the screen never ran
        one. A whole-scene jitter the screen-camera consumer applies."""
        recorder = (self._recorders.get(self._screen_recorder_id)
                    if self._screen_recorder_id is not None else None)
        return recorder.oscillator_spans() if recorder is not None else ()

    def player_oscillator_spans(self, name: str) -> tuple:
        """Oscillator spans for an engine player actor
        (`PlayerP1`/`PlayerP2`), or () when it ran no effect. gat's t~8-48
        bounce/bob/wag drives `Plr(pn)` = these player recorders (they are
        engine actors fetched via GetChild, not tree actors), so the field
        producer reads their oscillators from here."""
        table = self._screen_children.get(name)
        recorder = self._recorder_for_table(table) if table is not None \
            else None
        return recorder.oscillator_spans() if recorder is not None else ()

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

    # -- per-frame update integration -------------------------------------

    def run_update_integration(self, body, windows, to_seconds, to_beats,
                               tick_step, max_ticks) -> dict:
        """Run the recorded `UpdateCommand` `body` on a fixed tick grid
        over the beat `windows` (each a (start_beat, end_beat) pair),
        harvesting each tick's actor pokes onto the existing recorders.

        Recorders enter sampling mode so an update driver reading a source
        quad gets its value at the tick's time; the `mods`/`mods2`/
        `mod_actions` tables are emptied so the window reader and action
        loop inside the body no-op (other passes own them). Direct
        `ApplyGameCommand` mods from the per-frame drivers (the walking
        `movey` family) accumulate into per-tick windowed events, returned
        so the caller folds them into the mod timeline."""
        stripped = _strip_lua_wrapper(body)
        clock = [self._load_seconds]
        self._integration_clock = clock
        self._integration_applied = {}
        self._integration_faults = 0
        actions = self._sorted_mod_actions()
        saved_tables = self._detach_isolated_tables()
        for recorder in self._recorders.values():
            recorder.begin_sampling(clock)
        try:
            ticks = self._run_update_ticks(stripped, windows, actions,
                                           to_seconds, to_beats, clock,
                                           tick_step, max_ticks)
        finally:
            for recorder in self._recorders.values():
                recorder.end_sampling()
            self._restore_isolated_tables(saved_tables)
            applied = self._integration_applied
            faults = self._integration_faults
            self._integration_clock = None
            self._integration_applied = None
            self._clock_beat = self._start_beat
        events = _tick_applied_events(applied, to_seconds, tick_step)
        return {'ran': True, 'ticks': ticks, 'windows': len(windows),
                'applied': len(events), 'applied_events': events,
                'faults': faults}

    def _sorted_mod_actions(self):
        """Every mod_actions closure/broadcast as (beat, payload), beat-
        sorted. Re-fired during integration (recording frozen) so a tick
        sees the time-varying globals the closures maintain
        (`gat_walkerdir`, `gat_scrollspd`, `gat_screen_zoom`) at the value
        the engine's per-frame action loop would have set by that beat -
        the walking `movey` amplitude and the screen-zoom camera read
        them."""
        rows = []
        for row in self.mod_actions:
            payload = row.get(2) if isinstance(row, dict) else None
            if callable(payload) or isinstance(payload, str):
                rows.append((_beat_of(row), payload))
        rows.sort(key=lambda pair: pair[0])
        return rows

    def _run_update_ticks(self, chunk, windows, actions, to_seconds,
                          to_beats, clock, tick_step, max_ticks) -> int:
        cursor = 0
        ticks = 0
        for start_beat, end_beat in windows:
            t = to_seconds(start_beat)
            t_end = to_seconds(end_beat)
            while t <= t_end and ticks < max_ticks:
                beat = to_beats(t)
                cursor = self._fire_pending_actions(actions, cursor, beat)
                self._tick_update(chunk, t, beat, clock)
                ticks += 1
                t += tick_step
        return ticks

    def _fire_pending_actions(self, actions, cursor, beat) -> int:
        """Fire every action whose beat has passed by `beat`, advancing the
        monotonic cursor exactly as the engine's curaction loop does.
        Recording is frozen so only global state evolves - the closures'
        actor pokes were already captured by the one-shot replay. A window
        gap can skip a run of actions at once (the cursor jumps to `beat`),
        matching an editor that seeks past them; their globals still take
        their final pre-`beat` value because they fire in order."""
        self._recording_frozen = True
        try:
            while cursor < len(actions) and actions[cursor][0] <= beat:
                self._clock_beat = actions[cursor][0]
                self._fire_action_payload(actions[cursor][1])
                cursor += 1
        finally:
            self._recording_frozen = False
        return cursor

    def _fire_action_payload(self, payload) -> None:
        try:
            if callable(payload):
                payload()
            else:
                self._dispatch_message(payload)
        except Exception:
            self._integration_faults += 1

    def _tick_update(self, chunk, seconds, beat, clock) -> None:
        """One integration tick at song time `seconds` (chart beat `beat`):
        aim the shared sample clock and every recorder's local clock at
        this tick, point the engine beat/time clocks at it, then run the
        Update body. A faulting tick is swallowed (the body may reach an
        actor we do not model); its partial pokes stand."""
        clock[0] = seconds
        self._integration_beat = beat
        self._clock_beat = beat
        self._host.env['mod_time'] = seconds
        self._reset_recorder_clocks(seconds)
        try:
            self._host.run(chunk, name='update-integration')
        except Exception:
            self._integration_faults += 1

    def _detach_isolated_tables(self) -> dict:
        """Empty the mods/mods2/mod_actions globals (other passes own
        them) for the integration, keeping the originals to restore."""
        saved = {}
        for name in _ISOLATED_TABLES:
            saved[name] = self._host.env[name]
            self._host.env[name] = self._host.to_lua({})
        return saved

    def _restore_isolated_tables(self, saved) -> None:
        for name, value in saved.items():
            self._host.env[name] = value

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
        top_screen = self._make_top_screen()
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
        host.expose('__screen_get_child', self._screen_get_child_by_name)
        host.expose('Trace', lambda *_a: None)
        host.expose('print', lambda *_a: None)

        # Seed the sandbox PRNG so `math.random` scatter (the FUCK pool's
        # tween targets, chara pool placement) records the same positions
        # every compile - the determinism contract above.
        host.expose('__rng_seed', float(self._rng_seed))
        host.run('math.randomseed(__rng_seed)', name='rng-seed')

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
        self._record_applied_mod(rest.strip(), _to_int(pn))

    def _apply_modifiers(self, _self, modstring=None, *_a) -> None:
        """`GAMESTATE:ApplyModifiers('<modstring>')` (the raw form). Same
        one-shot recording as the 'mod,' game command."""
        if isinstance(modstring, str):
            self._record_applied_mod(modstring.strip(), None)
        else:
            self._swallowed += 1

    def _record_applied_mod(self, modstring, player) -> None:
        """Route an `ApplyGameCommand('mod,X')` to the right harvest.

        Outside a per-frame update pass it is a one-shot spike (the
        mod_actions replay path). During integration it is a per-frame
        driver's mod (the walking `movey`/`confusionoffset` families): the
        tick beat feeds a running window per (mod-name, player, base
        modstring) so a value that steps every tick becomes one continuous
        keyed channel, not thousands of one-frame spikes."""
        if self._integration_applied is None:
            self._applied_mods.append((self._clock_beat, modstring, player))
            return
        key = (_mod_name(modstring), player)
        self._integration_applied.setdefault(key, []).append(
            (self._integration_beat, modstring, player))


# Tables the per-frame Update body reads that other compile passes own
# (window reader / action loop). Emptied for the integration so those
# inner loops no-op; restored after.
_ISOLATED_TABLES = ('mods', 'mods2', 'mod_actions')


def _mod_name(modstring: str) -> str:
    """The bare mod name of the LAST token in a modstring
    (`*10000 40 movey0` -> `movey0`), the key a per-frame driver steps.
    Used to group a driver's per-tick values into one keyed channel."""
    token = str(modstring).split(',')[-1].strip().lower()
    return token.split()[-1] if token.split() else token


def _tick_applied_events(applied, to_seconds, tick_step):
    """Per-tick ApplyGameCommand recordings -> contiguous mod windows.

    `applied` maps (mod-name, player) to the list of (beat, modstring,
    player) a per-frame driver produced across its ticks. Consecutive
    ticks holding the SAME modstring coalesce into one window (the engine
    re-applies the identical value each frame, so only value CHANGES need
    a keyframe); the window runs from a value's first tick to the tick
    after it last held, and the driver's final value reverts a tick later.
    Emitted in the normalized `_mod_event` shape (seconds already
    resolved) so the caller folds them straight into the mod timeline."""
    events = []
    for records in applied.values():
        events.extend(_coalesce_ticks(records, to_seconds, tick_step))
    return events


def _coalesce_ticks(records, to_seconds, tick_step):
    """One (beat, modstring, player) run -> minimal windows: a new window
    only where the modstring changes, each ending a tick step past the
    last tick that held it."""
    windows = []
    for beat, modstring, player in records:
        t = to_seconds(beat)
        if windows and windows[-1]['modstring'] == modstring:
            windows[-1]['t_end'] = t + tick_step
        else:
            windows.append({
                'beat': beat, 'len_beats': 0.0, 'modstring': modstring,
                'apply_type': 'perframe', 'player': player,
                't_start': t, 't_end': t + tick_step, 'time_based': True,
            })
    return windows


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
