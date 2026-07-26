"""SimEnvironment: the engine surface the simulated chart runs against.

One LuaHost carrying the permissive bootstrap and recorder-table
bridge (the shared `_PERMISSIVE_BOOTSTRAP` in lua_api), routed onto
SimActors and ONE timeline:

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

import re
from pathlib import Path

from analysis.games.notitg.lua_api import (
    COMMAND_NAMES, GETTER_NAMES, SIM_GETTER_NAMES, _PERMISSIVE_BOOTSTRAP,
    _as_int, _lua_name_set)
from analysis.games.notitg.sim.actor import SimActor
from analysis.games.notitg.xml_actors import (
    _lua50_compat, _lua_expr_body, _strip_lua_wrapper,
    is_lua_function_literal, parse_command_string)
from analysis.player.render.expr.surface import UNRESOLVED
from analysis.player.render.lua import LuaHost
from analysis.player.render.lua.host import LuaScriptError

# Bound on load-time include expansions (the actorgen self-include loop
# generates one actor per iteration; real generators empty in tens).
_MAX_INCLUDE_EXPANSIONS = 512

# SM omits image extensions freely (ActorUtil's directory-listing
# resolution); the asset resolver retries a bare reference with each.
_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

# A classic-command arg that may be a Lua expression over identifiers
# (globals, screen constants) rather than a plain number.
_IDENT_CHAR_RE = re.compile(r'[A-Za-z_]')

# The renderer identity the stubs report. Charts probe the video vendor
# to compensate AFT preserve-texture blending: on opaque-texture GPUs
# ('nvidia') they multiply their feedback alpha down themselves, while
# elsewhere they rely on GL alpha-buffer decay. Our composited captures
# are OPAQUE, so the compensating branch is the one that matches our
# semantics - reporting an nvidia-class renderer keeps chart feedback
# trails at their authored decay instead of saturating.
_VIDEO_VENDOR = 'NVIDIA Corporation'
# Typed like the engine's: GetPreference returns the preference's real
# type, and charts do arithmetic straight off the numeric ones (the
# getfucked2 clock rig adds GlobalOffsetSeconds every frame - a ''
# default there cascades thousands of downstream nil faults).
_PREFERENCES = {
    'VideoRenderers': 'opengl',
    'LastSeenVideoDriver': _VIDEO_VENDOR,
    'GlobalOffsetSeconds': 0.0,
    'InputDuplication': False,
    'Autoplay': False,
    # SM 3.95 stock timing windows; charts derive prank judgment
    # timing from these.
    'JudgeWindowSecondsGreat': 0.090,
    'JudgeWindowSecondsBoo': 0.180,
    'PercentScoreWeightMarvelous': 3,
}

# Broadcast/command recursion guard: a handler may broadcast a message
# whose handlers broadcast again. Depth-only, deliberately: the engine
# has no TOTAL dispatch budget, and a whole-chart sim legitimately runs
# hundreds of thousands of self-scheduled command fires (every chara /
# pool actor loops via sleep+queuecommand). A global total cap starves
# every chain mid-song at once - it silently no-ops the body whose tail
# re-arms the loop. Runaway protection is the depth cap here plus the
# per-update drain cap and tween-overflow guard on the actor.
_MAX_DISPATCH_DEPTH = 24

# Budget of command dispatches within ONE update-body call. A normal
# tick dispatches a handful; only a runaway self-feeding loop (a chart
# action cursor rewound against an unmodeled clock) reaches this.
_MAX_TICK_DISPATCHES = 20000

# Engine starting positions for the real players (ScreenGameplay:
# each enabled player at its PlayerP{n}X style metric, Y=center; the
# classic versus split is center +-160 in the 640 design space).
_PLAYER_START_X = {'PlayerP1': 160.0, 'PlayerP2': 480.0}

# The sim bridge routes two extra getters (GetSecsIntoEffect/GetText)
# the harvest path leaves unrouted; swap the generated __GETTER set
# literal inside the shared bootstrap. The guard catches literal drift -
# a silently failed replace would break the round-trip clocks charts
# build on GetText.
_SIM_BOOTSTRAP = _PERMISSIVE_BOOTSTRAP.replace(
    _lua_name_set(GETTER_NAMES), _lua_name_set(SIM_GETTER_NAMES))
if _SIM_BOOTSTRAP == _PERMISSIVE_BOOTSTRAP:
    raise RuntimeError('sim getter-set substitution found no match')

# Redefine __make_recorder with real GetChild semantics (a later Lua
# definition wins; the screen recorder wraps whichever definition is
# live when it is built). The harvest bridge let GetChild fall through
# to the poke path, RETURNING THE PARENT table - so a proxy targeting
# P1:GetChild('Combo') looked identical to one targeting P1, and pokes
# meant for a child landed on the parent. Here GetChild resolves
# through Python to the XML child by Name, or to a persistent synthetic
# child recorder (engine children like NoteField/Judgment that exist
# without XML nodes).
_SIM_BOOTSTRAP += """
function __make_recorder(id)
    local t = {__recorder_id = id}
    local __GETTER = __SIM_GETTER_SET
    local __COMMAND = __SIM_COMMAND_SET
    setmetatable(t, {__index = function(_, key)
        if key == 'GetChild' then
            return function(_self, name) return __actor_get_child(id, name) end
        end
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
""".replace('__SIM_GETTER_SET', _lua_name_set(SIM_GETTER_NAMES)) \
   .replace('__SIM_COMMAND_SET', _lua_name_set(COMMAND_NAMES))


class SimEnvironment:
    """The singletons, actor registry, and dispatch for one sim run.
    The loop owns time: it calls `set_time(t, beat)` then `drain(t)`
    each tick; everything else happens through the chart's own Lua."""

    def __init__(self, load_seconds: float, rng_seed: int = 0,
                 to_seconds=None, song_dir=None):
        self._song_dir = Path(song_dir) if song_dir is not None else None
        # observe_globals: the sandbox becomes a store-backed proxy that
        # reports every global write, including ones through a computed name
        # (`_G['uksrt_p'..pn..'bonus'] = 0`). That is the only way to know when
        # a cached symbol went stale - chart Lua rebinds globals from command
        # bodies and mod-action payloads that Python never sees into.
        # Raw table ops on `_host.env` no longer see anything: use
        # `_host.has_global` / `_host.global_items`.
        self._host = LuaHost(dialect='luajit21', observe_globals=True)
        # Fan-out for global-write notifications. The host reports one
        # callback; several consumers want the signal (each compiled body
        # caching symbols), so the env owns the list.
        self._global_write_observers: list = []
        # Settled-property mirror sinks, installed by a consumer that can serve
        # actor reads without crossing. None until then, so nothing pays.
        self._prop_mirror = None
        self._tween_mirror = None
        self._host.observe_global_writes(self._notify_global_write)
        self._host.run(_SIM_BOOTSTRAP, name='bootstrap')
        self._load_seconds = float(load_seconds)
        self._to_seconds = to_seconds or (lambda beat: float(beat))
        self._now = float(load_seconds)
        self._beat = 0.0
        self._actors: dict = {}
        self._tables: dict = {}
        self._named_commands: dict = {}
        self._message_commands: dict = {}
        self._load_bodies: list = []
        self._labels: dict = {}
        self._xml_dirs: dict = {}
        self._children: dict = {}
        self._next_id = 0
        self._active: list = []
        self._applied_mods: list = []
        self._shader_flags: list = []
        self._warnings: list = []
        self._faults = 0
        self._fault_messages: list = []
        self._dropped_verbs: dict = {}
        self._deferred_verbs: dict = {}
        self._dispatch_depth = 0
        self._tick_dispatches = 0
        self._tick_budget_warned = False
        self._rng_seed = int(rng_seed)
        self._screen_id: int | None = None
        self._screen_children: dict = {}
        self._xml_child_names: dict = {}
        self._synthetic_children: dict = {}
        self._update_chunk = None
        self._body_chunks: dict = {}
        self._staged_actions: list = []
        self._next_action = 0
        # Queue-carried command names the sweep owns and _fire_queued
        # must NOT execute (the rig's Update re-arm: the sweep runs the
        # update body itself, so the queue-borne copy would double-run
        # the drivers at frozen drain clocks).
        self.suppressed_queued_commands: frozenset = frozenset()
        self._classic_cache: dict = {}
        # Compiled `return (<arg>)` chunks for identifier-bearing
        # classic-command args, keyed by raw arg text (False = does not
        # compile). Evaluated at fire time so globals are current.
        self._arg_chunks: dict = {}
        self._queued: set = set()
        self._include_expansions = 0
        # Phase 3 opt-in: run the per-frame Update body through the AST
        # interpreter (frame_eval) instead of Lua. Off by default so the Lua
        # path stays the baseline until the compiled path passes the same
        # tests; `run_declarative` flips it via `use_compiled_body`.
        self.use_compiled_body = False
        # Opt-in further: run the compiled body through the Rust-native core
        # (notitg_frame_native) instead of the Python interpreter - the same AST,
        # the residue tick loop ported native. Requires `use_compiled_body`; the
        # native path falls back to Python when the wheel is absent.
        self.use_native_body = False
        # Run the per-frame Update body through the C op-stream executor
        # (native_c/: opstream.py -> exec.c computed-goto, no Lua). Byte-
        # identical to the Python CompiledBody (keyframe_diff-gated, 25/25 corpus
        # charts 0-divergence), Lua-free, and the fastest self-contained path.
        # This is the DEFAULT for the compiled-body path: it is preferred
        # whenever `use_compiled_body` is set and falls back to the Python
        # CompiledBody when the .so is absent or a body does not compile, so a
        # missing build never breaks a run. Takes precedence over
        # use_native_body when both are set. Set False to force the Python
        # interpreter for the compiled path (e.g. differential testing).
        self.use_opstream_body = True
        # Bumped whenever control passes to HOST code that could rebind a
        # global without us seeing which one - a command body, a classic body,
        # a mod-action payload. A cross-tick symbol cache has to drop
        # everything when this moves, because the write itself is invisible
        # from here (lupa does not report `X = 5` inside a chunk).
        self._global_gen = 0
        self._compiled_body = None
        self._native_host = None
        self._install()

    @property
    def _native(self):
        """The lazily-built native execution host (interpreter over the shared
        lupa namespace). Load-pass chunks run through this instead of compiling
        as lupa chunks; see sim.native_host."""
        if self._native_host is None:
            from analysis.games.notitg.sim.native_host import NativeHost
            self._native_host = NativeHost(self)
        return self._native_host

    # -- declarative tables (the classic template's event data) ----------

    def read_table(self, name: str) -> list:
        """A classic-template global Lua array (`mods`/`mods2`/
        `mod_actions`) as a list of row dicts, or [] when absent. These
        are DECLARATIVE events - `{beat, len, modstring, end|len, pn}` -
        the load pass populates; reading them straight is the fast path
        that needs no per-frame simulation."""
        table = (self._host.env[name]
                 if self._host.has_global(name) else None)
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

    @staticmethod
    def _row_to_dict(row):
        if not (hasattr(row, '__getitem__')
                and not isinstance(row, (str, bytes))):
            return row
        out = {}
        for key in (1, 2, 3, 4, 5):
            try:
                value = row[key]
            except (KeyError, TypeError):
                value = None
            if value is not None:
                out[key] = value
        return out

    def run_update_body(self, body: str, name: str = 'update-body',
                        rec_id: int | None = None) -> None:
        """Run the `UpdateCommand` body once at the current sim time,
        driving its per-frame closures (a walker reading another actor's
        GetX, a rotator, the proxy grid). The body compiles ONCE (cached)
        and the compiled function runs per tick - per-call recompilation
        of an 800-line chunk would dominate the window sweep. Recorders
        are synced to `now` first so pokes timestamp here;
        `mod_time`/`GetSongBeat` see this tick. The mod_actions cursor
        inside the body no-ops (already fired in the replay pass). Faults
        are swallowed and counted."""
        if self.use_compiled_body:
            self._run_update_body_compiled(body, name, rec_id)
            return
        if self._update_chunk is False:
            return
        if self._update_chunk is None:
            # An expression command (`%prefix.update`) was captured as a
            # function at actor creation; call THAT - re-evaluating the
            # attr per tick breaks once the template's cleanup nils its
            # globals.
            resolved = self._named_commands.get(rec_id, {}).get('Update')
            if callable(resolved):
                table = self._tables[rec_id]
                self._update_chunk = lambda: resolved(table)
            else:
                try:
                    self._update_chunk = self._host.compile(body, name=name)
                except Exception as exc:
                    # A body that does not compile no-ops in the engine
                    # too (a bad command never aborts the chart).
                    self._update_chunk = False
                    self._warnings.append(f'{name}: {exc}')
                    return
        # No eager all-actor sync: pokes and getter reads _sync lazily,
        # so only the actors the body actually touches advance per tick.
        # `self` binds to the Update actor's table exactly as
        # `_run_lua_body` binds it for every other command body - a
        # body reading self:GetSecsIntoEffect() must see its own actor,
        # not whatever the last load body left behind.
        self._host.env['mod_time'] = self._now
        saved = self._host.env['self']
        if rec_id is not None:
            self._host.env['self'] = self._tables.get(rec_id, saved)
        try:
            self._update_chunk()
        except Exception as exc:
            self._record_fault(name, exc)
        finally:
            self._host.env['self'] = saved

    def _run_update_body_compiled(self, body, name, rec_id) -> None:
        """The interpreter-driven Update body (frame_eval, no Lua). Builds one
        persistent CompiledBody the first tick and re-runs it thereafter, so a
        body's accumulator globals carry across ticks. `mod_time` is set like
        the Lua path so a body reading it sees this tick.

        `use_native_body` selects the Rust core (`NativeCompiledBody`) over the
        Python interpreter - the SAME AST, the residue tick loop ported native.
        It falls back to the Python `CompiledBody` when the native wheel is
        absent or does not compile the body, so a missing wheel never breaks a
        run."""
        from analysis.games.notitg.sim.compiled_body import (
            CompiledBody, NativeCompiledBody, OpStreamCompiledBody)

        if self._compiled_body is None:
            if getattr(self, 'use_opstream_body', False):
                ops = OpStreamCompiledBody(self, body, rec_id, name)
                self._compiled_body = ops if ops._ok \
                    else CompiledBody(self, body, rec_id, name)
            elif getattr(self, 'use_native_body', False):
                native = NativeCompiledBody(self, body, rec_id, name)
                self._compiled_body = native if native._ok \
                    else CompiledBody(self, body, rec_id, name)
            else:
                self._compiled_body = CompiledBody(self, body, rec_id, name)
        self._host.env['mod_time'] = self._now
        self._compiled_body.run()

    def replay_mod_actions(self):
        """Fire every `mod_actions` entry once in beat order, each at its
        own fire time so a `GetSongBeat` branch inside sees that moment.
        Function payloads are scheduled closures (actor pokes, spawns);
        string payloads are `MESSAGEMAN:Broadcast(str)`. Each entry runs
        guarded so one fault cannot abort the replay. Returns (fired,
        failed)."""
        rows = self.read_table('mod_actions')
        ordered = sorted(enumerate(rows),
                         key=lambda pair: (_beat_of(pair[1]), pair[0]))
        fired = failed = 0
        for _order, row in ordered:
            payload = row.get(2) if isinstance(row, dict) else None
            if not (callable(payload) or isinstance(payload, str)):
                continue
            beat = _beat_of(row)
            self.set_time(self._to_seconds(beat), beat)
            # Advance every live queue to this fire time BEFORE the
            # action runs: queued tweens begin at their true times, so a
            # later finishtweening collapses only what is genuinely
            # still in flight - not the whole song's accumulated queue.
            self.drain(self._now)
            fired += 1
            try:
                if callable(payload):
                    payload()
                else:
                    self._broadcast(None, payload)
            except Exception as exc:
                failed += 1
                self._record_fault(f'action@{beat}', exc)
        return fired, failed

    def prepare_mod_actions(self) -> int:
        """Stage the `mod_actions` rows for INTERLEAVED firing: the
        declarative sweep calls `fire_mod_actions_until(t)` each tick, so
        actions fire at their true times WITHIN the one walk of song
        time. Queue state then evolves contemporaneously with the update
        body's reads - a driver sampling `GetX` at a sweep tick sees the
        value in force at that moment, not the whole-song end state the
        ahead-of-time replay left behind. Returns the staged count."""
        rows = self.read_table('mod_actions')
        ordered = sorted(enumerate(rows),
                         key=lambda pair: (_beat_of(pair[1]), pair[0]))
        self._staged_actions = [
            (self._to_seconds(_beat_of(row)), _beat_of(row), payload)
            for _order, row in ordered
            if (payload := row.get(2) if isinstance(row, dict) else None)
            is not None and (callable(payload) or isinstance(payload, str))]
        self._next_action = 0
        # The classic template's update body replays the same table
        # itself (`while curaction <= table.getn(mod_actions) ...`).
        # With the sweep firing every action at its true time, that
        # in-body replay would run each action a SECOND time one tick
        # later (tween chains restart 1+ chain-lengths late, leaving
        # e.g. a decayed-to-zero effectmagnitude frozen at a stale
        # nonzero value). Park the template's cursor past the end so
        # its loop never enters.
        if self._staged_actions and self._host.has_global('curaction'):
            self._host.env['curaction'] = float(len(rows) + 1)
        return len(self._staged_actions)

    def fire_mod_actions_until(self, t: float) -> None:
        """Fire every staged action with fire time <= `t`, in order, each
        at its own clock (time set + queues drained to the fire moment,
        exactly as the standalone replay did). The caller re-asserts its
        own tick time afterwards."""
        while self._next_action < len(self._staged_actions):
            fire_s, beat, payload = self._staged_actions[self._next_action]
            if fire_s > t:
                return
            self._next_action += 1
            self.set_time(fire_s, beat)
            self.drain(fire_s)
            try:
                if callable(payload):
                    payload()
                else:
                    self._broadcast(None, payload)
            except Exception as exc:
                self._record_fault(f'action@{beat}', exc)

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

    def _verb_dropped(self, rec_id, verb) -> None:
        """A poke fell through every dispatch with no IGNORED/DEFERRED
        entry: the verb is unmapped, or mapped but never routed (the
        Actor:cmd class of bug - claimed handled, silently lost). One
        warning per verb names the first offender; the count is
        probe-visible via `dropped_verbs`."""
        count = self._dropped_verbs.get(verb, 0)
        self._dropped_verbs[verb] = count + 1
        if count == 0:
            self._warnings.append(
                f"verb '{verb}' dropped (no dispatch matched) - first "
                f'from {self._label(rec_id)} at t={self._now:.2f}')

    @property
    def dropped_verbs(self) -> dict:
        """verb -> silently-dropped poke count for this run."""
        return dict(self._dropped_verbs)

    def _verb_deferred(self, verb) -> None:
        self._deferred_verbs[verb] = self._deferred_verbs.get(verb, 0) + 1

    @property
    def deferred_verbs(self) -> dict:
        """verb -> poke count for DEFERRED (documented-unbuilt) verbs
        this chart exercised - the coverage report's raw feed."""
        return dict(self._deferred_verbs)

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

    # -- recorded-state surface (the shapes the modfile element/screen/
    # field producers consume off a sim env directly) -------------------

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
        for key, value in self._host.global_items():
            if not isinstance(key, str):
                continue
            actor = self._actor_for_table(value)
            if actor is not None:
                out[key] = actor
        return out

    def named_actor_ids(self) -> dict:
        """recorder id -> bound global name, for labeling producer output.
        When several globals bind the same actor, the lexicographically
        smallest name wins - a STABLE tiebreak, because `_host.env.items()`
        iteration order is not deterministic (a hash-seed artifact), and the
        chosen label is what keyframe_diff joins on."""
        out = {}
        for key, value in sorted(
                ((k, v) for k, v in self._host.global_items()
                 if isinstance(k, str)), key=lambda kv: kv[0]):
            rec_id = self._table_rec_id(value)
            if rec_id is not None and rec_id not in out:
                out[rec_id] = key
        return out

    def named_actor_id(self, name: str) -> int | None:
        """recorder id of the actor bound to global `name` (`name = self`),
        or None when the global is unbound / not one of our recorder tables.
        The inverse of the label map: the lazy field producer reads the SAME
        actor eager's `named_actor_keyframes()[name]` came from."""
        value = (self._host.env[name]
                 if self._host.has_global(name) else None)
        return self._table_rec_id(value) if value is not None else None

    def screen_child_ids(self) -> dict:
        """screen child name ('PlayerP1', ...) -> recorder id."""
        return dict(self._screen_children)

    def _actor_for_table(self, table) -> SimActor | None:
        rec_id = self._table_rec_id(table)
        return self._actors.get(rec_id) if rec_id is not None else None

    @staticmethod
    def _table_rec_id(table) -> int | None:
        if not hasattr(table, '__getitem__'):
            return None
        try:
            rec_id = table['__recorder_id']
        except (KeyError, TypeError):
            return None
        return _as_int(rec_id) if rec_id is not None else None

    # -- load pass ---------------------------------------------------------

    def load_actors(self, root) -> list:
        """Load the tree in engine creation order (Actor::LoadFromNode):
        per actor, its Condition gates it, registration + InitCommand
        fire at creation, then its children load - so a child's
        Condition or InitCommand may read a global the parent's
        InitCommand just bound (the XGML template's `prefix`).
        OnCommand then runs over the loaded tree in a second tree-order
        pass (screen start); it may read a global a LATER actor's
        InitCommand bound - the AFT rigs depend on this. The chart's
        self-scheduling Update chain arms itself here (its queuecommand
        lands on the real queue)."""
        self._load_actor(root)
        self._run_load(root, 'OnCommand')
        return self._warnings

    def _load_actor(self, actor) -> bool:
        """Create one actor and its subtree; False when the actor's
        Condition drops it (the caller removes the subtree, so it never
        registers, draws, or receives dispatch). Conditions are engine
        gates (ActorUtil skips falsy actors), and templates also use
        them as load-time setup code (`Condition="(function() ...
        end)()"` defining helpers the InitCommands call). A condition
        that faults keeps its actor: a permissive-stub error must not
        drop real content."""
        if self._condition_falsy(actor):
            return False
        self._resolve_at_attrs(actor)
        self._expand_includes(actor)
        rec_id = self._register_one(actor)
        # NotITG's Var= extension binds the actor into a Lua global at
        # load (4400+ uses corpus-wide: `Var="spriteAscii"` then
        # `spriteAscii:GetTexture()` from another actor).
        var = actor.attrs.get('Var', '')
        if var:
            self._host.env[var] = self._tables[rec_id]
        value = actor.attrs.get('InitCommand', '')
        if value:
            self._load_bodies.append((rec_id, value))
        if value.startswith('%'):
            self._run_lua_body(rec_id, _strip_lua_wrapper(value),
                               f'{self._label(rec_id)}.InitCommand',
                               load=True)
        elif value:
            self._run_classic_body(rec_id, value)
        for child in list(actor.children):
            if not self._load_actor(child):
                actor.children.remove(child)
        self._children[rec_id] = [self._id_for(c) for c in actor.children]
        for child in actor.children:
            child_name = child.attrs.get('Name', '')
            if child_name:
                self._xml_child_names[(rec_id, child_name)] = \
                    self._id_for(child)
        return True

    def _condition_falsy(self, actor) -> bool:
        expr = actor.attrs.get('Condition', '').strip()
        if not expr:
            return False
        name = f'{self._actor_label(actor, "?")}.Condition'
        result, error = self._native.eval_expr(expr)
        if error is not None or result is UNRESOLVED:
            self._warnings.append(f'{name}: {error or "unresolved"}')
            # A gate the host cannot ANSWER keeps a plain actor (a permissive-
            # stub gap must not drop real content) but DROPS a looped include -
            # re-expanding on a broken condition would spin the actorgen loop
            # forever. UNRESOLVED belongs here and not below: it is the
            # interpreter's reply where lupa raised ("attempt to call a nil
            # value" for a name the surface does not model), so reading it as a
            # nil - a chart saying "skip this" - deletes real content.
            return getattr(actor, '_expand_include', None) is not None
        return result is None or result is False

    def _resolve_at_attrs(self, actor) -> None:
        """`@expr` attribute values evaluate as Lua at actor load
        (actorgen's `Type="@actorgen.Type()"`); the result replaces the
        value. An expression that faults - or that the host cannot answer -
        leaves the raw value."""
        for attr, value in list(actor.attrs.items()):
            if not value.startswith('@'):
                continue
            name = f'{self._actor_label(actor, "?")}.{attr}@'
            result, error = self._native.eval_expr(value[1:].strip())
            if error is not None or result is UNRESOLVED:
                self._warnings.append(f'{name}: {error or "unresolved"}')
                continue
            if isinstance(result, (str, int, float)) and not isinstance(
                    result, bool):
                actor.attrs[attr] = str(result)

    def _expand_includes(self, actor) -> None:
        """Run any deferred include expansion (the actorgen self-include
        loop; `File="@expr"` dynamic includes) now that the actor's
        Condition passed - the spliced subtree loads as this actor's
        children. Bounded so a runaway generator cannot spin the load
        forever."""
        expand = getattr(actor, '_expand_include', None)
        dynamic = getattr(actor, '_expand_dynamic_include', None)
        if expand is None and dynamic is None:
            return
        if self._include_expansions >= _MAX_INCLUDE_EXPANSIONS:
            if self._include_expansions == _MAX_INCLUDE_EXPANSIONS:
                self._warnings.append('include expansion cap reached')
                self._include_expansions += 1
            return
        self._include_expansions += 1
        label = f'{self._actor_label(actor, "?")}.File'
        try:
            if expand is not None:
                actor._expand_include = None
                expand()
            elif dynamic is not None:
                actor._expand_dynamic_include = None
                dynamic(actor.attrs.get('File', ''))
        except Exception as exc:
            self._warnings.append(f'{label}: {exc}')

    def _register_one(self, actor) -> int:
        rec_id = self._id_for(actor)
        self._labels[rec_id] = self._actor_label(actor, rec_id)
        base_dir = getattr(actor, '_base_dir', None)
        if base_dir is not None:
            self._xml_dirs[rec_id] = f'{base_dir}/'
        reference = (actor.attrs.get('Texture') or actor.attrs.get('Load')
                     or actor.attrs.get('File'))
        if reference:
            # The actor's own texture file, so GetTexture() can answer
            # with a 'file:' marker (what uniformTexture binds samplers
            # to). Unresolvable references (dynamic includes, theme
            # paths) just leave the marker unavailable.
            self._actors[rec_id].texture_file = self._resolve_asset(
                rec_id, reference)
        if actor.kind == 'ActorFrameTexture':
            # An `<... Type="ActorFrameTexture">` render target: name it so
            # its GetTexture() marker reaches copy/post-process sprites
            # even when the chart never calls SetTextureName (getfucked2
            # references AFTs by Lua global, not name).
            self._actors[rec_id].mark_aft(f'aft#{rec_id}')
        for message, body in actor.message_commands().items():
            resolved = self._load_resolve(rec_id, f'msg:{message}', body)
            if resolved is not None:
                self._message_commands.setdefault(message, []).append(
                    (rec_id, resolved))
        named = {name: self._load_resolve(rec_id, f'cmd:{name}', body)
                 for name, body in actor.named_commands().items()}
        named = {name: body for name, body in named.items()
                 if body is not None}
        if named:
            self._named_commands[rec_id] = named
        return rec_id

    def _load_resolve(self, rec_id, suffix, body):
        """Command bodies stay strings, except `%expr` expression
        commands (not function literals): the engine evaluates those
        ONCE at actor creation and stores the result - `%prefix.update`
        must capture the function before the template's cleanup nils
        the `prefix` global. Returns the body string, a captured Lua
        function, or None for a command that resolved to nothing."""
        if not body.startswith('%') or is_lua_function_literal(body):
            return body
        name = f'{self._label(rec_id)}.{suffix}'
        try:
            result = self._host.compile(
                f'return ({_lua_expr_body(body)})', name=name)()
        except Exception as exc:
            self._warnings.append(f'{name}: {exc}')
            return None
        return result if callable(result) or isinstance(result, str) \
            else None

    def _run_load(self, actor, attr) -> None:
        rec_id = self._id_for(actor)
        value = actor.attrs.get(attr, '')
        if value:
            self._load_bodies.append((rec_id, value))
        if value.startswith('%'):
            self._run_lua_body(rec_id, _strip_lua_wrapper(value),
                               f'{self._label(rec_id)}.{attr}', load=True)
        elif value:
            self._run_classic_body(rec_id, value)
        for child in actor.children:
            self._run_load(child, attr)

    @staticmethod
    def _actor_label(actor, rec_id) -> str:
        """Fault/chunk label carrying the actor's source XML file and its
        Name (kind#rec-id when anonymous), so a fault or Lua syntax error
        says WHICH actor in WHICH file it came from."""
        name = actor.attrs.get('Name') or f'{actor.kind}#{rec_id}'
        src = getattr(actor, '_src_xml', '')
        return f'{src}:{name}' if src else name

    def _label(self, rec_id) -> str:
        return self._labels.get(rec_id, f'actor#{rec_id}')

    def actor_tree_paths(self) -> dict:
        """rec_id -> a STABLE structural identity (its path from the root by
        actor kind + sibling index, e.g. `ActorFrame[3]/Quad[1]`). Unlike the
        `kind#rec_id` label or the `actor#rec_id` fallback, this is chart-
        intrinsic - identical across any run of the same chart regardless of
        rec-id assignment order - so a cross-run diff can join anonymous actors
        that carry no `file:Name`. Roots are the registered actors with no
        parent, ordered by rec_id (load order is deterministic)."""
        parent_of = {}
        for parent, kids in self._children.items():
            for child in kids:
                parent_of[child] = parent
        paths = {}

        def kind_of(rid):
            actor = self._actors.get(rid)
            return getattr(actor, 'kind', 'Actor') if actor else 'Actor'

        def path_of(rid):
            if rid in paths:
                return paths[rid]
            parent = parent_of.get(rid)
            if parent is None:
                seg = f'{kind_of(rid)}'
            else:
                siblings = self._children.get(parent, ())
                idx = siblings.index(rid) if rid in siblings else 0
                seg = f'{path_of(parent)}/{kind_of(rid)}[{idx}]'
            paths[rid] = seg
            return seg

        for rid in sorted(self._labels):
            path_of(rid)
        return paths

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
        actor.queue_notify = lambda: self._queued.add(rec_id)
        actor.dropped_notify = lambda verb: self._verb_dropped(rec_id, verb)
        actor.deferred_notify = self._verb_deferred
        actor.asset_resolver = lambda raw: self._resolve_asset(rec_id, raw)
        self._actors[rec_id] = actor
        self._tables[rec_id] = self._host.env["__make_recorder"](rec_id)
        self._active.append(rec_id)
        return rec_id

    # -- time --------------------------------------------------------------

    # The clock CONTRACT a guard/interpreter Surface reads (guard_surface.py,
    # frame_eval). Named apart from the private `_beat`/`_now` fields so the
    # value surface has a stable seam that does not reach into internals: a
    # guard reading `beat` or `mod_time` resolves through these, seeing the
    # exact tick clock the chart's own Update body sees (one source of truth).

    @property
    def _clock_beat(self) -> float:
        return self._beat

    def _song_time(self) -> float:
        return self._now

    def set_time(self, t: float, beat: float) -> None:
        self._now = float(t)
        self._beat = float(beat)
        # Each tick gets a fresh dispatch budget; only a runaway loop
        # inside one tick exhausts it.
        self._tick_dispatches = 0

    def drain(self, t: float, defer_queued: bool = True) -> None:
        """Advance every LIVE tween queue to `t`, firing queue-borne
        commands as the drains reach them. Only actors in the queued set
        (notified when their queue went non-empty) iterate; an actor
        whose queue drained empty leaves the set until re-armed - so a
        whole-song 60Hz drain sweep costs proportional to actual queue
        activity, not the actor count. `defer_queued=False` expands
        self-requeue chains to quiescence (the loop's final drain)."""
        for rec_id in list(self._queued):
            actor = self._actors[rec_id]
            if actor._tweens:
                self._drain_actor(rec_id, actor, t, defer_queued)
            if not actor._tweens:
                self._queued.discard(rec_id)

    def _drain_actor(self, rec_id, actor, t, defer_queued=True) -> None:
        actor.update_to(t, lambda name: self._fire_queued(rec_id, name),
                        defer_queued=defer_queued)

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
            elif name not in self.suppressed_queued_commands:
                body = self._named_commands.get(rec_id, {}).get(name)
                if body is not None:
                    self._run_command_body(
                        rec_id, body, f'{self._label(rec_id)}.cmd:{name}')
        except Exception as exc:
            self._record_fault(f'fire:{name}', exc)

    # -- dispatch ----------------------------------------------------------

    def _run_command_body(self, rec_id, body, name) -> None:
        if self._dispatch_depth >= _MAX_DISPATCH_DEPTH:
            return
        self._tick_dispatches += 1
        if self._tick_dispatches > _MAX_TICK_DISPATCHES:
            if not self._tick_budget_warned:
                self._tick_budget_warned = True
                self._warnings.append(
                    f'tick dispatch budget exhausted at {name}')
            return
        self._dispatch_depth += 1
        try:
            if not isinstance(body, str):
                self._call_command_fn(rec_id, body, name)
            elif body.startswith('%'):
                self._run_lua_body(rec_id, _strip_lua_wrapper(body), name)
            else:
                self._run_classic_body(rec_id, body)
        finally:
            self._dispatch_depth -= 1

    def _call_command_fn(self, rec_id, fn, name) -> None:
        """A load-resolved `%expr` command: call the captured function
        with the actor's recorder as `self`."""
        self._global_gen += 1
        self._sync(rec_id)
        saved = self._host.env['self']
        self._host.env['self'] = self._tables[rec_id]
        try:
            fn(self._tables[rec_id])
        except Exception as exc:
            self._record_fault(name, exc)
        finally:
            self._host.env['self'] = saved

    def install_prop_mirror(self, on_prop, on_tween) -> None:
        """Register a consumer for settled property writes and tween-queue
        transitions. Registers NOTHING on any actor yet - see `mirror_actor`."""
        self._prop_mirror = on_prop
        self._tween_mirror = on_tween

    def mirror_actor(self, rec_id: int) -> None:
        """Start reporting this actor's settled state to the mirror consumer.

        Per actor, on demand, because the populations are wildly lopsided: on
        gat 147 actors have a mirrored property written while the body READS
        only 6 of them. Mirroring all of them cost more than the reads it
        served (measured +11.5%); mirroring the read ones is nearly free.

        The caller invokes this the first time it has to answer a read for an
        actor the hard way, so the first read pays and the rest are served."""
        actor = self._actors.get(rec_id)
        if actor is None or actor.prop_notify is not None:
            return
        on_prop, on_tween = self._prop_mirror, self._tween_mirror
        if on_prop is None:
            return
        actor.prop_notify = (
            lambda prop, value, r=rec_id: on_prop(r, prop, value))
        actor.tween_notify = lambda busy, r=rec_id: on_tween(r, busy)
        # Seed: the actor has been poked since creation and the mirror is empty.
        for prop, value in actor._current.items():
            on_prop(rec_id, prop, value)
        on_tween(rec_id, bool(actor._tweens))

    def observe_global_write(self, callback) -> None:
        """Call `callback(name)` whenever chart Lua writes that global - a
        computed name included. The sandbox is proxied precisely so this is
        knowable; see LuaHost(observe_globals=True)."""
        self._global_write_observers.append(callback)

    def _notify_global_write(self, name) -> None:
        for observer in self._global_write_observers:
            observer(name)

    def _run_lua_body(self, rec_id, body, name, load=False) -> None:
        """Run a command body with `self` = the actor's recorder. Bodies
        fire many times (a chara Idle chain re-queues itself for the
        whole song), so each body string compiles ONCE and the cached
        chunk is called thereafter - `self` resolves through the sandbox
        env at call time, so one chunk serves every fire."""
        self._global_gen += 1
        chunk = self._body_chunks.get(body)
        if chunk is None:
            try:
                chunk = self._host.compile(body, name=name)
            except Exception as exc:
                if load:
                    self._warnings.append(f'{name}: {exc}')
                else:
                    self._record_fault(name, exc)
                return
            self._body_chunks[body] = chunk
        self._sync(rec_id)
        saved = self._host.env['self']
        self._host.env['self'] = self._tables[rec_id]
        try:
            chunk()
        except Exception as exc:
            if load:
                self._warnings.append(f'{name}: {exc}')
            else:
                self._record_fault(name, exc)
        finally:
            self._host.env['self'] = saved

    def _run_classic_body(self, rec_id, value) -> None:
        self._global_gen += 1
        self._sync(rec_id)
        actor = self._actors[rec_id]
        parsed = self._classic_cache.get(value)
        if parsed is None:
            parsed = parse_command_string(value)
            self._classic_cache[value] = parsed
        for verb, args in parsed:
            if verb in ('queuecommand', 'playcommand') and args:
                self._actor_command(rec_id, verb, args[0])
            else:
                actor.poke(verb, [self._classic_arg(a) for a in args])

    def _classic_arg(self, arg):
        """NotITG evaluates classic-command args as Lua EXPRESSIONS, so
        charts write `linear,my_dur*2` and `x,SCREEN_CENTER_X-220` -
        identifiers resolved against the live globals at fire time. Args
        with no identifier characters keep the numeric fast path; an arg
        whose evaluation is not a number keeps its raw string
        (`blend,add`, `effectclock,music`)."""
        if not isinstance(arg, str) or not _IDENT_CHAR_RE.search(arg):
            return arg
        chunk = self._arg_chunks.get(arg)
        if chunk is None:
            try:
                chunk = self._host.compile(f'return ({arg})', name='arg')
            except LuaScriptError:
                chunk = False
            self._arg_chunks[arg] = chunk
        if chunk is False:
            return arg
        try:
            value = chunk()
        except Exception:
            return arg
        return float(value) if isinstance(value, (int, float)) else arg

    def _broadcast(self, _self, name=None, *_a) -> None:
        """MESSAGEMAN:Broadcast - run <name>MessageCommand on every
        registering actor, at the current time."""
        if not isinstance(name, str):
            return
        for rec_id, body in self._message_commands.get(name, ()):
            self._run_command_body(
                rec_id, body, f'{self._label(rec_id)}.msg:{name}')

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
                self._run_command_body(
                    rec_id, commands[name],
                    f'{self._label(rec_id)}.cmd:{name}')
        for child_id in self._children.get(rec_id, ()):
            self._command_subtree(child_id, verb, name)

    # -- Lua bridge callbacks ----------------------------------------------

    def _actor_poke(self, rec_id, verb, *args) -> None:
        rec_id = _as_int(rec_id)
        actor = self._actors.get(rec_id)
        if actor is None:
            return
        if verb == 'SetTarget':
            target_id = self._table_rec_id(args[0]) if args else None
            if target_id is not None and target_id in self._actors:
                actor.proxy_target = target_id
            return
        if verb == 'cmd':
            # Actor:cmd(s) parses s as an actor-command string and runs it
            # NOW (Actor::RunCommands @0075b280) - the same dispatch as a
            # classic XML command body on this actor.
            if args and isinstance(args[0], str):
                self._run_classic_body(rec_id, args[0])
            return
        self._sync(rec_id)
        actor.poke(verb, list(args))

    def _actor_get(self, rec_id, verb):
        # Numeric getters charts do arithmetic on: a permissive-table
        # answer cascades type faults downstream, so these answer with
        # real numbers (GetNumChildren) or the empty-count 0.
        if verb == 'GetNumChildren':
            return float(len(self._children.get(_as_int(rec_id), ())))
        if verb in ('GetNumTapsInRange', 'GetNumVertices'):
            return 0.0
        if verb == 'GetXMLDir':
            return self._xml_dirs.get(_as_int(rec_id), '')
        actor = self._actors.get(_as_int(rec_id))
        if actor is None:
            return self._host.env['__permissive']()
        value = actor.read(verb)
        if value is None:
            return self._host.env['__permissive']()
        return value

    def _actor_get_child(self, rec_id, name=None):
        """`actor:GetChild(name)` - the XML child bound to that Name, or
        a persistent synthetic child recorder (engine children like
        NoteField/Judgment/Combo exist without XML nodes; proxies target
        them and their pokes must not land on the parent)."""
        if not isinstance(name, str):
            return self._host.env['__permissive']()
        parent_id = _as_int(rec_id)
        child_id = self._xml_child_names.get((parent_id, name))
        if child_id is None:
            child_id = self._synthetic_children.get((parent_id, name))
        if child_id is None:
            child_id = self._new_actor()
            self._synthetic_children[(parent_id, name)] = child_id
        return self._tables[child_id]

    def synthetic_child_ids(self) -> dict:
        """(parent recorder id, child name) -> recorder id, for the
        producers' proxy-source resolution."""
        return dict(self._synthetic_children)

    def _screen_get_child(self, name):
        """SCREENMAN:GetTopScreen():GetChild(name) - a persistent
        per-name recorder. The players' pokes record the base-field
        visibility stream (P1:hidden(1)); any other child (NoteField,
        Judgment) is a harmless poke-able target for proxy SetTarget.

        Player recorders are seeded with the ENGINE's starting state
        (ScreenGameplay places each player at its style X metric,
        Y=center): a chart's first position tween eases FROM that value,
        exactly as SetX eases from the current engine position - without
        it the intro bounce would ease from 0 (offscreen)."""
        if not isinstance(name, str):
            return self._host.env['__permissive']()
        rec_id = self._screen_children.get(name)
        if rec_id is None:
            rec_id = self._new_actor()
            self._screen_children[name] = rec_id
            start_x = _PLAYER_START_X.get(name)
            if start_x is not None:
                actor = self._actors[rec_id]
                actor.poke('x', [start_x])
                actor.poke('y', [240.0])
        return self._tables[rec_id]

    def _resolve_asset(self, rec_id, raw: str) -> str | None:
        """A chart-authored texture path (a File= attribute,
        Sprite:Load's argument) resolved to an absolute image file, or
        None when nothing exists there. Charts build these from
        GetSongDir() (absolute already) or relative to their own XML;
        both roots are tried in that order, with SM's omitted-extension
        leniency (`File="noise"` finds noise.png)."""
        path = Path(raw)
        if path.is_absolute():
            return self._existing_image(path)
        for base in (self._xml_dirs.get(rec_id), self._song_dir):
            if base:
                found = self._existing_image(Path(base) / raw)
                if found:
                    return found
        return None

    @staticmethod
    def _existing_image(candidate: Path) -> str | None:
        if candidate.is_file():
            return str(candidate)
        for suffix in _IMAGE_SUFFIXES:
            with_ext = candidate.with_name(candidate.name + suffix)
            if with_ext.is_file():
                return str(with_ext)
        return None

    def _loadfile(self, path=None):
        """Sandboxed `loadfile`: charts load their template libraries
        from the song directory (the Mirin/actorgen rigs' `xero.
        loadscript`). Only files under the song dir resolve - anything
        else returns nil, exactly as loadfile does for a missing file.
        Sources pass the 5.0 lexer rewrite like every other chunk."""
        if self._song_dir is None or not isinstance(path, str):
            return None
        text = path
        root = str(self._song_dir)
        if text.startswith(root):
            text = text[len(root):]
        candidate = (self._song_dir / text.lstrip('/')).resolve()
        if not (candidate.is_relative_to(self._song_dir.resolve())
                and candidate.is_file()):
            return None
        source = _lua50_compat(
            candidate.read_text(encoding='utf-8', errors='replace'))
        try:
            return self._host.compile(source, name=text.lstrip('/'))
        except Exception as exc:
            self._warnings.append(f'loadfile {text}: {exc}')
            return None

    # -- engine singletons -------------------------------------------------

    def _install(self) -> None:
        host = self._host
        singleton = host.env['__make_singleton']

        host.expose('__actor_poke', self._actor_poke)
        host.expose('__actor_get', self._actor_get)
        host.expose('__actor_command', self._actor_command)
        host.expose('__actor_get_child', self._actor_get_child)
        host.expose('__screen_get_child', self._screen_get_child)

        host.expose('loadfile', self._loadfile)
        # os.clock/os.time read the SIM clock: charts derive per-frame
        # deltaTime from them (Government Knows' CatUpdater), and the
        # deterministic sweep time is the engine-true answer here. A
        # chart assigning its own `os` global simply shadows this.
        host.expose('os', host.to_lua({'clock': lambda: self._now,
                                       'time': lambda: self._now}))
        host.expose('SOUND', singleton(host.to_lua({
            'PlayOnce': lambda _self, *_a: None,
            'PlayMusicPart': lambda _self, *_a: None,
            'StopMusic': lambda _self, *_a: None,
        })))
        song = singleton(host.to_lua({
            'GetSongDir': lambda _self: (
                f'{self._song_dir}/' if self._song_dir else ''),
        }))
        host.expose('GAMESTATE', singleton(host.to_lua({
            'GetCurrentSong': lambda _self: song,
            # Charts warp song time (heart's choose-your-minigame rig
            # rewinds its action cursor after SetSongBeat); honoring the
            # poke keeps their within-tick control flow consistent - the
            # sweep clock re-asserts the real beat next tick.
            'SetSongBeat': self._set_song_beat,
            'GetSongBeat': lambda _self: self._beat,
            'GetSongBeatNoOffset': lambda _self: self._beat,
            'GetSongTime': lambda _self: self._now,
            # ArrowEffects::GetZPos(player, col, yoffset): the note z
            # from active z mods. Our sim world has no z-spline state
            # (SetZSpline is unconsumed), so 0.0 is its true value; a
            # numeric answer also keeps the caller's Update tick alive
            # (the permissive-table fallback faulted math.abs and
            # killed everything after it in the same guard).
            'GetZ': lambda _self, *_a: 0.0,
            # Charts gate their whole modfile on a minimum engine build
            # (tonumber(GetVersionDate()) >= ...); report a modern one.
            'GetVersionDate': lambda _self: '20990101',
            'SetShaderFlag': self._set_shader_flag,
            'SetShaderFlagNum': self._set_shader_flag_num,
            'ApplyGameCommand': self._apply_game_command,
            'ApplyModifiers': self._apply_modifiers,
        })))
        host.expose('PREFSMAN', singleton(host.to_lua({
            'GetPreference': lambda _self, key=None: _PREFERENCES.get(
                str(key), ''),
        })))
        host.expose('MESSAGEMAN', singleton(host.to_lua({
            'Broadcast': self._broadcast,
        })))
        self._screen_id = self._new_actor()
        top_screen = self._host.env["__make_screen_recorder"](self._screen_id)
        self._tables[self._screen_id] = top_screen
        host.expose('SCREENMAN', singleton(host.to_lua({
            'SystemMessage': lambda _self, *_a: None,
            'PostScreenMessage': lambda _self, *_a: None,
            'GetTopScreen': lambda _self, *_a: top_screen,
        })))
        host.expose('DISPLAY', singleton(host.to_lua({
            'GetDisplayWidth': lambda _self: 640.0,
            'GetDisplayHeight': lambda _self: 480.0,
            'GetVendor': lambda _self: _VIDEO_VENDOR,
        })))
        # STATSMAN's score chain returns numbers (end-of-song bonus
        # closures do arithmetic on GetPossibleDancePoints); a permissive
        # table there faults the final action.
        player_stats = singleton(host.to_lua({
            'GetPossibleDancePoints': lambda _self: 0.0,
            'GetActualDancePoints': lambda _self: 0.0,
            'SetActualDancePoints': lambda _self, *_a: None,
        }))
        stage_stats = singleton(host.to_lua({
            'GetPlayerStageStats': lambda _self, _p=None: player_stats,
        }))
        host.expose('STATSMAN', singleton(host.to_lua({
            'GetCurStageStats': lambda _self: stage_stats,
        })))
        for name in ('SONGMAN', 'THEME', 'GAMEMAN',
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

    def _set_song_beat(self, _self, beat=None, *_a) -> None:
        if isinstance(beat, (int, float)):
            self._beat = float(beat)

    def _apply_modifiers(self, _self, modstring=None, pn=None,
                         *_a) -> None:
        if isinstance(modstring, str):
            self._applied_mods.append(
                (self._now, self._beat, modstring.strip(), _as_int(pn)))


def _beat_of(row) -> float:
    value = row.get(1) if isinstance(row, dict) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
