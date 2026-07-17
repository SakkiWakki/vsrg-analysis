"""Stub SM5 environment for running Etterna (.lua) modfile actor trees.

Etterna modcharts are wired through `#FGCHANGES:<beat>=<file>.lua=...`
and, unlike NotITG's XML actor trees with `ApplyModifiers` strings, they
are SM5 actor-tree Lua: a script builds `Def.ActorFrame{...}` /
`Def.Actor{...}` / `Def.Quad{...}` tables and `return`s the root. The
mod timeline is driven by METHOD calls on a PlayerOptions object:

    local po = GAMESTATE:GetPlayerState(PLAYER_1)
                        :GetPlayerOptions('ModsLevel_Preferred')
    po:Drunk(0.5, 3)   -- set drunk to 50% approaching at speed 3

so there is no modstring to parse; the method NAME is the mod and its
first two args are `(value, approach_speed)` (the FLOAT_INTERFACE Lua
binding, etterna PlayerOptions.cpp / OptionsBinding.h).

Two surfaces are recorded:

- MOD EVENTS: each `po:Drunk(v, s)` (etc.) becomes a `ModEvent`-shaped
  dict at the calling actor's CURRENT command-clock time. SM5 tween
  verbs (`sleep`/`linear`/...) advance that clock, so a script that
  sleeps between poptions calls lays down a mod TIMELINE. Unlike
  NotITG's per-frame `clearall` windows, SM5 poptions are PERSISTENT
  (the engine holds the value until the next call), so an event carries
  no auto-revert; the channel compiler holds the last value forward.

- ACTOR VISUALS: `Def.Quad`/`Def.Sprite`/`Def.BitmapText` actors with
  InitCommand/OnCommand FUNCTIONS that poke `self` (`self:xy(..)`,
  `self:diffuse(..)`, `self:zoomto(..)`) record onto a `RecordingActor`,
  compiled to storyboard Elements by the modfile module.

Every engine global is permissive: unknown singletons/methods resolve to
a dummy that swallows any call and returns another permissive value, so
an interactive modfile (SOUND callbacks, input hooks, chart queries -
the minesweeper ceiling) loads without crashing, harvesting whatever mod
and actor data it CAN reach before the unsupported call no-ops.

Per-frame `Update` functions (`self:SetUpdateFunction(f)`) are recorded
but NOT executed: their body reads live gameplay state we do not
simulate, mirroring the NotITG mod_actions precedent (record-don't-run,
counted).
"""
from __future__ import annotations

from analysis.games.etterna.recording_actor import ActorClock, RecordingActor
from analysis.player.render.lua import LuaHost

# The player index a poptions object was obtained for. Modfiles almost
# always drive PLAYER_1; PLAYER_2 is exposed so a dual-player script does
# not fault, and its events route to channel player 1.
_PLAYER_ENUMS = {'PlayerNumber_P1': 0, 'PlayerNumber_P2': 1}

_DEFAULT_APPROACH_SPEED = 1.0

# SM5 bootstrap: Def.* factories, the PlayerOptions recorder, and the
# recording-actor bridge, all as Lua-side tables so colon-method calls
# (`Def.Quad{...}`, `po:Drunk(..)`, `self:xy(..)`) dispatch naturally.
_BOOTSTRAP = """
-- Def.<Kind>{ props } records the actor class on the returned table and
-- hands the table back verbatim, so InitCommand/OnCommand FUNCTIONS,
-- Name=, children (t[#t+1]=...) survive for the Python compiler to read.
_G.Def = setmetatable({}, {__index = function(_, kind)
    return function(props)
        props = props or {}
        props.Class = kind
        return props
    end
end})

-- A permissive value: any field read, call, arithmetic, comparison, or
-- concatenation yields a benign result, so an unmodeled engine chain
-- (A:B():C().d, 'x'..thing, thing + 1) degrades instead of aborting the
-- harvest. Arithmetic collapses to 0 and concatenation to '' so a
-- deeply interactive modfile (SOUND/input callbacks, DSP) runs as far as
-- it can before its real work no-ops.
local permissive
local _PMT = {}
_PMT.__index = function(_, _k) return permissive() end
_PMT.__call = function(_, ...) return permissive() end
_PMT.__concat = function(a, b)
    local sa = type(a) == 'table' and '' or tostring(a)
    local sb = type(b) == 'table' and '' or tostring(b)
    return sa .. sb
end
_PMT.__tostring = function() return '' end
_PMT.__add = function() return 0 end
_PMT.__sub = function() return 0 end
_PMT.__mul = function() return 0 end
_PMT.__div = function() return 0 end
_PMT.__mod = function() return 0 end
_PMT.__pow = function() return 0 end
_PMT.__unm = function() return 0 end
_PMT.__len = function() return 0 end
_PMT.__lt = function() return false end
_PMT.__le = function() return false end
_PMT.__eq = function() return false end
permissive = function()
    return setmetatable({}, _PMT)
end
_G.__permissive = permissive

-- A singleton with named overrides; any other method resolves to a
-- callable that returns permissive (so `GAMESTATE:Anything()` no-ops).
function __make_singleton(overrides)
    local t = overrides or {}
    setmetatable(t, {__index = function(_, _k)
        return function(...) return permissive() end
    end})
    return t
end

-- A PlayerOptions object: EVERY method (`po:Drunk`, `po:XMod`, ...) is a
-- FLOAT_INTERFACE getter/setter. It routes to Python (records the set),
-- returns the recorder's stored (value, speed) pair like the engine, and
-- returns `po` when the SM `true` self-return flag is passed.
function __make_poptions(pid)
    local t = {}
    setmetatable(t, {__index = function(_, name)
        return function(_self, value, speed, want_self)
            local cur, curspeed = __po_call(pid, name, value, speed)
            if want_self == true then return _self end
            return cur, curspeed
        end
    end})
    return t
end

-- A recording actor table: every method call routes to __actor_poke and
-- returns the table so chained `self:linear(1):xy(0,0)` works; getters
-- route to __actor_get so a driver read hands back a number.
local __GETTER = {
    GetX=true, GetY=true, GetZoom=true, GetZoomX=true, GetZoomY=true,
    GetRotationX=true, GetRotationY=true, GetRotationZ=true, GetName=true,
    GetParent=true,
}
function __make_recorder(id)
    local t = {__recorder_id = id}
    setmetatable(t, {__index = function(_, key)
        if __GETTER[key] then
            return function(_self, ...) return __actor_get(id, key) end
        end
        return function(_self, ...)
            __actor_poke(id, key, ...)
            return t
        end
    end})
    return t
end

-- Etterna embeds Lua 5.1; a handful of scripts call 5.0 names.
if math.mod == nil then math.mod = math.fmod end
if table.getn == nil then table.getn = function(t) return #t end end

-- SM5 global helpers themes expose to every actor script (Actor.lua /
-- 02 Colors.lua). `color('#RRGGBB')` -> an {r,g,b,a} table; the numeric
-- helpers are the theme's math utilities. Scripts call these as free
-- globals (not methods), so they must exist or the chunk aborts.
_G.color = _G.color or function(_s) return {1, 1, 1, 1} end
_G.Color = _G.Color or _G.color
_G.lerp = _G.lerp or function(t, a, b) return a + (b - a) * (t or 0) end
_G.scale = _G.scale or function(x, l1, h1, l2, h2)
    if h1 == l1 then return l2 end
    return ((x - l1) * (h2 - l2)) / (h1 - l1) + l2
end
_G.clamp = _G.clamp or function(x, lo, hi)
    if x < lo then return lo elseif x > hi then return hi end
    return x
end
_G.round = _G.round or function(x) return math.floor((x or 0) + 0.5) end
_G.random = _G.random or math.random
_G.wait = _G.wait or function() end

-- LAST RESORT: any global the script reads that we did not define
-- (engine functions like GetPlayerOrMachineProfile, theme utilities)
-- resolves to a permissive value, so an interactive modfile runs as far
-- as it can instead of aborting on the first unmodeled global. __index
-- fires only for ABSENT keys, so our defined globals and the script's
-- own `foo = ...` assignments (real keys) read back normally.
setmetatable(_G, {__index = function(_, _k) return permissive() end})
"""


def _channel_player(pid) -> int:
    """PlayerOptions player number (1 or 2) -> mod-channel index (0 or 1).
    Anything unrecognized falls to player 0, the modfile default."""
    n = _to_int(pid)
    return 1 if n == 2 else 0


class ModRecorder:
    """Accumulates PlayerOptions method calls as mod events keyed by the
    calling actor's command-clock time (seconds)."""

    def __init__(self):
        self._values: dict = {}   # (mod, player) -> (value, speed)
        self._events: list = []

    def call(self, t: float, pid, name, value, speed):
        """Record one `po:<name>(value, speed)` fired at clock time `t`.
        Returns the previously-held (value, speed) pair so the Lua
        FLOAT_INTERFACE getter contract holds. A call with no value
        argument is a pure getter: it reads, it does not emit an event.
        `speed` defaults to the mod's currently-held approach speed when
        the call passes only a value."""
        player = _channel_player(pid)
        key = (name, player)
        current, current_speed = self._values.get(
            key, (0.0, _DEFAULT_APPROACH_SPEED))
        v = _to_float(value)
        if v is None:
            return current, current_speed

        s = _to_float(speed)
        s = current_speed if s is None else s
        self._values[key] = (v, s)
        self._events.append({'t': float(t), 'mod': str(name),
                             'value': v, 'speed': s, 'player': player})
        return current, current_speed

    @property
    def events(self) -> list:
        return list(self._events)


class Sm5Environment:
    """One LuaJIT host with the SM5 actor-tree stubs installed. Runs a
    modfile script, exposing its returned root actor table plus the
    harvested mod events and per-actor recorders."""

    def __init__(self, to_seconds=None):
        self._to_seconds = to_seconds or (lambda beat: float(beat))
        self._clock = ActorClock()
        self._mods = ModRecorder()
        self._recorders: dict = {}
        self._next_recorder_id = 0
        self._update_functions = 0
        self._swallowed = 0
        self._host = LuaHost(dialect='luajit21')
        self._host.run(_BOOTSTRAP, name='sm5-bootstrap')
        self._install()

    def run_script(self, source: str, name: str):
        """Run the modfile chunk; returns its `return`ed root actor table
        (a Lua table) or None."""
        return self._host.run(source, name=name)

    @property
    def host(self):
        return self._host

    @property
    def mod_events(self) -> list:
        return self._mods.events

    @property
    def update_functions(self) -> int:
        """Count of per-frame `SetUpdateFunction` closures recorded but
        not executed (the unsupported dynamic tail)."""
        return self._update_functions

    @property
    def swallowed(self) -> int:
        return self._swallowed

    def new_recorder_table(self):
        """A fresh recording-actor Lua table sharing the environment's
        command clock, plus its id. The compiler binds one per drawable
        actor before running that actor's command function."""
        rec_id = self._next_recorder_id
        self._next_recorder_id += 1
        self._recorders[rec_id] = RecordingActor(self._clock)
        return self._host.env['__make_recorder'](rec_id), rec_id

    def recorder(self, rec_id: int) -> RecordingActor | None:
        return self._recorders.get(rec_id)

    def reset_clock(self, seconds: float) -> None:
        """Point the shared command clock at an actor's creation time
        before running its command function."""
        self._clock.reset(seconds)

    # -- engine surface ---------------------------------------------------

    def _install(self) -> None:
        host = self._host
        singleton = host.env['__make_singleton']

        player_state = singleton(host.to_lua({
            'GetPlayerOptions': self._get_player_options,
        }))
        song_options = singleton(host.to_lua({'MusicRate': lambda _s, *_a: 1.0}))
        gamestate = singleton(host.to_lua({
            'GetPlayerState': lambda _s, *_a: player_state,
            'GetSongOptionsObject': lambda _s, *_a: song_options,
            'GetCurrentSong': lambda _s, *_a: host.env['__permissive'](),
            'GetSongBeat': lambda _s: self._clock.beat(self._to_seconds),
        }))
        host.expose('GAMESTATE', gamestate)
        host.expose('PREFSMAN', singleton(host.to_lua({
            'GetPreference': lambda _s, _k=None: 1.0,
            'PreferenceExists': lambda _s, _k=None: False,
        })))
        for name in ('SCREENMAN', 'SOUND', 'MESSAGEMAN', 'THEME', 'SONGMAN',
                     'STATSMAN', 'NOTESKIN', 'PROFILEMAN', 'DISPLAY'):
            host.expose(name, singleton(None))

        for enum, _idx in _PLAYER_ENUMS.items():
            host.expose(enum.replace('PlayerNumber_', 'PLAYER_'), enum)
        host.expose('PLAYER_1', 'PlayerNumber_P1')
        host.expose('PLAYER_2', 'PlayerNumber_P2')

        host.expose('SCREEN_WIDTH', 640.0)
        host.expose('SCREEN_HEIGHT', 480.0)
        host.expose('SCREEN_CENTER_X', 320.0)
        host.expose('SCREEN_CENTER_Y', 240.0)
        host.expose('SCREEN_LEFT', 0.0)
        host.expose('SCREEN_RIGHT', 640.0)
        host.expose('SCREEN_TOP', 0.0)
        host.expose('SCREEN_BOTTOM', 480.0)

        host.expose('__po_call', self._po_call)
        host.expose('__actor_poke', self._actor_poke)
        host.expose('__actor_get', self._actor_get)
        host.expose('Trace', lambda *_a: None)
        host.expose('print', lambda *_a: None)

    def _get_player_options(self, _self, level=None):
        """`playerState:GetPlayerOptions('ModsLevel_*')`. The mod level is
        irrelevant to the compiled timeline (all levels resolve to one
        held value on a replay), so every level returns the same recorder;
        the player index is carried from the enclosing GetPlayerState,
        which we do not thread - default player 1."""
        return self._host.env['__make_poptions'](1)

    def _po_call(self, pid=None, name=None, value=None, speed=None):
        """Bridge for `po:<name>(value, speed)`. Records the timed event
        (when a value was supplied) and returns the previously-held
        (value, speed) pair for the Lua getter contract."""
        if not isinstance(name, str):
            self._swallowed += 1
            return 0.0, _DEFAULT_APPROACH_SPEED
        return self._mods.call(self._clock.now(), pid, name, value, speed)

    def _actor_poke(self, rec_id, verb=None, *args) -> None:
        recorder = self._recorders.get(_to_int(rec_id))
        if recorder is None or not isinstance(verb, str):
            return
        if verb == 'SetUpdateFunction':
            self._update_functions += 1
            return
        recorder.poke(verb, list(args))

    def _actor_get(self, rec_id, verb=None):
        recorder = self._recorders.get(_to_int(rec_id))
        if recorder is not None and isinstance(verb, str):
            value = recorder.read(verb)
            if value is not None:
                return value
        return self._host.env['__permissive']()


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    f = _to_float(value)
    return int(f) if f is not None else None
