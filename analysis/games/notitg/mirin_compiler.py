"""Mirin declarative front-end for NotITG charts.

The Mirin template (XeroOl/mirin-template) is a DIFFERENT, cleaner mod
dialect than the classic `default.xml` reader that `modfile.py` handles.
Mirin is Lua we run once: its runtime data structures (`eases`, `funcs`,
`nodes`, `auxes`, `default_mods`) are plain tables we harvest after the
template's `ready_command` populates them, then compile the declarative
subset analytically. See MIRIN_ALIGNMENT.md for the full semantics map.

# Harvest mechanics

We stand up a dedicated LuaJIT runtime (luajit21, for `setfenv` /
`loadstring` / `math.pow`, which the shared sandbox host deliberately
forbids) with a minimal engine mock (`GAMESTATE` / `SCREENMAN` /
`DISPLAY` / screen constants / an actor stub), bootstrap the REAL
template files verbatim exactly as `template/main.xml`'s InitCommand
does (`package` -> `std` -> `sort` -> `ease` -> `template`), run the
chart's `lua/mods.lua` through the template's own module loader (so it
runs in the `xero` environment with `ease`/`add`/`set`/... in scope),
and drive On + Ready so `sort_tables` / `resolve_aliases` /
`compile_nodes` run.

The declarative tables (`eases` / `funcs` / `nodes` / `auxes` /
`default_mods`) are LOCALS in template.lua, not exports. We reach them
with `debug.getupvalue` off the exported closures (`xero.ease` is the
error-checking wrapper whose `fn` upvalue is the real `ease`, whose
`eases` upvalue is the sorted timeline). This reads the real compiled
program with ZERO edits to the vendored template.

# Baking policy

Mirin has NO engine approach-chase and NO per-frame clearall (it emits
`*-1` snap values every frame, easing in Lua itself), so we must NOT run
the classic `mod_channels` approach compiler. Instead we replicate
`run_eases` (the target/offset/relative activation math) in Python -
validated exactly against the template's own `spec/*_spec.lua` oracle -
then EVALUATE the template's own Lua ease functions on a sample grid and
emit dense value keyframes as `*-1` (snap) `ModEvent`s. `ModChannels`
samples the piecewise-linear interpolant through those snap breakpoints,
so a nonlinear ease (`outExpo`, the transient `bounce`/`pulse` family,
`blendease` combinators) is reproduced to within the grid resolution -
osu-enum easing ids are irrelevant here.

Density: an ease is baked with `_BAKE_SAMPLES` interior points across
its span, PLUS the exact endpoint, when its curve is nonlinear; a
`linear` / `instant` / `set` / `acc` ease bakes to its 2 (or 1)
analytic breakpoints (no sampling error to hide). Baseline values
(`default_mods`) seed each channel's rest.

# Deferred list

`funcs` (one-shot scheduled Lua), `perframe`s (per-frame bodies), and
actor-poking `node`s are the irreducible dynamic tail; they are NOT
compiled here. `compile_mirin` returns a structured `deferred` list
describing them (kind + beat/len) for the integrator path
(`update_integrator.py`) to consume later.

# Output shape

`compile_mirin(sm_path)` returns a dict shaped like
`modfile.compile_modfile`'s output, but carrying a pre-built
`ModChannels` under `mod_channels` (already baked, bypassing the
approach compiler) plus `default_mods`, `deferred`, and `dialect:
'mirin'`. The 2-line adapter/modfile wiring diff is documented at the
bottom of this module.
"""
from __future__ import annotations

from pathlib import Path

from analysis.games.etterna import sm_chart
from analysis.player.render.mods.channels import ModChannels, ModEvent

# The vendored Mirin template, relative to the repo root (refs/ is
# gitignored; a chart may also bundle its own `template/` dir, preferred
# when present so a chart pinned to an older template compiles faithfully).
_VENDORED_TEMPLATE = (
    Path(__file__).resolve().parents[3] / 'refs' / 'notitg' /
    'mirin-template' / 'template')

# Interior sample points baked across a nonlinear ease's span (plus the
# exact endpoints). ~48 breakpoints reproduce the sharpest template
# eases (spike, the elastic family) below visible error at 60Hz playback
# while keeping the compiled channel small.
_BAKE_SAMPLES = 48

# Mirin's instant-snap convention: a `*-1` approach makes `ModChannels`
# hold-then-step (no easing), used to pin the first breakpoint and any
# vertical jump. Interior breakpoints emit as `chase` events whose speed
# lands the ramp exactly on the NEXT breakpoint, so `ModChannels`
# reproduces the dense-baked curve as a piecewise-linear interpolant
# rather than a staircase.
_SNAP = -1.0


class _Row:
    """One harvested-and-resolved ease row for a single (mod, player):
    its span in beats, the Lua ease callable, its stick `offset`, and the
    per-mod effective magnitude + any instant table-jump baseline. This
    is the post-`run_eases`-activation form (relative deltas resolved,
    targets committed), ready to sample."""

    __slots__ = ('start', 'length', 'easefn', 'offset', 'eff', 'instant_a',
                 'mod', 'player', 'linear')

    def __init__(self, start, length, easefn, offset, eff, instant_a, mod,
                 player, linear):
        self.start = start
        self.length = length
        self.easefn = easefn
        self.offset = offset
        self.eff = eff
        self.instant_a = instant_a
        self.mod = mod
        self.player = player
        self.linear = linear


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def is_mirin_chart(lua_dir) -> bool:
    """True when a chart's modfile directory is a Mirin chart.

    Fingerprint (MIRIN_ALIGNMENT.md 2.4): the strongest signal is a
    bundled Mirin `template/` (its `template.lua` sets `setfenv(1,
    xero.strict)` and the `main.xml` bootstrap builds the single `xero`
    global) OR a `main.xml` pointing at one. Failing a bundled copy, a
    chart whose FGCHANGES references `template/main.xml` AND ships a
    `lua/mods.lua` is a Mirin chart running against a shared template
    (the standard Mirin distribution shape). A classic (gat)
    `default.xml` has neither signal; a Cat/bespoke framework has its
    own globals. We sniff files, never directory names."""
    lua_dir = Path(lua_dir)
    if _has_mirin_template(lua_dir) or _has_mirin_template(lua_dir.parent):
        return True
    main_xml = lua_dir / 'main.xml'
    if main_xml.is_file() and _mentions_xero(main_xml):
        return True
    return _fgchanges_reference_is_mirin(lua_dir)


def _fgchanges_reference_is_mirin(lua_dir: Path) -> bool:
    """True when a sibling `Song.sm` FGCHANGES names `template/main.xml`
    (the Mirin loader entry) and the chart ships `lua/mods.lua`. This is
    the shared-template case: no bundled `template/` beside the chart,
    but the FGCHANGES entry is unambiguously Mirin's."""
    if not (lua_dir / 'lua' / 'mods.lua').is_file():
        return False
    for sm in lua_dir.glob('*.sm'):
        text = sm.read_text(encoding='utf-8', errors='replace').lower()
        if 'fgchanges' in text and 'template/main.xml' in text:
            return True
    return False


def _has_mirin_template(base: Path) -> bool:
    template = base / 'template' / 'template.lua'
    return template.is_file() and _mentions_xero(template)


def _mentions_xero(path: Path) -> bool:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return False
    return 'xero' in text


# --------------------------------------------------------------------------
# Lua runtime + harvest
# --------------------------------------------------------------------------

# The engine mock: the minimum GAMESTATE / SCREENMAN / DISPLAY / actor
# surface the template's InitCommand + On + Ready touch. Mirrors the
# template's own `spec/mock.lua` harness, trimmed to what a mods-only
# harvest exercises. `GetElapsedTimeFromBeat` calls back into the chart's
# real BPM map (`BEAT_TO_TIME`, set from Python before bootstrap).
_ENGINE_MOCK = r"""
CURBEAT = 0
Song = {}
function Song:GetSongDir() return SONGDIR end
function Song:GetElapsedTimeFromBeat(b) return BEAT_TO_TIME(b) end
function Song:SetNumSpellCards(_) end
function Song:SetSpellCardTiming(_, _, _) end
function Song:SetSpellCardName(_, _) end
function Song:SetSpellCardDifficulty(_, _) end
function Song:SetSpellCardColor(_, _, _, _, _) end
GAMESTATE = {}
function GAMESTATE:GetCurrentSong() return Song end
function GAMESTATE:ApplyModifiers(str, pn) end
function GAMESTATE:GetFileStructure(path)
    local f = io.open(path); if f then f:close(); return true end; return false
end
function GAMESTATE:GetSongBeat() return CURBEAT end
function GAMESTATE:GetSongTime() return BEAT_TO_TIME(CURBEAT) end
function GAMESTATE:FinishSong() end
DISPLAY = {}
function DISPLAY:GetDisplayWidth() return 800 end
function DISPLAY:GetDisplayHeight() return 600 end
SCREEN_CENTER_X, SCREEN_CENTER_Y = 320, 240
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

local Actor = {}
local Actor_mt = {__index = Actor}
local ActorFrame = setmetatable({}, Actor_mt)
local ActorFrame_mt = {__index = ActorFrame}
local Player = setmetatable({}, Actor_mt)
local Player_mt = {__index = Player}
ALL_ACTORS = {}
local function make(mt, name, framelike)
    local r = {_name = name, _commands = {}, _queue = {}}
    if framelike then r._children = {} end
    r = setmetatable(r, mt)
    table.insert(ALL_ACTORS, r)
    return r
end
function newactor(name) return make(Actor_mt, name, false) end
function newframe(name) return make(ActorFrame_mt, name, true) end
function newplayer(name) return make(Player_mt, name, true) end
function Actor:sleep(_) end
function Actor:GetName() return self._name or "" end
function Actor:hidden(i) self._hidden = (i ~= 0) end
function Actor:addcommand(n, c) self._commands[n] = c end
function Actor:queuecommand(n) table.insert(self._queue, n) end
function Actor:playcommand(n)
    if self._commands[n] then self._commands[n](self) end
    if self._children then
        for _, v in ipairs(self._children) do v:playcommand(n) end
    end
end
function Actor:removecommand(n) self._commands[n] = nil end
function Actor:luaeffect(_) end
function Actor:effectclock(_) end
function Actor:tween(_, fn) self._tween = fn end
function Actor:GetSecsIntoEffect() return BEAT_TO_TIME(CURBEAT) end
-- The player-proxy / AFT setup calls a handful of no-op configurators.
local function noop() end
Actor.SetTarget = noop; Actor.SetFarDist = noop; Actor.xy = noop
Actor.x = noop; Actor.y = noop; Actor.basezoomx = noop; Actor.basezoomy = noop
Actor.SetWidth = noop; Actor.SetHeight = noop; Actor.Create = noop
Actor.EnableDepthBuffer = noop; Actor.EnableAlphaBuffer = noop
Actor.EnableFloat = noop; Actor.EnablePreserveTexture = noop
Actor.GetTexture = noop; Actor.SetTexture = noop
function ActorFrame:GetNumChildren() return #self._children end
function ActorFrame:GetChildAt(n) return self._children[n + 1] end
function Player:IsAwake()
    return self._name == "PlayerP1" or self._name == "PlayerP2"
end
-- `P[pn]('Child')` fetches a child actor by name (the mods.lua judgment/
-- combo-proxy boilerplate does `P[pn]('Judgment')`); a fresh stub actor
-- satisfies the sleep/hidden/SetTarget calls that follow.
Actor_mt.__call = function(self, _name) return newactor(_name) end
ActorFrame_mt.__call = Actor_mt.__call
Player_mt.__call = Actor_mt.__call
local screen_world = {}
SCREENMAN = setmetatable({}, {__call = function(_, arg)
    if arg and tostring(arg):match("^PlayerP[1-8]$") then
        screen_world[arg] = screen_world[arg] or newplayer(arg)
    else
        screen_world[arg] = screen_world[arg] or newactor(arg)
    end
    return screen_world[arg]
end})
function SCREENMAN:SystemMessage(_) end
function add_child(p, c) table.insert(p._children, c) end
"""

# Bootstrap the template exactly as template/main.xml's InitCommand,
# then drive Init -> On -> Ready. `TEMPLATE_DIR` and `SONGDIR` are set
# from Python; `mods` loads through the template's own `xero.require`.
_BOOTSTRAP = r"""
local fg = newframe()
xero = {
    MIRIN_VERSION = 'compiled',
    foreground = fg,
    dir = GAMESTATE:GetCurrentSong():GetSongDir(),
}
local package = loadfile(TEMPLATE_DIR .. 'package.lua')()
xero.package = package
xero.require = package.require
loadfile(TEMPLATE_DIR .. 'std.lua')()
loadfile(TEMPLATE_DIR .. 'sort.lua')()
loadfile(TEMPLATE_DIR .. 'ease.lua')()
loadfile(TEMPLATE_DIR .. 'template.lua')()

local tpl = newactor()
add_child(fg, tpl)
tpl:addcommand('Init', xero.init_command)
tpl:playcommand('Init')

-- The theme's layout.xml declares Named ActorProxies (PP/PC/PJ per
-- player) that the standard mods.lua boilerplate reads as xero.PP[pn]
-- etc. We don't splice layout.xml, so we register these six named
-- proxies directly (exactly as the template's own spec/helper.lua does);
-- scan_named_actors (run in On) resolves the names into the xero table.
local layout = newframe()
add_child(fg, layout)
for _, base in ipairs {'PP', 'PC', 'PJ'} do
    for pn = 1, 2 do
        add_child(layout, newactor(base .. '[' .. pn .. ']'))
    end
end

-- On resolves Name= actors and queues Ready; Ready loads mods.lua and
-- sorts/compiles the tables. Draining the queue runs Ready.
for _, v in ipairs(ALL_ACTORS) do
    if v._commands.On then v:playcommand('On') end
end
for _, v in ipairs(ALL_ACTORS) do
    while #v._queue >= 1 do v:playcommand(table.remove(v._queue, 1)) end
end
"""


class _Harvest:
    """A completed Mirin template run: the harvested tables plus the
    runtime that owns them (kept alive so the Lua ease callables in
    `eases` remain valid while Python samples them)."""

    def __init__(self, runtime, globals_, eases, funcs, nodes, auxes,
                 default_mods):
        self._rt = runtime
        self._g = globals_
        self.eases = eases
        self.funcs = funcs
        self.nodes = nodes
        self.auxes = auxes
        self.default_mods = default_mods


def _new_runtime():
    import lupa
    return lupa.luajit21.LuaRuntime(
        unpack_returned_tuples=True,
        register_eval=False,
        register_builtins=False)


def _upvalue(rt, fn, name):
    """The named upvalue of a Lua function, via `debug.getupvalue`."""
    getupvalue = rt.eval('debug.getupvalue')
    i = 1
    while True:
        upname, value = getupvalue(fn, i)
        if upname is None:
            return None
        if upname == name:
            return value
        i += 1


def _real_fn(rt, exported):
    """The template's real declarative function behind its exported
    error-checking wrapper (`export`'s `inner` closes over `fn`)."""
    return _upvalue(rt, exported, 'fn')


def _run_template(template_dir: Path, song_dir: Path, beat_to_time):
    """Bootstrap the real template + chart mods, returning a `_Harvest`.

    `beat_to_time(beat) -> seconds` is the chart's real BPM conversion,
    exposed to Lua as `Song:GetElapsedTimeFromBeat`."""
    rt = _new_runtime()
    g = rt.globals()
    g.SONGDIR = str(song_dir).rstrip('/') + '/'
    g.TEMPLATE_DIR = str(template_dir).rstrip('/') + '/'
    g.BEAT_TO_TIME = lambda beat: float(beat_to_time(float(beat)))
    rt.execute(_ENGINE_MOCK)
    rt.execute(_BOOTSTRAP)

    xero = g.xero
    ease_fn = _real_fn(rt, xero.ease)
    return _Harvest(
        rt, g,
        eases=_upvalue(rt, ease_fn, 'eases'),
        funcs=_upvalue(rt, _real_fn(rt, xero.func), 'funcs'),
        nodes=_upvalue(rt, _real_fn(rt, xero.node), 'nodes'),
        auxes=_upvalue(rt, _real_fn(rt, xero.aux), 'auxes'),
        default_mods=_upvalue(rt, _real_fn(rt, xero.setdefault),
                              'default_mods'))


# --------------------------------------------------------------------------
# run_eases port: resolve activation deltas, produce _Row list
# --------------------------------------------------------------------------

def _lua_len(row) -> int:
    """The `#row` array length of a harvested Lua ease (positional slots
    1..N, before the string keys like `plr`/`start_time`)."""
    n = 0
    i = 1
    while row[i] is not None:
        n += 1
        i += 1
    return n


def _is_lua_table(value) -> bool:
    """True for a Lua table percent `{a, b}` (has integer keys), as
    opposed to a plain number magnitude."""
    return hasattr(value, 'keys')


def _resolve_rows(harvest: _Harvest):
    """Replicate `run_eases`' activation pass over the sorted `eases`
    timeline, producing one `_Row` per (ease, mod) with relative deltas
    resolved and stick targets committed.

    This is the deterministic, non-per-frame core of Mirin's mod model:
    for each ease in sorted order we compute its stick `offset`
    (`easefn(1) >= 0.5`), turn absolute magnitudes into deltas off the
    running `targets` table, and commit sticking deltas back - exactly as
    template.lua L945-973, verified against the template's ease spec.

    Returns `(rows, default_mods)`."""
    default_mods = {k: float(harvest.default_mods[k])
                    for k in harvest.default_mods.keys()}
    targets: dict = {}

    def target(player, mod):
        per_player = targets.setdefault(player, {})
        if mod not in per_player:
            per_player[mod] = default_mods.get(mod, 0.0)
        return per_player[mod]

    rows = []
    for ease in _values(harvest.eases):
        rows.extend(_resolve_one(ease, target, targets))
    return rows, default_mods


def _resolve_one(ease, target, targets) -> list:
    player = int(ease.plr)
    easefn = ease[3]
    offset = 1.0 if easefn(1) >= 0.5 else 0.0
    linear = _is_linear_ease(easefn)
    length = _num(ease[2])
    rows = []
    n = _lua_len(ease)
    idx = 4
    while idx <= n:
        pct = ease[idx]
        mod = ease[idx + 1]
        instant_a = 0.0
        if _is_lua_table(pct):
            a, b = _num(pct[1]), _num(pct[2])
            targets.setdefault(player, {})[mod] = (
                (target(player, mod) if ease.relative else 0.0) + a)
            instant_a = a
            pct = b
        eff = _num(pct) if ease.relative else (_num(pct) - target(player, mod))
        if offset >= 0.5:
            targets.setdefault(player, {})[mod] = target(player, mod) + eff
        rows.append(_Row(
            start=_num(ease[1]), length=length, easefn=easefn, offset=offset,
            eff=eff, instant_a=instant_a, mod=str(mod),
            player=max(0, player - 1), linear=linear))
        idx += 2
    return rows


def _is_linear_ease(easefn) -> bool:
    """Whether an ease can be reproduced by 2 breakpoints without dense
    sampling. A curve equal to its own chord at the quarter points is
    treated as linear (`linear` and the analytic set/acc `instant` step
    qualify); anything else samples densely. The test only ever ADDS
    sampling when unsure, so a mislabel never loses fidelity."""
    lo, hi = easefn(0.0), easefn(1.0)
    for quarter in (0.25, 0.5, 0.75):
        if abs(easefn(quarter) - (lo + (hi - lo) * quarter)) > 1e-9:
            return False
    return True


def _num(value) -> float:
    return float(value)


# --------------------------------------------------------------------------
# Baking: _Row list -> ModEvent breakpoints -> ModChannels
# --------------------------------------------------------------------------

def _bake_channels(rows, default_mods, beat_to_time) -> ModChannels:
    """Dense-bake the resolved rows into `ModEvent` breakpoints and
    compile a `ModChannels`.

    Each (mod, player) channel is the SUM of its rows at any instant (the
    additive ease stack), resting at the mod's default. We build a sorted
    set of sample beats - each row contributes its endpoints, plus a dense
    interior grid when its ease is nonlinear - evaluate the exact live sum
    at each, and emit `ModEvent`s that make `ModChannels` reproduce the
    resulting piecewise-linear curve exactly: the first breakpoint snaps
    to its value, each interior breakpoint chases the NEXT value at a
    speed sized to arrive precisely at the next breakpoint's time (so the
    engine's constant-rate chase IS our linear segment). A vertical jump
    (two breakpoints sharing a time) snaps. Times are seconds (identity
    clock), matching the eased curve within the grid step."""
    by_channel: dict = {}
    for row in rows:
        by_channel.setdefault((row.mod, row.player), []).append(row)

    events = []
    for (mod, player), channel_rows in by_channel.items():
        breakpoints = _channel_breakpoints(channel_rows, default_mods, mod,
                                           beat_to_time)
        events.extend(_breakpoint_events(breakpoints, mod, player))
    return ModChannels.compile(events)


def _channel_breakpoints(rows, default_mods, mod, beat_to_time) -> list:
    """The `(time_seconds, value)` breakpoints of one channel, in time
    order, seeded so the channel rests at the mod's default: a nonzero
    default gets an explicit baseline point at the first sample beat so
    the curve holds the default before the first ease rather than
    `ModChannels`' 0 rest."""
    default = default_mods.get(mod, 0.0)
    points = [(beat_to_time(beat), _live_value(rows, beat, default))
              for beat in _sample_beats(rows)]
    if default != 0.0 and points:
        # Hold the default from song start (t=0) until the first sample,
        # so a mod with a nonzero baseline (zoom=100, grain=400) rests
        # there rather than at `ModChannels`' 0 before its first ease.
        baseline_t = min(0.0, points[0][0] - _PRE_START_EPS)
        points.insert(0, (baseline_t, default))
    return points


def _breakpoint_events(points, mod, player) -> list:
    """`(t, value)` breakpoints -> `ModEvent`s that `ModChannels`
    recompiles into exactly this piecewise-linear curve.

    The first point snaps to its value. Each subsequent segment emits a
    chase from the current point toward the next value at
    `|dv| / dt` per second, so `ModChannels`' constant-rate approach
    arrives precisely at the next breakpoint's time - i.e. a straight
    line between them. A zero-`dt` step (a vertical jump) snaps."""
    events = []
    for i, (t, value) in enumerate(points):
        if i == 0:
            events.append(ModEvent(t, value, _SNAP, mod, player))
            continue
        t_prev, v_prev = points[i - 1]
        dt = t - t_prev
        if dt <= 0.0:
            events.append(ModEvent(t, value, _SNAP, mod, player))
        else:
            speed = abs(value - v_prev) / dt
            events.append(ModEvent(t_prev, value, speed or _SNAP, mod, player))
    return events


def _sample_beats(rows) -> list:
    """Sorted, de-duplicated sample beats for one channel: every row's
    start and end, plus interior grid points for nonlinear rows, plus one
    beat just before each start so the channel sits at rest (not
    interpolating up from the previous value) until the ease begins."""
    beats = set()
    for row in rows:
        start = row.start
        beats.add(start)
        beats.add(start + row.length)
        if start > 0:
            beats.add(_before(start))
        if not row.linear and row.length > 0:
            for i in range(1, _BAKE_SAMPLES):
                beats.add(start + row.length * i / _BAKE_SAMPLES)
    return sorted(beats)


# A hair before an ease's start, to pin the resting value so the snap at
# `start` steps up rather than the channel ramping toward it from the
# previous breakpoint.
_PRE_START_EPS = 1e-4


def _before(beat: float) -> float:
    return beat - _PRE_START_EPS


def _live_value(rows, beat, default) -> float:
    """The exact Mirin live value of one channel at `beat`: the mod's
    default plus, for every row that has started, its committed stick
    (`offset * eff`) and instant table-jump baseline, plus the active
    overlay (`(easefn(progress) - offset) * eff`) while the ease runs.
    This is the summation of template.lua's `run_eases` inner loop."""
    value = default
    for row in rows:
        if beat < row.start:
            continue
        active = row.length == 0 or beat < row.start + row.length
        progress = 1.0 if row.length == 0 else min(
            1.0, (beat - row.start) / row.length)
        overlay = (row.easefn(progress) - row.offset) if active else 0.0
        value += row.instant_a + row.offset * row.eff + overlay * row.eff
    return value


# --------------------------------------------------------------------------
# Deferred tail: funcs / perframes / actor-poking nodes
# --------------------------------------------------------------------------

def _collect_deferred(harvest: _Harvest) -> list:
    """The irreducible dynamic tail as structured descriptors for the
    integrator path (NOT compiled here).

    `funcs` entries with no length are one-shot `func`s; with a length
    they are `perframe`s (or the persist tail of a `func_ease`). We record
    each one's start beat, start time in seconds, and length; the
    integrator later runs the bodies on a tick grid over that span."""
    deferred = []
    for func in _values(harvest.funcs):
        length = func[2] if _lua_len(func) >= 2 else None
        deferred.append({
            'kind': 'perframe' if length is not None else 'func',
            'beat': _num(func[1]),
            'start_time': (_num(func.start_time)
                           if func.start_time is not None else None),
            'len_beats': _num(length) if length is not None else 0.0,
        })
    return deferred


def _values(lua_table) -> list:
    """A Lua array table's positional values (1..#t) as a Python list."""
    if lua_table is None:
        return []
    count = len(list(lua_table.values()))
    return [lua_table[i] for i in range(1, count + 1)]


# The template's OWN built-in nodes (zoom/movex/y/z fan-outs, xmod) are
# infrastructure every chart carries; only chart-authored nodes are
# interesting to the node compiler. A node whose inputs are all built-in
# mod names is one of these (plus the auto-appended write-back
# terminators, which have empty outputs).
_BUILTIN_NODE_INPUTS = ('move', 'zoom', 'xmod', 'cmod')


def _collect_nodes(harvest: _Harvest) -> list:
    """Chart-authored `node` / `definemod` descriptors for the node
    compiler (item 52 grid-sampling), NOT compiled here.

    Each entry lists the node's input and output mod names; the compiler
    later samples the input channels, runs the node fn over them, and
    emits the output channels (recursing until engine mods) or, for an
    actor-poking fn, records via the integrator. We surface names only -
    the fn stays a live Lua closure we do not classify pure-vs-actor
    here (that needs running it). Built-in fan-out nodes and the
    auto-appended write-back terminators (empty outputs) are dropped."""
    nodes = []
    for node in _values(harvest.nodes):
        inputs = [str(v) for v in _values(node[1])]
        outputs = [str(v) for v in _values(node[2])]
        if not inputs or not outputs:
            continue
        if all(inp.startswith(_BUILTIN_NODE_INPUTS) for inp in inputs):
            continue
        nodes.append({'inputs': inputs, 'outputs': outputs})
    return nodes


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def _resolve_template_dir(lua_dir: Path) -> Path | None:
    """The Mirin `template/` dir to run: a chart-bundled copy wins (a
    chart pinned to an older template compiles against it), else the
    vendored clone."""
    for base in (lua_dir, lua_dir.parent):
        bundled = base / 'template'
        if (bundled / 'template.lua').is_file():
            return bundled
    return (_VENDORED_TEMPLATE
            if (_VENDORED_TEMPLATE / 'template.lua').is_file() else None)


def _mirin_lua_dir(sm_path) -> Path | None:
    """The directory holding the chart's `lua/mods.lua`, or None. The
    Mirin FGCHANGES points at `template/main.xml`; the mods live in a
    sibling `lua/` dir (the template's `package.path` is `lua/?.lua`),
    so the song dir is the harvest root."""
    song_dir = Path(sm_path).parent
    return song_dir if (song_dir / 'lua' / 'mods.lua').is_file() else None


def _beat_to_time_fn(sm_data, chart):
    bpms = sm_data['bpms']
    offset = sm_data['offset']

    def convert(beat):
        return sm_chart.beat_to_time(
            float(beat), bpms, offset, stops=chart.get('stops'),
            delays=chart.get('delays'), warps=chart.get('warps'))
    return convert


def compile_mirin(sm_path) -> dict | None:
    """Compile a Mirin NotITG chart's mods, or None when the chart is not
    a resolvable Mirin chart.

    Runs the real template once (harvesting `eases`/`funcs`/`nodes`/
    `auxes`/`default_mods`), resolves + dense-bakes the declarative ease
    subset into a pre-built `ModChannels` (snap keyframes, NO approach
    chase), and surfaces the func/perframe tail as `deferred`. Returns a
    dict shaped like `modfile.compile_modfile`, distinguished by
    `dialect: 'mirin'` and carrying `mod_channels` (already compiled)
    instead of raw `mod_events`. Never raises: a community chart must
    load."""
    try:
        return _compile_mirin(sm_path)
    except Exception as exc:
        return {
            'dialect': 'mirin', 'mod_channels': ModChannels.compile([]),
            'default_mods': {}, 'deferred': [], 'nodes': [], 'players': (),
            'warnings': [f'mirin compile aborted: {exc}'],
        }


def _compile_mirin(sm_path):
    lua_dir = _mirin_lua_dir(sm_path)
    if lua_dir is None or not is_mirin_chart(lua_dir):
        return None
    template_dir = _resolve_template_dir(lua_dir)
    if template_dir is None:
        return None

    sm_data = sm_chart.parse_sm(sm_path)
    chart = (sm_data.get('charts') or [{}])[0]
    beat_to_time = _beat_to_time_fn(sm_data, chart)

    harvest = _run_template(template_dir, lua_dir, beat_to_time)
    rows, default_mods = _resolve_rows(harvest)
    channels = _bake_channels(rows, default_mods, beat_to_time)

    return {
        'dialect': 'mirin',
        'mod_channels': channels,
        'default_mods': default_mods,
        'deferred': _collect_deferred(harvest),
        'nodes': _collect_nodes(harvest),
        'players': channels.players,
        'warnings': [],
    }


# --------------------------------------------------------------------------
# Adapter / modfile wiring diff (for the coordinator; NOT applied here -
# adapter.py and modfile.py are owned elsewhere).
#
# In analysis/games/notitg/adapter.py `_compiled_modfile`, prefer the
# Mirin front-end when the chart is one, falling back to the classic
# compiler (both memoized on the replay identically):
#
#     from analysis.games.notitg.mirin_compiler import compile_mirin
#     compiled = compile_mirin(sm_path) or compile_modfile(sm_path)
#
# and in `note_mods`, use a pre-baked `mod_channels` when present
# (Mirin) instead of running the approach compiler over `mod_events`:
#
#     compiled = self._compiled_modfile(replay) or {}
#     channels = (compiled.get('mod_channels')
#                 or compile_mod_channels(compiled.get('mod_events') or []))
# --------------------------------------------------------------------------
