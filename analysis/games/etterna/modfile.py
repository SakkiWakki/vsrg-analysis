"""Etterna SM5 modfile compiler (FGCHANGES .lua -> mods + storyboard).

Etterna modcharts hang off `#FGCHANGES:<beat>=<file>.lua=...` in the
.sm/.ssc. The referenced `.lua` is an SM5 actor-tree script: it builds
`Def.ActorFrame{...}` / `Def.Quad{...}` / `Def.Sprite{...}` tables whose
InitCommand/OnCommand are Lua FUNCTIONS, and `return`s the root. Two
things are harvested by running the script under the stub SM5
environment (`sm5_env`):

- MOD TIMELINE: the actor functions call `poptions:Drunk(v, s)` (etc.)
  on a stubbed PlayerOptions object. Each call records a mod event at
  the calling actor's command-clock time; the clock advances through
  `self:sleep(n)` / tween verbs, so interleaved poptions calls lay down
  a timeline. The mod NAME is the PlayerOptions method (lowercased to
  the arrow_effects channel name); args are `(value, approach_speed)`.
  SM5 poptions are PERSISTENT (no per-frame clearall), so the channel
  compiler holds the last value forward - no revert events are emitted.

- STORYBOARD: drawable actors (Quad/Sprite/BitmapText) with command
  functions poking `self` (`self:xy(..)`, `self:diffuse(..)`) record
  onto a RecordingActor and compile to storyboard `Element`s in SM's
  640x480 screen space, nested as `group` Elements for ActorFrames.

`compile_modfile` never raises: any community file must load, so the
script and every command function run under try/except and a partial
harvest is fine (the interactive minesweeper pilot faults on its
SOUND/input callbacks and still yields its actor visuals).
"""
from __future__ import annotations

import re
from pathlib import Path

from analysis.games.etterna import sm_chart
from analysis.games.etterna.recording_actor import RecordingActor
from analysis.games.etterna.sm5_env import Sm5Environment
from analysis.player.render.mods.channels import ModChannels, ModEvent
from analysis.player.render.storyboard.model import Element, build_timelines

_DESIGN_W = 640.0
_DESIGN_H = 480.0

_SPRITE_KINDS = frozenset({'Sprite', 'Quad'})
_TEXT_KINDS = frozenset({'BitmapText'})

# Command attributes the engine fires on actor creation - the load-time
# moment we record. *MessageCommand / named-command bodies fire on later
# broadcasts and read the live `self`; running them at load is
# unsupported (they stay unharvested).
_LOAD_TIME_ATTRS = ('InitCommand', 'OnCommand')

# Storyboard properties the renderer samples. rotation_x/y (3D tilt) and
# the raw color are recorded for legibility but only 2D-drawable props
# reach the built element timelines.
_DRAWABLE_PROPS = frozenset({
    'x', 'y', 'scale_x', 'scale_y', 'rotation', 'alpha', 'color',
})


def parse_fgchanges(sm_path) -> list:
    """`#FGCHANGES`/`#BGCHANGES` entries as (start_beat, name) pairs, read
    straight from the file so the parse_sm/parse_ssc contract is
    untouched. The tag is `#FGCHANGES:<beat>=<name>=<rate>=...;`."""
    text = Path(sm_path).read_text(encoding='utf-8', errors='replace')
    entries = []
    for tag in ('FGCHANGES', 'BGCHANGES'):
        entries.extend(_iter_change_entries(text, tag))
    return entries


def _iter_change_entries(text: str, tag: str):
    match = re.search(r'#' + tag + r':([^;]*);', text, flags=re.DOTALL)
    if not match:
        return
    for raw in match.group(1).split(','):
        fields = raw.strip().split('=')
        if len(fields) >= 2 and fields[0].strip():
            try:
                yield (float(fields[0]), fields[1].strip())
            except ValueError:
                continue


def _resolve_lua_file(sm_path, entries) -> Path | None:
    """The FGCHANGES `name` for Etterna modcharts is a `.lua` file
    beside the simfile (`minesweeper.lua`, `01 Undiscovered Colors.lua`).
    Returns the first existing `.lua` referenced, else None."""
    song_dir = Path(sm_path).parent
    for _beat, name in entries:
        if name.lower().endswith('.lua'):
            candidate = song_dir / name
            if candidate.exists():
                return candidate
    return None


def _parse_simfile(sm_path):
    path = str(sm_path)
    return (sm_chart.parse_ssc(path) if path.endswith('.ssc')
            else sm_chart.parse_sm(path))


def _beat_to_seconds(sm_data):
    """A beat -> seconds converter over the simfile's first chart timing
    (modfiles are chart-agnostic; the first chart's BPM/stop/warp map is
    the song timeline)."""
    bpms = sm_data['bpms']
    offset = sm_data['offset']
    chart = (sm_data.get('charts') or [{}])[0]

    def convert(beat):
        return sm_chart.beat_to_time(
            beat, chart.get('bpms', bpms), chart.get('offset', offset),
            stops=chart.get('stops'), delays=chart.get('delays'),
            warps=chart.get('warps'))
    return convert


def compile_modfile(sm_path) -> dict | None:
    """Compile an Etterna chart's modfile. Returns None when the chart
    references no resolvable `.lua`; otherwise a dict of harvested mod
    events, storyboard elements, and harvest counters. Never raises."""
    try:
        return _compile_modfile(sm_path)
    except Exception as exc:
        return _empty_result(f'compile aborted: {exc}')


def _empty_result(warning: str) -> dict:
    return {'mod_events': [], 'elements': [], 'tree': [],
            'update_functions': 0, 'actors': 0, 'warnings': [warning]}


def _compile_modfile(sm_path):
    entries = parse_fgchanges(sm_path)
    lua_file = _resolve_lua_file(sm_path, entries)
    if lua_file is None:
        return None

    sm_data = _parse_simfile(sm_path)
    to_seconds = _beat_to_seconds(sm_data)
    start_beat = min((b for b, _n in entries), default=0.0)
    start_time = to_seconds(start_beat)

    env = Sm5Environment(to_seconds=to_seconds)
    warnings = []
    try:
        root = env.run_script(
            lua_file.read_text(encoding='utf-8', errors='replace'),
            name=lua_file.name)
    except Exception as exc:
        return _empty_result(f'script failed: {exc}')

    tree, actors = _compile_tree(env, root, start_time, warnings)
    return {
        'mod_events': env.mod_events,
        'elements': _flatten(tree),
        'tree': tree,
        'update_functions': env.update_functions,
        'actors': actors,
        'swallowed': env.swallowed,
        'warnings': warnings,
    }


def _compile_tree(env, root, start_time, warnings):
    """Compile the returned actor table into storyboard Elements. The
    root is itself an actor (a script may `return Def.Sprite{...}` or a
    `Def.ActorFrame{...}` holding children). An ActorFrame with drawable
    descendants becomes a 'group' whose transform composes onto them; a
    Quad/Sprite/BitmapText becomes a leaf. Returns (elements, actor
    count) - a top-level frame yields one group, a bare drawable one
    leaf."""
    if not _is_lua_table(root):
        return [], 0
    counter = [0]
    element = _compile_actor(env, root, start_time, warnings, counter)
    elements = [element] if element is not None else []
    return elements, counter[0]


def _compile_actor(env, actor, start_time, warnings, counter):
    counter[0] += 1
    keyframes = _run_commands(env, actor, start_time, warnings)

    child_elements = []
    for child in _children(actor):
        element = _compile_actor(env, child, start_time, warnings, counter)
        if element is not None:
            child_elements.append(element)

    if child_elements:
        return _group_element(start_time, keyframes, tuple(child_elements))
    return _leaf_element(actor, start_time, keyframes)


def _run_commands(env, actor, start_time, warnings) -> dict:
    """Run this actor's load-time command functions on ONE recorder (so
    InitCommand then OnCommand share a clock and accumulate), returning
    its keyframes. Each command's clock starts at the modfile's creation
    time; a faulting command warns and is skipped so the harvest
    survives (the interactive pilot's callbacks fault here)."""
    recorder = None
    for attr in _LOAD_TIME_ATTRS:
        fn = actor[attr] if _has_key(actor, attr) else None
        if not callable(fn):
            continue
        env.reset_clock(start_time)
        rec_table, rec_id = env.new_recorder_table()
        recorder = env.recorder(rec_id)
        try:
            fn(rec_table)
        except Exception as exc:
            warnings.append(f'{_actor_kind(actor)}.{attr}: {exc}')
    return recorder.keyframes() if recorder is not None else {}


def _group_element(start_time, keyframes, children):
    return Element(
        kind='group', z=0, z_index=0,
        t_start=start_time, t_end=float('inf'),
        anchor=(0.5, 0.5), origin=(0.5, 0.5),
        timelines=build_timelines(keyframes=_drawable(keyframes)),
        children=children,
    )


def _leaf_element(actor, start_time, keyframes):
    kind = _element_kind(_actor_kind(actor))
    if kind is None:
        return None
    drawable = _drawable(keyframes)
    if not any(drawable.values()):
        return None

    text = actor['Text'] if _has_key(actor, 'Text') else ''
    asset = (actor['Texture'] if _has_key(actor, 'Texture')
             else actor['File'] if _has_key(actor, 'File') else None)
    return Element(
        kind=kind, z=0, z_index=0,
        t_start=start_time, t_end=float('inf'),
        anchor=(0.5, 0.5), origin=(0.5, 0.5),
        timelines=build_timelines(keyframes=drawable),
        asset=str(asset) if asset else None,
        text=str(text) if text else '',
    )


def _drawable(keyframes) -> dict:
    return {prop: frames for prop, frames in keyframes.items()
            if prop in _DRAWABLE_PROPS and frames}


def _element_kind(actor_kind: str):
    if actor_kind in _TEXT_KINDS:
        return 'text'
    if actor_kind in _SPRITE_KINDS:
        return 'sprite' if actor_kind == 'Sprite' else 'rect'
    return None


def _flatten(elements) -> list:
    """Flat element list (groups' children hoisted) for the storyboard
    fallback path, mirroring the NotITG adapter's tree-or-elements
    contract."""
    flat = []
    for element in elements:
        if element.kind == 'group':
            flat.extend(_flatten(element.children))
        else:
            flat.append(element)
    return flat


# -- Lua table helpers ----------------------------------------------------

def _is_lua_table(value) -> bool:
    return hasattr(value, '__getitem__') and not isinstance(
        value, (str, bytes))


def _has_key(table, key) -> bool:
    try:
        return table[key] is not None
    except (KeyError, TypeError):
        return False


def _actor_kind(actor) -> str:
    kind = actor['Class'] if _has_key(actor, 'Class') else None
    return str(kind) if kind else 'Actor'


def _children(actor):
    """Integer-indexed children of an actor table (`t[#t+1] = Def.X{}`),
    in order. Non-table entries (stray values) are skipped."""
    if not _is_lua_table(actor):
        return
    index = 1
    while True:
        try:
            child = actor[index]
        except (KeyError, TypeError):
            return
        if child is None:
            return
        if _is_lua_table(child):
            yield child
        index += 1


# -- mod-channel compilation ----------------------------------------------

def compile_mod_channels(mod_events) -> ModChannels:
    """Harvested PlayerOptions calls -> compiled mod channels.

    Each event is a `po:<Mod>(value, speed)` at time `t`: the mod name is
    lowercased to the arrow_effects channel name and the value/speed feed
    one `ModEvent`. No revert event is emitted - SM5 poptions persist, so
    the channel holds the last value forward (`ModChannels` sample). The
    events are already time-keyed (seconds), so compilation runs on the
    identity clock."""
    events = []
    for row in mod_events:
        events.append(ModEvent(float(row['t']), float(row['value']),
                               float(row['speed']), str(row['mod']).lower(),
                               int(row['player'])))
    return ModChannels.compile(events)
