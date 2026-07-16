"""NotITG modfile compiler (Mode-2 recorder + actor compiler).

Pre-Mirin "classic template" charts (the gat pilot) ship a `#FGCHANGES`
lua directory whose `default.xml` is an actor tree of CODE/Quad/Sprite
elements. The mod timeline lives in InitCommand Lua that fills plain
data tables (`mods`, `mods2`, `mod_actions`) via helpers defined in an
included `modhelpers.xml`. We compile this in two ways:

- Mode-2 recording (CODE chunks -> mod harvest): run the InitCommand
  Lua verbatim under the sandboxed LuaJIT host with stub engine globals,
  then read the tables back. The template's applier (default.xml, the
  `custom mod reader`) tells us the tuple semantics:

      {beat_or_time, len_or_end, modstring, apply_type, pn?}

  * `mods`  is beat-based, `mods2` is time-based (seconds).
  * apply_type 'len': active window [v1, v1 + v2]; 'end': [v1, v2].
  * pn (optional 5th field) is the player (1 or 2); absent = both.
  Shader pokes come through `mod_shader`/SetShaderFlag; `mod_actions`
  holds per-frame closures we do NOT execute (unsupported tail).

- Actor compilation (classic command strings -> storyboard IR):
  Sprites/Quads/BitmapText with `x,100;zoom,2;linear,1;...` commands
  become storyboard Elements with property Keyframes, timed from the
  actor's creation beat (the FGCHANGES start).

compile_modfile never raises: any community file must load, so every
chunk runs under try/except and partial output is fine.
"""
from __future__ import annotations

from pathlib import Path

from analysis.games.etterna import sm_chart
from analysis.games.notitg import xml_actors
from analysis.games.notitg.mod_stubs import StubEnvironment
from analysis.player.render.effects.timeline import Keyframe
from analysis.player.render.storyboard.model import Element, build_timelines

_DESIGN_W = 640.0
_DESIGN_H = 480.0

# Actor class -> storyboard element kind. Text actors declare a File=
# font (a StepMania BitmapText/LAER quirk) which we cannot render, so
# they compile as text elements carrying their literal Text=.
_SPRITE_KINDS = frozenset({'Sprite', 'Quad'})
_TEXT_KINDS = frozenset({'BitmapText'})

# classic command verb -> storyboard property + how to read its args.
_SCALAR_PROPS = {
    'x': 'x', 'y': 'y', 'zoom': ('scale_x', 'scale_y'),
    'zoomx': 'scale_x', 'zoomy': 'scale_y', 'rotationz': 'rotation',
    'diffusealpha': 'alpha',
}
_MAX_UNSUPPORTED_DESCRIBED = 20

# Commands the engine fires on actor creation (the load-time moment we
# record). *MessageCommand / named-command bodies fire on later engine
# broadcasts and index the live actor (`self`); running them at load
# just faults on nil `self`, so they stay in the deferred tail.
_LOAD_TIME_ATTRS = frozenset({'InitCommand', 'OnCommand'})


def parse_fgchanges(sm_path) -> list:
    """FGCHANGES entries as (start_beat, name) pairs. Read straight from
    the file so parse_sm's return contract stays untouched; the tag is
    `#FGCHANGES:<beat>=<name>=<rate>=...=...,<beat>=<name>=...;`."""
    text = Path(sm_path).read_text(encoding='utf-8', errors='replace')
    entries = []
    for tag in ('FGCHANGES', 'BGCHANGES'):
        for m in _iter_change_entries(text, tag):
            entries.append(m)
    return entries


def _iter_change_entries(text: str, tag: str):
    import re

    match = re.search(r'#' + tag + r':([^;]*);', text, flags=re.DOTALL)
    if not match:
        return
    for raw in match.group(1).split(','):
        fields = raw.strip().split('=')
        if len(fields) >= 2 and fields[0].strip():
            try:
                yield (float(fields[0]), fields[1].strip(), tag)
            except ValueError:
                continue


def _resolve_lua_dir(sm_path, entries) -> Path | None:
    """The FGCHANGES `name` for these charts is `lua`, a sibling dir
    holding default.xml. Fall back to any `lua/` dir next to the sm."""
    song_dir = Path(sm_path).parent
    for _beat, name, kind in entries:
        if kind == 'FGCHANGES' and name and name.lower() != 'bg':
            candidate = song_dir / name
            if (candidate / 'default.xml').exists():
                return candidate
    fallback = song_dir / 'lua'
    return fallback if (fallback / 'default.xml').exists() else None


def _load_document(lua_dir: Path):
    """Parse default.xml, splicing in any `<Layer File=...>` includes so
    included actors (modhelpers.xml helper definitions) are present in
    document order."""
    entry = lua_dir / 'default.xml'
    root_parsed = xml_actors.parse_actor_xml(
        entry.read_text(encoding='utf-8', errors='replace'))

    lua_chunks = list(root_parsed.lua_chunks)
    classic = list(root_parsed.classic_commands)
    _splice_includes(root_parsed.root, lua_dir, lua_chunks, classic)
    return root_parsed.root, lua_chunks, classic


def _splice_includes(actor, lua_dir, lua_chunks, classic) -> None:
    for child in list(actor.children):
        include = child.attrs.get('File', '')
        if include.lower().endswith('.xml'):
            included = lua_dir / include
            if included.exists():
                sub = xml_actors.parse_actor_xml(
                    included.read_text(encoding='utf-8', errors='replace'))
                child.children.append(sub.root)
                lua_chunks[:0] = sub.lua_chunks
                classic.extend(sub.classic_commands)
        _splice_includes(child, lua_dir, lua_chunks, classic)


def _timing(sm_data: dict):
    bpms = sm_data['bpms']
    offset = sm_data['offset']
    chart = (sm_data.get('charts') or [{}])[0]
    return bpms, offset, chart


def _beat_to_seconds(sm_data, chart):
    bpms, offset, _ = _timing(sm_data)

    def convert(beat):
        return sm_chart.beat_to_time(
            beat, bpms, offset, stops=chart.get('stops'),
            delays=chart.get('delays'), warps=chart.get('warps'))
    return convert


def _run_chunks(lua_chunks, start_beat):
    """Run every CODE chunk under a shared stubbed host, in document
    order, then harvest the mod tables. Per-chunk failures warn and are
    skipped so a partial harvest survives."""
    env = StubEnvironment(start_beat)
    warnings = []
    for chunk in lua_chunks:
        if chunk.attr not in _LOAD_TIME_ATTRS:
            continue
        try:
            env.run(chunk.body, name=f'{chunk.actor.kind}.{chunk.attr}')
        except Exception as exc:
            warnings.append(f'{chunk.actor.kind}.{chunk.attr}: {exc}')
    return env, warnings


def _normalize_mod_events(env, to_seconds):
    """mods (beat-based) + mods2 (time-based) -> normalized events.

    Beat-based rows resolve through the chart timing; time-based rows
    are already seconds. The template authors both in absolute chart
    beats/seconds (not relative to the actor's creation). apply_type
    gives the window end:
      'len': end = start_value + length
      'end': end = the second field directly."""
    events = []
    for row in env.mods:
        events.append(_mod_event(row, to_seconds, beat_based=True))
    for row in env.mods2:
        events.append(_mod_event(row, to_seconds, beat_based=False))
    return [e for e in events if e is not None]


def _mod_event(row, to_seconds, beat_based):
    if not isinstance(row, dict) or len(row) < 4:
        return None
    v1 = _as_float(row.get(1))
    v2 = _as_float(row.get(2))
    modstring = row.get(3)
    apply_type = row.get(4)
    if v1 is None or v2 is None or not isinstance(modstring, str):
        return None
    if apply_type not in ('len', 'end'):
        return None

    end_field = (v1 + v2) if apply_type == 'len' else v2
    if beat_based:
        start_s = to_seconds(v1)
        end_s = to_seconds(end_field)
    else:
        start_s, end_s = v1, end_field
    player = _as_int(row.get(5))
    return {
        'beat': v1, 'len_beats': v2, 'modstring': modstring.strip(),
        'apply_type': apply_type, 'player': player,
        't_start': start_s, 't_end': end_s, 'time_based': not beat_based,
    }


def _normalize_shader_flags(env, to_seconds):
    flags = []
    for beat, key, which in env.shader_flags:
        flags.append({
            'beat': beat, 't': to_seconds(beat),
            'key': key, 'which': which,
        })
    return flags


def _describe_unsupported(env):
    described = []
    for row in env.mod_actions[:_MAX_UNSUPPORTED_DESCRIBED]:
        beat = row.get(1) if isinstance(row, dict) else None
        payload = row.get(2) if isinstance(row, dict) else None
        described.append({'beat': _as_float(beat),
                          'payload': _kind_of(payload)})
    return {'count': len(env.mod_actions), 'described': described}


def _kind_of(value) -> str:
    if callable(value):
        return 'function'
    if isinstance(value, str):
        return f'message:{value}'
    return type(value).__name__


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    f = _as_float(value)
    return int(f) if f is not None else None


def _compile_elements(classic_commands, to_seconds, start_beat):
    """Actors with classic command strings -> storyboard Elements. One
    element per actor that carries renderable geometry commands."""
    by_actor = {}
    for command in classic_commands:
        by_actor.setdefault(id(command.actor), (command.actor, []))
        by_actor[id(command.actor)][1].append(command)

    start_time = to_seconds(start_beat)
    elements = []
    for actor, commands in by_actor.values():
        element = _actor_element(actor, commands, start_time)
        if element is not None:
            elements.append(element)
    return elements


def _actor_element(actor, commands, start_time):
    kind = _element_kind(actor.kind)
    if kind is None:
        return None

    keyframes = _collect_keyframes(commands, start_time)
    if not any(keyframes.values()):
        return None

    text = actor.attrs.get('Text', '')
    asset = actor.attrs.get('Texture') or actor.attrs.get('File')
    return Element(
        kind=kind, z=0, z_index=0,
        t_start=start_time, t_end=float('inf'),
        anchor=(0.5, 0.5), origin=(0.5, 0.5),
        timelines=build_timelines(keyframes={
            prop: kfs for prop, kfs in keyframes.items() if kfs}),
        asset=str(asset) if asset else None,
        text=str(text),
    )


def _element_kind(actor_kind: str):
    if actor_kind in _TEXT_KINDS:
        return 'text'
    if actor_kind in _SPRITE_KINDS:
        return 'sprite' if actor_kind == 'Sprite' else 'rect'
    return None


def _collect_keyframes(commands, start_time):
    """Replay each command list as the engine would: a tween verb sets
    the pending duration+easing for the next property command, `sleep`
    advances the clock, property verbs emit a Keyframe at the running
    time. Only OnCommand/InitCommand run on creation; Message commands
    fire on engine broadcasts we do not model, so they are skipped."""
    props = {prop: [] for prop in
             ('x', 'y', 'scale_x', 'scale_y', 'rotation', 'alpha', 'color')}
    for command in commands:
        if command.attr not in ('OnCommand', 'InitCommand'):
            continue
        _replay_commands(command.commands, start_time, props)
    return props


def _replay_commands(commands, start_time, props):
    clock = start_time
    pending_dur = 0.0
    pending_ease = 0
    for verb, args in commands:
        if xml_actors.is_tween_verb(verb):
            duration = _as_float(args[0]) if args else 0.0
            if verb == 'sleep':
                clock += duration or 0.0
                pending_dur, pending_ease = 0.0, 0
            else:
                pending_dur = duration or 0.0
                pending_ease = xml_actors.tween_easing(verb) or 0
            continue
        _emit_property(verb, args, clock, pending_dur, pending_ease, props)
        pending_dur, pending_ease = 0.0, 0


def _emit_property(verb, args, clock, duration, easing, props):
    if verb == 'diffuse':
        values = _rgb(args)
        if values is not None:
            props['color'].append(Keyframe(clock, values, duration, easing))
        alpha = _as_float(args[3]) if len(args) > 3 else None
        if alpha is not None:
            props['alpha'].append(Keyframe(clock, (alpha,), duration, easing))
        return

    prop = _SCALAR_PROPS.get(verb)
    if prop is None:
        return
    value = _as_float(args[0]) if args else None
    if value is None:
        return
    targets = prop if isinstance(prop, tuple) else (prop,)
    for target in targets:
        props[target].append(Keyframe(clock, (value,), duration, easing))


def _rgb(args):
    channels = [_as_float(a) for a in args[:3]]
    if len(channels) < 3 or any(c is None for c in channels):
        return None
    return tuple(channels)


def compile_modfile(sm_path) -> dict | None:
    """Compile a NotITG chart's modfile. Returns None when the chart has
    no resolvable lua modfile; otherwise a dict of harvested mod events,
    shader flags, unsupported tail, storyboard elements, and warnings.
    Never raises: chart loading must survive any community file."""
    try:
        return _compile_modfile(sm_path)
    except Exception as exc:
        return {'mod_events': [], 'shader_flags': [], 'unsupported':
                {'count': 0, 'described': []}, 'elements': [],
                'warnings': [f'compile aborted: {exc}']}


def _compile_modfile(sm_path):
    entries = parse_fgchanges(sm_path)
    lua_dir = _resolve_lua_dir(sm_path, entries)
    if lua_dir is None:
        return None

    sm_data = sm_chart.parse_sm(sm_path)
    _root, lua_chunks, classic_commands = _load_document(lua_dir)

    _bpms, _offset, chart = _timing(sm_data)
    to_seconds = _beat_to_seconds(sm_data, chart)
    start_beat = min((b for b, _n, k in entries if k == 'FGCHANGES'),
                     default=0.0)

    env, chunk_warnings = _run_chunks(lua_chunks, start_beat)
    return {
        'mod_events': _normalize_mod_events(env, to_seconds),
        'shader_flags': _normalize_shader_flags(env, to_seconds),
        'unsupported': _describe_unsupported(env),
        'elements': _compile_elements(classic_commands, to_seconds,
                                      start_beat),
        'warnings': chunk_warnings,
    }
