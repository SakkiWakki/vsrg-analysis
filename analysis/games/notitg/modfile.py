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

- One-shot replay (mod_actions closures): the template's `mod_actions`
  are SCHEDULED ONE-SHOTS - the per-frame reader fires each closure
  exactly once when its beat passes (curaction advances monotonically,
  never resets). We execute each closure once, in beat order, against
  the recording stub, capturing the `SetShaderFlag`/`ApplyGameCommand`
  pokes it makes. A closure that pokes an actor we do not model faults
  harmlessly (per-closure try/except).

  PERSISTENCE of message-applied mods: an `ApplyGameCommand('mod,X')`
  fired from a closure is NOT persistent. The reader runs `mod,clearall`
  every frame and only reapplies windows from the `mods`/`mods2` tables;
  a closure's mod is not in those tables, so the very next frame's
  clearall wipes it. It therefore lives for one frame (~20ms). We encode
  each as a ZERO-LENGTH window [beat, beat] (a one-frame spike), NOT a
  persistent [beat, +inf) start - verified against gat's reader
  (default.xml ~line 3999 clearall + ~4680 monotonic action loop).

- Actor compilation (classic command strings -> storyboard IR):
  Sprites/Quads/BitmapText with `x,100;zoom,2;linear,1;...` commands
  become storyboard Elements with property Keyframes, timed from the
  actor's creation beat (the FGCHANGES start).

compile_modfile never raises: any community file must load, so every
chunk runs under try/except and partial output is fine.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from analysis.games.etterna import sm_chart
from analysis.games.notitg import aft_drivers, sprite_sheet, xml_actors
from analysis.games.notitg.mod_stubs import StubEnvironment
from analysis.games.notitg.paths import find_notitg_dirs
from analysis.games.notitg.recording_actor import RecordingActor
from analysis.player.render.effects.timeline import EventTimeline
from analysis.player.render.storyboard import bitmap_font
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


def _font_search_dirs(lua_dir) -> list:
    """Where a BitmapText `File=` font resolves: the chart's own lua dir
    first (chart-bundled fonts win), then the NotITG theme Fonts dirs
    (gat's `_eurostile normal`, `Common normal`, ... live there). The
    install root is derived from the chart's own path (its `Songs`
    ancestor) so the compile is self-contained, then the runtime path
    override backfills if the chart sits outside a `Songs` tree."""
    dirs = [str(lua_dir)]
    for theme_fonts in _theme_font_dirs(_install_root(lua_dir)):
        dirs.append(str(theme_fonts))
    return dirs


def _install_root(lua_dir):
    for parent in Path(lua_dir).parents:
        if parent.name == 'Songs':
            return parent.parent
    return find_notitg_dirs().get('root')


def _theme_font_dirs(root):
    if not root:
        return
    themes = Path(root) / 'Themes'
    if themes.is_dir():
        for theme in themes.iterdir():
            if (theme / 'Fonts').is_dir():
                yield theme / 'Fonts'


def _font_resolver(lua_dir):
    """A cached `reference -> BitmapFont | None` resolver over the font
    search dirs. Cached so a chart with many BitmapText actors sharing
    one font parses the .ini once."""
    search_dirs = _font_search_dirs(lua_dir)
    cache: dict = {}

    def resolve(reference):
        if not reference:
            return None
        if reference not in cache:
            cache[reference] = bitmap_font.load_font(reference, search_dirs)
        return cache[reference]
    return resolve


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


def _load_document(lua_dir: Path, bg_stem=''):
    """Parse default.xml, splicing in any `<Layer File=...>` includes so
    included actors (modhelpers.xml helper definitions, the chara
    subtree) are present in document order. Every actor is annotated with
    the directory its XML came from (`_base_dir`) so a Sprite's `Texture=`
    / `File=` reference resolves against ITS file's location, not the top
    lua dir (chara sprites reference `shame/idle.sprite` relative to
    `lua/chara`). `bg_stem` (the #BACKGROUND image stem) tags any include
    subtree that draws it as a background layer (behind the notes)."""
    entry = lua_dir / 'default.xml'
    root_parsed = xml_actors.parse_actor_xml(
        entry.read_text(encoding='utf-8', errors='replace'))

    lua_chunks = list(root_parsed.lua_chunks)
    classic = list(root_parsed.classic_commands)
    _tag_base_dir(root_parsed.root, lua_dir)
    _splice_includes(root_parsed.root, lua_dir, lua_chunks, classic, bg_stem)
    return root_parsed.root, lua_chunks, classic


def _tag_base_dir(actor, base_dir) -> None:
    actor._base_dir = base_dir
    for child in actor.children:
        _tag_base_dir(child, base_dir)


def _splice_includes(actor, lua_dir, lua_chunks, classic, bg_stem='') -> None:
    for child in list(actor.children):
        # Recurse into the child's OWN children first (captured now, so
        # an appended include subtree - already fully spliced - is not
        # re-processed under the wrong base dir).
        _splice_includes(child, lua_dir, lua_chunks, classic, bg_stem)
        included = _include_path(child.attrs.get('File', ''), lua_dir)
        if included is None:
            continue
        sub = xml_actors.parse_actor_xml(
            included.read_text(encoding='utf-8', errors='replace'))
        _tag_base_dir(sub.root, included.parent)
        _splice_includes(sub.root, included.parent, sub.lua_chunks,
                         sub.classic_commands, bg_stem)
        # An include that renders the #BACKGROUND image is a BGCHANGES-
        # style background layer: SM draws it BEHIND the notefield (later
        # FG actors sit in front). Tag its subtree so the compiler routes
        # it to a below-the-notes z band.
        if bg_stem and _subtree_draws_background(sub.root, bg_stem):
            _tag_background_layer(sub.root)
        child.children.append(sub.root)
        lua_chunks[:0] = sub.lua_chunks
        classic.extend(sub.classic_commands)


def _subtree_draws_background(actor, bg_stem) -> bool:
    """True when any actor in a spliced subtree loads the chart's
    #BACKGROUND image (by File=/Texture=/Load= filename stem)."""
    reference = (actor.attrs.get('File') or actor.attrs.get('Texture')
                 or actor.attrs.get('Load') or '')
    if reference and Path(reference).stem.casefold() == bg_stem:
        return True
    return any(_subtree_draws_background(c, bg_stem) for c in actor.children)


def _tag_background_layer(actor) -> None:
    actor._background_layer = True
    for child in actor.children:
        _tag_background_layer(child)


def _include_path(include, lua_dir) -> Path | None:
    """The actor-XML file a `File=` reference includes, or None. A `.xml`
    reference is the file directly; a bare directory name (`File="chara"`,
    gat's `<ZZZZZLAER File="chara"/>`) resolves to that dir's default.xml
    (SM's implicit directory-actor rule). Anything else (a texture path)
    is not an actor include."""
    if not include:
        return None
    if include.lower().endswith('.xml'):
        candidate = lua_dir / include
        return candidate if candidate.exists() else None
    directory = lua_dir / include
    entry = directory / 'default.xml'
    return entry if entry.is_dir() is False and entry.exists() else None


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


def _run_chunks(root, start_beat, to_seconds):
    """Load the actor tree under a shared stubbed host: one persistent
    recorder per actor, InitCommand/OnCommand run with `self` bound to
    it, and every `<Name>MessageCommand` / `<Name>Command` registered so
    later broadcasts and play/queuecommands run on the SAME recorder.
    A trailing `NAME = self` still binds a global (the poke target for the
    mod_actions closures). Per-chunk failures warn; a partial harvest
    survives."""
    env = StubEnvironment(start_beat, to_seconds=to_seconds)
    warnings = env.load_actors(root)
    return env, warnings


_BIND_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*self\b')


def _bound_global_name(actor):
    """The Lua global an actor's InitCommand/OnCommand self-assigns
    (`gat_g_rot_intro = self`), or None. This is the name the scheduled
    mod_actions closures poke, so it links the actor to its recorder."""
    for attr in _LOAD_TIME_ATTRS:
        value = actor.attrs.get(attr, '')
        if value.startswith('%'):
            match = _BIND_RE.search(value)
            if match and match.group(1) != 'self':
                return match.group(1)
    return None


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


def _normalize_applied_mods(env, to_seconds):
    """`ApplyGameCommand('mod,X')` recordings from the one-shot replay ->
    zero-length mod windows [beat, beat]. See the module docstring: these
    are one-frame spikes (wiped by the next frame's clearall), not
    persistent windows, so start and end coincide."""
    events = []
    for beat, modstring, player in env.applied_mods:
        if not isinstance(modstring, str) or not modstring:
            continue
        t = to_seconds(beat)
        events.append({
            'beat': beat, 'len_beats': 0.0, 'modstring': modstring,
            'apply_type': 'oneshot', 'player': player,
            't_start': t, 't_end': t, 'time_based': False,
        })
    return events


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


def _compile_elements(classic_commands, to_seconds, start_beat,
                      named_keyframes=None):
    """FLAT compile (kept for charts/tests with no hierarchy): actors
    with classic command strings -> storyboard Elements, one per actor,
    with recorded pokes merged in when the actor bound a global name."""
    by_actor = {}
    order = []
    for command in classic_commands:
        if id(command.actor) not in by_actor:
            by_actor[id(command.actor)] = command.actor
            order.append(command.actor)

    start_time = to_seconds(start_beat)
    named_keyframes = named_keyframes or {}
    elements = []
    for actor in order:
        element = _leaf_element(actor, start_time, named_keyframes)
        if element is not None:
            elements.append(element)
    return elements


def compile_element_tree(root, to_seconds, start_beat, named_keyframes=None,
                         fonts=None, actor_keyframes=None):
    """HIERARCHICAL compile: the actor tree becomes a tree of storyboard
    Elements. An ActorFrame with drawable descendants becomes a 'group'
    whose transform composes onto its children; a Sprite/Quad/BitmapText
    becomes a leaf. Each actor's timeline merges its XML command-string
    keyframes with the pokes recorded onto the global it self-assigned.

    `fonts` resolves a BitmapText actor's `File=` to a parsed SM font
    (bitmaptext elements); None falls back to system-font text.

    Empty subtrees (frames with no drawable descendants and no own
    animation) are pruned, so a chart with no real hierarchy collapses
    back to a flat element list.

    Background-layer subtrees (BGCHANGES actors tagged by _load_document)
    are HOISTED to top-level elements with a below-the-notes z, so SM's
    'background behind the notefield, foreground in front' draw order is
    honoured (the StoryboardEffect only bands top-level elements)."""
    start_time = to_seconds(start_beat)
    named_keyframes = named_keyframes or {}
    children = []
    below = []
    for child in root.children:
        element = _compile_actor(child, start_time, named_keyframes, fonts,
                                 below, actor_keyframes)
        if element is not None:
            children.append(element)
    return below + children


def _compile_actor(actor, start_time, named_keyframes, fonts, below=None,
                   actor_keyframes=None):
    if below is None:
        below = []
    if getattr(actor, '_background_layer', False):
        element = _compile_background_actor(actor, start_time, named_keyframes,
                                           fonts, actor_keyframes)
        if element is not None:
            below.append(element)
        return None

    child_elements = []
    for child in actor.children:
        element = _compile_actor(child, start_time, named_keyframes, fonts,
                                 below, actor_keyframes)
        if element is not None:
            child_elements.append(element)

    keyframes = _merged_keyframes(actor, start_time, named_keyframes,
                                  actor_keyframes)
    if child_elements:
        return _group_element(actor, start_time, keyframes,
                              tuple(child_elements))
    return _leaf_element(actor, start_time, named_keyframes,
                         precomputed=keyframes, fonts=fonts)


# The background band z: below the notes/field (z=0) but a valid
# storyboard slot. gat's whole BGCHANGES tree lands here.
_BACKGROUND_Z = -100


def _compile_background_actor(actor, start_time, named_keyframes, fonts,
                              actor_keyframes=None):
    """Compile a background-layer subtree as one top-level element at the
    background z. The subtree is self-contained (its own gat_all_bg /
    gat_bg transforms), so hoisting it past the identity top frames keeps
    its placement while moving it behind the notes."""
    child_elements = []
    for child in actor.children:
        element = _compile_background_actor(child, start_time,
                                           named_keyframes, fonts,
                                           actor_keyframes)
        if element is not None:
            child_elements.append(element)

    keyframes = _merged_keyframes(actor, start_time, named_keyframes,
                                  actor_keyframes)
    if child_elements:
        element = _group_element(actor, start_time, keyframes,
                                tuple(child_elements))
    else:
        element = _leaf_element(actor, start_time, named_keyframes,
                               precomputed=keyframes, fonts=fonts)
    return _with_z(element, _BACKGROUND_Z) if element is not None else None


def _with_z(element, z):
    return replace(element, z=z)


def _merged_keyframes(actor, start_time, named_keyframes, actor_keyframes=None):
    """All keyframes for one actor.

    When the actor carries a load-pass `_recorder_id` (the modfile compile
    path), its recorder already holds the COMPLETE poke stream - its
    InitCommand/OnCommand AND every message / play / queuecommand body
    dispatched onto it during load and the mod_actions replay. That stream
    is authoritative (the anonymous bg Layer children get their BG2/BG3/BG4
    crossfades only here - they self-assign no global), so we use it
    directly.

    The FLAT path (tests / charts compiled without a StubEnvironment) has
    no recorder ids: it re-poks the classic InitCommand/OnCommand strings
    on a fresh recorder and merges whatever the actor's bound global
    recorded from the closures."""
    recorded = _recorded_keyframes(actor, actor_keyframes)
    if recorded is not None:
        return recorded

    recorder = RecordingActor(clock=start_time)
    for attr in ('InitCommand', 'OnCommand'):
        value = actor.attrs.get(attr, '')
        if value and not value.startswith('%'):
            for verb, args in xml_actors.parse_command_string(value):
                recorder.poke(verb, args)
    keyframes = recorder.keyframes()

    name = _bound_global_name(actor)
    if name and name in named_keyframes:
        for prop, frames in named_keyframes[name].items():
            keyframes.setdefault(prop, []).extend(frames)
    return keyframes


def _recorded_keyframes(actor, actor_keyframes):
    """The actor's load-pass recorder keyframes (a fresh dict of lists so
    the caller may mutate it), or None when no recorder id is available."""
    if not actor_keyframes:
        return None
    rec_id = getattr(actor, '_recorder_id', None)
    if rec_id is None or rec_id not in actor_keyframes:
        return None
    return {prop: list(frames)
            for prop, frames in actor_keyframes[rec_id].items()}


def _group_element(actor, start_time, keyframes, children):
    return Element(
        kind='group', z=0, z_index=0,
        t_start=start_time, t_end=float('inf'),
        anchor=(0.0, 0.0), origin=(0.5, 0.5),
        timelines=build_timelines(keyframes=_drawable_props(keyframes)),
        children=children,
    )


def _leaf_element(actor, start_time, named_keyframes, precomputed=None,
                  fonts=None):
    text = actor.attrs.get('Text', '')
    font = _resolve_font(actor, fonts)
    if font is not None:
        asset, spec, states = (None, sprite_sheet.AssetSizeSpec(), ())
    else:
        asset, spec, states = _resolve_sprite(actor)
    kind = _element_kind(actor.kind, has_text=bool(text), font=font,
                         has_image=_is_image_asset(asset))
    if kind is None:
        return None

    keyframes = precomputed if precomputed is not None \
        else _merged_keyframes(actor, start_time, named_keyframes)
    drawable = _drawable_props(keyframes)
    state_pin = _state_pin(keyframes)
    # A frame pin is real content (a sprite animated purely by
    # setstate/animate pokes), so it keeps an otherwise-untweened actor
    # alive alongside any transform/color keyframes.
    if not any(drawable.values()) and state_pin is None:
        return None
    return Element(
        kind=kind, z=0, z_index=0,
        t_start=start_time, t_end=float('inf'),
        anchor=(0.0, 0.0), origin=(0.5, 0.5),
        timelines=build_timelines(keyframes=drawable),
        asset=asset,
        text=str(text), font=font,
        sheet_cols=spec.cols, sheet_rows=spec.rows, sheet_states=states,
        size_spec=spec, state_pin=state_pin,
    )


# Frame-index keyframes the recorder emits from setstate/animate pokes.
# When present, a pin timeline overrides the sheet's auto-animation.
_STATE_PIN_PROP = 'frame'


def _state_pin(keyframes):
    """An EventTimeline of the sprite's frame index over time, from
    recorded `setstate`/`animate` pokes, or None when the actor never
    pinned a frame (the sheet then auto-animates through its states)."""
    frames = keyframes.get(_STATE_PIN_PROP)
    if not frames:
        return None
    return EventTimeline(frames, rest=(0.0,))


# StepMania's built-in flat-color texture: the renderer synthesizes it,
# so the reference stays a name (never resolved to a file on disk).
_BUILTIN_TEXTURES = frozenset({'white'})


def _resolve_asset(actor) -> str | None:
    """Absolute path to a Sprite's texture, resolved against the actor's
    OWN XML directory (`_base_dir`), or None. The reference comes from
    `Texture=` / `Load=` / `File=`; SM resolves it relative to the file
    that declared the actor, so a chara sprite's `shame/idle.sprite`
    resolves under `lua/chara`, not the top lua dir. A `.sprite` file is
    an animation manifest whose first `Texture=` names the real image; we
    resolve that image so the sprite shows its idle frame. `white` stays
    a name (the renderer synthesizes it)."""
    reference = (actor.attrs.get('Texture') or actor.attrs.get('Load')
                 or actor.attrs.get('File'))
    if not reference:
        return None
    if reference in _BUILTIN_TEXTURES:
        return reference
    base_dir = getattr(actor, '_base_dir', None)
    if base_dir is None:
        return reference
    return _resolve_texture_path(reference, Path(base_dir))


_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')


def _resolve_texture_path(reference: str, base_dir: Path) -> str | None:
    """Resolve a texture reference to an existing image file under
    `base_dir`, following SM's leniencies: an explicit path is used as
    is; a `.sprite`/`.actor` manifest yields its inner `Texture=`; a bare
    name matches a file with any image extension (SM omits them)."""
    candidate = (base_dir / reference)
    if candidate.suffix.lower() == '.sprite':
        return _sprite_manifest_texture(candidate, base_dir)
    if candidate.exists():
        return str(candidate)
    for suffix in _IMAGE_SUFFIXES:
        with_ext = candidate.with_name(candidate.name + suffix)
        if with_ext.exists():
            return str(with_ext)
    return str(candidate)


def _sprite_manifest_texture(sprite_path: Path, base_dir: Path) -> str | None:
    """The image an SM `.sprite` manifest points at: its `[Sprite]`
    section's `Texture=<image>`, resolved beside the manifest. Returns the
    manifest path itself when it has no readable Texture line."""
    if not sprite_path.exists():
        return str(sprite_path)
    for line in sprite_path.read_text(
            encoding='utf-8', errors='replace').splitlines():
        key, _sep, value = line.partition('=')
        if key.strip().lower() == 'texture' and value.strip():
            return _resolve_texture_path(value.strip(), sprite_path.parent)
    return str(sprite_path)


def _resolve_sprite(actor) -> tuple:
    """A sprite's `(asset_path, size_spec, sheet_states)`.

    `size_spec` is the SM size conventions its filename encodes (NxM grid
    from `GetFrameDimensionsFromFileName`, plus `(doubleres)`/`res`
    hints) - the record the renderer resolves against raw pixels to get
    the frame's logical size. The state list is the `.sprite` manifest's
    `Frame%04d=`/`Delay%04d=` pairs when the reference is a manifest, else
    SM's default sequential animation (one state per frame). A plain
    single-frame sprite yields (path, 1x1 spec, ())."""
    asset = _resolve_asset(actor)
    if asset is None or asset in _BUILTIN_TEXTURES:
        return (asset, sprite_sheet.AssetSizeSpec(), ())

    spec = sprite_sheet.size_spec_from_filename(asset)
    frame_count = spec.cols * spec.rows
    states = _manifest_states(actor, frame_count)
    if not states and frame_count > 1:
        states = sprite_sheet.default_states(frame_count)
    return (asset, spec, states)


def _manifest_states(actor, frame_count: int) -> tuple:
    """The `.sprite` state list for this actor's texture reference, or ()
    when the reference is not a `.sprite` manifest (or defines no
    frames). The manifest's `Frame`/`Delay` pairs override SM's default
    sequence exactly as `Sprite::LoadFromNode` does."""
    reference = (actor.attrs.get('Texture') or actor.attrs.get('Load')
                 or actor.attrs.get('File'))
    base_dir = getattr(actor, '_base_dir', None)
    if not reference or base_dir is None:
        return ()
    manifest = Path(base_dir) / reference
    if manifest.suffix.lower() != '.sprite' or not manifest.exists():
        return ()
    text = manifest.read_text(encoding='utf-8', errors='replace')
    return sprite_sheet.parse_sprite_states(text, frame_count)


def _resolve_font(actor, fonts):
    """The parsed SM bitmap font a text actor draws with, or None. A
    `File=` that resolves to a font .ini marks a BitmapText (gat's
    `<LAER File="_eurostile normal" Text=...>`); a Texture= sprite or an
    unresolvable name stays a plain sprite/text."""
    if fonts is None or actor.attrs.get('Texture'):
        return None
    reference = actor.attrs.get('Font') or actor.attrs.get('File')
    return fonts(reference) if reference else None


# Storyboard properties the renderer samples. `hidden` is SM's hard
# visibility bit (1 hidden, 0 shown), kept separate from `alpha` so a
# `hidden,1` gate and a diffusealpha crossfade coexist. The recorder also
# captures 3D-only channels (z, rotation_x/y, skew, scale_z) with no 2D
# analogue; they are kept out of the built element timelines.
_DRAWABLE_PROPS = frozenset({
    'x', 'y', 'scale_x', 'scale_y', 'rotation', 'alpha', 'color', 'hidden',
    # Absolute on-screen size (SM zoomto/setsize). Rest is the unset
    # sentinel in model._SCALAR_RESTS; when set it overrides natural*scale.
    'size_x', 'size_y',
})


def _drawable_props(keyframes):
    return {prop: frames for prop, frames in keyframes.items()
            if prop in _DRAWABLE_PROPS and frames}


def _is_image_asset(asset) -> bool:
    """True when a resolved asset is a real image reference (an existing
    file or the synthesized `white`), so an untyped `Actor`/`Layer` that
    loads one - gat's chara sprites, `<Actor File="shame/idle.sprite">` -
    counts as a Sprite even without a `Type=`."""
    if not asset:
        return False
    if asset in _BUILTIN_TEXTURES:
        return True
    return Path(asset).exists()


def _element_kind(actor_kind: str, has_text=False, font=None,
                  has_image=False):
    if font is not None:
        return 'bitmaptext'
    if actor_kind in _TEXT_KINDS or has_text:
        return 'text'
    if actor_kind in _SPRITE_KINDS:
        return 'sprite' if actor_kind == 'Sprite' else 'rect'
    return 'sprite' if has_image else None


def compile_modfile(sm_path) -> dict | None:
    """Compile a NotITG chart's modfile. Returns None when the chart has
    no resolvable lua modfile; otherwise a dict of harvested mod events,
    shader flags, unsupported tail, storyboard elements, and warnings.
    Never raises: chart loading must survive any community file."""
    try:
        return _compile_modfile(sm_path)
    except Exception as exc:
        return {'mod_events': [], 'shader_flags': [], 'unsupported':
                {'count': 0, 'described': []}, 'elements': [], 'tree': [],
                'named_actors': 0, 'recorded_keyframes': 0,
                'warnings': [f'compile aborted: {exc}']}


def _compile_modfile(sm_path):
    entries = parse_fgchanges(sm_path)
    lua_dir = _resolve_lua_dir(sm_path, entries)
    if lua_dir is None:
        return None

    sm_data = sm_chart.parse_sm(sm_path)
    bg_stem = Path(_sm_background_name(sm_path)).stem.casefold()
    root, _lua_chunks, classic_commands = _load_document(lua_dir, bg_stem)

    _bpms, _offset, chart = _timing(sm_data)
    to_seconds = _beat_to_seconds(sm_data, chart)
    start_beat = min((b for b, _n, k in entries if k == 'FGCHANGES'),
                     default=0.0)

    env, chunk_warnings = _run_chunks(root, start_beat, to_seconds)
    fired, failed = env.replay_mod_actions()
    named_keyframes = env.named_actor_keyframes()
    named_meta = env.named_actor_meta()
    actor_keyframes = env.actor_keyframes()

    mod_events = _normalize_mod_events(env, to_seconds)
    mod_events.extend(_normalize_applied_mods(env, to_seconds))
    fonts = _font_resolver(lua_dir)
    tree = compile_element_tree(root, to_seconds, start_beat, named_keyframes,
                                fonts=fonts, actor_keyframes=actor_keyframes)
    return {
        'mod_events': mod_events,
        'shader_flags': _normalize_shader_flags(env, to_seconds),
        'unsupported': _describe_unsupported(env),
        'elements': _compile_elements(classic_commands, to_seconds,
                                      start_beat, named_keyframes),
        'tree': tree,
        'has_background': _has_background_actors(tree, sm_path),
        'field_copies': _field_copies(root, named_keyframes, named_meta,
                                      to_seconds, start_beat, actor_keyframes),
        'aft_bg_visible': _aft_bg_visible_timeline(root, bg_stem,
                                                   actor_keyframes),
        'base_field_hidden': _base_field_hidden_timeline(env),
        'named_actors': len(named_keyframes),
        'recorded_keyframes': _count_recorded_keyframes(named_keyframes),
        'replay': {'fired': fired, 'failed': failed,
                   'applied_mods': len(env.applied_mods),
                   'swallowed': env.swallowed},
        'warnings': chunk_warnings,
    }


def _count_recorded_keyframes(named_keyframes) -> int:
    return sum(len(frames) for props in named_keyframes.values()
               for frames in props.values())


def _has_background_actors(tree, sm_path) -> bool:
    """True when the compiled actor tree draws the chart's own background
    image (gat's `bg/` BGCHANGES tree renders bg.png). When it does, the
    built-in MapBackgroundEffect is a duplicate and the adapter drops it
    (`background_path` -> None), leaving the modfile's animated background
    (which rides the actor/AFT transforms) as the sole one."""
    background = _sm_background_name(sm_path)
    if not background:
        return False
    stem = Path(background).stem.casefold()
    return any(_is_background_sprite(el, stem) for el in _iter_elements(tree))


def _sm_background_name(sm_path) -> str:
    match = re.search(r'#BACKGROUND:([^;]*);',
                      Path(sm_path).read_text(encoding='utf-8',
                                              errors='replace'))
    return match.group(1).strip() if match else ''


def _is_background_sprite(element, stem) -> bool:
    asset = getattr(element, 'asset', None)
    return bool(asset and Path(asset).stem.casefold() == stem)


def _iter_elements(elements):
    for element in elements:
        yield element
        yield from _iter_elements(element.children)


# Proxy-actor globals (Proxy(pn) = _G['P<n>p']) re-render a player's
# whole notefield elsewhere - true field copies, same as an AFT-copy
# sprite, so they feed the field producer alongside the AFT copies.
_PROXY_NAMES = frozenset({'P1p', 'P2p', 'P3p', 'P4p'})

# Field-copy transform properties the producer samples. base_scale
# folds into scale (SM's separate pre-multiplier); rotation is degrees.
# `hidden` is SM's hard visibility bit (the proxy ActorFrames rest
# `hidden,1` until their section shows them), so the producer drops a
# copy whose hidden bit is set even when its diffusealpha is still 1.
_FIELD_PROPS = ('x', 'y', 'rotation', 'scale_x', 'scale_y',
                'base_scale_x', 'base_scale_y', 'alpha', 'hidden')


def _field_copies(root, named_keyframes, named_meta, to_seconds,
                  start_beat, actor_keyframes=None) -> list:
    """Actors that draw a copy of the playfield - AFT-screen copy
    sprites (`aft_source` set) and Proxy notefield actors - as
    ready-to-sample transform timelines for the field producer.

    Each entry: {'name', 'source', 'timelines'} where `timelines` is one
    EventTimeline per _FIELD_PROPS, built from the actor's FULL merged
    poke stream (InitCommand base position + closure pokes), same as the
    tree elements - so a copy's base screen-center placement and its
    relative addx/addy moves resolve together. Ordinary named actors
    (mod-driver quads, rotators) are excluded: only actors whose texture
    IS the captured field become field instances.

    Copies with a per-frame data-holder-quad DRIVER (aft_drivers) also
    get compile-time grid-sampled keyframes over the driver window,
    reading the compiled quad curves - the drivers never poke the copy's
    own timeline, so without this the copy would sit at its base."""
    start_time = to_seconds(start_beat)
    quad_timelines = _quad_source_timelines(named_keyframes)
    copies = []
    for actor in _iter_actors(root):
        name = _bound_global_name(actor)
        source = _field_source(name, named_meta)
        if source is None:
            continue
        keyframes = _merged_keyframes(actor, start_time, named_keyframes,
                                      actor_keyframes)
        field_keyframes = {prop: keyframes[prop] for prop in _FIELD_PROPS
                           if keyframes.get(prop)}
        _merge_driven_keyframes(field_keyframes, name, quad_timelines,
                                to_seconds)
        if not field_keyframes:
            continue
        copies.append({
            'name': name, 'source': source,
            'timelines': build_timelines(rests=_FIELD_RESTS,
                                         keyframes=field_keyframes),
        })
    return copies


# The engine player actors the chart hides while proxies stand in
# (`P1:hidden(1)`). PlayerP1 is player 0's real NoteField; hiding it means
# the copies replace the base field, so the renderer skips the base draw.
_BASE_PLAYER_NAME = 'PlayerP1'


def _base_field_hidden_timeline(env):
    """An EventTimeline sampling 1.0 while the real player-0 NoteField is
    hidden (the chart poked `P1:hidden(1)` to let proxies stand in), 0.0
    otherwise. None when the chart never hides it (every non-notitg chart,
    and unmodded notitg). Sourced from the recorder the screen stub hands
    back for `GetChild('PlayerP1')`."""
    hidden = env.player_keyframes(_BASE_PLAYER_NAME).get('hidden')
    if not hidden:
        return None
    return EventTimeline(hidden, rest=(0.0,))


def _aft_bg_visible_timeline(root, bg_stem, actor_keyframes):
    """An EventTimeline sampling 1.0 while an ActorFrameTexture capture
    includes the background, 0.0 otherwise.

    gat toggles a fullscreen bg.png quad between its `ShowAFT` (no bg) and
    `ShowAFTBG` (bg) states: the quad is `hidden,1` until `ShowAFTBG`
    shows it and `HideAFT` hides it again. Its recorded `hidden` timeline
    (inverted to bg-visible) IS the AFT bg-in-capture signal. Multiple
    such quads OR together (bg visible if any is shown). None when the
    chart has no AFT-bg quad."""
    visibles = []
    for actor in _iter_actors(root):
        if not _is_aft_bg_quad(actor, bg_stem):
            continue
        hidden = _actor_hidden_keyframes(actor, actor_keyframes)
        if hidden:
            visibles.append(EventTimeline(
                [_invert_hidden(kf) for kf in hidden], rest=(1.0,)))
    if not visibles:
        return None
    return visibles[0] if len(visibles) == 1 else _AnyVisible(visibles)


def _is_aft_bg_quad(actor, bg_stem) -> bool:
    """True for a fullscreen background sprite/quad the AFT rig toggles:
    it responds to `ShowAFTBG` and draws the chart's #BACKGROUND image."""
    if 'ShowAFTBG' not in actor.message_commands():
        return False
    reference = (actor.attrs.get('File') or actor.attrs.get('Texture')
                 or actor.attrs.get('Load') or '')
    return bool(bg_stem) and Path(reference).stem.casefold() == bg_stem


def _actor_hidden_keyframes(actor, actor_keyframes):
    if not actor_keyframes:
        return None
    rec_id = getattr(actor, '_recorder_id', None)
    if rec_id is None:
        return None
    return actor_keyframes.get(rec_id, {}).get('hidden')


def _invert_hidden(keyframe):
    """A `hidden` keyframe (1 hidden / 0 shown) as a bg-visible keyframe
    (0 hidden -> 1 visible). Zero-duration step, so the boolean flips
    exactly at the poke time."""
    hidden = keyframe.values[0]
    return replace(keyframe, values=(1.0 - hidden,), duration=0.0)


class _AnyVisible:
    """Samples 1.0 when ANY of several bg-visible timelines is set (the
    AFT includes bg if any of its bg quads is shown)."""

    def __init__(self, timelines):
        self._timelines = tuple(timelines)

    def sample(self, t) -> tuple:
        return (max(tl.sample(t)[0] for tl in self._timelines),)


def _quad_source_timelines(named_keyframes) -> dict:
    """Compiled {prop: EventTimeline} per named actor, so a copy's driver
    can sample the data-holder quads it reads (gat_aftx GetX, ...)."""
    return {name: build_timelines(keyframes=props)
            for name, props in named_keyframes.items()}


def _merge_driven_keyframes(field_keyframes, name, quad_timelines,
                            to_seconds) -> None:
    if not aft_drivers.has_driver(name):
        return
    driven = aft_drivers.driven_keyframes(name, quad_timelines, to_seconds)
    for prop, frames in driven.items():
        field_keyframes.setdefault(prop, []).extend(frames)


def _field_source(name, named_meta):
    """The playfield capture a named actor draws, or None. AFT-copy
    sprites carry it in their recorder meta; Proxy actors ARE their own
    source (they re-render a player's notefield directly)."""
    if not name:
        return None
    aft_source = named_meta.get(name, {}).get('aft_source')
    if aft_source:
        return aft_source
    return name if name in _PROXY_NAMES else None


def _iter_actors(actor):
    yield actor
    for child in actor.children:
        yield from _iter_actors(child)


_FIELD_RESTS = {
    'x': 0.0, 'y': 0.0, 'rotation': 0.0,
    'scale_x': 1.0, 'scale_y': 1.0,
    'base_scale_x': 1.0, 'base_scale_y': 1.0, 'alpha': 1.0,
    'hidden': 0.0,
}
