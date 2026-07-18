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

import hashlib
import math
import re
from dataclasses import replace
from pathlib import Path

from analysis.games.etterna import sm_chart
from analysis.games.notitg import (
    aft_drivers, sprite_sheet, update_integrator, xml_actors)
from analysis.games.notitg.mod_stubs import StubEnvironment
from analysis.games.notitg.paths import find_notitg_dirs
from analysis.games.notitg.recording_actor import RecordingActor
from analysis.player.render.effects.timeline import EventTimeline, Keyframe
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
    _tag_source(root_parsed.root, entry)
    _splice_includes(root_parsed.root, lua_dir, lua_chunks, classic, bg_stem,
                     stack=(entry,))
    return root_parsed.root, lua_chunks, classic


def _tag_source(actor, xml_path) -> None:
    """Annotate a subtree with where its XML lives: `_base_dir` for
    asset resolution, `_src_xml` (dir/file, e.g. `chara/default.xml`)
    for fault and Lua-chunk naming."""
    actor._base_dir = xml_path.parent
    actor._src_xml = '/'.join(xml_path.parts[-2:])
    for child in actor.children:
        _tag_source(child, xml_path)


def _splice_includes(actor, lua_dir, lua_chunks, classic, bg_stem='',
                     stack=()) -> None:
    for child in list(actor.children):
        # Recurse into the child's OWN children first (captured now, so
        # an appended include subtree - already fully spliced - is not
        # re-processed under the wrong base dir).
        _splice_includes(child, lua_dir, lua_chunks, classic, bg_stem,
                         stack)
        include = child.attrs.get('File', '')
        if include.startswith('@'):
            # A dynamic include: the engine evaluates the `@expr` at
            # actor load. The sim resolves the value then and calls the
            # closure with it (a non-XML result is a texture, not an
            # include).
            child._expand_dynamic_include = _dynamic_expander(
                child, lua_dir, lua_chunks, classic, bg_stem, stack)
            continue
        included = _include_path(include, lua_dir)
        if included is None:
            continue
        if included in stack:
            # The actorgen LOOP idiom: a file includes ITSELF behind
            # Condition="actorgen.HasNext()", generating one actor per
            # iteration until the generator empties. Eager splicing
            # recurses forever; the sim expands the include at LOAD
            # TIME, after the engine's Condition gate has passed.
            child._expand_include = _deferred_expander(
                child, included, lua_chunks, classic, bg_stem, stack)
            continue
        _splice_file(child, included, lua_chunks, classic, bg_stem, stack)


def _deferred_expander(child, included, lua_chunks, classic, bg_stem, stack):
    def expand():
        _splice_file(child, included, lua_chunks, classic, bg_stem, stack)
    return expand


def _dynamic_expander(child, lua_dir, lua_chunks, classic, bg_stem, stack):
    def expand(resolved: str):
        included = _include_path(resolved, lua_dir)
        if included is not None:
            _splice_file(child, included, lua_chunks, classic, bg_stem,
                         stack)
    return expand


def _splice_file(child, included, lua_chunks, classic, bg_stem,
                 stack=()) -> None:
    sub = xml_actors.parse_actor_xml(
        included.read_text(encoding='utf-8', errors='replace'))
    _tag_source(sub.root, included)
    _splice_includes(sub.root, included.parent, sub.lua_chunks,
                     sub.classic_commands, bg_stem, stack + (included,))
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
    reference is the file directly; a bare name resolves like SM does:
    the sibling `<name>.xml` file (`<Layer File="easing" />`) or a
    directory's default.xml (`File="chara"`, the implicit
    directory-actor rule). Anything else (a texture path) is not an
    actor include."""
    if not include:
        return None
    if include.lower().endswith('.xml'):
        candidate = lua_dir / include
        return candidate if candidate.exists() else None
    sibling = lua_dir / f'{include}.xml'
    if sibling.exists():
        return sibling
    entry = lua_dir / include / 'default.xml'
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


def _run_chunks(root, start_beat, to_seconds, rng_seed=0):
    """Load the actor tree under a shared stubbed host: one persistent
    recorder per actor, InitCommand/OnCommand run with `self` bound to
    it, and every `<Name>MessageCommand` / `<Name>Command` registered so
    later broadcasts and play/queuecommands run on the SAME recorder.
    A trailing `NAME = self` still binds a global (the poke target for the
    mod_actions closures). Per-chunk failures warn; a partial harvest
    survives. `rng_seed` seeds the sandbox `math.random` so the chart's
    random spawner scatter (the FUCK datamosh pool) records reproducibly."""
    env = StubEnvironment(start_beat, to_seconds=to_seconds, rng_seed=rng_seed)
    warnings = env.load_actors(root)
    return env, warnings


def _chart_rng_seed(lua_dir) -> int:
    """A stable per-chart RNG seed from the chart's modfile directory, so
    `math.random`-driven spawns record the same scatter every compile
    while different charts get different scatters. Content-agnostic (the
    directory IS the chart's modfile identity) and 32-bit for LuaJIT's
    `math.randomseed`."""
    digest = hashlib.sha1(str(lua_dir).encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big')


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
    # The template reader applies a row only while `start <= now <= end`;
    # an inverted window (an 'end' row whose author meant 'len', gat's
    # `{847.5, 1, ..., 'end'}`) can never satisfy both and never applies.
    if end_field < v1:
        return None
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
                         fonts=None, actor_keyframes=None, osc_context=None):
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
                                 below, actor_keyframes, osc_context)
        if element is not None:
            children.append(element)
    return below + children


def _compile_actor(actor, start_time, named_keyframes, fonts, below=None,
                   actor_keyframes=None, osc_context=None):
    if below is None:
        below = []
    if getattr(actor, '_aft_fill', False):
        # An AFT-rig curtain quad: re-emitted as a 'fill' field instance
        # at its tree position (see producers._mark_aft_fills), never a
        # storyboard element.
        return None
    if getattr(actor, '_background_layer', False):
        element = _compile_background_actor(actor, start_time, named_keyframes,
                                           fonts, actor_keyframes, osc_context)
        if element is not None:
            below.append(element)
        return None

    child_elements = []
    for child in actor.children:
        element = _compile_actor(child, start_time, named_keyframes, fonts,
                                 below, actor_keyframes, osc_context)
        if element is not None:
            child_elements.append(element)

    keyframes = _merged_keyframes(actor, start_time, named_keyframes,
                                  actor_keyframes, osc_context)
    if child_elements:
        return _group_element(actor, start_time, keyframes,
                              tuple(child_elements))
    leaf = _leaf_element(actor, start_time, named_keyframes,
                         precomputed=keyframes, fonts=fonts)
    if leaf is not None and _is_aft_backdrop(actor):
        # The AFT rig's fullscreen backdrops (the ShowAFT black quad,
        # the ShowAFTBG bg image) sit UNDER the proxies in engine tree
        # order; as ordinary elements they would draw over the field
        # and copies (they only started rendering once fills gained
        # w/h), so they route to the below-field band instead.
        below.append(_with_z(leaf, _AFT_BACKDROP_Z))
        return None
    return leaf


# The background band z: below the notes/field (z=0) but a valid
# storyboard slot. gat's whole BGCHANGES tree lands here.
_BACKGROUND_Z = -100

# AFT backdrop quads sit above the background band but still under the
# field/copies (engine tree order: the rig's backdrops precede the
# proxies).
_AFT_BACKDROP_Z = -50

# Message commands identifying the AFT rig's fullscreen backdrops.
_AFT_BACKDROP_MESSAGES = ('ShowAFT', 'ShowAFTBG', 'HideAFT')


def _is_aft_backdrop(actor) -> bool:
    commands = actor.message_commands()
    return any(name in commands for name in _AFT_BACKDROP_MESSAGES)


def _compile_background_actor(actor, start_time, named_keyframes, fonts,
                              actor_keyframes=None, osc_context=None):
    """Compile a background-layer subtree as one top-level element at the
    background z. The subtree is self-contained (its own gat_all_bg /
    gat_bg transforms), so hoisting it past the identity top frames keeps
    its placement while moving it behind the notes."""
    child_elements = []
    for child in actor.children:
        element = _compile_background_actor(child, start_time,
                                           named_keyframes, fonts,
                                           actor_keyframes, osc_context)
        if element is not None:
            child_elements.append(element)

    keyframes = _merged_keyframes(actor, start_time, named_keyframes,
                                  actor_keyframes, osc_context)
    if child_elements:
        element = _group_element(actor, start_time, keyframes,
                                tuple(child_elements))
    else:
        element = _leaf_element(actor, start_time, named_keyframes,
                               precomputed=keyframes, fonts=fonts)
    return _with_z(element, _BACKGROUND_Z) if element is not None else None


def _with_z(element, z):
    return replace(element, z=z)


def _merged_keyframes(actor, start_time, named_keyframes, actor_keyframes=None,
                      osc_context=None):
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
    recorded from the closures.

    An actor that ran an effect oscillator (`osc_context` carries its
    span) has the sine synthesised into dense keyframes on the affected
    property, riding its tweened base motion."""
    keyframes = _recorded_keyframes(actor, actor_keyframes)
    if keyframes is None:
        keyframes = _flat_keyframes(actor, start_time, named_keyframes)
    return _apply_oscillators(actor, keyframes, osc_context)


def _flat_keyframes(actor, start_time, named_keyframes):
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


def _apply_oscillators(actor, keyframes, osc_context):
    """Fold this actor's oscillator spans into its keyframes, when it ran
    any. Keyed by the actor's load-pass recorder id; a chart with no
    oscillators (osc_context None) or an actor that ran none returns the
    keyframes unchanged."""
    if osc_context is None:
        return keyframes
    rec_id = getattr(actor, '_recorder_id', None)
    spans = osc_context.spans_by_id.get(rec_id) if rec_id is not None else None
    if not spans:
        return keyframes
    # Recorded ends only, NEVER the open-span extension: the bake
    # replaces the actor's base keyframes across the span window, so an
    # extended-to-compile-end span would wipe the actor's remaining
    # animation and freeze it at the sample cap. The live field channels
    # (OscDeltaChannel) are the extension-capable form; the tree bake
    # keeps engine behavior only over the span's recorded activity.
    return compile_oscillator_keyframes(spans, keyframes, osc_context.clock,
                                        osc_context.rng)


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


# Effect-oscillator synthesis ------------------------------------------
#
# SM's Actor::UpdateInternal drives an actor's own pos/rotation by a sine
# of the effect clock every frame (Actor.cpp). We synthesise that motion
# into dense keyframes on the affected 2D property so the static-keyframe
# renderer animates it. Deterministic: the clock is song beat/time and
# vibrate's RNG is seeded, so a chart always compiles the same shake.

# Dense-sample spacing in SECONDS. Oscillator periods are ~1 beat
# (~0.29s at gat's 205bpm); ~120Hz keeps >30 samples per period so the
# linear-interpolated keyframes trace the sine without aliasing (the same
# over-sampling logic as the hold-body sampler, memory item 78).
_OSC_SAMPLE_STEP_S = 1.0 / 120.0

# Cap per span so a very long open effect (a vibrate left running for the
# whole song) cannot explode the keyframe count; at ~120Hz this bounds a
# single span to ~50s of dense samples, past which it holds its last value
# (a longer shake reads the same).
_OSC_MAX_SAMPLES = 6000

# Each oscillator kind's per-property contribution as f(pct, mag, beat_or
# _time_elapsed). Returns {property: delta} added onto the actor's base
# value at that sample. `pct` is the SM fraction through the period
# (Actor.cpp:276); `mag` is the (x, y, z) effect magnitude in force;
# `elapsed` is beats (beat clock) or seconds (time clock) since the span
# start, used by spin's continuous accumulation.
_TWO_PI = 2.0 * math.pi


def _osc_deltas(kind, pct, mag, elapsed, rng):
    """The 2D-property deltas one oscillator kind contributes at a sample.

    - bob: pos += mag * sin(pct*2pi)          (Actor.cpp:353)
    - bounce: pos += mag * sin(pct*pi)        (Actor.cpp:344, abs-sine)
    - wag: rotation += mag.z * sin(pct*2pi)   (Actor.cpp:332; z drives 2D)
    - spin: rotation += mag.z * elapsed        (Actor.cpp:599, continuous)
    - vibrate: pos += mag * randomf(-1,1)      (Actor.cpp:338, seeded RNG)

    Only the x/y position and z rotation have 2D-storyboard analogues; the
    z-position and x/y-rotation components of bob/bounce/vibrate are 3D and
    dropped (same as the recorder's 3D-channel handling)."""
    match kind:
        case 'bob':
            s = math.sin(pct * _TWO_PI)
            return {'x': mag[0] * s, 'y': mag[1] * s}
        case 'bounce':
            s = math.sin(pct * math.pi)
            return {'x': mag[0] * s, 'y': mag[1] * s}
        case 'wag':
            return {'rotation': mag[2] * math.sin(pct * _TWO_PI)}
        case 'spin':
            return {'rotation': mag[2] * elapsed}
        case 'vibrate':
            return {'x': mag[0] * rng.uniform(-1.0, 1.0),
                    'y': mag[1] * rng.uniform(-1.0, 1.0)}
    return {}


class _OscillatorClock:
    """Maps a sample time (seconds) to the effect clock's phase source.

    A beat clock (`bgm`/`beat`) reads the song BEAT at the sample time
    (SM's `m_fSecsIntoEffect = g_fCurrentBGMBeat`); a time clock
    (`timer`/`music`) reads the song SECOND directly. Built once per
    compile over the span range so the beat inverter is shared."""

    def __init__(self, to_seconds, beat_range):
        self._to_seconds = to_seconds
        self._to_beats = _sec_to_beat_inverter(to_seconds, beat_range)

    def phase_source(self, clock_name, seconds) -> float:
        if clock_name in _EFFECT_TIME_CLOCK_NAMES:
            return seconds
        return self._to_beats(seconds)


# Clock-name sets mirrored from recording_actor so the synthesis reads the
# same vocabulary the recorder tagged the span with.
_EFFECT_TIME_CLOCK_NAMES = frozenset({'timer', 'music'})


def _sec_to_beat_inverter(to_seconds, beat_range):
    """A `seconds -> beat` inverter over `beat_range = (lo, hi)`, by bisect
    on a dense (beat, seconds) table (to_seconds is monotonic in beat).
    Mirror of update_integrator's inverter, local so the synthesis is
    self-contained."""
    from bisect import bisect_right

    lo, hi = beat_range
    if hi <= lo:
        hi = lo + 1.0
    steps = max(2, int((hi - lo) * 8.0) + 1)
    beats = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    times = [to_seconds(b) for b in beats]

    def to_beats(t):
        idx = bisect_right(times, t) - 1
        if idx < 0:
            return beats[0]
        if idx >= steps - 1:
            return beats[-1]
        span = times[idx + 1] - times[idx]
        frac = (t - times[idx]) / span if span > 0 else 0.0
        return beats[idx] + (beats[idx + 1] - beats[idx]) * frac
    return to_beats


def _effective_end(span, end_seconds) -> float:
    """The time an oscillator span actually stops. A span the chart
    closed itself (stopeffect / a replacing kind verb) ends exactly
    there; a span still running when recording ended keeps going in the
    engine, so it extends to the compile end when one is known."""
    if span.explicit_end or end_seconds is None:
        return span.end
    return max(span.end, float(end_seconds))


def _span_keyframes(span, osc_clock, rng, end=None):
    """Dense step keyframes per affected 2D property for one oscillator
    span over [span.start, end] (default: the span's own recorded end),
    or {} when it produces no 2D motion.

    Samples the analytic sine at `_OSC_SAMPLE_STEP_S`, computing SM's
    `pct = frac(phase + offset, period)` from the effect clock's phase
    source at each sample. Each sample is a STEP keyframe (duration 0)
    that holds until the next - piecewise-constant, so the value AT a
    sample time is exactly that sample (the EventTimeline eases
    prev->target over a keyframe's duration, so a nonzero duration would
    put a sample's own value one step in its future). At 120Hz over a
    ~1-beat period this traces the sine finely; a trailing rest returns the
    delta to zero when the effect stops."""
    start = span.start
    if end is None:
        end = span.end
    if end <= start:
        return {}
    n = min(_OSC_MAX_SAMPLES, int((end - start) / _OSC_SAMPLE_STEP_S) + 1)
    phase0 = osc_clock.phase_source(span.clock, start)
    per_prop: dict = {}
    for i in range(n + 1):
        t = min(end, start + i * _OSC_SAMPLE_STEP_S)
        phase = osc_clock.phase_source(span.clock, t)
        pct = _effect_pct(phase, span.period, span.offset)
        mag = span.magnitude_at(t)
        deltas = _osc_deltas(span.kind, pct, mag, phase - phase0, rng)
        for prop, delta in deltas.items():
            per_prop.setdefault(prop, []).append(
                Keyframe(t, (delta,), 0.0, _STEP_EASE))
    _append_rest(per_prop, end)
    return per_prop


# Step keyframes hold their value until the next (duration 0); the easing
# id is unused then, but 0 (linear) is the neutral default.
_STEP_EASE = 0


def _effect_pct(phase, period, offset) -> float:
    """SM's fraction through the effect period at a phase value
    (Actor.cpp:271-278): `secsIntoPeriod = fmod(phase + offset, period)`;
    `pct = secsIntoPeriod / period`, clamped to [0, 1]. Delay is 0 for
    every gat effect, so total period == period."""
    if period <= 0:
        return 0.0
    into = math.fmod(phase + offset, period)
    if into < 0:
        into += period
    return max(0.0, min(1.0, into / period))


def _append_rest(per_prop, end) -> None:
    """A trailing rest (delta 0) a hair past the span end, so the
    oscillator delta returns to zero when the effect stops rather than
    holding its last sample forever (the renderer holds the last keyframe
    otherwise)."""
    for frames in per_prop.values():
        frames.append(Keyframe(end + _OSC_SAMPLE_STEP_S, (0.0,), 0.0,
                               _STEP_EASE))


def compile_oscillator_keyframes(spans, base_keyframes, osc_clock, rng,
                                 end_seconds=None):
    """Fold an actor's oscillator spans into its base keyframe dict.

    The oscillator drives a DELTA on top of the actor's tweened base
    position/rotation. We cannot add curves inside the EventTimeline model,
    so we bake the base value at each dense sample and store base+delta as
    the keyframe value on that property - the synthesised stream then
    supersedes the sparse base keyframes for the property's oscillating
    span, and the trailing rest keyframe hands motion back to the base
    after (a stopeffect). Properties the oscillator never touches are left
    untouched.

    Mutates and returns `base_keyframes` (a dict of {prop: [Keyframe]})."""
    for span in spans:
        end = _effective_end(span, end_seconds)
        deltas = _span_keyframes(span, osc_clock, rng, end)
        for prop, delta_frames in deltas.items():
            # x, y and rotation all rest at 0, so the base sits at 0 before
            # its first keyframe - the delta rides whatever base motion is
            # tweened in.
            base_tl = EventTimeline(base_keyframes.get(prop, []), rest=(0.0,))
            merged = [_add_base(kf, base_tl) for kf in delta_frames]
            base_keyframes[prop] = _merge_base_outside(
                base_keyframes.get(prop, []), merged, span.start, end)
    return base_keyframes


def _add_base(delta_kf, base_tl):
    """A delta keyframe rebased onto the actor's tweened value at its time:
    keyframe value = base(t) + delta, so the shake rides the base motion."""
    base = base_tl.sample(delta_kf.t)[0]
    return replace(delta_kf, values=(base + delta_kf.values[0],))


def _merge_base_outside(base_frames, synth_frames, start, end):
    """Keep the actor's base keyframes OUTSIDE [start, end] (before start /
    after the trailing rest) and replace the inside with the synthesised
    stream, so the sparse base motion still plays either side of the
    oscillator. Sorted by time for the EventTimeline."""
    margin = _OSC_SAMPLE_STEP_S
    kept = [kf for kf in base_frames
            if kf.t < start or kf.t > end + margin]
    return sorted(kept + synth_frames, key=lambda k: k.t)


class _OscContext:
    """Everything the tree compiler needs to synthesise oscillators: the
    per-recorder-id spans, the shared effect clock, the seeded RNG for
    baked vibrate, the chart-stable integer seed for live vibrate
    channels, and the compile end open spans extend to. Absent (None)
    when no actor ran an effect oscillator, so the common no-oscillator
    chart pays nothing."""

    __slots__ = ('spans_by_id', 'clock', 'rng', 'seed', 'end_seconds')

    def __init__(self, spans_by_id, clock, rng, seed, end_seconds):
        self.spans_by_id = spans_by_id
        self.clock = clock
        self.rng = rng
        self.seed = seed
        self.end_seconds = end_seconds


def _build_osc_context(env, to_seconds, start_beat, lua_dir,
                       end_seconds=None):
    """Build the oscillator compile context, or None when the chart has no
    effect oscillators. The RNG and the integer seed are per-chart (the
    same determinism contract as the spawner scatter, mod_stubs), so a
    chart's vibrate shake compiles identically every run. `end_seconds`
    (the compile end) is what still-open spans run to; None keeps every
    span at its recorded end."""
    import random
    import zlib

    spans_by_id = env.actor_oscillator_spans()
    if not spans_by_id:
        return None
    clock = _OscillatorClock(
        to_seconds, _osc_beat_range(spans_by_id, to_seconds, start_beat,
                                    end_seconds))
    rng = random.Random(f'notitg-osc:{lua_dir}')
    seed = zlib.crc32(f'notitg-osc:{lua_dir}'.encode())
    return _OscContext(spans_by_id, clock, rng, seed, end_seconds)


def _osc_beat_range(spans_by_id, to_seconds, start_beat, end_seconds=None):
    """(lo_beat, hi_beat) covering every oscillator span's time range
    (including the compile end still-open spans extend to), for the beat
    inverter. Spans carry SECOND clocks; since `to_seconds` is monotone
    in beat, we double `hi_beat` from a small start until its mapped time
    passes the last span end - a handful of steps, bounded so a
    pathological span cannot loop forever."""
    last_end = max((_effective_end(span, end_seconds)
                    for spans in spans_by_id.values() for span in spans),
                   default=to_seconds(start_beat))
    hi = start_beat + 8.0
    for _ in range(32):
        if to_seconds(hi) >= last_end:
            break
        hi = start_beat + (hi - start_beat) * 2.0
    return (start_beat, hi)


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
    drawable = _fill_size_as_wh(kind, _drawable_props(keyframes))
    state_pin = _state_pin(keyframes, states, start_time)
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
        additive=_is_additive(actor),
        sheet_cols=spec.cols, sheet_rows=spec.rows, sheet_states=states,
        size_spec=spec, state_pin=state_pin,
    )


def _is_additive(actor) -> bool:
    """Whether the actor sets additive blending (`blend,add`) in its
    load-time commands. gat's TargOn split judgment lines blend add so
    the red bars glow over the field; the renderer honours Element.additive
    with a plus composition mode."""
    return any(_is_blend_add(verb, args)
               for verb, args in _load_classic_verbs(actor))


def _is_blend_add(verb, args) -> bool:
    return verb == 'blend' and bool(args) and str(args[0]).strip() == 'add'


def _load_classic_verbs(actor):
    """(verb, args) pairs from an actor's classic (non-Lua) load-time
    command strings."""
    for attr in _LOAD_TIME_ATTRS:
        value = actor.attrs.get(attr, '')
        if value and not value.startswith('%'):
            yield from xml_actors.parse_command_string(value)


# Keyframe channels the recorder emits from setstate/animate pokes.
_STATE_PROP = 'frame'
_STATE_PAUSED_PROP = 'frame_paused'


def _state_pin(keyframes, states, start_time):
    """A `StateAnchors` sampler of the sprite's frame index over time,
    from recorded `setstate`/`animate` pokes, or None when the actor
    never poked its state (the sheet then auto-animates). A `setstate`
    is a RESTART anchor - the state list keeps playing from that state
    (SM `SetState`) - while an `animate(off)` pause holds the anchored
    frame until resumed."""
    state_kfs = keyframes.get(_STATE_PROP) or ()
    paused_kfs = keyframes.get(_STATE_PAUSED_PROP) or ()
    if not state_kfs and not paused_kfs:
        return None

    events = sorted([(kf.t, _STATE_PROP, kf.values[0]) for kf in state_kfs]
                    + [(kf.t, _STATE_PAUSED_PROP, kf.values[0])
                       for kf in paused_kfs])
    anchors = []
    state, paused = 0.0, False
    for t, prop, value in events:
        if prop == _STATE_PROP:
            state = value
        else:
            paused = value != 0.0
        anchors.append((t, state, not paused))
    return sprite_sheet.StateAnchors(anchors, states, t_start=start_time)


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
    return _prefix_globbed(candidate) or str(candidate)


def _prefix_globbed(candidate: Path) -> str | None:
    """SM's directory-listing match (`ActorUtil` globs `<reference>*`):
    a bare name resolves to a file whose name merely STARTS with it, so
    `File="laugh"` finds `laugh 2x1.png` (the frame-dimensions token
    lives only in the on-disk name). First image match in name order; a
    `.sprite` manifest match resolves through its Texture= line."""
    if not candidate.parent.is_dir():
        return None
    for path in sorted(candidate.parent.glob(candidate.name + '*')):
        if path.suffix.lower() == '.sprite':
            return _sprite_manifest_texture(path, path.parent)
        if path.suffix.lower() in _IMAGE_SUFFIXES and path.is_file():
            return str(path)
    return None


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
# `hidden,1` gate and a diffusealpha crossfade coexist. The 3D scene
# channels (z, rotation_x/y, scale_z, skew, fov) flow through too: the
# storyboard renderer projects an element through its frame chain's
# perspective camera, so an actor tilted/pushed in z or inside a
# fov frame renders in true 3D. They rest at identity, so a flat actor
# is unchanged.
_DRAWABLE_PROPS = frozenset({
    'x', 'y', 'scale_x', 'scale_y', 'rotation', 'alpha', 'color', 'hidden',
    # Absolute on-screen size (SM zoomto/setsize). Rest is the unset
    # sentinel in model._SCALAR_RESTS; when set it overrides natural*scale.
    'size_x', 'size_y',
    # SM crop family (recorded onto crop_* by the crop setters); the
    # storyboard renderer insets the drawn/source rect by these fractions.
    'crop_top', 'crop_bottom', 'crop_left', 'crop_right',
    # 3D scene channels (rest at identity -> flat actors unchanged).
    'rotation_x', 'rotation_y', 'z', 'scale_z', 'skew_x', 'skew_y', 'fov',
})


def _drawable_props(keyframes):
    return {prop: frames for prop, frames in keyframes.items()
            if prop in _DRAWABLE_PROPS and frames}


# Element kinds with no natural (pixmap/text) size: the renderer sizes
# them from the model's `w`/`h` timelines (fluXis rect width/height).
# NotITG Quads (`<Quad>` -> 'rect') carry their size in `zoomto`/`setsize`
# -> `size_x`/`size_y` instead, so a fill quad recorded that way renders
# at zero size unless we also expose it as `w`/`h`.
_FILL_SIZE_KINDS = frozenset({'rect', 'ellipse', 'outline_rect',
                              'outline_ellipse'})


def _fill_size_as_wh(kind, drawable):
    """Expose a fill primitive's `zoomto`/`setsize` size (recorded on
    `size_x`/`size_y`) as the `w`/`h` the renderer sizes fill kinds from.

    A Quad's absolute size is its whole size (no natural basis), so the
    two representations agree; copying it to `w`/`h` lets the renderer's
    fill-kind sizing path see it (the gat fullscreen flash quad and the
    split TargOn judgment lines both size this way). Sprites keep their
    pixmap size and are untouched."""
    if kind not in _FILL_SIZE_KINDS:
        return drawable
    for size_prop, wh_prop in (('size_x', 'w'), ('size_y', 'h')):
        frames = drawable.get(size_prop)
        if frames:
            drawable[wh_prop] = frames
    return drawable


def _is_image_asset(asset) -> bool:
    """True when a resolved asset is a real image reference (an existing
    file or the synthesized `white`), so an untyped `Actor`/`Layer` that
    loads one - gat's chara sprites, `<Actor File="shame/idle.sprite">` -
    counts as a Sprite even without a `Type=`."""
    if not asset:
        return False
    if asset in _BUILTIN_TEXTURES:
        return True
    # A directory reference (`File="../bg"`, a BGAnimation dir) is
    # include-spliced by _splice_includes, not drawn as a texture.
    return Path(asset).is_file()


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

    env, chunk_warnings = _run_chunks(root, start_beat, to_seconds,
                                      rng_seed=_chart_rng_seed(lua_dir))
    fired, failed = env.replay_mod_actions()
    integration = update_integrator.integrate_update(env, root, to_seconds)
    named_keyframes = env.named_actor_keyframes()
    named_meta = env.named_actor_meta()
    actor_keyframes = env.actor_keyframes()
    osc_context = _build_osc_context(env, to_seconds, start_beat, lua_dir)

    mod_events = _normalize_mod_events(env, to_seconds)
    mod_events.extend(_normalize_applied_mods(env, to_seconds))
    mod_events.extend(integration.get('applied_events') or [])
    proxy_grid = env.proxy_grid()
    fonts = _font_resolver(lua_dir)
    tree = compile_element_tree(root, to_seconds, start_beat, named_keyframes,
                                fonts=fonts, actor_keyframes=actor_keyframes,
                                osc_context=osc_context)
    return {
        'mod_events': mod_events,
        'shader_flags': _normalize_shader_flags(env, to_seconds),
        'unsupported': _describe_unsupported(env),
        'elements': _compile_elements(classic_commands, to_seconds,
                                      start_beat, named_keyframes),
        'tree': tree,
        'has_background': _has_background_actors(tree, sm_path),
        'field_copies': _all_field_copies(
            root, named_keyframes, named_meta, to_seconds, start_beat,
            actor_keyframes, proxy_grid, osc_context),
        'screen_transform': _screen_transform_timelines(env),
        'screen_oscillator': _screen_oscillator_timelines(env, osc_context),
        'field_oscillators': _field_oscillator_timelines(env, osc_context),
        'field_vanish': _field_vanish_timelines(env),
        'aft_bg_visible': _aft_bg_visible_timeline(root, bg_stem,
                                                   actor_keyframes),
        'base_field_hidden': _base_field_hidden_timeline(env),
        'named_actors': len(named_keyframes),
        'recorded_keyframes': _count_recorded_keyframes(named_keyframes),
        'replay': {'fired': fired, 'failed': failed,
                   'applied_mods': len(env.applied_mods),
                   'swallowed': env.swallowed},
        'integration': {'ran': integration.get('ran', False),
                        'ticks': integration.get('ticks', 0),
                        'windows': integration.get('windows', 0),
                        'applied': integration.get('applied', 0),
                        'faults': integration.get('faults', 0)},
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
                  start_beat, actor_keyframes=None, osc_context=None) -> list:
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

    A copy actor that ran an effect oscillator (gat's `Proxy(pn):vibrate()`
    /`wag()`) has the sine baked into its x/y/rotation here, so the copy
    shakes; `osc_context` carries the spans.

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
                                      actor_keyframes, osc_context)
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


# Proxy-grid copies source to their player's notefield: player 1 -> P1p,
# player 2 -> P2p (the same NoteField the split-screen proxies re-render),
# so field_instances gives them the field-only ('field') capture scope.
_GRID_SOURCE = {1: 'P1p', 2: 'P2p'}


def _all_field_copies(root, named_keyframes, named_meta, to_seconds,
                      start_beat, actor_keyframes, proxy_grid,
                      osc_context=None) -> list:
    """AFT/Proxy field copies plus the gat_updateproxies 3x3 grid copies.
    The grid frames self-assign no global (they live in the `gat_proxies`
    table), so they come through `env.proxy_grid()` rather than the
    named-actor path, and their world transform is composed here."""
    copies = _field_copies(root, named_keyframes, named_meta, to_seconds,
                           start_beat, actor_keyframes, osc_context)
    copies.extend(_proxy_grid_copies(proxy_grid))
    return copies


def _proxy_grid_copies(proxy_grid) -> list:
    """The gat_updateproxies proxy grid as field copies.

    Each `gat_proxies[pn][i]` frame is a notefield proxy whose SCREEN
    position is the SM ActorFrame composition
        world = gat_allproxies + gat_allproxiesc + frame_local
    (translations sum in the frame hierarchy). `gat_allproxies` is the
    per-frame accumulator the update integrator drives (the scatter/scroll
    with wrap-around); `gat_allproxiesc` is a static centering offset;
    the frame carries its StartShit2 grid slot and its per-frame rotation /
    visibility cull. Scale comes from the `gat_proxiesc` content proxy's
    zoom. Emitting each as a field copy replicates the notefield across the
    3x3 grid exactly where the live per-frame code places it."""
    if not proxy_grid:
        return []
    parent = _timelines_for(proxy_grid.get('parent'), _FIELD_RESTS)
    offset = _timelines_for(proxy_grid.get('parent_offset'), _FIELD_RESTS)
    spans = proxy_grid.get('spans') or ()
    copies = []
    for index, frame in enumerate(proxy_grid.get('frames') or []):
        copy = _proxy_grid_copy(frame, parent, offset, index)
        if copy is not None:
            if spans:
                copy['timelines']['hidden'] = _SpanGatedTimeline(
                    copy['timelines']['hidden'], spans)
            copies.append(copy)
    return copies


def _proxy_grid_copy(frame, parent, offset, index):
    player = frame.get('player', 1)
    frame_tl = _timelines_for(frame.get('frame'), _FIELD_RESTS)
    content_tl = _timelines_for(frame.get('content'), _FIELD_RESTS)
    timelines = dict(frame_tl)
    timelines['x'] = _SumTimeline((parent['x'], offset['x'], frame_tl['x']))
    timelines['y'] = _SumTimeline((parent['y'], offset['y'], frame_tl['y']))
    # The content proxy's zoom is the copy's scale (0.8 in StartShit2); the
    # frame itself does not zoom. Fall back to the frame's own scale when a
    # content proxy never recorded one.
    for scale in ('scale_x', 'scale_y'):
        if frame.get('content', {}).get(scale):
            timelines[scale] = content_tl[scale]
    return {
        'name': f'gat_proxy_{player}_{index}', 'source': _GRID_SOURCE[player],
        'timelines': timelines,
    }


def _timelines_for(keyframes, rests):
    return build_timelines(rests=rests, keyframes=keyframes or {})


# Grace beyond a driven span before a copy stops rendering: one-ish
# integration tick, so the last tick's placement still draws.
_DRIVEN_SPAN_MARGIN = 0.1


class _SpanGatedTimeline:
    """Visibility gate for per-frame-driven copies: inside a span in which
    the integrator actually poked the driver, defer to the recorded hidden
    channel; outside every span the copy is simply not rendered (samples
    hidden=1). A per-frame-driven visual has no compiled definition beyond
    its driver's lifetime, so the last recorded cull state must not hold
    forever."""

    def __init__(self, child, spans):
        self._child = child
        self._spans = tuple(spans)

    def sample(self, t) -> tuple:
        for start, end in self._spans:
            if start - _DRIVEN_SPAN_MARGIN <= t <= end + _DRIVEN_SPAN_MARGIN:
                return self._child.sample(t)
        return (1.0,)


class _SumTimeline:
    """A timeline sampling the SUM of several child timelines - the field
    producer samples `timeline.sample(t)`, and an ActorFrame's world
    translation is the sum of its own and its ancestors' offsets, so a
    grid copy's world x/y composes as this sum without materializing a
    merged keyframe stream."""

    def __init__(self, timelines):
        self._timelines = tuple(timelines)

    def sample(self, t) -> tuple:
        return (sum(tl.sample(t)[0] for tl in self._timelines),)


def _screen_transform_timelines(env) -> dict | None:
    """The whole-scene camera the per-frame update drives via the top
    screen: gat_updateproxies zooms and offsets `SCREENMAN:GetTopScreen()`
    for a screen-zoom camera. Returns {prop: EventTimeline} (x/y/scale) or
    None when nothing poked the screen (every non-gat chart). The screen's
    `effectmagnitude` vibrate is a SEPARATE channel (`screen_oscillator`),
    since it is a synthesised sine, not a tweened transform."""
    keyframes = env.screen_keyframes()
    moved = {prop: keyframes[prop] for prop in ('x', 'y', 'scale_x',
             'scale_y')
             if _deviates(keyframes.get(prop), _SCREEN_RESTS[prop])}
    if not moved:
        return None
    return build_timelines(rests=_SCREEN_RESTS, keyframes=moved)


def _screen_oscillator_timelines(env, osc_context) -> dict | None:
    """The whole-scene vibrate the screen's effect oscillator drives, as
    `{prop: EventTimeline}` of the x/y jitter DELTA, or None when the
    screen ran no oscillator. gat's datamosh section (t~312-382) pokes
    `screen:vibrate()` with a per-frame `effectmagnitude(gat_vib:GetX()..)`
    envelope - a scene shake the screen-camera consumer adds onto its
    transform. Delta only (like the field oscillators), so it composes onto
    the screen transform without baking a base in."""
    if osc_context is None:
        return None
    return oscillator_delta_channels(env.screen_oscillator_spans(),
                                     osc_context, seed=osc_context.seed)


def _deviates(frames, rest) -> bool:
    """True when any keyframe value differs from the channel rest - a
    channel poked only with its rest value (gat_updateproxies writing
    `screen:zoom(1); x(0); y(0)` every frame) carries no motion and is
    dropped, so the screen camera stays inactive."""
    return bool(frames) and any(abs(kf.values[0] - rest) > 1e-6
                                for kf in frames)


_SCREEN_RESTS = {'x': 0.0, 'y': 0.0, 'scale_x': 1.0, 'scale_y': 1.0}


# The engine player actors the chart hides while proxies stand in
# (`P1:hidden(1)`). PlayerP1 is player 0's real NoteField; hiding it means
# the copies replace the base field, so the renderer skips the base draw.
_BASE_PLAYER_NAME = 'PlayerP1'


def _field_oscillator_timelines(env, osc_context):
    """Per-player field oscillator deltas as `{player: {prop:
    OscDeltaChannel}}` (live channels, sampled at frame time) for the
    field consumers, or None when neither player field oscillates.

    gat's t~8-48 section pokes `Plr(pn)` (= the engine PlayerP1/PlayerP2
    NoteFields, fetched via GetChild) with bounce/bob/wag - a whole-field
    shake/rotate the field-instances/field-3d layer applies, not a
    storyboard element. Each player's spans become x/y/rotation delta
    channels (delta only - the field's base transform is the note
    pipeline's, so these ADD onto it, unlike the tree path that bakes
    base+delta). None when no oscillator ran (the common case)."""
    if osc_context is None:
        return None
    out = {}
    for player, name in enumerate(_PLAYER_FIELD_NAMES, start=1):
        spans = env.player_oscillator_spans(name)
        deltas = oscillator_delta_channels(spans, osc_context,
                                           seed=osc_context.seed + player)
        if deltas:
            out[player] = deltas
    return out or None


# The engine player actor names, in player order (Plr(1) -> PlayerP1).
_PLAYER_FIELD_NAMES = ('PlayerP1', 'PlayerP2')

# The 2D properties each oscillator kind drives (mirrors _osc_deltas).
_SPAN_PROPS = {
    'bob': ('x', 'y'), 'bounce': ('x', 'y'), 'vibrate': ('x', 'y'),
    'wag': ('rotation',), 'spin': ('rotation',),
}

# The engine re-rolls vibrate's random offset once per rendered frame;
# 60Hz cells reproduce that cadence deterministically at any render rate
# (a faster display holds each cell, a slower one skips cells).
_VIBRATE_CELL_HZ = 60.0
_U64 = (1 << 64) - 1


def _rand_unit(seed, axis, cell) -> float:
    """Deterministic uniform in [-1, 1) from a splitmix64-style hash of
    (seed, axis, cell). The engine draws `randomf(-1,1)` per rendered
    frame; a stateless hash reproduces that at any sample time with no
    stored sequence (pure arithmetic, a rust port boundary)."""
    x = (seed * 0x9E3779B97F4A7C15 + axis * 0xBF58476D1CE4E5B9
         + cell * 0x94D049BB133111EB) & _U64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64
    x ^= x >> 31
    return x / float(1 << 63) - 1.0


class OscDeltaChannel:
    """LIVE oscillator delta for one 2D property - the scheduler-model
    channel form of an actor's effect spans, evaluated at frame time and
    never baked (no sample cap, no freeze). `.sample(t) -> (delta,)` is
    the EventTimeline surface, so field_compose.overlay_deltas sums it
    onto an instance link unchanged.

    Sine kinds compute SM's analytic waveform from the effect clock.
    Vibrate re-randomizes per frame cell with a seeded hash - the
    per-frame teleport that reads as duplicated receptors (the mirage) -
    scaled by the actor's zoom exactly as the engine multiplies GetZoom
    into the offset."""

    def __init__(self, spans, prop, clock, seed, end_seconds=None,
                 zoom=None):
        self._spans = tuple(spans)
        self._prop = prop
        self._clock = clock
        self._seed = int(seed)
        self._end = end_seconds
        self._zoom = zoom

    def sample(self, t):
        t = float(t)
        total = 0.0
        for span in self._spans:
            if span.start <= t < _effective_end(span, self._end) \
                    and self._prop in _SPAN_PROPS.get(span.kind, ()):
                total += self._span_delta(span, t)
        return (total,)

    def _span_delta(self, span, t) -> float:
        if span.kind == 'vibrate':
            axis = 0 if self._prop == 'x' else 1
            magnitude = span.magnitude_at(t)[axis]
            if magnitude == 0.0:
                return 0.0
            cell = int((t - span.start) * _VIBRATE_CELL_HZ)
            zoom = self._zoom.sample(t)[0] if self._zoom is not None else 1.0
            return magnitude * _rand_unit(self._seed, axis, cell) * zoom
        phase = self._clock.phase_source(span.clock, t)
        pct = _effect_pct(phase, span.period, span.offset)
        elapsed = phase - self._clock.phase_source(span.clock, span.start)
        deltas = _osc_deltas(span.kind, pct, span.magnitude_at(t), elapsed,
                             rng=None)
        return deltas.get(self._prop, 0.0)


def oscillator_delta_channels(spans, osc_context, seed, zoom=None):
    """{prop: OscDeltaChannel} of the raw oscillator DELTA (no base baked
    in) for one actor's spans, or None when they drive no 2D property.
    Used for the field-instance layer, which sums the delta onto the
    instance's own transform link (field_compose.overlay_deltas), so the
    delta must stay separated from any base."""
    props = sorted({prop for span in spans
                    for prop in _SPAN_PROPS.get(span.kind, ())})
    if not props:
        return None
    return {prop: OscDeltaChannel(spans, prop, osc_context.clock, seed,
                                  osc_context.end_seconds, zoom=zoom)
            for prop in props}


def _field_vanish_timelines(env):
    """Per-player fov vanish-point streams as `{player: {'vanish_x',
    'vanish_y': EventTimeline}}`, or None when no player field recorded a
    SetVanishPoint. gat drives `SetVanishPoint(GetX(), GetY())` per frame on
    the Proxy actors (P1p..P6p), the source the 3D field projection reads to
    project off-centre; players 1/2 map to P1p/P2p (the P1/P2 notefield
    proxies). The field_3d consumer projects through these instead of the
    default screen-centre vanish."""
    out = {}
    for player, name in enumerate(_VANISH_PROXY_NAMES, start=1):
        vanish = _proxy_vanish_timelines(env, name)
        if vanish:
            out[player] = vanish
    return out or None


# The proxy actors whose per-frame SetVanishPoint drives each player field's
# perspective centre (Proxy(1) -> P1p). Only P1p/P2p feed the two rendered
# player fields; P3p..P6p are extra copies read by the field producer.
_VANISH_PROXY_NAMES = ('P1p', 'P2p')
_VANISH_RESTS = {'vanish_x': 320.0, 'vanish_y': 240.0}


def _proxy_vanish_timelines(env, name):
    """{'vanish_x'/'vanish_y': EventTimeline} for one proxy's recorded
    SetVanishPoint stream, or None when it never set one. Read from the
    named-actor keyframes (the proxy self-assigns its global `P1p = self`)
    so the full merged stream - InitCommand + per-frame vanish pokes - is
    present."""
    keyframes = env.named_actor_keyframes().get(name) or {}
    vanish = {prop: keyframes[prop] for prop in _VANISH_RESTS
              if keyframes.get(prop)}
    if not vanish:
        return None
    return {prop: EventTimeline(frames, rest=(_VANISH_RESTS[prop],))
            for prop, frames in vanish.items()}


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
