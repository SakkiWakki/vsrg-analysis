"""Sim run -> the compiled-modfile dict.

`compile_via_sim` produces the same contract `modfile.compile_modfile`
emits, from ONE engine-loop run instead of the load/replay/integration
harvest passes. The generic downstream machinery is reused directly:
`compile_element_tree` consumes the sim actors' keyframes (the tree
compiler keys by `_recorder_id`, which the loop tags), the screen/field
producers consume the env's mirrored harvest surface, and
`compile_mod_channels` consumes the coalesced ApplyModifiers windows.

Deliberately NOT reproduced from the harvest path:

- `unsupported` is always empty: there is no mod_actions residue - every
  closure executed inside the sim.
- No `field_copies`: this compiler emits `field_instances` - the
  generic list (player fields + proxy/AFT copies) of composed transform
  channels (field_compose) - and the 3x3 proxy-grid frames come through
  the same proxy-bind walk, replacing the harvest grid special-case.
"""
from __future__ import annotations

import os
from pathlib import Path

from analysis.games.notitg import aft_chains, field_compose, modfile
from analysis.games.notitg.sim.loop import (
    load_chart, run_declarative, run_sim)
from analysis.games.notitg.sim.record import chase_events, coalesce_applied
from analysis.player.render.effects.timeline import EventTimeline
from analysis.player.render.mods.channels import ModChannels

# Opt-in: run each chart's per-frame Update body through the AST interpreter
# (frame_eval, no Lua) instead of lupa. Set VSRG_NOTITG_COMPILED_BODY=1 in the
# environment to see the compiled path drive the real app. Off by default -
# the Lua path stays the baseline until the compiled path is the default.
_TRUE = {'1', 'true', 'yes', 'on'}


def _compiled_body_flag() -> bool:
    on = os.environ.get('VSRG_NOTITG_COMPILED_BODY', '').lower() in _TRUE
    if on and not _compiled_body_flag._announced:
        _compiled_body_flag._announced = True
        import sys
        print('[notitg] VSRG_NOTITG_COMPILED_BODY set: per-frame Update '
              'bodies run through the AST interpreter (no Lua)', file=sys.stderr)
    return on


_compiled_body_flag._announced = False


def compile_via_sim(sm_path, end_seconds: float | None = None) -> dict | None:
    """The compiled-modfile dict via the engine loop, or None when the
    chart has no modfile. `end_seconds` defaults to the chart's last
    measure plus a tail. Never raises (same contract as
    `compile_modfile`)."""
    try:
        return _compile_via_sim(sm_path, end_seconds)
    except Exception as exc:
        return {'mod_events': [], 'shader_flags': [], 'unsupported':
                {'count': 0, 'described': []}, 'elements': [], 'tree': [],
                'named_actors': 0, 'recorded_keyframes': 0,
                'warnings': [f'sim compile aborted: {exc}']}


def _compile_via_sim(sm_path, end_seconds):
    doc = load_chart(sm_path)
    if doc is None:
        return None
    if end_seconds is None:
        end_seconds = doc.end_seconds
    result = run_declarative(doc.root, doc.to_seconds, doc.start_beat,
                             end_seconds, rng_seed=doc.rng_seed,
                             song_dir=doc.lua_dir.parent,
                             use_compiled_body=_compiled_body_flag())
    env = result.env

    named_keyframes = env.named_actor_keyframes()
    named_meta = env.named_actor_meta()
    actor_keyframes = env.actor_keyframes()
    osc_context = _osc_context(env, doc, end_seconds)
    fonts = modfile._font_resolver(doc.lua_dir)
    _mark_aft_fills(doc, env)
    tree = modfile.compile_element_tree(
        doc.root, doc.to_seconds, doc.start_beat, named_keyframes,
        fonts=fonts, actor_keyframes=actor_keyframes,
        osc_context=osc_context)

    # Mod events from TWO deterministic sources, no whole-song sim:
    #   1. the declarative mods/mods2 tables (the bulk) - read straight
    #      via the proven normalizer, exactly as the harvest path did;
    #   2. the applied stream (ApplyGameCommand mods injected by the
    #      per-frame drivers during the bounded UpdateCommand windows +
    #      the mod_actions replay).
    declarative = modfile._normalize_mod_events(_TableView(env), doc.to_seconds)
    applied = _mod_events(result)
    mod_events = declarative + applied
    # Precompiled channels: the declarative windows compiled through
    # the approach-chase resolver, plus the driver-injected chase.
    mod_channels = _compile_channels(mod_events)

    field_oscillators = modfile._field_oscillator_timelines(env, osc_context)
    field_instances = _sim_field_instances(
        doc, env, actor_keyframes, osc_context, named_keyframes,
        field_oscillators, mod_channels, t0=result.load_seconds)

    return {
        'mod_events': mod_events,
        'mod_channels': mod_channels,
        'shader_flags': [{'beat': beat, 't': t, 'key': key, 'which': which}
                         for t, beat, key, which in result.shader_flags],
        'unsupported': {'count': 0, 'described': []},
        'elements': modfile._compile_elements(
            doc.classic_commands, doc.to_seconds, doc.start_beat,
            named_keyframes),
        'tree': tree,
        'has_background': modfile._has_background_actors(tree, sm_path),
        'field_instances': field_instances,
        'screen_transform': modfile._screen_transform_timelines(env),
        'screen_oscillator': modfile._screen_oscillator_timelines(
            env, osc_context),
        'field_oscillators': field_oscillators,
        'field_vanish': modfile._field_vanish_timelines(env),
        'chart_shaders': _chart_shaders(doc, env, actor_keyframes),
        'aft_bg_visible': modfile._aft_bg_visible_timeline(
            doc.root, _bg_stem(sm_path), actor_keyframes),
        'base_field_hidden': modfile._base_field_hidden_timeline(env),
        # The P1/P2 poke streams for consumers that read the player
        # actors directly (field_3d; the player-placement work) - saves
        # them a private recompile.
        'player_field_keyframes': {
            name: named_keyframes[name] for name in ('P1', 'P2')
            if named_keyframes.get(name)},
        'named_actors': len(named_keyframes),
        'recorded_keyframes': sum(
            len(kfs) for frames in actor_keyframes.values()
            for kfs in frames.values()),
        'replay': {'fired': 0, 'failed': 0, 'applied_mods': 0,
                   'swallowed': 0},
        'integration': {'ran': True, 'ticks': result.ticks,
                        'windows': (), 'applied': len(result.applied_mods),
                        'faults': result.faults},
        'warnings': list(result.warnings) + list(env.fault_messages),
    }


# A mod applied on frame N holds until the reader's next clearall +
# reapply, one frame later - so every window's effective end is one
# frame past its last call. Without this, a single-call spike coalesces
# to a zero-length window and its target never establishes (the classic
# '*100000 1000 drunk' one-frame slam would vanish).
_FRAME_HOLD_S = 1.0 / 60.0

# A per-frame driver re-applies the same mod with a CHANGING value every
# body tick (`ApplyGameCommand('mod,*10000 '..driver:GetX()..' ...')`),
# and between the engine's frames NOTHING reverts - each apply's target
# simply holds until the next. Consecutive same-(player, mods) windows
# within this gap chain end-to-start so the resolved target is the
# driver's staircase, never a rest dip the resolver would otherwise
# insert in every inter-frame gap (the chase then integrates those dips
# into values that neither track the driver nor come to rest).
_CHAIN_GAP_S = 0.05


class _TableView:
    """Adapts a SimEnvironment to the `.mods`/`.mods2` attribute surface
    `modfile._normalize_mod_events` reads, so the proven declarative-table
    normalizer runs against the sim's load-populated tables unchanged."""

    def __init__(self, env):
        self.mods = env.read_table('mods')
        self.mods2 = env.read_table('mods2')


def _compile_channels(mod_events):
    """Mod channels from the combined event set (declarative windows +
    driver-injected windows), all in the one row shape the approach-chase
    resolver reads."""
    from analysis.games.notitg.mod_channels import compile_mod_channels
    return compile_mod_channels(mod_events)


def _mod_events(result) -> list:
    """Coalesced ApplyModifiers windows in `compile_mod_channels`'
    row shape, with driver bursts chained (see _CHAIN_GAP_S). `apply_type`
    marks the provenance; the channel compiler reads only
    t_start/t_end/modstring/player."""
    from analysis.games.notitg.mod_channels import parse_modstring

    groups: dict = {}
    for window in coalesce_applied(result.applied_mods):
        names = tuple(sorted(
            name for _p, _s, name in parse_modstring(window.modstring)))
        groups.setdefault((window.player, names), []).append(window)

    rows = []
    for group in groups.values():
        group.sort(key=lambda w: w.t_start)
        for window, successor in zip(group, [*group[1:], None]):
            end = window.t_end + _FRAME_HOLD_S
            if (successor is not None
                    and successor.t_start - window.t_end <= _CHAIN_GAP_S):
                end = successor.t_start
            rows.append({
                'beat': window.beat_start,
                'modstring': window.modstring,
                'apply_type': 'sim',
                # Window players are engine channel INDEXES (0/1); the
                # row contract carries the chart's 1-based numbers.
                'player': window.player + 1,
                't_start': window.t_start,
                't_end': end,
                'time_based': True,
            })
    rows.sort(key=lambda r: r['t_start'])
    return rows


# -- generalized field instances ---------------------------------------------
#
# A field instance is ANY actor that draws a playfield capture: the two
# player field groups themselves, an ActorProxy with a recorded
# SetTarget bind onto a player (the bind, not a name list, is the
# marker), or a sprite whose texture is an AFT capture. Each gets ONE
# composed transform channel (field_compose.TransformChannel) built from
# its parent-chain of recorded curves in engine order, so copy lifetimes
# are engine-true with no span-gating heuristics: the chart itself hides
# the grid's accumulator frame at section end, and the chain's hidden
# carries it.

import re as _re

# `PlayerP{n}` screen-child name -> 1-based player number. A chart
# `GetChild('PlayerP3')`s any player it wants a field for; NotITG allows
# up to 8, and SRT charts use the extra slots as field SOURCES for their
# proxy copies (each plays the SAME notes with its OWN per-player mods).
# We never hardcode a count: the referenced players are exactly those the
# chart touched (screen children + the players its ApplyModifiers named).
_PLAYER_NAME_RE = _re.compile(r'^PlayerP(\d+)$')


def _player_number(child_name):
    m = _PLAYER_NAME_RE.match(child_name)
    return int(m.group(1)) if m else None


def _base_players(mod_channels) -> list:
    """The 1-based real gameplay players that render an always-drawn
    base field: the ones the chart actually mods (ApplyModifiers(str,
    pn) -> mod channels). A lone player [1] keeps the direct-draw fast
    path; two+ means a versus/dual layout. Extra `GetChild('PlayerP3')`
    slots are NOT base players - they exist only as proxy sources."""
    players = {p + 1 for p in mod_channels.players}  # 0-based -> 1-based
    players.add(1)
    return sorted(players)


def _proxy_source_players(env) -> list:
    """Every 1-based player number a proxy could target: the PlayerP{n}
    screen children the chart accessed. A proxy of P{n}'s notefield
    re-renders that player's field (the SRT charts' decorative copies),
    so its target must resolve to a player number even for n > the real
    gameplay count."""
    return sorted(n for name in env.screen_child_ids()
                  if (n := _player_number(name)) is not None)

# Drawable leaf kinds an AFT-rig fill can be (the rig's fullscreen
# curtains are Quads and bg.png sprites).
_FILL_KINDS = frozenset({'Sprite', 'Quad', 'Layer'})


def _mark_aft_fills(doc, env) -> None:
    """Tag the AFT rig's curtain quads (`actor._aft_fill = True`) so the
    element compiler skips them and `_sim_field_instances` re-emits them
    as ordered fill instances.

    The rig's fullscreen quads (gat's ShowAFT/ShowAFT2/ShowAFT3 black
    and white curtains) sit BETWEEN the field proxies and the
    aft-sampler sprites in engine draw order: they black out the
    proxies underneath while the frozen captures flash above. As
    storyboard elements they draw outside the field-blit pass - either
    over the samplers (a solid black screen) or under the proxies
    (receptors showing through the blackout) - so draw order can only
    be honoured by blitting them inside the instance pass at their
    tree position. A quad belongs to the rig when it shares a message
    command with an AFT node or sampler sprite."""
    rig_messages: set = set()
    fill_candidates = []
    for actor in _iter_xml(doc.root):
        sim = env.actors.get(env.actor_id(actor))
        if sim is None:
            continue
        if sim.is_aft or sim.aft_source:
            rig_messages.update(actor.message_commands())
        elif actor.kind in _FILL_KINDS and not actor.children:
            fill_candidates.append(actor)
    for actor in fill_candidates:
        if rig_messages & set(actor.message_commands()):
            actor._aft_fill = True


def _sim_field_instances(doc, env, actor_keyframes, osc_context,
                         named_keyframes, field_oscillators,
                         mod_channels, t0) -> list:
    # `t0` is the sim's load anchor: channel samples clamp to it so
    # pre-chart times hold the load state (see TransformChannel).
    parents: dict = {}
    _map_parents(doc.root, None, parents, env)
    player_ids = env.screen_child_ids()
    synthetic = env.synthetic_child_ids()
    proxy_players = {}
    for number in _proxy_source_players(env):
        player_id = player_ids.get(f'PlayerP{number}')
        if player_id is None:
            continue
        proxy_players[player_id] = number
        notefield = synthetic.get((player_id, 'NoteField'))
        if notefield is not None:
            proxy_players[notefield] = number
    notefields = {synthetic.get((pid, 'NoteField'))
                  for pid in player_ids.values()} - {None}
    names = env.named_actor_ids()
    aft_nodes = {sim.aft_texture_name: rec_id
                 for rec_id, sim in env.actors.items() if sim.is_aft}
    chain_graph = _aft_chain_graph(doc, env, aft_nodes, proxy_players)

    instances = []
    base_players = _base_players(mod_channels)
    if _multi_players(base_players):
        oscillators = field_oscillators or {}
        for number in base_players:
            instances.append(field_compose.player_instance(
                number, named_keyframes.get(f'P{number}'),
                oscillators.get(number), t0=t0))
    for actor in _iter_xml(doc.root):
        rec_id = env.actor_id(actor)
        sim = env.actors.get(rec_id)
        if sim is None:
            continue
        aft_order = None
        aft_live = None
        color = None
        capture_source = None
        if sim.aft_source:
            kind, player = 'aft', 0
            node = aft_nodes.get(sim.aft_source)
            aft_order = ('pre' if node is not None and rec_id < node
                         else 'post')
            aft_live = _aft_node_visible(env, node)
            # If the source AFT is a 2-stage chain node (it captured a
            # single isolated upstream node, not the whole screen), the
            # consumer blits that isolated capture; None = whole screen
            # (the gat 1 path, byte-identical).
            capture_source = chain_graph.capture_of(sim.aft_source)
        elif sim.proxy_target in proxy_players:
            kind, player = 'proxy', proxy_players[sim.proxy_target]
        elif getattr(actor, '_aft_fill', False):
            kind, player = 'fill', 0
            color = EventTimeline(
                (actor_keyframes.get(rec_id) or {}).get('color', []),
                rest=(1.0, 1.0, 1.0))
        else:
            continue
        chain = _chain(actor, parents)
        links = [_instance_link(link_actor, env, actor_keyframes,
                                osc_context)
                 for link_actor in reversed(chain)]
        if kind == 'proxy':
            # The engine draws a proxied target WITH the target's own
            # transform composed inside the proxy's frame. WHICH
            # transform depends on the bind: `SetTarget(P1)` re-renders
            # the whole player (seat + pokes + oscillators), while
            # `SetTarget(P1:GetChild('NoteField'))` targets the CHILD -
            # the player frame's transform never applies, only the
            # notefield's own recorded pokes do (composing the player's
            # too would double every seat/motion offset).
            if sim.proxy_target in notefields:
                links.append(_notefield_link(sim.proxy_target,
                                             actor_keyframes))
            else:
                links.append(field_compose.player_link(
                    player, named_keyframes.get(f'P{player}'),
                    (field_oscillators or {}).get(player),
                    ignore_hidden=True))
        name = names.get(rec_id)
        if name is None:
            ancestor = next((names[env.actor_id(a)] for a in chain
                             if env.actor_id(a) in names), 'copy')
            name = f'{ancestor}_{rec_id}'
        inst = field_compose.instance(name, kind, player, links,
                                      t0=t0, aft_order=aft_order,
                                      aft_live=aft_live, color=color)
        if capture_source is not None:
            inst['capture_source'] = capture_source
        instances.append(inst)
    return instances


def _aft_chain_graph(doc, env, aft_nodes, proxy_players):
    """The compile-time AFT render-target chain graph (aft_chains): each
    AFT node's isolated upstream capture source (or None = whole screen)
    resolved from engine draw order. Feeds each aft consumer's
    `capture_source` so a 2-stage chain node's sampler blits the isolated
    upstream content instead of the finished frame."""
    node_by_id = {rec_id: name for name, rec_id in aft_nodes.items()}
    blit_sources = {}
    screen_content_ids = set()
    draw_order = []
    for actor in _iter_xml(doc.root):
        rec_id = env.actor_id(actor)
        sim = env.actors.get(rec_id)
        if sim is None:
            continue
        draw_order.append(rec_id)
        if sim.aft_source and rec_id not in node_by_id:
            blit_sources[rec_id] = sim.aft_source
        if sim.proxy_target in proxy_players:
            screen_content_ids.add(rec_id)
    return aft_chains.build_chain_graph(
        node_by_id, blit_sources, draw_order, screen_content_ids)


def _notefield_link(notefield_id, actor_keyframes) -> dict:
    """The proxied NoteField child's own transform link, hidden pinned
    visible (proxies draw their target regardless of its hidden bit)."""
    link = field_compose.link_timelines(actor_keyframes.get(notefield_id))
    link['hidden'] = EventTimeline([], rest=(0.0,))
    return link


def _aft_node_visible(env, node_id):
    """A 0/1 timeline of the source AFT node's visibility, or None when
    the node is unknown. An AFT captures only while it draws
    (`EnablePreserveTexture` holds the last capture across hidden
    frames), so a sampler shows a FROZEN capture whenever its node is
    hidden - a still-frames rig flashes its node visible for a few
    hundredths of a second to grab one freeze, and hiding the node
    freezes the toss capture."""
    sim = env.actors.get(node_id)
    if sim is None:
        return None
    hidden = sim.keyframes().get('hidden')
    if not hidden:
        return None
    return _HiddenAsVisible(EventTimeline(hidden, rest=(0.0,)))


class _HiddenAsVisible:
    """`sample(t) -> (1.0 visible,)` over a hidden channel."""

    def __init__(self, hidden):
        self._hidden = hidden

    def sample(self, t):
        return (0.0 if self._hidden.sample(t)[0] >= 0.5 else 1.0,)


def _multi_players(players) -> bool:
    """More than one player field is in play. Then each referenced
    player renders as its own instance with its own capture (its own
    per-player mods); a lone player keeps the direct-draw fast path."""
    return len(players) > 1


def _map_parents(actor, parent, parents, env) -> None:
    parents[env.actor_id(actor)] = parent
    for child in actor.children:
        _map_parents(child, actor, parents, env)


def _chain(actor, parents) -> list:
    chain = [actor]
    current = parents.get(getattr(actor, '_sim_id', None))
    while current is not None:
        chain.append(current)
        current = parents.get(getattr(current, '_sim_id', None))
    return chain


def _instance_link(actor, env, actor_keyframes, osc_context) -> dict:
    """One chain link: the actor's recorded transform timelines with its
    oscillator spans overlaid as LIVE delta channels, never baked - the
    vibrate mirage is per-frame randomness, so it must evaluate at frame
    time (and a long-running span never freezes at a sample cap). The
    link's own scale_x channel feeds the vibrate amplitude, as the
    engine scales the offset by the actor's zoom."""
    rec_id = env.actor_id(actor)
    link = field_compose.link_timelines(actor_keyframes.get(rec_id))
    spans = (osc_context.spans_by_id.get(rec_id)
             if osc_context is not None else None)
    if spans:
        deltas = modfile.oscillator_delta_channels(
            spans, osc_context, seed=osc_context.seed + rec_id * 7919,
            zoom=link['scale_x'])
        link = field_compose.overlay_deltas(link, deltas)
    return link


def _iter_xml(actor):
    yield actor
    for child in actor.children:
        yield from _iter_xml(child)


def _chart_shaders(doc, env, actor_keyframes) -> list:
    """The map-supplied fragment-shader passes (shader_bridge's
    `chart_shaders` contract), harvested from the actor tree.

    A `Frag=` attribute binds a per-actor GLSL program that samples the
    actor's own texture (openitg Sprite::DrawPrimitives -> sampler0 =
    m_pTexture). Only actors whose texture is a whole-screen
    ActorFrameTexture capture (`aft_source` set - they drew
    `SetTexture(<aft>:GetTexture())`) are fullscreen-expressible: there
    sampler0 IS our finished-frame capture, so the frag maps onto the
    fullscreen `u_tex` contract exactly. Per-actor frags on ordinary
    small textures are Stage B (they would wrongly post-process the
    whole screen) and skipped - the same fullscreen/per-actor split the
    engine has no concept of but that our fullscreen pipeline requires.

    Each pass carries the resolved `.frag` path, the per-uniform value
    streams the actor's `GetShader():uniform*` pokes recorded, and a 0/1
    visibility window from its `hidden` channel (these sprites sit
    `hidden,1` until their section's show message)."""
    passes = []
    for actor in _iter_xml(doc.root):
        frag = actor.attrs.get('Frag')
        rec_id = env.actor_id(actor)
        if not frag or rec_id is None:
            continue
        sim = env.actors.get(rec_id)
        if sim is None or not sim.aft_source:
            continue
        frag_path = _resolve_frag(actor, frag)
        if frag_path is None:
            continue
        frames = actor_keyframes.get(rec_id, {})
        uniforms = {prop[len('uniform:'):]: _uniform_events(kfs)
                    for prop, kfs in frames.items()
                    if prop.startswith('uniform:')}
        passes.append({
            'name': f'gf{rec_id}_{Path(frag).stem}',
            'frag_path': str(frag_path),
            'uniforms': uniforms,
            'windows': _visibility_events(frames.get('hidden')),
        })
    return passes


def _resolve_frag(actor, frag) -> Path | None:
    """Absolute path to a `Frag=` reference, resolved against the actor's
    own XML directory (`_base_dir`), like every other chart asset. `./`
    prefixes and bare `shaders/x.frag` both resolve under that dir."""
    base_dir = getattr(actor, '_base_dir', None)
    if base_dir is None:
        return None
    candidate = Path(base_dir) / frag.lstrip('./')
    return candidate if candidate.is_file() else None


def _uniform_events(keyframes) -> list:
    """A uniform's `Keyframe` stream as the bridge's `.ffx`-shaped event
    dicts (ms times, `strength` value): the same shape the shader-flag
    path emits, so shader_bridge stays the one stable contract."""
    return [{'time': kf.t * 1000.0, 'duration': kf.duration * 1000.0,
             'ease': kf.easing, 'strength': kf.values[0]}
            for kf in keyframes]


def _visibility_events(hidden_kfs) -> list:
    """A pass-liveness window stream from the actor's `hidden` channel
    (1 hidden, 0 shown) inverted to `strength` (1 live, 0 off), or []
    when the actor never toggled hidden (always live). Instant steps -
    the hidden bit is immediate, never tweened."""
    if not hidden_kfs:
        return []
    return [{'time': kf.t * 1000.0, 'duration': 0.0, 'ease': 0,
             'strength': 0.0 if kf.values[0] else 1.0}
            for kf in hidden_kfs]


def _osc_context(env, doc, end_seconds):
    """The oscillator synthesis context over the sim env's spans, via
    the same modfile machinery the harvest path uses. `end_seconds` is
    the compile end still-open spans extend to."""
    return modfile._build_osc_context(env, doc.to_seconds, doc.start_beat,
                                      doc.lua_dir, end_seconds=end_seconds)


def _bg_stem(sm_path) -> str:
    return Path(modfile._sm_background_name(sm_path)).stem.casefold()
