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

from pathlib import Path

from analysis.games.notitg import field_compose, modfile
from analysis.games.notitg.sim.loop import (
    load_chart, run_declarative, run_sim)
from analysis.games.notitg.sim.record import chase_events, coalesce_applied
from analysis.player.render.mods.channels import ModChannels


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
                             end_seconds, rng_seed=doc.rng_seed)
    env = result.env

    named_keyframes = env.named_actor_keyframes()
    named_meta = env.named_actor_meta()
    actor_keyframes = env.actor_keyframes()
    osc_context = _osc_context(env, doc, end_seconds)
    fonts = modfile._font_resolver(doc.lua_dir)
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
    row shape. `apply_type` marks the provenance; the channel compiler
    reads only t_start/t_end/modstring/player."""
    return [{
        'beat': window.beat_start,
        'modstring': window.modstring,
        'apply_type': 'sim',
        # Window players are engine channel INDEXES (0/1); the row
        # contract carries the chart's 1-based numbers.
        'player': window.player + 1,
        't_start': window.t_start,
        't_end': window.t_end + _FRAME_HOLD_S,
        'time_based': True,
    } for window in coalesce_applied(result.applied_mods)]


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

# Screen children in player order: a proxy targeting a player's recorder
# (or its NoteField child) re-renders that player's notefield.
_PLAYER_CHILDREN = ('PlayerP1', 'PlayerP2')


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
    for number, child in enumerate(_PLAYER_CHILDREN, start=1):
        player_id = player_ids.get(child)
        if player_id is None:
            continue
        proxy_players[player_id] = number
        notefield = synthetic.get((player_id, 'NoteField'))
        if notefield is not None:
            proxy_players[notefield] = number
    names = env.named_actor_ids()
    aft_nodes = {sim.aft_texture_name: rec_id
                 for rec_id, sim in env.actors.items() if sim.is_aft}

    instances = []
    if _dual_players(mod_channels, named_keyframes):
        oscillators = field_oscillators or {}
        for number in (1, 2):
            instances.append(field_compose.player_instance(
                number, named_keyframes.get(f'P{number}'),
                oscillators.get(number), t0=t0))
    for actor in _iter_xml(doc.root):
        rec_id = env.actor_id(actor)
        sim = env.actors.get(rec_id)
        if sim is None:
            continue
        aft_order = None
        if sim.aft_source:
            kind, player = 'aft', 0
            node = aft_nodes.get(sim.aft_source)
            aft_order = ('pre' if node is not None and rec_id < node
                         else 'post')
        elif sim.proxy_target in proxy_players:
            kind, player = 'proxy', proxy_players[sim.proxy_target]
        else:
            continue
        chain = _chain(actor, parents)
        links = [_instance_link(link_actor, env, actor_keyframes,
                                osc_context)
                 for link_actor in reversed(chain)]
        name = names.get(rec_id)
        if name is None:
            ancestor = next((names[env.actor_id(a)] for a in chain
                             if env.actor_id(a) in names), 'copy')
            name = f'{ancestor}_{rec_id}'
        instances.append(field_compose.instance(name, kind, player, links,
                                                t0=t0, aft_order=aft_order))
    return instances


def _dual_players(mod_channels, named_keyframes) -> bool:
    """A second real player is in play: the chart mods player 2's
    channels or poked its PlayerP2 group. Then both player fields render
    every frame, each an instance with its own capture."""
    return 1 in mod_channels.players or bool(named_keyframes.get('P2'))


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


def _osc_context(env, doc, end_seconds):
    """The oscillator synthesis context over the sim env's spans, via
    the same modfile machinery the harvest path uses. `end_seconds` is
    the compile end still-open spans extend to."""
    return modfile._build_osc_context(env, doc.to_seconds, doc.start_beat,
                                      doc.lua_dir, end_seconds=end_seconds)


def _bg_stem(sm_path) -> str:
    return Path(modfile._sm_background_name(sm_path)).stem.casefold()
