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
- The 3x3 proxy-grid copies (`env.proxy_grid()`) are not emitted yet;
  the grid frames' driven transforms ARE recorded on their actors, and
  the generalized copy producer (proxy/AFT binds + parent-chain
  composition) replaces the grid special-case in the parity phase. The
  compile-diff report tracks this as the known gap.
"""
from __future__ import annotations

from pathlib import Path

from analysis.games.notitg import modfile
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
    osc_context = _osc_context(env, doc)
    fonts = modfile._font_resolver(doc.lua_dir)
    tree = modfile.compile_element_tree(
        doc.root, doc.to_seconds, doc.start_beat, named_keyframes,
        fonts=fonts, actor_keyframes=actor_keyframes,
        osc_context=osc_context)

    field_copies = _sim_field_copies(doc, env, actor_keyframes,
                                     osc_context)

    # Mod events from TWO deterministic sources, no whole-song sim:
    #   1. the declarative mods/mods2 tables (the bulk) - read straight
    #      via the proven normalizer, exactly as the harvest path did;
    #   2. the applied stream (ApplyGameCommand mods injected by the
    #      per-frame drivers during the bounded UpdateCommand windows +
    #      the mod_actions replay).
    declarative = modfile._normalize_mod_events(_TableView(env), doc.to_seconds)
    applied = _mod_events(result)
    mod_events = declarative + applied

    return {
        'mod_events': mod_events,
        # Precompiled channels: the declarative windows compiled through
        # the approach-chase resolver, plus the driver-injected chase.
        'mod_channels': _compile_channels(mod_events),
        'shader_flags': [{'beat': beat, 't': t, 'key': key, 'which': which}
                         for t, beat, key, which in result.shader_flags],
        'unsupported': {'count': 0, 'described': []},
        'elements': modfile._compile_elements(
            doc.classic_commands, doc.to_seconds, doc.start_beat,
            named_keyframes),
        'tree': tree,
        'has_background': modfile._has_background_actors(tree, sm_path),
        'field_copies': field_copies,
        'screen_transform': modfile._screen_transform_timelines(env),
        'screen_oscillator': modfile._screen_oscillator_timelines(
            env, osc_context),
        'field_oscillators': modfile._field_oscillator_timelines(
            env, osc_context),
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


# -- generalized field copies ------------------------------------------------
#
# A field copy is ANY actor that draws the captured playfield: an
# ActorProxy with a recorded SetTarget bind onto a player (the bind, not
# a name list, is the marker), or a sprite whose texture is an AFT
# capture. Its on-screen transform is the PARENT-CHAIN composition of
# recorded curves - x/y/rotation sum, scales/alpha multiply, hidden is
# any-ancestor-hides - which is also what makes copy lifetimes
# engine-true with no span-gating heuristics: the chart itself hides the
# grid's accumulator frame at section end, and the composed hidden
# carries it. (Curve folds, not baked matrices, so the document tree can
# take over composition later; rotation-around-offset composition is the
# document tree's job.)

class _SumTimeline:
    def __init__(self, timelines):
        self._timelines = timelines

    def sample(self, t):
        return (sum(tl.sample(t)[0] for tl in self._timelines),)


class _ProductTimeline:
    def __init__(self, timelines):
        self._timelines = timelines

    def sample(self, t):
        value = 1.0
        for tl in self._timelines:
            value *= tl.sample(t)[0]
        return (value,)


class _MaxTimeline:
    def __init__(self, timelines):
        self._timelines = timelines

    def sample(self, t):
        return (max(tl.sample(t)[0] for tl in self._timelines),)


_SUM_PROPS = ('x', 'y', 'rotation')
_PRODUCT_PROPS = ('scale_x', 'scale_y', 'base_scale_x', 'base_scale_y',
                  'alpha')

# ActorProxy sources: a proxy targeting a player's recorder re-renders
# that player's notefield; 'P1p'/'P2p' are the source tags the field
# consumer already maps to per-player field captures.
_PLAYER_SOURCES = (('PlayerP1', 'P1p'), ('PlayerP2', 'P2p'))


def _sim_field_copies(doc, env, actor_keyframes, osc_context) -> list:
    from analysis.player.render.effects.timeline import EventTimeline

    parents: dict = {}
    _map_parents(doc.root, None, parents, env)
    player_ids = env.screen_child_ids()
    synthetic = env.synthetic_child_ids()
    proxy_sources = {}
    for child, source in _PLAYER_SOURCES:
        player_id = player_ids.get(child)
        if player_id is None:
            continue
        proxy_sources[player_id] = source
        notefield = synthetic.get((player_id, 'NoteField'))
        if notefield is not None:
            proxy_sources[notefield] = source
    names = env.named_actor_ids()

    copies = []
    for actor in _iter_xml(doc.root):
        rec_id = env.actor_id(actor)
        sim = env.actors.get(rec_id)
        if sim is None:
            continue
        source = sim.aft_source or (
            proxy_sources.get(sim.proxy_target)
            if sim.proxy_target is not None else None)
        if source is None:
            continue
        chain = _chain(actor, parents)
        timelines = {}
        for prop in modfile._FIELD_PROPS:
            rest = modfile._FIELD_RESTS[prop]
            curves = [EventTimeline(kfs, rest=(rest,)) for kfs in
                      _chain_keyframes(chain, prop, env, actor_keyframes,
                                       osc_context)]
            if not curves:
                timelines[prop] = EventTimeline([], rest=(rest,))
            elif len(curves) == 1:
                timelines[prop] = curves[0]
            elif prop in _SUM_PROPS:
                timelines[prop] = _SumTimeline(curves)
            elif prop in _PRODUCT_PROPS:
                timelines[prop] = _ProductTimeline(curves)
            else:
                timelines[prop] = _MaxTimeline(curves)
        name = names.get(rec_id)
        if name is None:
            ancestor = next((names[env.actor_id(a)] for a in chain
                             if env.actor_id(a) in names), 'copy')
            name = f'{ancestor}_{rec_id}'
        copies.append({'name': name, 'source': source,
                       'timelines': timelines})
    return copies


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


def _chain_keyframes(chain, prop, env, actor_keyframes, osc_context):
    for actor in chain:
        keyframes = actor_keyframes.get(env.actor_id(actor))
        if not keyframes:
            continue
        if osc_context is not None:
            keyframes = modfile._apply_oscillators(actor, keyframes,
                                                   osc_context)
        if keyframes.get(prop):
            yield keyframes[prop]


def _iter_xml(actor):
    yield actor
    for child in actor.children:
        yield from _iter_xml(child)


def _osc_context(env, doc):
    """The oscillator synthesis context over the sim env's spans, via
    the same modfile machinery the harvest path uses."""
    return modfile._build_osc_context(env, doc.to_seconds, doc.start_beat,
                                      doc.lua_dir)


def _bg_stem(sm_path) -> str:
    return Path(modfile._sm_background_name(sm_path)).stem.casefold()
