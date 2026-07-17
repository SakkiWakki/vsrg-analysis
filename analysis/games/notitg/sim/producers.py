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
from analysis.games.notitg.sim.loop import load_chart, run_sim
from analysis.games.notitg.sim.record import chase_events, coalesce_applied
from analysis.player.render.mods.channels import ModChannels


def compile_via_sim(sm_path, end_seconds: float) -> dict | None:
    """The compiled-modfile dict via the engine loop, or None when the
    chart has no modfile. Never raises (same contract as
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
    result = run_sim(doc.root, doc.to_seconds, doc.start_beat, end_seconds,
                     rng_seed=doc.rng_seed)
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

    field_copies = modfile._field_copies(
        doc.root, named_keyframes, named_meta, doc.to_seconds,
        doc.start_beat, actor_keyframes, osc_context)

    return {
        'mod_events': _mod_events(result),
        # Precompiled channels from frame-resolved retarget events - the
        # exact engine chase, no window reconstruction. Consumers prefer
        # this over recompiling mod_events (the mirin-dict pattern).
        'mod_channels': ModChannels.compile(
            chase_events(result.applied_mods)),
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


def _osc_context(env, doc):
    """The oscillator synthesis context over the sim env's spans, via
    the same modfile machinery the harvest path uses."""
    return modfile._build_osc_context(env, doc.to_seconds, doc.start_beat,
                                      doc.lua_dir)


def _bg_stem(sm_path) -> str:
    return Path(modfile._sm_background_name(sm_path)).stem.casefold()
