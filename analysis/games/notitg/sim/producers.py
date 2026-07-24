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
import re
from functools import lru_cache
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


class _LiveBaseHidden:
    """Lazy twin of modfile._base_field_hidden_timeline: samples the
    real PlayerP1 screen child's hidden channel from the live sim.
    gat-family charts hide the base fields whole-file and show notes
    only through copies; the eager path honoured this, but the lazy
    path hardcoded None - so the app drew the hidden base fields
    (fullscreen notes + receptors) into every scene AND every AFT
    capture, and feedback rigs multiplied them (the cyriak
    branch-density excess). The screen child binds when the chart first
    calls GetChild('PlayerP1') - AFTER load - so the curve resolves on
    first sample, not at compile."""

    def __init__(self, live):
        self._live = live
        self._curve = None

    def sample(self, t):
        if self._curve is None:
            rec = self._live.env.screen_child_ids().get('PlayerP1')
            if rec is None:
                return (0.0,)
            from analysis.games.notitg.sim.seg_read import curve_for
            self._curve = curve_for(self._live, rec, 'hidden', (0.0,))
        return self._curve.sample(t)


def _base_field_hidden_live(live):
    return _LiveBaseHidden(live)


def _lazy_flag() -> bool:
    """LAZY REPLAY (the DEFAULT compile path): `compile` stores a LiveSim
    (instant open, no whole-song bake) and the storyboard element tree + field
    instances + mods sample it live at draw time; a background sweep fills in the
    driver-injected mods, the complete proxy/AFT/NoteField topology, the
    body-populated declarative mods, and freezes the never-poked storyboard
    curves - all hot-swapped in place. Full-corpus verified (989 charts): 83%
    field-perfect vs the eager bake, 0 missing instances/mods; the residual is a
    sub-visible motion-frame-phase difference. Set VSRG_NOTITG_LAZY=0 to force
    the legacy eager whole-song bake (the byte-exact reference the golden/keyframe
    -diff harnesses compare against)."""
    return os.environ.get('VSRG_NOTITG_LAZY', '1').lower() in _TRUE


def compile_via_sim(sm_path, end_seconds: float | None = None) -> dict | None:
    """The compiled-modfile dict via the engine loop, or None when the
    chart has no modfile. `end_seconds` defaults to the chart's last
    measure plus a tail. Never raises (same contract as
    `compile_modfile`)."""
    try:
        if _lazy_flag():
            return _compile_live(sm_path, end_seconds)
        return _compile_via_sim(sm_path, end_seconds)
    except Exception as exc:
        return {'mod_events': [], 'shader_flags': [], 'unsupported':
                {'count': 0, 'described': []}, 'elements': [], 'tree': [],
                'named_actors': 0, 'recorded_keyframes': 0,
                'warnings': [f'sim compile aborted: {exc}']}


def _compile_live(sm_path, end_seconds) -> dict | None:
    """LAZY compile (near-instant): build a `LiveSim` (load + store, NO ticking)
    and the storyboard element tree whose value timelines are LiveCurves reading
    that sim at draw time. The whole-song outputs (mod_channels, field_instances,
    oscillators) are DEFERRED for this first cut - the storyboard element tree is
    the bulk of what renders; those effects appear as they are lazy-completed."""
    from analysis.games.notitg.sim.loop import LiveSim

    doc = load_chart(sm_path)
    if doc is None:
        return None
    end = doc.end_seconds if end_seconds is None else end_seconds
    live = LiveSim(doc.root, doc.to_seconds, doc.start_beat, end,
                   rng_seed=doc.rng_seed, song_dir=doc.lua_dir.parent,
                   use_compiled_body=_compiled_body_flag())
    fonts = modfile._font_resolver(doc.lua_dir)
    tree = modfile.compile_element_tree(
        doc.root, doc.to_seconds, doc.start_beat, named_keyframes={},
        fonts=fonts, actor_keyframes=None, sim=live)

    # DECLARATIVE mods (the bulk of the note-mod windows: the mods/mods2 tables
    # the load pass populated) need NO sim - read them straight from the live
    # env's tables, exactly as the eager path does. This restores the scroll/
    # tipsy/drunk/etc. note mods instantly. The DRIVER-injected `applied` mods
    # (ApplyGameCommand from the per-frame body) still need the sim run and are
    # deferred - a minority for most charts.
    declarative = modfile._normalize_mod_events(_TableView(live.env),
                                                doc.to_seconds)
    mod_channels = _compile_channels(declarative)

    # The whole-scene camera (gat's screen zoom) is a single actor - read it
    # LIVE like the element tree, so the screen-zoom camera works instantly.
    screen_transform = modfile._screen_transform_live(live)

    # The note-field proxy/AFT instances: a PROVIDER that rebuilds from the live
    # sim as proxy/AFT bindings fire (topology grows throughout the chart). The
    # effect calls it each frame; transform values are LiveCurves.
    field_instances = _LiveFieldInstances(doc, live, mod_channels,
                                          t0=live._load_s)

    # BACKGROUND upgrade (one daemon sweep to chart end): resolve the driver-
    # injected mods (gat's drunk/tipsy/hallway approach-chase bursts need the
    # full pass to resolve their ramps), freeze the tree's never-poked LiveCurves
    # to constants (~86%), and hand the field provider the swept env as its
    # COMPLETE topology source (the playback sim only knows proxy/AFT/NoteField
    # binds up to the playhead, so proxy-heavy charts render missing copies until
    # the sweep lands). All hot-swap in place, so instant open pays for none of
    # it: the driver mods fill in, the storyboard gets cheaper, and the full
    # field-instance set appears a few seconds after open.
    from analysis.games.notitg.sim.seg_read import segtl_enabled
    shared_sim = live if segtl_enabled() else None
    preview_note = _attach_preview(live) if shared_sim is not None else ''
    painter_rows = getattr(live, 'evaluated_applied', None)
    if painter_rows:
        # Painter mods (per-frame ApplyModifiers curves) recovered by the
        # residue evaluation join the instant channel compile; the sweep's
        # completion swap later replaces this with the exact stream.
        full = _compile_channels(
            declarative + _mod_events(_SweptResult(painter_rows)))
        mod_channels._channels = full._channels
        mod_channels._players = full._players
    # The chart's own fragment-shader passes need the swept uniform +
    # visibility channels, so open hands the renderer an EMPTY live
    # effect and the sweep swaps the real passes in (adapter recognizes
    # the prebuilt object; previously the lazy path hardcoded [] and
    # every shader moment silently rendered as nothing).
    from analysis.games.notitg.shader_bridge import ChartShaderEffect
    chart_shaders = ChartShaderEffect(())
    screen_shake = ScreenShakeHandle()
    scroll_mult = ScrollMultiplierHandle()
    _spawn_background_upgrade(mod_channels, tree, field_instances,
                             sm_path, end, live_sim=shared_sim,
                             to_seconds=doc.to_seconds, doc=doc,
                             chart_shaders=chart_shaders,
                             screen_shake=screen_shake,
                             scroll_mult=scroll_mult)

    return {
        'mod_events': declarative, 'mod_channels': mod_channels,
        'shader_flags': [], 'unsupported': {'count': 0, 'described': []},
        'elements': [], 'tree': tree,
        'has_background': modfile._has_background_actors(tree, sm_path),
        'field_instances': field_instances, 'screen_transform': screen_transform,
        'screen_oscillator': screen_shake, 'field_oscillators': [],
        'field_vanish': None, 'chart_shaders': chart_shaders,
        'scroll_multiplier_timeline': scroll_mult,
        'aft_bg_visible': None,
        'base_field_hidden': _base_field_hidden_live(live),
        '_live_sim': live,
        'named_actors': 0, 'recorded_keyframes': 0,
        'warnings': list(live.warnings) + [
            'lazy replay (VSRG_NOTITG_LAZY): element tree + declarative '
            'mods + field instances live; driver-applied mods fill in via '
            'background compile'
            + ('; segment-timeline reads (VSRG_NOTITG_SEGTL)' + preview_note
               if segtl_enabled() else '')],
    }


class _LiveFieldInstances:
    """A provider that rebuilds the field-instance list from the LIVE sim as its
    topology grows (proxy/AFT bindings fire throughout the chart). Called by the
    NotitgFieldInstances effect each frame; rebuilds ONLY when the topology
    SIGNATURE (the set of bound proxy_target / aft_source / is_aft actors)
    changes - a rebuild is ~1ms and topology changes are rare, so per-frame cost
    is a cheap signature check. Transform values are LiveCurves (read live), so
    only STRUCTURE triggers a rebuild."""

    def __init__(self, doc, live, mod_channels, t0):
        self._doc = doc
        self._live = live
        self._mod_channels = mod_channels
        self._t0 = t0
        # The seconds<->beat inverter is the expensive part of the oscillator
        # context (a dense table over the whole chart tempo map) and is STATIC -
        # it depends on the chart, not the live span set - so build it once here
        # and reuse it every frame. Without this the provider rebuilt an ~8000-
        # point inverter per draw (the dominant per-frame cost, ~16k tempo-map
        # lookups/frame).
        self._osc_clock = modfile._osc_clock_for_end(
            doc.to_seconds, doc.start_beat, doc.end_seconds)
        self._cache = None
        self._cache_sig = None
        self._cache_expiry = 0.0
        # The topology source: which actors are proxies/AFTs/fills, their
        # targets, and the NoteField synthetic-child mapping. The PLAYBACK sim
        # only knows the topology GetChild/SetTarget calls have fired UP TO the
        # playhead, so proxy-heavy charts (SRT allproxies) render missing copies
        # until the chart reaches those binds - and NoteField proxies never bind
        # if GetChild('NoteField') fires late. The background sweep runs the full
        # sim, so it holds the COMPLETE topology; once it lands here (via
        # set_topology_source) the provider reads structure from the swept env
        # while still sampling transforms live from the playback sim. None until
        # the sweep completes (early frames use the growing live topology).
        self._topology_env = None

    def set_topology_source(self, swept_env) -> None:
        """Called by the background upgrade with the fully-swept env (complete
        proxy/AFT/NoteField topology). Invalidates the cache so the next build
        emits the full instance set."""
        self._topology_env = swept_env
        self._cache = None
        self._cache_sig = None

    def __call__(self):
        # Instance-list caching: the rebuild is ~1.2ms and 94% of it is the
        # topology walk + per-link LiveCurve build, which only CHANGES when a
        # proxy/AFT binds. The transforms are LiveCurves (re-read the sim), so a
        # cached list stays live for FREE - EXCEPT the field oscillator, whose
        # OscDeltaChannel snapshots a COPY of the open span at build (actor.
        # oscillator_spans() copies _osc_open), so a cached instance freezes the
        # vibrate phase while a span is OPEN. So: cache keyed on topology, and
        # while ANY player field has an open oscillator span, bypass the cache
        # and rebuild every frame (gat oscillates only ~t8-48; the other ~8min
        # reuse the cache). This is why an earlier topology-only cache froze the
        # field - it missed the open-span exception.
        # In segment-read mode the ONE sim races to the chart end on a
        # daemon thread while this runs per-frame on the render thread;
        # a dict resized mid-iteration raises RuntimeError. The rebuild
        # is a pure read, so on a collision serve the previous frame's
        # list - the next frame retries (and the sweep is done within
        # seconds of launch, after which the env is quiescent).
        self._nudge_sweep()
        try:
            return self._build_instances()
        except RuntimeError:
            return self._cache or []

    def _nudge_sweep(self) -> None:
        """A WALL-budgeted slice of the sweep on the render thread (~3ms
        per frame): a guaranteed frontier floor that no GIL handoff can
        starve, costing a bounded fraction of the frame regardless of
        how expensive the chart region is to simulate. Non-blocking: if
        the worker holds the lock it is already making progress."""
        import time as _time

        live = self._live
        lock = getattr(live, 'sweep_lock', None)
        if lock is None or live.frontier >= live._end_seconds:
            return
        live.render_seen = _time.monotonic()
        if lock.acquire(blocking=False):
            try:
                deadline = _time.perf_counter() + 0.003
                while (_time.perf_counter() < deadline
                       and live.frontier < live._end_seconds):
                    live.advance_to(min(live.frontier + 0.02,
                                        live._end_seconds))
            finally:
                lock.release()

    def _build_instances(self):
        env = self._live.env
        # Before the sweep hands over the topology, a list built from the
        # RACING env can be torn yet carry the final signature - never
        # pin it. But rebuilding every frame during the sweep steals
        # render headroom exactly when the sweep needs it, so pre-handover
        # frames reuse the last build for a short TTL: a torn list can
        # survive at most one TTL, and the handover invalidation ends the
        # regime entirely.
        import time as _time
        if self._topology_env is None:
            now = _time.monotonic()
            if self._cache is not None and now < self._cache_expiry:
                return self._cache
            instances = self._rebuild(env)
            self._cache = instances
            self._cache_sig = None
            self._cache_expiry = now + 0.25
            return instances

        oscillating = self._oscillating(env)
        sig = None if oscillating else self._topology_sig(env)
        if sig is not None and sig == self._cache_sig:
            return self._cache

        instances = self._rebuild(env)
        # Cache only closed-oscillator frames; an oscillating frame's list holds
        # a frozen open-span copy and must never be reused.
        self._cache = instances if not oscillating else None
        self._cache_sig = sig
        return instances

    def _rebuild(self, env):
        # Oscillator state is LIVE (playback env); the instance STRUCTURE uses
        # the swept env once available (complete topology) so proxy/AFT/fill
        # copies all appear. rec_ids match between the two sims, so the links'
        # LiveCurves (keyed by rec_id) still sample the playback sim live.
        osc_context = _osc_context(env, self._doc, self._doc.end_seconds,
                                   clock=self._osc_clock)
        field_oscillators = modfile._field_oscillator_timelines(
            env, osc_context)
        topology_env = self._topology_env if self._topology_env is not None \
            else env
        # `named_keyframes` feeds ONLY the eager (`live_sim is None`) branches of
        # _sim_field_instances; the live path reads the sim directly. Computing
        # env.named_actor_keyframes() here (which re-simplifies every named
        # actor's poke stream, invalidated each tick during playback) was pure
        # per-frame waste, so pass an empty map.
        return _sim_field_instances(
            self._doc, topology_env, None, osc_context,
            {}, field_oscillators,
            self._mod_channels, t0=self._t0, live_sim=self._live)

    def _oscillating(self, env) -> bool:
        """Any player field with an OPEN oscillator span (its delta channel
        snapshots a stale copy, so the instance list can't be cached)."""
        actors = (env.player_actor(name) for name in modfile._PLAYER_FIELD_NAMES)
        return any(a is not None and a._osc_open is not None for a in actors)

    def _topology_sig(self, env):
        """A cheap hashable signature of everything _sim_field_instances'
        STRUCTURE depends on: proxy binds, AFT binds + visibility, and the
        screen child set. Unchanged signature => an identical instance list
        (transforms stay live via LiveCurves), so the cache is reused."""
        proxy = tuple(sorted((rec_id, a.proxy_target)
                             for rec_id, a in env.actors.items()
                             if a.proxy_target is not None))
        aft = tuple(sorted((rec_id, a.aft_source, a.is_aft)
                           for rec_id, a in env.actors.items()
                           if a.aft_source is not None or a.is_aft))
        return (proxy, aft, tuple(sorted(env.screen_child_ids().items())))

    def __iter__(self):
        # The provider IS the instance list: any consumer that iterates it
        # (e.g. adapter._field_owned's topology check) gets the current
        # snapshot. The base players bind at t0, so 'player' ownership is
        # decidable from the first rebuild.
        return iter(self())


class _SweptResult:
    """The `.applied_mods` surface `_mod_events` reads, from a LiveSim swept to
    the chart end (no baking - just the accumulated ApplyGameCommand stream)."""

    __slots__ = ('applied_mods',)

    def __init__(self, applied_mods):
        self.applied_mods = applied_mods


def _attach_preview(live) -> str:
    """Schedule-lower the staged actions into preview lanes on the live
    sim's actors (the beyond-frontier read layer). Conservative by
    construction; a failure just means no preview, never a bad one."""
    from analysis.games.notitg import schedule_lower

    try:
        preview = schedule_lower.lower_actions(live.env,
                                               to_beats=live._to_beats,
                                               to_seconds=live._to_seconds)
    except Exception as exc:
        return f'; action preview failed: {exc}'
    for rec_id, lanes in preview.lanes.items():
        actor = live.env._actors.get(rec_id)
        if actor is not None:
            actor._seg_preview = lanes

    # The Update body's closed-form half fills in body-driven props the
    # actions never touch; action lanes win where both exist.
    body_lanes, body_note = schedule_lower.lower_update_body(live)
    for rec_id, lanes in body_lanes.items():
        actor = live.env._actors.get(rec_id)
        if actor is None:
            continue
        for prop, lane in lanes.items():
            actor._seg_preview.setdefault(prop, lane)

    eval_note = _attach_evaluated_residue(live, preview)
    return (f'; action preview: {preview.lifted_handlers} handlers -> '
            f'{sum(len(v) for v in preview.lanes.values())} channels '
            f'({preview.residue_handlers} residue)') + body_note + eval_note


def _attach_evaluated_residue(live, preview) -> str:
    """The pure lane-backed evaluation of the residue windows: body
    pokes over known collection membership become preview emissions,
    MERGED with the action emissions per channel so a channel driven by
    both composes in time order. Tainted channels stay sweep-owned.

    DEFAULT ON (VSRG_NOTITG_LANE_EVAL=0 reverts): the gate harness
    (tests/local/lane_eval_gate.py) passes every evaluating chart in
    the 29-chart sweep against the swept truth, and a wrong channel is
    bounded by design - tainted channels never leave the sweep, and
    everything here is beyond-frontier preview that the sweep
    overwrites as it advances."""
    if os.environ.get('VSRG_NOTITG_LANE_EVAL', '1').lower() not in _TRUE:
        return ''
    from analysis.games.notitg import residue_eval, schedule_lower

    try:
        result = residue_eval.evaluate_residue(
            live, preview.registrations,
            global_sets=preview.global_sets,
            seed_pokes=preview.seed_pokes)
    except Exception as exc:
        return f'; residue eval failed: {exc}'
    if result is None:
        return ''

    merged: dict = {}
    kept = 0
    for (rec_id, prop), pokes in result.emissions.items():
        if (rec_id, prop) in result.tainted:
            continue
        kept += 1
        merged.setdefault(rec_id, []).extend(
            (t, prop, value) for t, value in pokes)
    for rec_id, extra in merged.items():
        actor = live.env._actors.get(rec_id)
        if actor is None:
            continue
        events = list(preview.emissions.get(rec_id, ())) + extra
        actor._seg_preview = schedule_lower._finish_lanes(
            live.env, rec_id, events)

    live.evaluated_applied = list(result.applied)
    return (f'; residue eval: {result.ticks} ticks -> {kept} channels '
            f'({len(result.tainted)} tainted, '
            f'{len(result.applied)} painter rows)')


class ScrollMultiplierHandle:
    """The chart's eased scroll-multiplier timeline for the lazy path:
    rest (1.0) until the background sweep resolves the applied xmod
    stream (the per-frame reader carries the baseline and every burst,
    none of which the instant compile has). The sv renderer samples this
    every frame, so the swap needs no player rebuild - the same hot-swap
    shape as ScreenShakeHandle."""

    __slots__ = ('timeline',)

    def __init__(self, timeline=None):
        self.timeline = timeline

    def sample(self, t):
        inner = self.timeline
        return inner.sample(t) if inner is not None else (1.0,)


class ScreenShakeHandle:
    """The screen's oscillator-delta channels for the lazy path: None
    until the background sweep computes them (the delta needs the swept
    effect spans + oscillator context). The screen camera samples
    `.channels` every frame, so the shake appears at handover without
    the adapter rebuilding anything."""

    __slots__ = ('channels',)

    def __init__(self, channels=None):
        self.channels = channels


def _spawn_background_upgrade(mod_channels, tree, field_provider,
                             sm_path, end_seconds, live_sim=None,
                             to_seconds=None, doc=None,
                             chart_shaders=None, screen_shake=None,
                             scroll_mult=None):
    """Background pass on a daemon thread: sweep to the chart end, then
    hot-swap three things the instant compile left approximate. With
    `live_sim` (the segment-read default) the sweep advances THAT sim -
    the one recording sim serves both playback reads (behind its
    frontier) and this pass; without it (LiveCurve fallback) a separate
    sweep sim is built so the playback sim stays at the playhead.

    1. Driver-injected mods: resolve the applied ApplyGameCommand stream and
       replace `mod_channels`' internals in place (the player holds that exact
       object and reads it every frame).
    2. Static storyboard timelines: ~86% of the tree's per-property LiveCurves
       belong to a property the actor NEVER pokes (avg 1.7 animated props out of
       38), so each frame they advance the sim + look up the actor + read a
       channel only to return the same rest. Replace every never-poked
       property's LiveCurve with a constant EventTimeline (its rest) IN the
       element's mutable `timelines` dict, so the live effect samples a cheap
       constant. Never-poked is exact: a prop absent from the swept actor's
       `_frames` (and not a set rotation_order/quat) is provably constant.
    3. Field-instance topology: the playback sim only knows the proxy/AFT/
       NoteField binds that fired up to the playhead, so proxy-heavy charts (SRT
       allproxies) render missing copies. The swept env holds the COMPLETE
       topology, so tag its AFT fills onto the provider's tree and hand it the
       swept env as the topology source (transforms still sample the playback
       sim live - rec_ids match, so the LiveCurves resolve either way)."""
    import threading

    from analysis.games.notitg.sim.loop import LiveSim

    def worker():
        sweep_doc = doc
        if live_sim is not None:
            sweep = live_sim
            sweep_seconds = to_seconds
        else:
            sweep_doc = load_chart(sm_path)
            if sweep_doc is None:
                return
            sweep = LiveSim(sweep_doc.root, sweep_doc.to_seconds,
                            sweep_doc.start_beat, end_seconds,
                            rng_seed=sweep_doc.rng_seed,
                            song_dir=sweep_doc.lua_dir.parent,
                            use_compiled_body=_compiled_body_flag())
            sweep_seconds = sweep_doc.to_seconds
        # Chunked advance with progress to stderr: on a loaded machine
        # the sweep can take minutes, and "still compiling to X" vs
        # "broken" must be decidable from the terminal. Progress keys
        # on sweep.now, not on this thread's own advances: in-app the
        # render thread's _nudge_sweep drives the sweep while the
        # worker parks, and the sweep still has to report.
        import sys
        import time as _time
        sweep_start = _time.monotonic()
        last_print = 0.0
        lock = getattr(sweep, 'sweep_lock', None)
        while sweep.now < end_seconds:
            # While frames are rendering, the render-thread nudge owns
            # the sweep: a starved worker holding the lock through a
            # slow chunk would lock the healthy thread out of its own
            # floor (measured as a 0x inversion). The worker drives only
            # when no frame has arrived recently (launch, pause, menu).
            if (lock is not None and
                    _time.monotonic() - getattr(sweep, 'render_seen', 0.0)
                    < 0.5):
                _time.sleep(0.05)
            else:
                target = min(sweep.now + 0.25, end_seconds)
                if lock is not None:
                    with lock:
                        sweep.advance_to(target)
                    _time.sleep(0.001)
                else:
                    sweep.advance_to(target)
            if sweep.now - last_print >= 30.0:
                last_print = sweep.now
                print(f'[notitg] background compile: {sweep.now:.0f}s '
                      f'/ {end_seconds:.0f}s '
                      f'({_time.monotonic() - sweep_start:.0f}s elapsed)',
                      file=sys.stderr)
        # Re-read the declarative mods from the SWEPT env, not the load-time
        # capture: some charts populate their mods/mods2 tables from the Update
        # BODY (e.g. a beat-gated ApplyModifiers loop), so at load the table is
        # empty and the instant-compile `declarative` misses them entirely. The
        # table entries carry their own beat/time windows, so reading the fully
        # populated table once at the end preserves the time-windowing.
        swept_declarative = modfile._normalize_mod_events(
            _TableView(sweep.env), sweep_seconds)
        applied = _mod_events(_SweptResult(sweep.env.applied_mods))
        full = _compile_channels(swept_declarative + applied)
        if scroll_mult is not None:
            # The applied stream carries the chart's xmod baseline and
            # bursts (the per-frame reader re-applies them); resolve them
            # into the eased multiplier timeline the sv renderer samples.
            from analysis.games.notitg.mod_channels import (
                compile_scroll_multipliers)
            from analysis.player.render.effects.timeline import (
                EventTimeline, keyframes_from_events)
            scroll_events, _skipped = compile_scroll_multipliers(
                swept_declarative + applied)
            if scroll_events:
                keyframes = keyframes_from_events(
                    scroll_events, ('multiplier',), (1.0,))
                scroll_mult.timeline = EventTimeline(keyframes, rest=(1.0,))
        # Swap the resolved channels/players into the object the player holds.
        # ModChannels reads _channels/_players on every value() call, so the
        # single-statement rebind is atomic enough for a reader (GIL-guarded
        # dict/tuple reference swap); no half-updated state is ever observed.
        mod_channels._channels = full._channels
        mod_channels._players = full._players
        if tree:
            _freeze_static_timelines(tree, sweep.env)
        if field_provider is not None:
            # Tag AFT-rig fills on the PROVIDER's tree (the one _sim_field_
            # instances iterates) using the swept env's actor flags, then give
            # the provider the swept env as its complete topology source.
            _mark_aft_fills(field_provider._doc, sweep.env)
            field_provider.set_topology_source(sweep.env)
        if chart_shaders is not None and sweep_doc is not None:
            # The chart's Frag= passes: harvest against the swept env
            # (uniform pokes and hidden windows now recorded) and swap
            # into the live effect the renderer already holds.
            from analysis.games.notitg import shader_bridge
            actor_keyframes = {rec_id: actor.keyframes()
                               for rec_id, actor in sweep.env._actors.items()}
            entries = _chart_shaders(sweep_doc, sweep.env, actor_keyframes)
            chart_shaders.swap_passes(
                shader_bridge._build_chart_passes(entries))
        if screen_shake is not None and sweep_doc is not None:
            osc_ctx = _osc_context(sweep.env, sweep_doc, end_seconds)
            screen_shake.channels = modfile._screen_oscillator_timelines(
                sweep.env, osc_ctx)
        # Completion signal AFTER the hot-swaps: the channel swap above
        # is when driver-injected content actually appears on screen.
        print(f'[notitg] background compile done '
              f'({_time.monotonic() - sweep_start:.0f}s elapsed)',
              file=sys.stderr)
        _print_unimplemented(sweep.env, sweep_doc)

    threading.Thread(target=worker, daemon=True,
                     name='notitg-lazy-upgrade').start()


def _print_unimplemented(env, doc) -> None:
    """The chart's NOT-IMPLEMENTED report, printed once after the sweep:
    every DEFERRED verb the chart actually poked (with its documented
    reason), every silently-dropped verb, and the structural gaps
    (polygon actors, depth-capped chains). Silence here has repeatedly
    cost whole sessions of manual section-by-section checking - a gap
    the chart exercises must always announce itself."""
    import sys

    from analysis.games.notitg.sim import verb_surface

    lines = []
    for verb, count in sorted(env.deferred_verbs.items(),
                              key=lambda kv: -kv[1]):
        reason = verb_surface.DEFERRED.get(verb, '')
        lines.append(f'  DEFERRED {verb} x{count}: {reason}')
    for verb, count in sorted(env.dropped_verbs.items(),
                              key=lambda kv: -kv[1]):
        lines.append(f'  DROPPED {verb} x{count}: no dispatch matched '
                     '(unclassified - route or document it)')
    if doc is not None:
        polygons = sum(1 for a in _iter_xml(doc.root)
                       if a.kind == 'Polygon')
        if polygons:
            lines.append(f'  POLYGON actors x{polygons}: mesh tier not '
                         'built (SetDrawMode/SetNumVertices deferred)')
        aft_nodes = {a.aft_texture_name: rec_id
                     for rec_id, a in env.actors.items() if a.is_aft}
        graph = _aft_chain_graph(doc, env, aft_nodes, {})
        for name in graph.depth_capped:
            lines.append(f'  AFT chain at {name}: past MAX_CHAIN_DEPTH, '
                         'demoted to whole-screen')
    if lines:
        print('[notitg] not implemented (exercised by this chart):',
              file=sys.stderr)
        for line in lines:
            print(line, file=sys.stderr)


def _poked_props(actor) -> set:
    """Every storyboard property the swept actor ever WROTE: its recorded
    channel keys plus the transform-order channels that live outside `_frames`
    (rotation_order token / quat tuple). A property NOT in this set is never
    poked, so its LiveCurve is constant at the property's rest for all t."""
    poked = set(actor._frames)
    if getattr(actor, '_rotation_order', None) not in (None, 'ZYX'):
        poked.add('rotation_order')
    if getattr(actor, '_quat', None) is not None:
        poked.add('quat')
    return poked


def _freeze_static_timelines(tree, swept_env) -> None:
    """Replace never-poked LiveCurves in the element tree with constant
    EventTimelines (their rest), sampled once against the fully-swept env.
    Mutates each element's `timelines` dict in place (the frozen Element
    dataclass holds a mutable dict), so the live storyboard effect - which
    samples timelines every frame - sees the cheaper constants immediately."""
    from analysis.games.notitg.sim.seg_read import SegCurve
    from analysis.player.render.effects.timeline import EventTimeline
    from analysis.player.render.storyboard.model import LiveCurve

    actors = swept_env._actors
    poked_cache: dict = {}

    def poked_for(rec_id):
        if rec_id not in poked_cache:
            actor = actors.get(rec_id)
            poked_cache[rec_id] = _poked_props(actor) if actor else None
        return poked_cache[rec_id]

    def visit(elements):
        for element in elements:
            timelines = element.timelines
            for prop, curve in list(timelines.items()):
                if not isinstance(curve, (LiveCurve, SegCurve)):
                    continue
                poked = poked_for(curve._rec_id)
                # Unknown actor (never created by chart end) - leave live rather
                # than guess; freeze only the provably never-poked properties.
                if poked is not None and prop not in poked:
                    timelines[prop] = EventTimeline([], rest=curve._rest)
            if element.children:
                visit(element.children)

    visit(tree)


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
    base field: the JOINED sides (at most P1/P2 - the engine draws only
    joined players) that the chart actually mods (ApplyModifiers(str,
    pn) -> mod channels). A lone player [1] keeps the direct-draw fast
    path; two means a versus/dual layout. Players 3+ (`GetChild
    ('PlayerP3')` slots the chart also mods) are NEVER base fields: the
    screen does not draw them, they exist only as proxy sources - a
    reference chart's intro shows its extra players' content strictly
    through proxies, never a stacked center field."""
    players = {p + 1 for p in mod_channels.players if p + 1 <= 2}
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
    node_recs: dict = {}
    sampler_recs: dict = {}
    for actor in _iter_xml(doc.root):
        rec_id = env.actor_id(actor)
        sim = env.actors.get(rec_id)
        if sim is None:
            continue
        if sim.is_aft or sim.aft_source:
            rig_messages.update(actor.message_commands())
            if sim.is_aft:
                node_recs[sim.aft_texture_name] = rec_id
            if sim.aft_source:
                sampler_recs.setdefault(sim.aft_source, []).append(rec_id)
        elif actor.kind in _FILL_KINDS and not actor.children:
            fill_candidates.append((actor, rec_id))
    # A quad also belongs to a rig purely by POSITION: sandwiched
    # between an AFT node and that node's NEAREST post sampler with
    # nothing else drawable in between. gat 2's gf2_kek_black1 resets
    # the cyriak feedback base to black between node 402 and its
    # sampler blit - it carries no rig message, but the downstream
    # nodes' at-position captures must contain it (as a storyboard
    # element it draws after every capture, so the feedback loop
    # recycled the uncovered bright scene and blew out to white). The
    # adjacency requirement keeps ordinary section art out: a span
    # holding any non-frame, non-candidate actor is no curtain slot.
    candidate_recs = {rec_id: actor for actor, rec_id in fill_candidates}
    all_recs = {env.actor_id(a): a for a in _iter_xml(doc.root)
                if env.actor_id(a) in env.actors}
    for name, rec_n in node_recs.items():
        post = [r for r in sampler_recs.get(name, ()) if r > rec_n]
        if not post:
            continue
        between = [r for r in all_recs if rec_n < r < min(post)]
        if all(r in candidate_recs or all_recs[r].kind == 'ActorFrame'
               for r in between):
            for r in between:
                if r in candidate_recs:
                    candidate_recs[r]._aft_fill = True
    for actor, rec_id in fill_candidates:
        if rig_messages & set(actor.message_commands()):
            actor._aft_fill = True


def _sim_field_instances(doc, env, actor_keyframes, osc_context,
                         named_keyframes, field_oscillators,
                         mod_channels, t0, live_sim=None) -> list:
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
    node_by_id = {rec_id: name for name, rec_id in aft_nodes.items()}
    consumed_sources = {a.aft_source for a in env.actors.values()
                        if a.aft_source}
    slot_nodes = _chain_slot_nodes(chain_graph, aft_nodes, consumed_sources)
    seen_actors: dict = {}

    instances = []
    base_players = _base_players(mod_channels)
    if _multi_players(base_players):
        oscillators = field_oscillators or {}
        # The base field's transform comes from the chart's `P{n}` GLOBAL binding
        # (`P1 = self` in a field actor's InitCommand) - the exact actor eager's
        # `named_keyframes['P{n}']` came from - NOT the raw `PlayerP{n}` screen
        # child. When the chart binds P{n} to its field, that global IS the
        # screen child (gat: P1 -> screen 778, live x tracks the chart's moves).
        # When it does NOT (Ayakashi has no P1 global), eager falls to the
        # player_rest seat (+-160/center), and the screen child sits at the
        # engine's default multi-player X fan (design 288...) which is NOT where
        # the versus field renders - so lazy must fall to the seat too. Reading
        # named_actor_id (None when unbound) matches eager on both.
        for number in base_players:
            if live_sim is not None:
                rec_id = env.named_actor_id(f'P{number}')
                instances.append(field_compose.player_live_instance(
                    live_sim, number, rec_id, oscillators.get(number), t0=t0))
            else:
                instances.append(field_compose.player_instance(
                    number, named_keyframes.get(f'P{number}'),
                    oscillators.get(number), t0=t0))
    for actor in _iter_xml(doc.root):
        rec_id = env.actor_id(actor)
        sim = env.actors.get(rec_id)
        if sim is None:
            continue
        seen_actors[rec_id] = actor
        node_name = node_by_id.get(rec_id)
        if node_name is not None:
            inst = _node_instance(node_name, actor, chain_graph, slot_nodes,
                                  seen_actors, parents, env, actor_keyframes,
                                  osc_context, live_sim, t0)
            if inst is not None:
                instances.append(inst)
            continue
        aft_order = None
        aft_live = None
        color = None
        capture_source = None
        frag_path = None
        frag_uniforms = None
        if sim.aft_source:
            if actor.attrs.get('Frag') and _fullscreen_identity_draw(sim):
                # A fullscreen-identity Frag= sampler draws its capture
                # THROUGH its shader as the chart_shaders fullscreen
                # pass (see _chart_shaders); a plain blit would paper
                # the raw capture over everything drawn before it. A
                # TRANSFORMED sampler compiles no pass, so it KEEPS the
                # plain blit: the AFT curtain idiom blacks out the raw
                # scene expecting the sampler to redraw the capture on
                # top (kecak/afthell showed bare curtain without this),
                # and the raw transformed blit approximates that until
                # the per-actor shader tier draws it shaded.
                continue
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
            frag = actor.attrs.get('Frag')
            if frag:
                # A kept Frag= sampler (transformed draw): its blit runs
                # THROUGH the shader on the GL tier (raster falls back
                # to the unshaded blit). Uniforms ride their recorded
                # poke streams; the diffuse rgb tints even plain blits.
                resolved = _resolve_frag(actor, frag)
                if resolved is not None:
                    frag_path = str(resolved)
                    frag_uniforms = _uniform_curves(sim, rec_id, live_sim,
                                                    actor_keyframes,
                                                    frag_path)
            if live_sim is not None:
                from analysis.games.notitg.sim.seg_read import curve_for
                color = curve_for(live_sim, rec_id, 'color', (1.0, 1.0, 1.0))
                blend_add = curve_for(live_sim, rec_id, 'blend_add', (0.0,))
            else:
                frames = actor_keyframes.get(rec_id) or {}
                color = EventTimeline(frames.get('color', []),
                                      rest=(1.0, 1.0, 1.0))
                blend_add = EventTimeline(frames.get('blend_add', []),
                                          rest=(0.0,))
        elif sim.proxy_target in proxy_players:
            kind, player = 'proxy', proxy_players[sim.proxy_target]
        elif getattr(actor, '_aft_fill', False):
            kind, player = 'fill', 0
            if live_sim is not None:
                from analysis.games.notitg.sim.seg_read import curve_for
                color = curve_for(live_sim, rec_id, 'color', (1.0, 1.0, 1.0))
            else:
                color = EventTimeline(
                    (actor_keyframes.get(rec_id) or {}).get('color', []),
                    rest=(1.0, 1.0, 1.0))
        else:
            continue
        chain = _chain(actor, parents)
        links = [_instance_link(link_actor, env, actor_keyframes,
                                osc_context, live_sim)
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
                                             actor_keyframes, live_sim))
            elif live_sim is not None:
                pid = player_ids.get(f'PlayerP{player}')
                links.append(field_compose.player_live_link(
                    live_sim, player, pid,
                    (field_oscillators or {}).get(player),
                    ignore_hidden=True))
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
        if kind == 'aft':
            # The sampler's DIRECT source node: the entry point of the
            # stage-fold walk (a chain consumer composes every stage
            # transform down to the chain root's snapshot slot).
            inst['aft_node'] = sim.aft_source
            inst['blend_add'] = blend_add
            if frag_path is not None:
                inst['frag'] = frag_path
                inst['frag_uniforms'] = frag_uniforms
        z_group, z_link = _z_sort_group(chain, links, env)
        if z_group is not None:
            inst['z_group'] = z_group
            inst['z_sort'] = z_link['z']
        instances.append(inst)
    return instances


def _z_sort_group(chain, links, env):
    """(flagged frame rec_id, direct-child link) when the instance sits
    under a SetDrawByZPosition frame, else (None, None). The engine
    draws the flagged frame's DIRECT children stable-sorted by their z,
    ascending (ActorFrame.cpp:194-205 -> ActorUtil::SortByZPosition), so
    the sort key is the direct child on this instance's ancestor path;
    the NEAREST flagged ancestor wins for nested flags."""
    root_first = list(reversed(chain))
    for i in range(len(root_first) - 2, -1, -1):
        sim = env.actors.get(env.actor_id(root_first[i]))
        if sim is None:
            continue
        flag = sim.keyframes().get('draw_by_z')
        if flag and flag[-1].values[0] >= 0.5:
            return env.actor_id(root_first[i]), links[i + 1]
    return None, None


def _uniform_curves(sim, rec_id, live_sim, actor_keyframes,
                    frag_path) -> dict:
    """{uniform name: value timeline} from the actor's recorded
    `GetShader():uniform*` pokes. The REST value is the .frag's own
    declared initializer when it has one (GLSL 1.2 allows
    `uniform float pixelSize = 0.00001;` and NotITG honours it; the
    translation strips initializers for ES compatibility, and a 0.0
    rest turned lumikey's pre-poke frames into floor(x/0) NaN black -
    "notes only appear when the pixelation activates"), else 0.0."""
    defaults = _frag_uniform_defaults(frag_path)
    names = {prop[len('uniform:'):] for prop in sim.keyframes()
             if prop.startswith('uniform:')} | set(defaults)
    if live_sim is not None:
        from analysis.games.notitg.sim.seg_read import curve_for
        out = {}
        for name in names:
            curve = curve_for(live_sim, rec_id, f'uniform:{name}',
                              (defaults.get(name, 0.0),))
            kfs = sim.keyframes().get(f'uniform:{name}')
            if name in defaults and kfs:
                # A live lane seeds 0.0 before its first poke, ignoring
                # the passed rest - pin the declared default until then.
                curve = _RestUntil(curve, kfs[0].t, defaults[name])
            out[name] = curve
        return out
    frames = actor_keyframes.get(rec_id) or {}
    return {name: EventTimeline(frames.get(f'uniform:{name}', []),
                                rest=(defaults.get(name, 0.0),))
            for name in names}


class _RestUntil:
    """A sampleable returning `rest` before `first_t`, the wrapped curve
    after."""

    def __init__(self, curve, first_t, rest):
        self._curve = curve
        self._first_t = first_t
        self._rest = (rest,)

    def sample(self, t):
        if t < self._first_t:
            return self._rest
        return self._curve.sample(t)


@lru_cache(maxsize=64)
def _frag_uniform_defaults(frag_path) -> dict:
    """{name: value} for every scalar uniform the .frag declares WITH an
    initializer."""
    try:
        src = Path(frag_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return {}
    pattern = re.compile(
        r'^\s*uniform\s+(?:float|int)\s+([A-Za-z_]\w*)\s*=\s*'
        r'([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*;', re.MULTILINE)
    return {name: float(value) for name, value in pattern.findall(src)}


def _chain_slot_nodes(graph, aft_nodes, consumed) -> frozenset:
    """The AFT node names the composed-capture path materializes as
    at-position snapshot slots: every CONSUMED whole-screen node (plus
    every resolved chain's root and every feedback node). Isolating
    (stage) nodes never materialize - their transforms fold into
    consumers at sample time.

    At-position slots ARE the engine capture semantics: each node
    captures the composite as of its own draw position, so a multi-node
    cascade (gat 2's cyriak rig: 402 self-feeds via a pre sampler, 405
    carries it into 409, 409 self-feeds via three more, 410 feeds the
    visible copies) composes correctly - the single node-point capture
    could never contain a sampler's own blit, which is why the
    recursion rendered one level deep. A slot doubles as its node's
    previous-frame source: it updates at the node's document position,
    so a sampler drawn BEFORE the node reads last frame's content (the
    recursion/trail leg) and the snapshot then captures the composite
    including that blit."""
    stage_nodes = {name for name in aft_nodes if graph.capture_of(name)}
    roots = set()
    for name in stage_nodes:
        node = name
        while graph.capture_of(node):
            node = graph.capture_of(node)
        roots.add(node)
    screen_nodes = {name for name in consumed
                    if name in aft_nodes and name not in stage_nodes}
    return frozenset(roots | set(graph.feedback) | screen_nodes)


def _node_instance(name, actor, graph, slot_nodes, seen_actors, parents,
                   env, actor_keyframes, osc_context, live_sim, t0):
    """The composed-capture instance for one AFT node actor: 'capture'
    (an at-position snapshot slot) for chain roots and feedback nodes,
    'stage' (the captured sprite's transform, folded into consumers at
    sample time) for isolating nodes, None for nodes outside any chain
    (the legacy single-screen path). A 'capture' instance carries the
    NODE's own chain, so its hidden/alpha gate the slot update exactly
    like the engine's capture-only-while-drawn; a 'stage' instance
    carries the captured SPRITE's chain (it is what the node's texture
    contains)."""
    upstream = graph.capture_of(name)
    if upstream is not None:
        sprite = seen_actors.get(graph.stage_of(name))
        if sprite is None:
            return None
        chain = _chain(sprite, parents)
        links = [_instance_link(link_actor, env, actor_keyframes,
                                osc_context, live_sim)
                 for link_actor in reversed(chain)]
        inst = field_compose.instance(name, 'stage', 0, links, t0=t0)
        inst['source'] = upstream
        return inst
    if name not in slot_nodes:
        return None
    chain = _chain(actor, parents)
    links = [_instance_link(link_actor, env, actor_keyframes,
                            osc_context, live_sim)
             for link_actor in reversed(chain)]
    return field_compose.instance(name, 'capture', 0, links, t0=t0)


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


def _notefield_link(notefield_id, actor_keyframes, sim=None) -> dict:
    """The proxied NoteField child's own transform link, hidden pinned
    visible (proxies draw their target regardless of its hidden bit)."""
    link = field_compose.link_live_timelines(sim, notefield_id) if sim is not None \
        else field_compose.link_timelines(actor_keyframes.get(notefield_id))
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


def _instance_link(actor, env, actor_keyframes, osc_context, sim=None) -> dict:
    """One chain link: the actor's recorded transform timelines with its
    oscillator spans overlaid as LIVE delta channels, never baked - the
    vibrate mirage is per-frame randomness, so it must evaluate at frame
    time (and a long-running span never freezes at a sample cap). The
    link's own scale_x channel feeds the vibrate amplitude, as the
    engine scales the offset by the actor's zoom. `sim` (lazy) makes the
    base transform LiveCurves instead of baked keyframes."""
    rec_id = env.actor_id(actor)
    link = field_compose.link_live_timelines(sim, rec_id) if sim is not None \
        else field_compose.link_timelines(actor_keyframes.get(rec_id))
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
        if not _fullscreen_identity_draw(sim, frames):
            continue
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


# Per-channel values compatible with a fullscreen-identity draw: any
# other value means the quad covers a different region than the frame
# (rotated, scaled, skewed, cropped). base_scale_y additionally allows
# -1, the universal AFT flip-cancel idiom (basezoomy(-1) uprights the
# GL-flipped capture; our capture is already upright, so the net draw
# is identity - every corpus AFT sampler pokes it). Position channels
# are deliberately absent: a translated fullscreen quad still shows the
# whole capture, so a position shake compiles to an unshaken pass - an
# approximation, not a wrong region.
_FULLSCREEN_ALLOWED = {
    'rotation': (0.0,), 'rotation_x': (0.0,), 'rotation_y': (0.0,),
    'scale_x': (1.0,), 'scale_y': (1.0,),
    'base_scale_x': (1.0,), 'base_scale_y': (1.0, -1.0),
    'skew_x': (0.0,), 'skew_y': (0.0,),
    'crop_top': (0.0,), 'crop_bottom': (0.0,),
    'crop_left': (0.0,), 'crop_right': (0.0,),
}


def _fullscreen_identity_draw(sim, frames=None) -> bool:
    """Whether a Frag= capture sampler's DRAW stays fullscreen-identity,
    the condition for compiling it to a fullscreen pass: the pass output
    replaces the whole frame, so it is only faithful when the actor
    draws the frame back over itself unchanged. A sampler the chart
    transforms or oscillates is a picture-in-picture element; as a pass
    it would paint over the entire scene, so it is skipped - that draw
    belongs to the per-actor shader tier (composed captures).

    Examples: getfucked2's horizon sampler bounces at 0.8 zoom with
    rotation (chart t 134-161/218-260 - as a pass it blacked out both
    windows) while its monitor sampler stays untransformed and keeps
    its fullscreen pass."""
    if getattr(sim, '_osc_spans', ()) or getattr(sim, '_osc_open', None):
        return False
    if frames is None:
        # The lazy field-instance rebuild has no compiled keyframe map;
        # the sim actor's own recorded stream is the same data.
        frames = sim.keyframes()
    return not any(value not in allowed
                   for prop, allowed in _FULLSCREEN_ALLOWED.items()
                   for kf in frames.get(prop, ())
                   for value in kf.values)


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


def _osc_context(env, doc, end_seconds, clock=None):
    """The oscillator synthesis context over the sim env's spans, via
    the same modfile machinery the harvest path uses. `end_seconds` is
    the compile end still-open spans extend to. `clock` reuses a
    precomputed seconds<->beat inverter (the lazy per-frame fast path)."""
    return modfile._build_osc_context(env, doc.to_seconds, doc.start_beat,
                                      doc.lua_dir, end_seconds=end_seconds,
                                      clock=clock)


def _bg_stem(sm_path) -> str:
    return Path(modfile._sm_background_name(sm_path)).stem.casefold()
