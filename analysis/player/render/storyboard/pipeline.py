"""Opt-in end-to-end Drawable pipeline for the chart region.

Under the env flag ``VSRG_DRAWABLE_PIPELINE=1`` the renderer's per-frame
field-instance blitting (``qt_renderer._blit_field_instances``) is routed
through the game-agnostic Drawable core instead of the legacy capture
machinery:

    compiled chart --drawable_doc.build_static_doc--> Evaluator (Seam A)
    per frame:  Evaluator.frame(t) (Seam B)
                --> GLExecutor.render_and_present --> chart_rect (GL quad)

STATIC TREE-ORDER DOC (this wave): the pipeline crosses Seam A with
``drawable_doc.build_static_doc(compiled)`` - the field-instance topology
compiled ONCE, in engine tree order, as an ``item_link`` channel chain the
evaluator samples itself. This is what makes storyboard BACKGROUNDS render:
the static doc bands the chart's storyboard elements (``compiled['tree']``)
around the field-instance stream and emits their sprite art as SRC_IMAGE
items. There are no dynamic segments - the doc's Snapshots are static
commands - so the per-frame path is just ``evaluator.frame(t)`` with NO
feeds, which the executor presents.

    (The feed-model sibling, ``drawable_bridge`` - build_doc + per-frame
    feed_frame - stays the reference implementation and is untouched. This
    pipeline no longer imports it.)

GL-ONLY (user directive): the executor is a ``GLExecutor`` that binds the
renderer's live capture FBO textures directly and presents the composite
onto the painter's GL target with no QImage readback. It is built only when
the delegate's painter is on a GL engine (``gl_capture.usable``); a non-GL
painter DISABLES the pipeline with a one-line log - there is no raster app
path and no readback fallback. ``RasterExecutor`` stays a reference/test
backend only, never constructed here.

DEFAULT OFF: with the flag unset the renderer never constructs this object
and behaves byte-for-byte as before. The hook in the renderer is a single
guarded delegate at the top of ``_blit_field_instances``; everything else
lives here.

Compiled-data plumbing (documented choice): the pipeline reaches the
NotITG compiled document the same way every other adapter surface does -
``player._adapter._compiled_modfile(player.replay)``, which returns the
per-replay-memoized ``compile_via_sim`` dict (the lazy document with the
``field_instances`` provider, ``base_field_hidden``, ``player_fields``,
``tree`` and ``_live_sim``). That dict is exactly the ``compiled`` argument
``build_static_doc`` takes, so no new plumbing is introduced.

Element images (this wave): ``id_maps['images']`` is
``{image_id -> absolute path}``. Each path is loaded LAZILY as a ``QImage``
and handed to the GL executor's image table (which uploads it on first use),
so SRC_IMAGE element blits draw the real sprite art. An unreadable path is
logged once and skipped - a missing image draws nothing, never a crash.

Staleness (documented choice): the compiled provider's instance list GROWS
during the lazy sim sweep (proxy/AFT binds fire as the chart plays), and the
static doc reflects a SNAPSHOT of that list. The pipeline polls a cheap
topology signature per frame - ``(instance count, last instance name)`` -
and rebuilds the doc (re-applying clears / resolution / image table) when it
changes. Count catches growth; the last name catches an in-place swap that
keeps the count (a whole-list replacement of equal length) - together they
are the cheapest signature that never misses a topology change while never
sampling every instance every frame.

Degradation rule (the glGenTextures lesson): the pipeline is individually
fallible and never crashes a frame. ANY exception during build or a frame
logs ONCE and permanently disables the pipeline for the rest of the
session; the caller then falls through to the normal render path.

The doc compiler (``analysis.games.notitg.drawable_doc``) is imported
lazily and guarded: if it is absent (built concurrently) the pipeline
reports itself unavailable and the renderer uses the normal path.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)


def _load_doc():
    """Return the NotITG static-doc compiler module, or None if unavailable.

    The module owns ``build_static_doc``; a missing one means the pipeline
    simply degrades to the normal render path.
    """
    try:
        from analysis.games.notitg import drawable_doc
    except Exception:
        return None
    if not hasattr(drawable_doc, 'prepare_static_doc'):
        return None
    return drawable_doc


def _native_available() -> bool:
    try:
        import storyboard_native  # noqa: F401
    except Exception:
        return False
    return True


_PLAYER_ATTR = '_drawable_pipeline'
_UNAVAILABLE = object()


def pipeline_for(player):
    """The Drawable pipeline for ``player``, built lazily once and cached
    on the player so it survives across frames yet is per-replay.

    Returns a healthy ``DrawablePipeline`` or None (use the normal path).
    A one-time build that yields no pipeline (wrong game, no compiled doc,
    native/doc-compiler absent) is remembered as unavailable so the probe
    does not repeat every frame. Never raises.
    """
    cached = getattr(player, _PLAYER_ATTR, None)
    if cached is _UNAVAILABLE:
        return None
    if cached is not None:
        return cached if cached.healthy else None
    try:
        built = build_pipeline(player)
    except Exception:
        built = None
    setattr(player, _PLAYER_ATTR, built if built is not None else _UNAVAILABLE)
    return built


def build_pipeline(player):
    """Construct a pipeline for ``player`` if every dependency is present,
    else None. Called at most once per player by ``pipeline_for``.

    Returns None (not an exception) when the game is not NotITG, the
    compiled document is absent, the native core is unbuilt, or the
    static-doc compiler has not landed - all "use the normal path"
    conditions, not failures.
    """
    doc = _load_doc()
    if doc is None or not _native_available():
        return None
    compiled = _compiled_for(player)
    if not compiled or not compiled.get('field_instances'):
        return None
    return DrawablePipeline(player, compiled, doc)


def _compiled_for(player):
    """The NotITG compiled document for this player, via the adapter seam,
    or None for any game/state that lacks one."""
    adapter = getattr(player, '_adapter', None)
    replay = getattr(player, 'replay', None)
    getter = getattr(adapter, '_compiled_modfile', None)
    if getter is None or replay is None:
        return None
    try:
        return getter(replay)
    except Exception:
        return None


def _topology_signature(compiled):
    """A cheap per-frame topology signature: ``(instance count, last
    instance name)``. The lazy provider's instance list grows during the
    sim sweep, so the static doc must rebuild when it changes; this pair is
    the cheapest signature that catches both growth (count) and an
    equal-length in-place swap (last name) without sampling every instance.
    Any failure to read the provider yields None (treated as unchanged)."""
    provider = compiled.get('field_instances')
    if provider is None:
        return None
    try:
        instances = list(provider() if callable(provider) else provider)
    except Exception:
        return None
    last = instances[-1].get('name') if instances else None
    return (len(instances), last)


def _sweep_in_progress(compiled):
    """True/False when the chart's lazy background upgrade has yet to land,
    or None when it has none (nothing to wait for).

    The signal is the producer's OWN handover event, not the frontier. The
    frontier reaching the chart end is necessary but not sufficient, and
    `producers._spawn_background_upgrade` says so in as many words: the
    hot-swaps happen after it, and one of them hands the provider the swept
    env - the COMPLETE proxy/AFT topology, which is the thing the static doc
    is built out of. Gating on the frontier started the first prepare on
    whatever had bound by the playhead, and the handover then grew the list
    and forced a rebuild: gat 1 built its doc twice, gat 2 three times, each
    one the full channel export.

    A never-signalled event reads as still in progress, which is the safe
    direction - the pipeline waits and the legacy path draws."""
    if not compiled:
        return None
    event = compiled.get('_upgrade_done')
    if event is not None:
        return not event.is_set()
    live = compiled.get('_live_sim')
    if live is None:
        return None
    try:
        return float(live.frontier) < float(live._end_seconds)
    except Exception:
        return None


class DrawablePipeline:
    """Owns the lazy Seam-A build (the static tree-order doc) and the
    per-frame Seam-B -> GL present for one player. Self-disables permanently
    on any error; rebuilds the doc when the provider's topology grows."""

    def __init__(self, player, compiled, doc) -> None:
        self._player = player
        self._compiled = compiled
        self._doc = doc
        self._disabled = False
        self._evaluator = None
        self._executor = None
        self._id_maps = None
        # The rebuild settle gate's tracking (see _rebuild_if_stale).
        self._settle_sig = None
        self._settle_since = 0.0
        self._signature_cache = None
        self._signature_polled = 0.0
        self._topology_final = False
        self._note_images = None
        self._note_generation = None
        self._note_feed_failed = False
        self._wait_logged = False
        # Async build stages (see _ensure_built).
        self._prepare_thread = None
        self._prepared = None
        self._prepare_error = None
        # Numbers the prepares in the log: a run does several, and without a
        # number the interleaved start/heartbeat/finish lines could not be
        # matched up to each other.
        self._prepare_n = 0
        # One-shot: a scope the doc declares but the renderer never feeds.
        self._unfed_logged = False
        self._live_prepare_n = 0
        self._assembly = None
        self._assembly_last_advance = None
        self._assembly_logged = 0.0
        self._signature = None
        self._res_scale = None

    @property
    def healthy(self) -> bool:
        return not self._disabled

    @property
    def doc_live(self) -> bool:
        """Whether an assembled doc is drawing frames RIGHT NOW. False all
        through prepare + assembly, when the legacy path still owns the
        frame - the renderer must keep drawing everything it normally
        would until this flips."""
        return self._evaluator is not None

    def delegate(self, frame, ctx, painter, field_captures=None,
                 overscan=None, per_player=None) -> bool:
        """Render the chart region through the Drawable core and blit it.

        ``field_captures`` maps a field scope ('field', 'field2', ...) to
        the renderer's LIVE GL capture handle for that scope this frame (the
        transparent field-layers capture and any per-player captures). Each
        handle's FBO texture is bound as its field drawable's content, so
        SRC_DRAWABLE field blits draw real notes. None = no field content
        this frame (the composite still runs; field drawables read empty).

        ``per_player`` drives the inline note feed: the pipeline hands it an
        emitter and it calls that once per player, each time with a context
        speaking that player's mods (see `_note_feed`). None emits once, for
        the ctx as given - every single-player chart.

        Returns True when the frame was drawn (the caller must then skip
        the normal path); False when the pipeline is disabled or could
        not draw (the caller falls through unchanged). Never raises.
        """
        if self._disabled:
            return False
        try:
            return self._delegate(frame, ctx, painter, field_captures,
                                  overscan, per_player)
        except Exception:
            self._disable("frame render failed")
            return False

    def _delegate(self, frame, ctx, painter, field_captures,
                  overscan, per_player=None) -> bool:
        if not self._ensure_built(painter):
            return False
        self._rebuild_if_stale(painter)
        self._apply_resolution(ctx, painter)
        self._ingest_field_captures(field_captures, overscan,
                                    ctx.chart_rect)
        u, f, uf = self._schedule_with_feeds(float(ctx.t_now), ctx,
                                             per_player)
        if u is None:
            return False
        # GL-ONLY present: composite onto the painter's GL target directly,
        # no QImage readback (the user directive). render_and_present returns
        # False if it could not draw (broken context, bind failure) - the
        # caller then falls through to the normal render path.
        return self._executor.render_and_present(u, f, painter,
                                                ctx.chart_rect, uf=uf)

    def _apply_resolution(self, ctx, painter) -> None:
        """Match the composite's FBO resolution to the chart rect's device
        size (before any target allocates): a 640x480-pixel composite
        stretched onto a ~1750px chart rect reads as ultra low res. Geometry
        stays logical; only allocation scales. Kept idempotent (the executor
        re-applies the same scale cheaply) so a doc rebuild - which drops the
        allocated targets - restores it on the next frame."""
        try:
            dpr = float(painter.device().devicePixelRatioF())
        except Exception:
            dpr = 1.0
        chart_w = float(ctx.chart_rect[2]) * dpr
        chart_h = float(ctx.chart_rect[3]) * dpr
        scale = max(chart_w / _SCREEN_W, chart_h / _SCREEN_H)
        if self._res_scale is not None and abs(scale - self._res_scale) < 1e-6:
            return
        self._res_scale = scale
        self._executor.set_resolution_scale(scale)

    def capture_scopes(self) -> frozenset:
        """The field-capture scopes this doc BINDS - empty when it draws its
        notes as inline items and reads no capture at all.

        The renderer renders the whole field layer group into a capture for
        these. With none of them, that render is work nobody reads: the
        notes are fed as items, and the group's other members (judgments,
        press marks, miss X, arrowpaths) go into a texture that is never
        bound. Asking here rather than guessing keeps the two sides' scope
        vocabularies in one place - they have disagreed silently before
        (`_report_unfed_scopes`)."""
        fields = (self._id_maps.get('fields')
                  if isinstance(self._id_maps, dict) else None)
        return frozenset(fields or ())

    def _ingest_field_captures(self, field_captures, overscan=None,
                               chart_rect=None) -> None:
        """Bind each live field capture into its mapped field drawable's
        content. A scope with no drawable in the doc (or no capture this
        frame) is skipped. The captures are the renderer's GL capture
        handles (``gl_capture._GLHandle``): the GL executor binds their FBO
        textures directly (no readback), which is the GL-only app path -
        the raster executor is reference/test-only and never runs here."""
        fields = self._id_maps.get('fields') if isinstance(self._id_maps, dict) else None
        if not fields:
            return
        for scope, handle in (field_captures or {}).items():
            drawable_id = fields.get(scope)
            if drawable_id is None:
                continue
            self._bind_capture(drawable_id, handle,
                               (overscan or {}).get(scope), chart_rect)
        self._report_unfed_scopes(field_captures or {}, fields)

    def _report_unfed_scopes(self, field_captures, fields) -> None:
        """Name, once, any field scope the DOC declares that the renderer
        never hands a capture for.

        Such a drawable carries only what is bound, so it reads EMPTY and its
        copies draw nothing - a whole section can go black with every
        transform correct and nothing else in the log. The doc's scope set and
        the renderer's capture set are produced by different code paths
        (`drawable_doc._field_scope` against `qt_renderer
        ._capture_second_field`), so they can disagree silently."""
        if self._unfed_logged:
            return
        unfed = sorted(s for s in fields
                       if field_captures.get(s) is None)
        if not unfed:
            return
        self._unfed_logged = True
        logger.warning(
            "DrawablePipeline: the doc declares field scope(s) %s that the "
            "renderer fed no capture for - those copies draw EMPTY. Fed this "
            "frame: %s", unfed,
            sorted(s for s, h in field_captures.items() if h is not None)
            or 'none')

    def _bind_capture(self, drawable_id, handle, margins=None,
                      chart_rect=None) -> None:
        """Bind one renderer capture handle as ``drawable_id``'s content.
        A GL capture handle resolves to (texture id, pixel w, h) and binds
        via the GL executor; None / an unresolvable handle un-binds the
        drawable so it reads empty this frame (a command-less field drawable
        carries only what is bound)."""
        resolved = _resolve_gl_texture(handle)
        if resolved is None:
            self._executor.set_drawable_texture(drawable_id, 0, 0, 0)
            return
        texture_id, w_px, h_px = resolved
        # The capture is WINDOW-sized plus overscan margins, with its
        # window origin at (+mx, +my) (qt_renderer._begin_field_capture:
        # `open(slot, painter, p.W + 2mx, p.H + 2my)` then
        # `translate(mx, my)`). The drawable's logical box corresponds to
        # the CHART RECT within that window - margins-only mapping
        # compressed the sidebar into the field box (the off-center bug).
        uv_rect = None
        mx, my = margins or (0, 0)
        lw = float(getattr(handle, 'w', 0) or 0)
        lh = float(getattr(handle, 'h', 0) or 0)
        if lw > 0 and lh > 0:
            cx, cy, cw, chh = (chart_rect if chart_rect is not None
                               else (0.0, 0.0, lw - 2 * mx, lh - 2 * my))
            uv_rect = ((mx + cx) / lw, (my + cy) / lh,
                       (mx + cx + cw) / lw, (my + cy + chh) / lh)
            if uv_rect == (0.0, 0.0, 1.0, 1.0):
                uv_rect = None
        self._executor.set_drawable_texture(drawable_id, texture_id,
                                            w_px, h_px, uv_rect)

    def _ensure_built(self, painter) -> bool:
        """Seam A without freezing a single frame. Three async stages:

        1. PREPARE (worker thread): `prepare_static_doc` - the tens of
           seconds of pure-Python channel export - recorded as plain ops
           (no PyO3 objects cross threads; the Evaluator is unsendable).
        2. REPLAY (render thread, BUDGETED): the recorded ops replay onto
           a real DocBuilder a few milliseconds per frame.
        3. ADOPT: finish() -> evaluator, mint the GL executor, apply
           clears; the pipeline starts (or resumes) drawing.

        Until adoption the delegate returns False and the normal path
        draws - the pipeline takes over seamlessly when ready. GL-ONLY:
        a non-GL painter disables the pipeline."""
        if self._advance_assembly():
            return True
        if self._assembly is not None:
            # A replay is in flight. It must finish before ANYTHING else
            # starts - falling through here treated "no prepare thread" as
            # "nothing happening" and spawned a duplicate 'superseded'
            # prepare EVERY assembly window; each duplicate's result then
            # sat in `_prepared` unconsumed (the live-doc early return sat
            # above the check), which also blocked every future rebuild.
            # The live doc, when there is one, keeps drawing meanwhile.
            return self._evaluator is not None
        if self._prepared is not None:
            # Consume a finished prepare BEFORE the live-doc early return,
            # or a supersede's result is never assembled.
            self._start_assembly()
            if self._advance_assembly():
                return True
            return self._evaluator is not None
        if self._evaluator is not None:
            return True
        if self._prepare_thread is not None and self._prepare_thread.is_alive():
            return False
        if self._prepare_error is not None:
            self._disable(self._prepare_error)
            return False
        from analysis.player.render.gl_capture import usable
        if not usable(painter):
            self._disable("painter is not on a GL engine (GL-only pipeline)")
            return False
        # The FIRST prepare also waits for the settle gate: its worker-side
        # dense channel sampling reads (and can advance) the LIVE sim, and
        # doing that while the background sweep thread is still driving the
        # same sim STALLED the sweep (the user's vanished 'background
        # compile' progress). Post-sweep the static timelines are frozen
        # and worker reads are safe - so no prepare starts until the
        # topology signature has stopped changing.
        if not self._signature_settled():
            if not self._wait_logged:
                self._wait_logged = True
                logger.warning("DrawablePipeline: waiting for the sweep to "
                               "settle before the first prepare (legacy "
                               "path draws until then)")
            return False
        self._start_prepare('first build' if self._prepare_n == 0
                             else 'the previous doc was superseded')
        return False

    def _signature_settled(self) -> bool:
        """True once it is safe and worthwhile to build the static doc.

        The real gate is SWEEP COMPLETION. A prepare dense-samples the lazy
        curves in a tight Python loop, which starves the sweep thread of the
        GIL - the sweep's progress stalls mid-chart - and the topology it
        would capture is incomplete until the sweep ends anyway, so a
        mid-sweep build is both harmful and wasted. Waiting for the frontier
        yields exactly one build, on complete data.

        Producers with no live sweep (any non-lazy chart) expose no sim; those
        fall back to the topology settle window."""
        sweeping = _sweep_in_progress(self._compiled)
        if sweeping is not None:
            return not sweeping
        return self._settled(self._poll_signature())

    def _settled(self, signature) -> bool:
        """Track `signature` against the settle window: any change restarts the
        clock, and only an unchanged signature that has aged past
        `_REBUILD_SETTLE_S` reports settled."""
        import time as _time
        now = _time.monotonic()
        if signature != self._settle_sig:
            self._settle_sig = signature
            self._settle_since = now
            return False
        return now - self._settle_since >= self._REBUILD_SETTLE_S

    def _poll_signature(self):
        """The provider's topology signature, refreshed at most every
        `_SIGNATURE_POLL_S`.

        Asking the provider is NOT free: the lazy NotITG provider runs a
        wall-budgeted slice of the sim sweep on the render thread and may
        rebuild its instance list, so polling it per frame doubles that work
        (the effect already calls it once itself). Settling is measured in
        seconds, so a few polls per second detects it just as fast."""
        import time as _time
        now = _time.monotonic()
        if now - self._signature_polled < self._SIGNATURE_POLL_S:
            return self._signature_cache
        self._signature_polled = now
        self._signature_cache = _topology_signature(self._compiled)
        return self._signature_cache

    def _start_prepare(self, why: str) -> None:
        """Kick the worker-side prepare. The signature is taken BEFORE the
        prepare so growth during it registers as stale later.

        `why` names the reason, because a run logs several of these and
        nothing said which was the first build and which was a rebuild the
        chart's own topology growth forced."""
        import threading
        signature = _topology_signature(self._compiled)
        self._prepare_n += 1
        run = self._prepare_n

        logger.warning("DrawablePipeline: prepare #%d started (%s)", run, why)

        def worker():
            try:
                ops, id_maps, _report = self._doc.prepare_static_doc(
                    self._compiled, screen_w=_SCREEN_W, screen_h=_SCREEN_H)
                self._prepared = (ops, id_maps, signature, run)
                logger.warning("DrawablePipeline: prepare #%d finished "
                               "(%d ops); assembling over the next frames",
                               run, len(ops))
            except Exception as exc:  # noqa: BLE001 - surfaced via disable
                logger.warning("DrawablePipeline: prepare #%d failed", run,
                               exc_info=True)
                self._prepare_error = f'static-doc prepare failed ({exc})'

        self._prepare_thread = threading.Thread(
            target=worker, daemon=True, name='drawable-doc-prepare')
        self._prepare_thread.start()

        def heartbeat(thread=self._prepare_thread):
            import time as _time
            t0 = _time.monotonic()
            while thread.is_alive():
                _time.sleep(10.0)
                if thread.is_alive():
                    logger.warning(
                        "DrawablePipeline: prepare #%d still running (%.0fs; "
                        "it runs on a worker thread, so the chart keeps "
                        "drawing through the previous doc)",
                        run, _time.monotonic() - t0)

        threading.Thread(target=heartbeat, daemon=True,
                         name='drawable-doc-prepare-heartbeat').start()

    # Replay budget per frame: the full replay is ~1.3s of FFI on gat 1
    # (7k ops); this bound keeps each frame's share invisible.
    _REPLAY_BUDGET_S = 0.006
    # The budget SCALES with the observed frame period, because until the
    # doc is adopted every frame pays the legacy path - the slow one the
    # pipeline exists to replace. A fixed 6ms is 40% of a 60fps frame but
    # 2% of the 300ms frames a heavy chart draws at, which stretched a
    # 20k-op assembly across ~600 frames: minutes of wall time stuck at
    # the slow rate, adoption receding as the frames it needs get rarer.
    # A quarter of the frame converges instead - the slower the legacy
    # frames, the harder the replay pushes to escape them.
    _REPLAY_BUDGET_SHARE = 0.25
    _REPLAY_BUDGET_MAX_S = 0.120

    def _start_assembly(self) -> None:
        ops, id_maps, signature, run = self._prepared
        # Which prepare this doc came from. A later prepare is usually
        # already running by now, so `_prepare_n` names the wrong one.
        self._live_prepare_n = run
        self._prepared = None
        self._prepare_thread = None
        import storyboard_native as sn
        builder = sn.DocBuilder(float(_SCREEN_W), float(_SCREEN_H))
        self._assembly = [builder, ops, 0, id_maps, signature]
        self._assembly_last_advance = None
        self._assembly_logged = 0.0

    def _advance_assembly(self) -> bool:
        """Replay a budget's worth of recorded ops; adopt the doc when the
        replay completes. Returns True only when a doc is ready THIS frame
        (adopted now or already live with no pending swap)."""
        if self._assembly is None:
            return False
        import time as _time
        builder, ops, index, id_maps, signature = self._assembly
        now = _time.perf_counter()
        deadline = now + self._replay_budget(now)
        try:
            while index < len(ops) and _time.perf_counter() < deadline:
                method, args, kwargs = ops[index]
                getattr(builder, method)(*args, **kwargs)
                index += 1
            if index < len(ops):
                self._assembly[2] = index
                self._log_assembly_progress(index, len(ops))
                return False
            evaluator = builder.finish()
        except Exception:
            self._assembly = None
            self._disable("static-doc assembly failed")
            return False
        self._assembly = None
        self._adopt(evaluator, id_maps, signature)
        return True

    def _replay_budget(self, now: float) -> float:
        """This frame's replay slice: a share of the observed frame period
        (the gap since the last advance), floored and capped. See
        `_REPLAY_BUDGET_SHARE`."""
        last = self._assembly_last_advance
        self._assembly_last_advance = now
        if last is None:
            return self._REPLAY_BUDGET_S
        return min(max(self._REPLAY_BUDGET_S,
                       (now - last) * self._REPLAY_BUDGET_SHARE),
                   self._REPLAY_BUDGET_MAX_S)

    def _log_assembly_progress(self, index: int, total: int) -> None:
        """A progress line every ~10s of ongoing assembly, so a slow
        adoption reads as work advancing rather than as a prepare that
        never ends (the heartbeat covers only the worker-side prepare)."""
        import time as _time
        now = _time.monotonic()
        if now - self._assembly_logged < 10.0:
            return
        self._assembly_logged = now
        logger.warning("DrawablePipeline: assembling prepare #%d - "
                       "%d/%d ops replayed (frame-budgeted)",
                       self._live_prepare_n, index, total)

    def _adopt(self, evaluator, id_maps, signature) -> None:
        """Swap the freshly assembled doc in: mint the GL executor, apply
        clears, force resolution + bindings to re-apply."""
        from analysis.player.render.storyboard.gl_executor import GLExecutor
        from analysis.player.render.storyboard.executor import (
            CLEAR_TRANSPARENT, SCREEN_ID)
        self._evaluator = evaluator
        self._id_maps = id_maps
        self._signature = signature
        # A doc assembled from a finished sweep is built on final topology:
        # nothing can grow behind it, so staleness polling can stop.
        self._topology_final = _sweep_in_progress(self._compiled) is False
        self._note_images = None
        self._note_generation = None
        images = _lazy_images(id_maps)
        images.warm()
        self._executor = GLExecutor(
            images,
            _drawable_sizes_of(id_maps, evaluator),
            image_grids=(id_maps.get('image_grids')
                         if isinstance(id_maps, dict) else None),
            image_specs=(id_maps.get('image_specs')
                         if isinstance(id_maps, dict) else None),
            meshes=(id_maps.get('meshes')
                    if isinstance(id_maps, dict) else None),
            note_samplers=(_note_sampler_map(id_maps)
                           if isinstance(id_maps, dict) else None))
        # The DOC says what its own screen surface is. A game whose doc holds
        # the whole scene declares an opaque clear (NotITG: the engine clears
        # its framebuffer black, and an AFT capture of a transparent screen
        # comes back a cutout); a game whose doc is an overlay on a scene
        # someone else paints stays transparent, which is the default.
        self._executor.set_clear(
            SCREEN_ID,
            (id_maps.get('screen_clear', CLEAR_TRANSPARENT)
             if isinstance(id_maps, dict) else CLEAR_TRANSPARENT))
        # Per-item `Frag=` programs, positional by the shader id the doc's
        # BLIT lanes carry. Without this the executor's table is empty, every
        # shader lane resolves to None, and a shaded sampler silently blits
        # through the default textured program.
        self._executor.set_shaders(
            (id_maps.get('shaders') or []) if isinstance(id_maps, dict) else [])
        # Says which prepare's doc went live: assembly is spread over frames,
        # so a later prepare is usually already RUNNING by the time an earlier
        # one is adopted, and the two lines interleave.
        logger.warning(
            "DrawablePipeline: prepare #%d now LIVE (drawables=%d, "
            "per-player fields=%s, note feeds=%s)", self._live_prepare_n,
            evaluator.drawable_count(),
            id_maps.get('fields') or 'none', id_maps.get('note_feeds'))
        # Field drawables are SLICES of the one screen surface: transparent
        # overlays, never opaque slabs (the black-chart-region fix). AFT slots
        # are NOT included: they are minted persistent (ClearMode::Retain) and
        # must keep their captured content across frames - clearing them would
        # destroy the engine capture/decay semantics.
        for field_id in (id_maps.get('fields') or {}).values():
            self._executor.set_clear(field_id, CLEAR_TRANSPARENT)
        self._res_scale = None

    # The settle gate: re-prepares wait until the topology signature is
    # stale AND has stopped changing (the sweep's churn ends) - combined
    # with the async prepare + budgeted replay, no rebuild ever blocks.
    _REBUILD_SETTLE_S = 2.0

    # How often the topology signature may be re-read (see _poll_signature).
    # Well under the settle window, so settling is still detected promptly.
    _SIGNATURE_POLL_S = 0.25

    def _rebuild_if_stale(self, painter) -> None:
        """Start a background re-prepare once the provider's topology
        signature has changed AND settled. The current doc keeps drawing
        until the replacement is assembled and adopted."""
        if self._evaluator is None:
            return
        if self._prepare_thread is not None and self._prepare_thread.is_alive():
            return
        if self._assembly is not None or self._prepared is not None:
            return
        if self._topology_final:
            # The live doc was built from post-sweep data, so the topology
            # cannot change again and there is nothing to detect. Asking is
            # not free (see _poll_signature), and at ~200ms per call on a
            # full topology it is the periodic frame-rate collapse - so once
            # the answer is knowably fixed, stop asking.
            return
        if _sweep_in_progress(self._compiled):
            # Same reason the first build waits: a rebuild mid-sweep starves
            # the sweep and captures topology that is still growing.
            return
        signature = self._poll_signature()
        if signature is None or signature == self._signature:
            return
        if self._settled(signature):
            self._start_prepare('the chart bound more proxies/AFTs')

    def _note_feed(self, ctx, per_player=None):
        """This frame's note items for the inline note slots, as
        `(slot_ids, counts, u_bytes, f_bytes)` with the items concatenated in
        slot order - or None when the doc has no slots (the
        captured-notefield path) or nothing is visible anywhere.

        ONE EMISSION PER PLAYER. `per_player` is the caller's driver (see
        `qt_renderer._per_player_notes`): it calls the emitter once per
        player, each time with a context speaking that player's mods, and
        returns `{scope: emission}`. A scope with no emission of its own
        falls back to the primary field's - a proxy of player 1, and every
        single-player chart, is exactly that case. Without the split every
        slot got player 1's arrows, so two independently-modded fields drew
        the same notes.

        Reuses the render context the caller already built - the emitter is a
        pure read of the prepass, so no second pass over the notes."""
        id_maps = self._id_maps or {}
        slots = id_maps.get('note_feeds')
        if not slots:
            slot = id_maps.get('notes_slot')
            slots = {'field': slot} if slot is not None else {}
        if not slots:
            return None
        image_map = self._ensure_note_images(ctx)
        if image_map is None:
            return None
        from analysis.player.render.storyboard import note_feed

        shader_ctx = self._note_shader_ctx(float(getattr(ctx, 't_now', 0.0)))

        def emit(c, number=1):
            # The ctx's field geometry is in SCREEN px (the adapter already
            # stretched its design grid onto the chart rect); this doc's
            # screen is the design box the executor stretches at present
            # time, so the feed must convert or the stretch lands twice.
            u_soa, f_soa, x_soa, count, _report = note_feed.feed_from_context(
                c, image_map, design=(_SCREEN_W, _SCREEN_H),
                note_shader=shader_ctx(number))
            return int(count), u_soa, f_soa.tobytes(), x_soa

        # An emission costs a whole per-player prepass rebuild, so ask the
        # doc which slots' consumers can DRAW right now and pay only for
        # those. A chart runs with up to 8 player fields but shows a
        # couple at a time; emitting for hidden ones was most of the
        # frame's Python (measured ~9ms/frame on gat 2, 5 players).
        live = self._live_feed_slots(ctx)
        wanted = frozenset(scope for scope, slot in slots.items()
                           if live is None or int(slot) in live)
        emissions = per_player(emit, wanted) if per_player is not None \
            else ({_DEFAULT_SCOPE: emit(ctx)} if _DEFAULT_SCOPE in wanted
                  or live is None else {})
        primary = emissions.get(_DEFAULT_SCOPE)
        if live is None and not any(e[0] for e in emissions.values()):
            # Without liveness the all-empty frame is indistinguishable
            # from a broken emitter, so decline it (the legacy path draws).
            # WITH liveness, empty is an answer: nothing on screen wants
            # notes, and the doc still owns the frame.
            return None
        empty = (0, None, b'', None)
        slot_ids, counts, u_parts, f_parts, x_parts = [], [], [], [], []
        x_base = 0
        for scope, slot in sorted(slots.items(), key=lambda kv: int(kv[1])):
            if scope in wanted:
                count, u_soa, f_bytes, x_soa = emissions.get(scope) \
                    or primary or empty
            else:
                count, u_soa, f_bytes, x_soa = empty
            slot_ids.append(int(slot))
            counts.append(count)
            f_parts.append(f_bytes)
            if count and x_soa is not None and len(x_soa):
                # Each emission's x-window offsets are local to its own
                # buffer; the evaluator sees ONE concatenated buffer, so
                # rebase them here (a primary emission reused by several
                # slots gets a fresh copy per slot).
                rows = u_soa.reshape(count, note_feed.FEED_U_STRIDE).copy()
                stamped = rows[:, 6] > 0
                rows[stamped, 5] += x_base
                u_parts.append(rows.tobytes())
                x_parts.append(x_soa.tobytes())
                x_base += len(x_soa)
            else:
                u_parts.append(u_soa.tobytes() if u_soa is not None else b'')
        return (slot_ids, counts, b''.join(u_parts), b''.join(f_parts),
                b''.join(x_parts))

    def _note_shader_ctx(self, t):
        """A per-player factory for this frame's note-shader stamp:
        `shader_ctx(number)` -> the `note_shader` dict
        `note_feed.feed_from_context` takes, or None while that player
        has no live bind. Chart uniform values are sampled here, once
        per player per frame - a handful of scalar curves."""
        note_shaders = (self._id_maps or {}).get('note_shaders')

        def dark(_number):
            return None

        if not note_shaders:
            return dark
        to_beat = note_shaders.get('to_beat')

        def shader_ctx(number):
            curves = note_shaders['players'].get(number)
            if not curves:
                return None
            active = {}
            for category, curve in curves.items():
                src = int(round(float(curve.sample(t)[0])))
                entry = (note_shaders['sources'].get(src)
                         if src >= 0 else None)
                if entry is None:
                    continue
                values = [float(entry['uniforms'][name].sample(t)[0])
                          for name in entry['uniform_names']]
                active[category] = (entry['shader_plus_one'], values)
            if not active:
                return None
            active['time'] = t
            active['beat'] = float(to_beat(t)) if to_beat is not None else 0.0
            active['to_beat'] = to_beat
            return active

        return shader_ctx

    def _live_feed_slots(self, ctx):
        """The doc's feed slots whose consumers can draw at the ctx's time,
        or None when the evaluator predates the query (feed everything -
        correct, just unpriced)."""
        query = getattr(self._evaluator, 'live_feed_slots', None)
        if query is None:
            return None
        return frozenset(int(s) for s in query(float(ctx.t_now)))

    # The note sprites the feed resolves, as (feed key, sprite-cache name).
    # The receptor is not a cache sprite - the field layer strokes a plain
    # notch - so it registers as a solid source the item's mat3 shapes into
    # the lane bar.
    _NOTE_SPRITES = (
        ('tap', 'tap_head'), ('ln_head', 'ln_head'), ('ln_tail', 'ln_tail'),
        ('ln_body', 'ln_body'), ('mine', 'mine'), ('lift', 'lift'),
        ('fake', 'fake'), ('ghost_tap', 'ghost_tap'),
    )
    _SPRITE_STATES = ('normal', 'tick', 'miss_tap', 'miss_ln', 'roll',
                      'released')

    def _ensure_note_images(self, ctx):
        """`{key: (image_id, w, h)}` for the note feed, registering each
        sprite into the executor's image table once.

        Every feed image declares natural size (1, 1): a fed item's mat3
        already carries its on-screen rect over a unit source box, so the
        sprite's own pixel dimensions must not scale it. Sprites are
        rasterised per (column, state) exactly as the raster cache keys them,
        so the feed resolves the same noteskin variant.

        The `(w, h)` the map carries is that rasterised pixmap's DESIGN SIZE,
        which is what the raster path blits it at (`notes._blit_lane_pixmap`
        draws the pixmap at its own dimensions, centred on the note's y). It
        has to travel with the id rather than be recomputed by the emitter:
        a sprite's box is skin-dependent (a circle head is `lane_w` square, a
        bar head is `note_h + 2 * HEAD_PAD` tall) and an adapter may replace
        the specs outright via `GameAdapter.note_sprites`, so the rasterised
        pixmap is the only honest source.

        Returns None when the sprite cache is unavailable - the caller then
        skips the feed rather than drawing wrong art."""
        generation = _sprite_generation(ctx)
        if self._note_images is not None and generation == self._note_generation:
            return self._note_images
        images = getattr(self._executor, '_images', None)
        cache = getattr(ctx, 'sprite_cache', None)
        put = getattr(images, 'put', None)
        if put is None or cache is None:
            return None
        from PySide6.QtGui import QColor, QImage

        # The cache just dropped its pixmaps (resize / skin toggle / palette
        # fade), so the ids minted below must re-upload rather than hit the
        # executor's texture cache holding the previous art.
        self._drop_note_textures()

        next_id = 1 + max((self._id_maps.get('images') or {}), default=-1)
        image_map = {}

        def register(key, image, size=None):
            nonlocal next_id
            put(next_id, image)
            self._executor._image_natural[next_id] = (1.0, 1.0)
            image_map[key] = (next_id, *(size or (image.width(), image.height())))
            next_id += 1

        # Neither the receptor notch nor a stroke is a cache sprite - the
        # raster field draws them with a brush - so both are the same white
        # solid, stretched to the rect the emitter names and tinted to the
        # colour it wants.
        solid = QImage(1, 1, QImage.Format.Format_ARGB32_Premultiplied)
        solid.fill(QColor(255, 255, 255, 255))
        register('receptor', solid, size=(1.0, 1.0))
        register('solid', solid, size=(1.0, 1.0))

        keycount = int(getattr(ctx, 'keycount', 0) or 0)
        for key, sprite in self._NOTE_SPRITES:
            for name, kwargs in _sprite_variants(cache, key, sprite, keycount,
                                                 self._SPRITE_STATES):
                pixmap = _sprite_image(cache, sprite, ctx, kwargs)
                if pixmap is not None:
                    register(name, pixmap)
        for name, kwargs in _judgment_variants(ctx):
            pixmap = _sprite_image(cache, 'miss_x', ctx, kwargs)
            if pixmap is not None:
                register(name, pixmap)
        self._note_images = image_map
        self._note_generation = generation
        return image_map

    def _drop_note_textures(self) -> None:
        """Forget the GPU textures behind the previously registered note
        sprites, so re-registering the same ids uploads the new art."""
        textures = getattr(self._executor, '_image_textures', None)
        for entry in (self._note_images or {}).values():
            if textures is not None:
                textures.pop(entry[0], None)

    def _schedule_with_feeds(self, t, ctx, per_player=None):
        """`_schedule`, plus this frame's inline note feed when the doc has
        one.

        A doc with NO feed slots draws its notes from a captured notefield, so
        the plain fold is the whole frame. A doc that HAS slots and gets no
        items is a different thing entirely: folding anyway composites a
        notefield with no notes in it, and because the slots are inline there
        is no captured field underneath to show through - the notes simply
        vanish, silently, for as long as the emitter keeps failing. Declining
        the frame instead (returning None) drops through to the legacy path in
        `_blit_field_instances`, which draws real notes. A visibly older
        rendering beats a chart with no notes."""
        try:
            fed = self._note_feed(ctx, per_player)
        except Exception:
            self._log_note_feed_failure()
            fed = None
        if fed is None:
            if self._has_note_slots():
                return None, None, None
            return self._schedule(t)
        slots, counts, u_bytes, f_bytes, x_bytes = fed
        u_raw, f_raw, uf_raw, n = self._evaluator.frame_with_feeds(
            float(t), slots, counts, u_bytes, f_bytes, x_bytes)
        u = np.frombuffer(u_raw, dtype=np.uint32).reshape(
            n, self._evaluator.u_stride)
        f = np.frombuffer(f_raw, dtype=np.float32).reshape(
            n, self._evaluator.f_stride)
        return u, f, np.frombuffer(uf_raw, dtype=np.float32)

    def _has_note_slots(self) -> bool:
        """Whether the doc draws its notes as INLINE fed items. When it does,
        an empty feed means an empty notefield with nothing behind it."""
        id_maps = self._id_maps or {}
        return bool(id_maps.get('note_feeds')
                    or id_maps.get('notes_slot') is not None)

    def _log_note_feed_failure(self) -> None:
        if not self._note_feed_failed:
            self._note_feed_failed = True
            logger.warning("DrawablePipeline: note feed failed; the notefield "
                           "draws empty this session", exc_info=True)

    def _schedule(self, t):
        """Fold the static doc into a DrawSchedule at ``t``: return the
        (u, f, uf) SoA record arrays for the executor. The static doc has no
        dynamic feeds - its Snapshots are static commands - so this is a plain
        ``evaluator.frame(t)`` (Seam B) with the Rust core sampling every
        channel itself.

        ``uf`` is the flat per-BLIT shader-uniform window each op indexes by
        its (offset, count) lanes. Dropping it leaves every chart uniform at
        its GLSL zero, which is not merely unstyled: monitor.frag divides by
        `fAmt`, so a zeroed uniform renders NaN black rather than a plain
        copy."""
        u_raw, f_raw, uf_raw, n = self._evaluator.frame(t)
        u = np.frombuffer(u_raw, dtype=np.uint32).reshape(
            n, self._evaluator.u_stride)
        f = np.frombuffer(f_raw, dtype=np.float32).reshape(
            n, self._evaluator.f_stride)
        return u, f, np.frombuffer(uf_raw, dtype=np.float32)

    def _disable(self, why: str) -> None:
        if not self._disabled:
            logger.warning(
                "DrawablePipeline disabled for the session (%s); "
                "falling back to the normal render path", why,
                exc_info=True)
        self._disabled = True


_SCREEN_W = 640
_SCREEN_H = 480

# The primary field's scope: player 1's notes, and the fallback emission for
# any slot with none of its own.
_DEFAULT_SCOPE = 'field'


def _sprite_generation(ctx):
    """The sprite cache's invalidation generation, paired with the keycount.

    The pipeline keeps its own copy of every rasterised sprite - uploaded as a
    GL texture, with the design box it draws at - so it has to re-read whenever
    the cache drops its pixmaps. `NoteSpriteCache.invalidate` is the one choke
    point for all of those (a resize, a SKIN TOGGLE, a palette fade), so its
    generation is the whole signal; watching `(lane_w, note_h)` instead misses
    a skin toggle entirely, which rasterises a different shape at the very same
    geometry. The keycount rides along because it decides how many per-column
    variants get registered at all."""
    cache = getattr(ctx, 'sprite_cache', None)
    return (getattr(cache, 'generation', 0), getattr(ctx, 'keycount', None))


def _judgment_variants(ctx):
    """The `(image_map name, cache kwargs)` pairs for the overlays that
    rasterise IN a judgment's colour rather than tinting to it.

    The miss X is a red outline plus an X in the judgment colour - two
    colours, so one tinted sprite cannot serve every judgment. The set is
    small and fixed (one per judgment window), and the emitter names the
    variant by the judgment's own name."""
    colors = getattr(getattr(ctx, 'player', None), 'judge_colors', None) or {}
    for judgment, color in colors.items():
        yield f'miss_x_{judgment}', {'jcolor': color}


def _sprite_variants(cache, key, sprite, keycount, states):
    """The `(image_map name, cache kwargs)` pairs to register for one sprite.

    Each sprite declares which key fields it varies on (`SpriteSpec.key_fields`)
    and they genuinely differ: a mine is one glyph for the whole field, a lift
    varies by column, a head varies by column AND judgment state, and an LN body
    additionally by roll-ness. Enumerating a fixed column x state grid for all of
    them registers every sprite under a name the emitter never looks up (a mine
    became `mine_0_normal` while `note_feed._lookup_sprite` asks for `mine`) and
    raises `KeyError('is_roll')` on the body - so mines, lifts, fakes and every
    LN body silently never drew.

    `is_roll` is derived rather than enumerated: the emitter's `_tail_state`
    already encodes roll-ness as the 'roll' state, so the two never disagree."""
    fields = _sprite_key_fields(cache, sprite)
    cols = range(keycount) if 'col' in fields else (None,)
    for col in cols:
        for state in (states if 'state' in fields else (None,)):
            kwargs = {}
            if col is not None:
                kwargs['col'] = col
            if state is not None:
                kwargs['state'] = state
            if 'is_roll' in fields:
                kwargs['is_roll'] = state == 'roll'
            name = key
            if col is not None:
                name += f'_{col}'
            if state is not None:
                name += f'_{state}'
            yield name, kwargs


def _sprite_key_fields(cache, sprite) -> tuple:
    """The key fields `sprite` varies on, or the head-like default when the
    cache does not expose its specs (a stubbed cache in a test)."""
    specs = getattr(cache, '_specs', None)
    spec = specs.get(sprite) if isinstance(specs, dict) else None
    return tuple(getattr(spec, 'key_fields', ('col', 'state')))


def _sprite_image(cache, name, ctx, kwargs):
    """One note sprite as a QImage, or None when the cache cannot produce it
    (not every sprite has every state).

    The cache rasterises QPixmaps for the raster path; the GL executor uploads
    QImages, so this converts at the boundary."""
    try:
        pixmap = cache.get(name, ctx, **kwargs)
    except Exception:
        return None
    if pixmap is None or pixmap.isNull():
        return None
    return pixmap.toImage()


def _resolve_gl_texture(handle):
    """Resolve a renderer capture handle to (texture id, pixel w, h) for GL
    binding, or None when it carries no live GL texture this frame.

    The renderer's GL capture backend hands ``gl_capture._GLHandle`` objects:
    ``.fbo`` is a ``QOpenGLFramebufferObject`` whose ``.texture()`` is the
    live capture texture, and ``.fbo.width()/.height()`` its pixel size.
    (After source normalization only the aspect matters, but the pixel size
    is the honest source dimension.) A None handle, a non-GL handle (a raster
    QPixmap - never expected on the GL-only app path), or an invalid FBO all
    resolve to None -> the drawable un-binds and reads empty this frame."""
    if handle is None:
        return None
    fbo = getattr(handle, 'fbo', None)
    if fbo is None:
        return None
    texture = fbo.texture()
    if not texture:
        return None
    return int(texture), int(fbo.width()), int(fbo.height())


class _LazyImages:
    """The GL executor's image table backed by ``{image_id -> path}``,
    loading each path as a ``QImage`` on first ``.get`` and caching it.

    The GL executor consumes its ``images`` map by ``self._images.get(id)``
    -> a QImage it uploads once; this table defers the file read until a
    SRC_IMAGE blit actually asks for the id (many charts reference art that
    never draws in a given run) and NEVER crashes on a bad path: an
    unreadable / null image logs once and resolves to None, exactly the
    "missing image draws nothing" degradation the executor already handles.

    Only ``.get`` is used by the executor; the other read paths are provided
    for API symmetry with a plain dict."""

    def __init__(self, paths: dict[int, str], texts: dict | None = None) -> None:
        self._paths = dict(paths)
        # image id -> (string, pixel size) for `text` elements. The doc
        # compiler cannot lay these out - it runs on a worker thread and stays
        # Qt-free - so it records the spec and the raster happens here.
        self._texts = dict(texts or {})
        self._cache: dict[int, object] = {}
        self._logged: set[int] = set()

    def get(self, image_id, default=None):
        if image_id in self._cache:
            image = self._cache[image_id]
            return default if image is None else image
        image = self._load(image_id)
        self._cache[image_id] = image
        return default if image is None else image

    def put(self, image_id, image) -> None:
        """Register an already-built QImage under `image_id` (no path). Used
        for the note sprites a per-frame feed references, which are rasterised
        by the sprite cache rather than read from disk."""
        self._cache[int(image_id)] = image

    def warm(self) -> None:
        """Decode every path on a daemon thread so `get` is a cache hit.

        Without this the first draw of each image pays a disk read + decode
        ON THE RENDER THREAD, inside the paint bracket - and elements enter
        their time windows throughout the chart, so the hitches recur as new
        art appears rather than being a one-off cost at load.

        QImage decoding is not GUI-thread-bound (unlike QPixmap), so this is
        safe off-thread. The cache is plain dict get/set, atomic under the
        GIL: a race can only decode the same image twice, never corrupt."""
        import threading

        def worker():
            # Text ids too: laying out and rasterising a caption is the same
            # kind of first-draw hitch a disk decode is, and QPainter onto a
            # QImage is no more GUI-bound than QImage decoding.
            for image_id in [*self._paths, *self._texts]:
                if image_id not in self._cache:
                    self._cache[image_id] = self._load(image_id)

        threading.Thread(target=worker, daemon=True,
                         name='drawable-image-warm').start()

    def __contains__(self, image_id) -> bool:
        return image_id in self._paths

    def __len__(self) -> int:
        return len(self._paths)

    # SM pseudo-assets: a Texture="white"/"black" names a solid color,
    # not a file (gat's curtains and glow quads use them). Synthesized as
    # tiny solid images instead of warning-and-skipping.
    _SOLID_COLORS = {'white': (255, 255, 255), 'black': (0, 0, 0)}

    def _load(self, image_id):
        import os

        from PySide6.QtGui import QColor, QImage

        spec = self._texts.get(image_id)
        if spec is not None:
            return self._render_text(*spec)
        path = self._paths.get(image_id)
        if not path:
            return None
        stem = os.path.splitext(os.path.basename(str(path)))[0].lower()
        if stem in self._SOLID_COLORS and not os.path.isfile(str(path)):
            image = QImage(8, 8, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(QColor(*self._SOLID_COLORS[stem]))
            return image
        try:
            image = QImage(str(path))
        except Exception:
            image = None
        if image is None or image.isNull():
            self._log_missing(image_id, path)
            return None
        return image

    def _render_text(self, text: str, font_px: float):
        """A `text` element's glyphs as a WHITE-on-transparent QImage sized
        exactly `(bounding width, metrics height)` with the baseline at
        `ascent` - the box legacy's `_element_size` reports and draws into.

        White, not the element's colour: the item's tint multiplies it, which
        is what legacy's `painter.setPen(color)` does, and it lets one raster
        serve a caption that recolours over time."""
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter

        font = QFont()
        font.setPixelSize(max(1, int(font_px or 32)))
        metrics = QFontMetricsF(font)
        width = max(1, int(round(metrics.boundingRect(text).width())))
        height = max(1, int(round(metrics.height())))
        image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 255))
        painter.drawText(QPointF(0.0, metrics.ascent()), text)
        painter.end()
        return image

    def _log_missing(self, image_id, path) -> None:
        if image_id in self._logged:
            return
        self._logged.add(image_id)
        logger.warning(
            "DrawablePipeline: element image %s unreadable (%s), "
            "drawing nothing", image_id, path)


def _lazy_images(id_maps):
    """A lazy image table from the static doc's ``id_maps['images']``
    ({image_id -> absolute path}) plus ``id_maps['text_images']``
    ({image_id -> (string, pixel size)}), which this table rasterises itself.
    Absent -> an empty table (a doc that references no image sources composes
    fine)."""
    maps = id_maps if isinstance(id_maps, dict) else {}
    paths = maps.get('images')
    texts = maps.get('text_images')
    return _LazyImages(paths if isinstance(paths, dict) else {},
                       texts if isinstance(texts, dict) else {})


def _note_sampler_map(id_maps) -> dict | None:
    """{shader id: {sampler name: file path}} for the doc's note-shader
    sources (their uniformTexture file binds), or None."""
    note_shaders = id_maps.get('note_shaders')
    if not note_shaders:
        return None
    return {entry['shader_plus_one'] - 1: entry['samplers']
            for entry in note_shaders['sources'].values()
            if entry.get('samplers')}


def _drawable_sizes_of(id_maps, evaluator) -> list:
    """Per-DrawableId logical sizes for the executor, from the doc's
    ``drawable_sizes`` table (each drawable is only as big as it needs to
    be). A producer that exports no table falls back to screen-sized
    drawables sized from the evaluator's drawable_count."""
    sizes = None
    if isinstance(id_maps, dict):
        sizes = id_maps.get('drawable_sizes')
    else:
        sizes = getattr(id_maps, 'drawable_sizes', None)
    if sizes:
        return [(float(w), float(h)) for (w, h) in sizes]
    return [(float(_SCREEN_W), float(_SCREEN_H))] * _drawable_count(evaluator)


def _drawable_count(evaluator) -> int:
    """The doc's drawable count (a method on the native Evaluator). Falls
    back to 1 when absent so the screen (id 0) always has a size."""
    attr = getattr(evaluator, 'drawable_count', None)
    count = attr() if callable(attr) else attr
    try:
        return max(1, int(count))
    except (TypeError, ValueError):
        return 1
