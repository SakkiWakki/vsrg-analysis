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
    """True/False when the chart has a lazy sweep still short of the chart
    end, or None when it has no live sim at all (nothing to wait for).

    Reads the published frontier, which the sweep advances monotonically -
    a passive read, never a seek."""
    live = compiled.get('_live_sim') if compiled else None
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
        self._note_feed_failed = False
        self._wait_logged = False
        # Async build stages (see _ensure_built).
        self._prepare_thread = None
        self._prepared = None
        self._prepare_error = None
        self._assembly = None
        self._signature = None
        self._res_scale = None

    @property
    def healthy(self) -> bool:
        return not self._disabled

    def delegate(self, frame, ctx, painter, field_captures=None,
                 overscan=None) -> bool:
        """Render the chart region through the Drawable core and blit it.

        ``field_captures`` maps a field scope ('field', 'field2', ...) to
        the renderer's LIVE GL capture handle for that scope this frame (the
        transparent field-layers capture and any per-player captures). Each
        handle's FBO texture is bound as its field drawable's content, so
        SRC_DRAWABLE field blits draw real notes. None = no field content
        this frame (the composite still runs; field drawables read empty).

        Returns True when the frame was drawn (the caller must then skip
        the normal path); False when the pipeline is disabled or could
        not draw (the caller falls through unchanged). Never raises.
        """
        if self._disabled:
            return False
        try:
            return self._delegate(frame, ctx, painter, field_captures,
                                  overscan)
        except Exception:
            self._disable("frame render failed")
            return False

    def _delegate(self, frame, ctx, painter, field_captures,
                  overscan) -> bool:
        if not self._ensure_built(painter):
            return False
        self._rebuild_if_stale(painter)
        self._apply_resolution(ctx, painter)
        self._ingest_field_captures(field_captures, overscan,
                                    ctx.chart_rect)
        u, f = self._schedule_with_feeds(float(ctx.t_now), ctx)
        if u is None:
            return False
        # GL-ONLY present: composite onto the painter's GL target directly,
        # no QImage readback (the user directive). render_and_present returns
        # False if it could not draw (broken context, bind failure) - the
        # caller then falls through to the normal render path.
        return self._executor.render_and_present(u, f, painter, ctx.chart_rect)

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

    def _ingest_field_captures(self, field_captures, overscan=None,
                               chart_rect=None) -> None:
        """Bind each live field capture into its mapped field drawable's
        content. A scope with no drawable in the doc (or no capture this
        frame) is skipped. The captures are the renderer's GL capture
        handles (``gl_capture._GLHandle``): the GL executor binds their FBO
        textures directly (no readback), which is the GL-only app path -
        the raster executor is reference/test-only and never runs here."""
        if not field_captures:
            return
        fields = self._id_maps.get('fields') if isinstance(self._id_maps, dict) else None
        if not fields:
            return
        for scope, handle in field_captures.items():
            drawable_id = fields.get(scope)
            if drawable_id is None:
                continue
            self._bind_capture(drawable_id, handle,
                               (overscan or {}).get(scope), chart_rect)

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
        if self._evaluator is not None:
            return True
        if self._prepared is not None:
            self._start_assembly()
            return self._advance_assembly()
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
        self._start_prepare()
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

    def _start_prepare(self) -> None:
        """Kick the worker-side prepare. The signature is taken BEFORE the
        prepare so growth during it registers as stale later."""
        import threading
        signature = _topology_signature(self._compiled)

        logger.warning("DrawablePipeline: prepare started")

        def worker():
            try:
                ops, id_maps, _report = self._doc.prepare_static_doc(
                    self._compiled, screen_w=_SCREEN_W, screen_h=_SCREEN_H)
                self._prepared = (ops, id_maps, signature)
                logger.warning("DrawablePipeline: prepare finished "
                               "(%d ops); assembling over the next frames",
                               len(ops))
            except Exception as exc:  # noqa: BLE001 - surfaced via disable
                logger.warning("DrawablePipeline: prepare failed",
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
                        "DrawablePipeline: prepare still running (%.0fs; "
                        "~25s standalone on gat, longer under a live "
                        "render loop)", _time.monotonic() - t0)

        threading.Thread(target=heartbeat, daemon=True,
                         name='drawable-doc-prepare-heartbeat').start()

    # Replay budget per frame: the full replay is ~1.3s of FFI on gat 1
    # (7k ops); this bound keeps each frame's share invisible.
    _REPLAY_BUDGET_S = 0.006

    def _start_assembly(self) -> None:
        ops, id_maps, signature = self._prepared
        self._prepared = None
        self._prepare_thread = None
        import storyboard_native as sn
        builder = sn.DocBuilder(float(_SCREEN_W), float(_SCREEN_H))
        self._assembly = [builder, ops, 0, id_maps, signature]

    def _advance_assembly(self) -> bool:
        """Replay a budget's worth of recorded ops; adopt the doc when the
        replay completes. Returns True only when a doc is ready THIS frame
        (adopted now or already live with no pending swap)."""
        if self._assembly is None:
            return False
        import time as _time
        builder, ops, index, id_maps, signature = self._assembly
        deadline = _time.perf_counter() + self._REPLAY_BUDGET_S
        try:
            while index < len(ops) and _time.perf_counter() < deadline:
                method, args, kwargs = ops[index]
                getattr(builder, method)(*args, **kwargs)
                index += 1
            if index < len(ops):
                self._assembly[2] = index
                return False
            evaluator = builder.finish()
        except Exception:
            self._assembly = None
            self._disable("static-doc assembly failed")
            return False
        self._assembly = None
        self._adopt(evaluator, id_maps, signature)
        return True

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
        images = _lazy_images(id_maps)
        images.warm()
        self._executor = GLExecutor(
            images,
            _drawable_sizes_of(id_maps, evaluator),
            image_grids=(id_maps.get('image_grids')
                         if isinstance(id_maps, dict) else None))
        self._executor.set_clear(SCREEN_ID, CLEAR_TRANSPARENT)
        logger.warning(
            "DrawablePipeline: static doc LIVE (drawables=%d, fields=%s, "
            "note_feeds=%s)", evaluator.drawable_count(),
            id_maps.get('fields'), id_maps.get('note_feeds'))
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
            self._start_prepare()

    def _note_feed(self, ctx):
        """This frame's note items for the inline note slots, as
        `(slot_ids, count, u_bytes, f_bytes)`, or None when the doc has no
        slots (the captured-notefield path) or nothing is visible.

        Every slot receives the same items (one emission, shared by the
        base field and each proxy/player consumer - the consumer's Feed
        composes its own chain over them natively).

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
        u_soa, f_soa, count, _report = note_feed.feed_from_context(
            ctx, image_map)
        if not count:
            return None
        slot_ids = sorted(int(s) for s in slots.values())
        return slot_ids, int(count), u_soa.tobytes(), f_soa.tobytes()

    # The note sprites the feed resolves, as (feed key, sprite-cache name)
    # with the per-(col, state) variants the cache keys on. The receptor is
    # not a cache sprite - the field layer strokes a plain notch - so it
    # registers as a solid source the item's mat3 shapes into the lane bar.
    _NOTE_SPRITES = (
        ('tap', 'tap_head'), ('ln_head', 'ln_head'), ('ln_tail', 'ln_tail'),
        ('ln_body', 'ln_body'), ('mine', 'mine'), ('lift', 'lift'),
        ('fake', 'fake'),
    )
    _SPRITE_STATES = ('normal', 'tick', 'miss_tap', 'miss_ln', 'roll',
                      'released')

    def _ensure_note_images(self, ctx):
        """`{key: image_id}` for the note feed, registering each sprite into
        the executor's image table once.

        Every feed image declares natural size (1, 1): a fed item's mat3
        already carries its on-screen rect over a unit source box, so the
        sprite's own pixel dimensions must not scale it. Sprites are
        rasterised per (column, state) exactly as the raster cache keys them,
        so the feed resolves the same noteskin variant.

        Returns None when the sprite cache is unavailable - the caller then
        skips the feed rather than drawing wrong art."""
        if self._note_images is not None:
            return self._note_images
        images = getattr(self._executor, '_images', None)
        cache = getattr(ctx, 'sprite_cache', None)
        put = getattr(images, 'put', None)
        if put is None or cache is None:
            return None
        from PySide6.QtGui import QColor, QImage

        next_id = 1 + max((self._id_maps.get('images') or {}), default=-1)
        image_map = {}

        def register(key, image):
            nonlocal next_id
            put(next_id, image)
            self._executor._image_natural[next_id] = (1.0, 1.0)
            image_map[key] = next_id
            next_id += 1

        solid = QImage(1, 1, QImage.Format.Format_ARGB32_Premultiplied)
        solid.fill(QColor(255, 255, 255, 255))
        register('receptor', solid)

        keycount = int(getattr(ctx, 'keycount', 0) or 0)
        for key, sprite in self._NOTE_SPRITES:
            for col in range(keycount):
                for state in self._SPRITE_STATES:
                    pixmap = _sprite_image(cache, sprite, ctx, col, state)
                    if pixmap is not None:
                        register(f'{key}_{col}_{state}', pixmap)
        self._note_images = image_map
        return image_map

    def _schedule_with_feeds(self, t, ctx):
        """`_schedule`, plus this frame's inline note feed when the doc has
        one. Falls back to the plain fold on any emitter failure - a missing
        note feed degrades to an empty notefield, never a dead frame."""
        try:
            fed = self._note_feed(ctx)
        except Exception:
            self._log_note_feed_failure()
            fed = None
        if fed is None:
            return self._schedule(t)
        slots, count, u_bytes, f_bytes = fed
        u_raw, f_raw, _uf_raw, n = self._evaluator.frame_with_feeds(
            float(t), slots, [count] * len(slots),
            u_bytes * len(slots), f_bytes * len(slots))
        u = np.frombuffer(u_raw, dtype=np.uint32).reshape(
            n, self._evaluator.u_stride)
        f = np.frombuffer(f_raw, dtype=np.float32).reshape(
            n, self._evaluator.f_stride)
        return u, f

    def _log_note_feed_failure(self) -> None:
        if not self._note_feed_failed:
            self._note_feed_failed = True
            logger.warning("DrawablePipeline: note feed failed; the notefield "
                           "draws empty this session", exc_info=True)

    def _schedule(self, t):
        """Fold the static doc into a DrawSchedule at ``t``: return the
        (u, f) SoA record arrays for the executor, or (None, None) on failure.
        The static doc has no dynamic feeds - its Snapshots are static
        commands - so this is a plain ``evaluator.frame(t)`` (Seam B) with the
        Rust core sampling every channel itself."""
        u_raw, f_raw, _uf_raw, n = self._evaluator.frame(t)
        u = np.frombuffer(u_raw, dtype=np.uint32).reshape(
            n, self._evaluator.u_stride)
        f = np.frombuffer(f_raw, dtype=np.float32).reshape(
            n, self._evaluator.f_stride)
        return u, f

    def _disable(self, why: str) -> None:
        if not self._disabled:
            logger.warning(
                "DrawablePipeline disabled for the session (%s); "
                "falling back to the normal render path", why,
                exc_info=True)
        self._disabled = True


_SCREEN_W = 640
_SCREEN_H = 480


def _sprite_image(cache, name, ctx, col, state):
    """One note sprite as a QImage, or None when the cache does not carry it
    in that (col, state) combination (not every sprite has every state).

    The cache rasterises QPixmaps for the raster path; the GL executor uploads
    QImages, so this converts at the boundary."""
    try:
        pixmap = cache.get(name, ctx, col=col, state=state)
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

    def __init__(self, paths: dict[int, str]) -> None:
        self._paths = dict(paths)
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
            for image_id in list(self._paths):
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

    def _log_missing(self, image_id, path) -> None:
        if image_id in self._logged:
            return
        self._logged.add(image_id)
        logger.warning(
            "DrawablePipeline: element image %s unreadable (%s), "
            "drawing nothing", image_id, path)


def _lazy_images(id_maps):
    """A lazy image table from the static doc's ``id_maps['images']``
    ({image_id -> absolute path}). Absent -> an empty table (a doc that
    references no image sources composes fine)."""
    paths = id_maps.get('images') if isinstance(id_maps, dict) else None
    return _LazyImages(paths if isinstance(paths, dict) else {})


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
