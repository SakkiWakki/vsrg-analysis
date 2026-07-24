"""Opt-in end-to-end Drawable pipeline for the chart region.

Under the env flag ``VSRG_DRAWABLE_PIPELINE=1`` the renderer's per-frame
field-instance blitting (``qt_renderer._blit_field_instances``) is routed
through the game-agnostic Drawable core instead of the legacy capture
machinery:

    compiled chart --bridge.build_doc--> Evaluator (Seam A, once)
    per frame:  bridge.feed_frame --> Evaluator.frame_with_feeds (Seam B)
                --> GLExecutor.render_and_present --> chart_rect (GL quad)

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
``field_instances`` provider, ``base_field_hidden`` and ``_live_sim``).
That dict is exactly the ``compiled`` argument the B3 bridge's
``build_doc`` / ``feed_frame`` take, so no new plumbing is introduced -
the pipeline consumes the existing effect/adapter seam read-only.

Degradation rule (the glGenTextures lesson): the pipeline is individually
fallible and never crashes a frame. ANY exception during build or a frame
logs ONCE and permanently disables the pipeline for the rest of the
session; the caller then falls through to the normal render path.

The B3 bridge (``analysis.games.notitg.drawable_bridge``) is imported
lazily and guarded: if it is absent (built concurrently) the pipeline
reports itself unavailable and the renderer uses the normal path.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _load_bridge():
    """Return the NotITG drawable bridge module, or None if unavailable.

    B3 owns this module and may not have landed yet; a missing bridge
    means the pipeline simply degrades to the normal render path.
    """
    try:
        from analysis.games.notitg import drawable_bridge
    except Exception:
        return None
    if not (hasattr(drawable_bridge, 'build_doc')
            and hasattr(drawable_bridge, 'feed_frame')):
        return None
    return drawable_bridge


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
    native/bridge absent) is remembered as unavailable so the probe does
    not repeat every frame. Never raises.
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
    compiled document is absent, the native core is unbuilt, or the B3
    bridge has not landed - all "use the normal path" conditions, not
    failures.
    """
    bridge = _load_bridge()
    if bridge is None or not _native_available():
        return None
    compiled = _compiled_for(player)
    if not compiled or not compiled.get('field_instances'):
        return None
    return DrawablePipeline(player, compiled, bridge)


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


class DrawablePipeline:
    """Owns the lazy Seam-A build and the per-frame Seam-B -> raster ->
    blit for one player. Self-disables permanently on any error."""

    def __init__(self, player, compiled, bridge) -> None:
        self._player = player
        self._compiled = compiled
        self._bridge = bridge
        self._disabled = False
        self._evaluator = None
        self._executor = None
        self._id_maps = None
        self._res_applied = False

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
        self._apply_resolution(ctx, painter)
        self._ingest_field_captures(field_captures, overscan)
        t = float(ctx.t_now)
        feed_ids, counts, feed_u, feed_f = _unpack_feed(
            self._bridge.feed_frame(self._compiled, t, self._id_maps))
        u, f = self._schedule(t, feed_ids, counts, feed_u, feed_f)
        if u is None:
            return False
        # GL-ONLY present: composite onto the painter's GL target directly,
        # no QImage readback (the user directive). render_and_present returns
        # False if it could not draw (broken context, bind failure) - the
        # caller then falls through to the normal render path.
        return self._executor.render_and_present(u, f, painter, ctx.chart_rect)

    def _apply_resolution(self, ctx, painter) -> None:
        """Match the composite's FBO resolution to the chart rect's
        device size ONCE (before any target allocates): a 640x480-pixel
        composite stretched onto a ~1750px chart rect reads as ultra low
        res. Geometry stays logical; only allocation scales."""
        if self._res_applied:
            return
        self._res_applied = True
        try:
            dpr = float(painter.device().devicePixelRatioF())
        except Exception:
            dpr = 1.0
        chart_w = float(ctx.chart_rect[2]) * dpr
        chart_h = float(ctx.chart_rect[3]) * dpr
        self._executor.set_resolution_scale(
            max(chart_w / _SCREEN_W, chart_h / _SCREEN_H))

    def _ingest_field_captures(self, field_captures, overscan=None) -> None:
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
                               (overscan or {}).get(scope))

    def _bind_capture(self, drawable_id, handle, margins=None) -> None:
        """Bind one renderer capture handle as ``drawable_id``'s content.
        A GL capture handle resolves to (texture id, pixel w, h) and binds
        via the GL executor; None / an unresolvable handle un-binds the
        drawable so it reads empty this frame (a command-less field drawable
        carries only what is fed)."""
        resolved = _resolve_gl_texture(handle)
        if resolved is None:
            self._executor.set_drawable_texture(drawable_id, 0, 0, 0)
            return
        texture_id, w_px, h_px = resolved
        # An overscanned capture's window origin sits at (+mx, +my) in
        # the capture (qt_renderer._overscan_blit); the drawable's
        # logical box corresponds to the inset sub-rect, expressed here
        # as texture fractions off the handle's LOGICAL size.
        uv_rect = None
        mx, my = margins or (0, 0)
        lw = float(getattr(handle, 'w', 0) or 0)
        lh = float(getattr(handle, 'h', 0) or 0)
        if (mx or my) and lw > 0 and lh > 0:
            uv_rect = (mx / lw, my / lh, (lw - mx) / lw, (lh - my) / lh)
        self._executor.set_drawable_texture(drawable_id, texture_id,
                                            w_px, h_px, uv_rect)

    def _ensure_built(self, painter) -> bool:
        """Cross Seam A once: build the doc + evaluator + GL executor. GL-ONLY
        (user directive): the executor binds the renderer's capture FBO
        textures directly, so it is built only when the delegate's painter is
        on a GL engine (the ``gl_capture.usable`` test). A non-GL painter (a
        raster host, a headless frame) DISABLES the pipeline with the one-line
        log - there is no raster app path and no QImage-readback fallback. A
        build that raises disables the pipeline; a build that yields no
        evaluator (bridge declined) reports unavailable without a crash."""
        if self._evaluator is not None:
            return True
        from analysis.player.render.gl_capture import usable
        if not usable(painter):
            self._disable("painter is not on a GL engine (GL-only pipeline)")
            return False
        evaluator, id_maps = self._bridge.build_doc(
            self._compiled, screen_w=_SCREEN_W, screen_h=_SCREEN_H)
        if evaluator is None:
            self._disable("bridge produced no evaluator")
            return False
        from analysis.player.render.storyboard.gl_executor import GLExecutor
        from analysis.player.render.storyboard.executor import (
            CLEAR_TRANSPARENT, SCREEN_ID)
        self._evaluator = evaluator
        self._id_maps = id_maps
        self._executor = GLExecutor(
            _images_of(id_maps),
            _drawable_sizes_of(id_maps, evaluator))
        # The screen root is minted OpaqueBlack (DocBuilder has no clear arg;
        # that opaque clear IS the black-chart-region baseline). Make it
        # TransparentBlack so the composed screen presents OVER the backdrop
        # the renderer already painted, instead of covering it.
        self._executor.set_clear(SCREEN_ID, CLEAR_TRANSPARENT)
        # Segment and field drawables are SLICES of the one screen surface,
        # not independent screens: their doc-minted OpaqueBlack clears made
        # every fullscreen segment blit an opaque slab that buried all
        # earlier segments' content (the black chart region). They must
        # composite as transparent overlays.
        for segment_id in id_maps.get('segments') or ():
            self._executor.set_clear(segment_id, CLEAR_TRANSPARENT)
        for field_id in (id_maps.get('fields') or {}).values():
            self._executor.set_clear(field_id, CLEAR_TRANSPARENT)
        return True

    def _schedule(self, t, feed_ids, counts, feed_u, feed_f):
        """Fold the doc + this frame's feeds into a DrawSchedule: return the
        (u, f) SoA record arrays for the executor, or (None, None) on failure.
        Feed buffers are two flat SoA arrays (u32 kinds/ids + f32 state); the
        evaluator ingests them zero-copy per the Seam-B contract."""
        raw_u, raw_f = _feed_bytes(
            feed_u, feed_f,
            self._evaluator.feed_u_stride, self._evaluator.feed_f_stride)
        u_raw, f_raw, _uf_raw, n = self._evaluator.frame_with_feeds(
            t, list(feed_ids or []), list(counts or []), raw_u, raw_f)
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


def _unpack_feed(result) -> tuple:
    """Normalize the bridge's feed_frame return to
    ``(feed_ids, counts, feed_u, feed_f)``.

    The bridge returns a 5-tuple whose trailing element is a coverage dict
    (diagnostics, not needed to draw); an earlier 4-tuple form omits it.
    Either way the first four members are the two id/count lists and the
    two feed buffers (bytes or numpy arrays - see _feed_bytes). Under feed
    v2 the id/count lists are per-INTER-CAPTURE-SEGMENT (the bridge splits
    the screen's entry stream at capture positions), and the feed f32
    stride is 18 (a mat3 crossing verbatim); both are consumed generically
    here - the evaluator getters (feed_f_stride, frame_with_feeds) carry
    the stride and segment routing, so this stays layout-agnostic."""
    feed_ids, counts, feed_u, feed_f = result[0], result[1], result[2], result[3]
    return feed_ids, counts, feed_u, feed_f


def _feed_bytes(feed_u, feed_f, u_stride, f_stride) -> tuple[bytes, bytes]:
    """Feed buffers as raw bytes for ``frame_with_feeds``. The bridge may
    hand them already serialized (bytes) or as numpy SoA arrays; both are
    accepted, and an empty/None side yields a zero-row buffer of the
    frozen feed stride."""
    return (_as_feed_bytes(feed_u, u_stride, np.uint32),
            _as_feed_bytes(feed_f, f_stride, np.float32))


def _as_feed_bytes(buf, stride: int, dtype) -> bytes:
    if isinstance(buf, (bytes, bytearray, memoryview)):
        return bytes(buf)
    if buf is not None and getattr(buf, 'size', 0) > 0:
        return np.ascontiguousarray(buf, dtype=dtype).tobytes()
    return np.zeros((0, stride), dtype=dtype).tobytes()


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


def _images_of(id_maps) -> dict:
    """Image-id -> QImage source textures from the bridge's id maps.
    Absent -> empty (a doc that references no image sources draws fine)."""
    if isinstance(id_maps, dict):
        images = id_maps.get('images')
        if isinstance(images, dict):
            return images
    images = getattr(id_maps, 'images', None)
    return images if isinstance(images, dict) else {}


def _drawable_sizes_of(id_maps, evaluator) -> list:
    """Per-DrawableId logical sizes for the raster executor. The bridge
    supplies them; absent, fall back to screen-sized drawables sized from
    the evaluator's drawable_count so the screen (id 0) is at least
    640x480."""
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
