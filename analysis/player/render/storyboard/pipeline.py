"""Opt-in end-to-end Drawable pipeline for the chart region.

Under the env flag ``VSRG_DRAWABLE_PIPELINE=1`` the renderer's per-frame
field-instance blitting (``qt_renderer._blit_field_instances``) is routed
through the game-agnostic Drawable core instead of the legacy capture
machinery:

    compiled chart --bridge.build_doc--> Evaluator (Seam A, once)
    per frame:  bridge.feed_frame --> Evaluator.frame_with_feeds (Seam B)
                --> RasterExecutor.execute --> QImage --> drawImage(chart_rect)

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
from PySide6.QtCore import QRectF

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

    @property
    def healthy(self) -> bool:
        return not self._disabled

    def delegate(self, frame, ctx, painter) -> bool:
        """Render the chart region through the Drawable core and blit it.

        Returns True when the frame was drawn (the caller must then skip
        the normal path); False when the pipeline is disabled or could
        not draw (the caller falls through unchanged). Never raises.
        """
        if self._disabled:
            return False
        try:
            return self._delegate(frame, ctx, painter)
        except Exception:
            self._disable("frame render failed")
            return False

    def _delegate(self, frame, ctx, painter) -> bool:
        if not self._ensure_built():
            return False
        t = float(ctx.t_now)
        feed_ids, counts, feed_u, feed_f = _unpack_feed(
            self._bridge.feed_frame(self._compiled, t, self._id_maps))
        image = self._evaluate(t, feed_ids, counts, feed_u, feed_f)
        if image is None:
            return False
        self._blit(painter, ctx, image)
        return True

    def _ensure_built(self) -> bool:
        """Cross Seam A once: build the doc + evaluator + executor. A
        build that raises disables the pipeline; a build that yields no
        evaluator (bridge declined) reports unavailable without a crash."""
        if self._evaluator is not None:
            return True
        evaluator, id_maps = self._bridge.build_doc(
            self._compiled, screen_w=_SCREEN_W, screen_h=_SCREEN_H)
        if evaluator is None:
            self._disable("bridge produced no evaluator")
            return False
        from analysis.player.render.storyboard.executor import RasterExecutor
        self._evaluator = evaluator
        self._id_maps = id_maps
        self._executor = RasterExecutor(
            _images_of(id_maps),
            _drawable_sizes_of(id_maps, evaluator))
        return True

    def _evaluate(self, t, feed_ids, counts, feed_u, feed_f):
        """Fold the doc + this frame's feeds into a DrawSchedule and run
        the raster executor. Returns the screen QImage or None."""
        # Feed buffers are two flat SoA arrays (u32 kinds/ids + f32 state);
        # the evaluator ingests them zero-copy per the Seam-B contract.
        raw_u, raw_f = _feed_bytes(
            feed_u, feed_f,
            self._evaluator.feed_u_stride, self._evaluator.feed_f_stride)
        u_raw, f_raw, _uf_raw, n = self._evaluator.frame_with_feeds(
            t, list(feed_ids or []), list(counts or []), raw_u, raw_f)
        u = np.frombuffer(u_raw, dtype=np.uint32).reshape(
            n, self._evaluator.u_stride)
        f = np.frombuffer(f_raw, dtype=np.float32).reshape(
            n, self._evaluator.f_stride)
        return self._executor.execute(u, f)

    def _blit(self, painter, ctx, image) -> None:
        """Blit the composed screen image into the chart rect. The screen
        drawable is 640x480 design units; drawImage stretches it into the
        real chart rectangle, exactly the mapping the legacy blit uses."""
        x, y, w, h = ctx.chart_rect
        painter.save()
        painter.drawImage(QRectF(float(x), float(y), float(w), float(h)),
                          image)
        painter.restore()

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
