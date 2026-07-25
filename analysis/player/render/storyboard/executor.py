"""Seam-B raster executor: a QPainter/QImage consumer of the
``DrawSchedule`` record stream produced by ``storyboard_native``.

The evaluator (Rust) folds the drawable tree and sampled channels into a
flat, fully-resolved op list crossing Seam B as two fixed-stride SoA
buffers (u32 records + f32 records). This module is the reference raster
backend: it walks those records with painter's-algorithm semantics onto
per-drawable ``QImage`` targets, mapping each target's logical units to
device pixels exactly once. It never sees a timeline, a tree, or a game.

Record layout: `storyboard.record` is the one Python statement of it, and
this module reads its lane offsets from there. Restating them here is how
this backend silently fell behind the GL one - it kept a 10/20 stride and
its own offsets while the doc grew origin, absolute size, scale-to-fit and
edge-fade lanes, so it drew every element top-left-anchored at its natural
size.

  BEGIN: a = drawable id, b = clear mode
  BLIT:  a = source kind, b = source id, c = frame index;
         shader+1/clip+1 are 1-based (0 = none); uf_offset/uf_count
         window into the third `uf` buffer of sampled uniform values.
  COPY:  a = destination drawable id (source is the OPEN target)
  END:   a = drawable id

Clip shapes and polyline vertices arrive out-of-band via the ctor
(``clips`` mirrors the doc's ClipDesc table; ``lines`` seeds polyline
sources, updated per frame by ``set_lines``). Shader uniform values
ride the ``uf`` buffer passed to ``execute``; the executor stashes the
last-seen dict per shader id on ``shader_uniforms`` (raster draws
unshaded but binds the uniforms).

Clear modes match ClearMode in doc.rs: TransparentBlack=0, OpaqueBlack=1,
Retain=2. Source kinds and op kinds are the module constants exported by
storyboard_native (SRC_*, OP_*).

Conventions (drawable-ir.md Seam B): sequential, painter's algorithm,
premultiplied alpha, y-down. Persistent (Retain) targets keep their
image across execute() calls - that is how feedback / AFT preservation
works. A COPY duplicates the CURRENT in-progress target image into the
named drawable (at-position Snapshot semantics), which is why sampling a
snapshot slot reads pre-curtain content, not a later fullscreen fill.
"""
from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF, QTransform

from analysis.player.render.storyboard import record as _rec

logger = logging.getLogger(__name__)

_SCREEN_ID = 0

# Lane offsets come from the record mirror, never restated here.
_U_KIND, _U_A, _U_B, _U_C = _rec.U_KIND, _rec.U_A, _rec.U_B, _rec.U_C
_U_BLEND, _U_SHADER, _U_CLIP = _rec.U_BLEND, _rec.U_SHADER, _rec.U_CLIP
_U_SCREEN_SPACE = _rec.U_SCREEN_SPACE
_U_UF_OFFSET, _U_UF_COUNT = _rec.U_UF_OFFSET, _rec.U_UF_COUNT
_F_OPACITY, _F_TINT, _F_CROP = _rec.F_OPACITY, _rec.F_TINT, _rec.F_CROP
_F_ORIGIN, _F_FADE = _rec.F_ORIGIN, _rec.F_FADE
_draw_box = _rec.draw_box

# Polyline stroke width, in the target's logical units (drawable-ir.md
# B2: fixed for now; a future width channel would replace this).
_LINES_WIDTH = 3.0

# Op / source / clear codes (kept in sync with storyboard_native; the
# module also exports them, but the executor must run without importing
# it at module load so tests can importorskip cleanly).
class _Op:
    """Op-kind codes as dotted names so `match` can use value patterns."""

    BEGIN, BLIT, COPY, END = 0, 1, 2, 3


_OP_BEGIN, _OP_BLIT, _OP_COPY, _OP_END = _Op.BEGIN, _Op.BLIT, _Op.COPY, _Op.END
_SRC_IMAGE, _SRC_DRAWABLE, _SRC_MESH, _SRC_FILL, _SRC_LINES = 0, 1, 2, 3, 4
_CLEAR_TRANSPARENT, _CLEAR_OPAQUE, _CLEAR_RETAIN = 0, 1, 2

# Public aliases for callers that drive set_clear / target the screen root
# (the pipeline overrides the screen's clear). Mirror ClearMode / the root
# DrawableId without reaching for the private names.
CLEAR_TRANSPARENT, CLEAR_OPAQUE, CLEAR_RETAIN = (
    _CLEAR_TRANSPARENT, _CLEAR_OPAQUE, _CLEAR_RETAIN)
SCREEN_ID = _SCREEN_ID

_QT_BLEND = {
    0: QPainter.CompositionMode.CompositionMode_SourceOver,  # Blend::SourceOver
    1: QPainter.CompositionMode.CompositionMode_Plus,        # Blend::Additive
}


def _new_image(w: int, h: int) -> QImage:
    img = QImage(max(1, w), max(1, h), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    return img


class RasterExecutor:
    """Runs DrawSchedule records onto retained per-drawable QImages.

    ``images`` maps ImageId -> QImage source textures. ``drawable_sizes``
    is indexed by DrawableId and gives each drawable's logical size; the
    executor allocates a device-pixel target of that integer size and
    maps logical -> device once (1 logical unit = 1 device px here - the
    reference backend keeps them equal; a scaled backend would differ
    only in this mapping).
    """

    def __init__(
        self,
        images: dict[int, QImage],
        drawable_sizes: list[tuple[float, float]],
        clips: list[tuple] | None = None,
        lines: dict[int, np.ndarray] | None = None,
    ) -> None:
        self._images = images
        self._sizes = drawable_sizes
        self._targets: dict[int, QImage] = {}
        # Per-drawable clear-mode overrides (drawable_id -> ClearMode code).
        # The doc's BEGIN carries a clear mode, but a caller with no clear
        # arg at doc-build (DocBuilder exposes none) sets it here; the
        # override wins over the record's mode. The pipeline uses this to
        # make the screen composite TRANSPARENT so its blit overlays the
        # normally-painted backdrop instead of covering it opaque black.
        self._clear_override: dict[int, int] = {}
        # Per-drawable retain-decay factors (DrawableId -> alpha kept per
        # frame). Engine PreserveTexture ACCUMULATES WITH DECAY: a Retain
        # drawable's retained content fades toward zero each frame rather
        # than persisting forever (the ghost-trail smear). At each execute()
        # a Retain BEGIN pre-multiplies the surviving content by the factor
        # BEFORE this frame's new content lands on top. Absent / 1.0 = no
        # decay (today's persist-forever behavior); 0.0 = one-frame content.
        self._decay: dict[int, float] = {}
        # Per-frame injected content: DrawableId -> QImage source, applied
        # into the target table at execute() start so SRC_DRAWABLE blits of
        # command-less drawables (the field-scope drawables the pipeline
        # feeds the renderer's live field capture into) read real pixels.
        # `_last_injected` tracks the ids seeded last run so an un-seeded id
        # drops its stale target instead of retaining it (command-less
        # field drawables are non-persistent - they carry only what is fed).
        self._injected: dict[int, QImage] = {}
        self._last_injected: set[int] = set()
        self._skipped_lanes: set[str] = set()
        # Clip shapes mirror the doc's ClipDesc table, indexed by ClipId
        # (the record carries clip+1 on lane 6). Prebuild the QPainterPath
        # for each shape once; a clipped blit intersects the target's own
        # bounds with it in TARGET logical space.
        self._clip_paths: list[QPainterPath] = [
            _clip_path(shape) for shape in (clips or [])
        ]
        # Polyline sources keyed by LinesId (the record's source id on
        # lane B). Vertices are (n, 2) float arrays in the source's own
        # logical units; set_lines() swaps them per frame (travelpath).
        self._lines: dict[int, np.ndarray] = dict(lines or {})
        # Last-seen sampled uniform VALUES per shader id, introspectable
        # after execute(): {shader_id: [f32, ...]} in binding order.
        self.shader_uniforms: dict[int, list[float]] = {}

    def set_lines(self, lines_id: int, verts: np.ndarray) -> None:
        """Update a polyline source's vertices (n, 2) in its own logical
        units. Travelpaths feed a new array each frame; a subsequent
        execute() strokes the latest one under the op's transform."""
        self._lines[int(lines_id)] = np.asarray(verts, dtype=np.float32).reshape(-1, 2)

    def set_clear(self, drawable_id: int, mode: int) -> None:
        """Override the clear mode a drawable's BEGIN op applies, keyed by
        DrawableId. `mode` is a ClearMode code (TransparentBlack=0,
        OpaqueBlack=1, Retain=2). Set once and it holds across execute()
        calls; the override wins over the record's own clear.

        This is the pipeline-side knob the spec calls for: DocBuilder
        exposes no clear arg, so the screen root (id 0) is minted with the
        engine-AFT OpaqueBlack default. The pipeline sets it TransparentBlack
        so the composed screen image overlays the normally-painted backdrop
        rather than covering it with opaque black (the black-chart-region
        baseline)."""
        self._clear_override[int(drawable_id)] = int(mode)

    def set_decay(self, drawable_id: int, factor_per_frame: float) -> None:
        """Set a Retain drawable's per-frame decay factor, keyed by
        DrawableId. Each execute() a Retain BEGIN pre-multiplies the
        surviving content's alpha (premultiplied: every channel) by
        ``factor_per_frame`` before this frame's content composes on top, so
        undrawn content fades geometrically toward transparent instead of
        persisting forever (the engine PreserveTexture accumulate-with-decay
        semantics; the ghost-trail smear). 1.0 = no decay (today's
        behavior); 0.0 = content lasts a single frame. Only affects Retain
        BEGINs - Transparent/Opaque clears wipe the surface regardless."""
        self._decay[int(drawable_id)] = max(0.0, min(1.0, float(factor_per_frame)))

    def set_drawable_image(self, drawable_id: int, image: QImage) -> None:
        """Seed a drawable's target content with `image`, keyed by
        DrawableId. Applied at the next execute()'s start (and every one
        after, until re-set), so a SRC_DRAWABLE blit of `drawable_id` reads
        these pixels.

        This is how the pipeline hands the renderer's LIVE field capture
        (the transparent field-layers pixmap, plus per-player field{N}
        captures) into the doc's command-less field drawables: those
        drawables carry no items of their own, so without injected content a
        field-scope blit would resolve to nothing. `image` is a QImage in
        the drawable's logical size; a None un-seeds it."""
        if image is None:
            self._injected.pop(int(drawable_id), None)
        else:
            self._injected[int(drawable_id)] = image

    def execute(
        self,
        u: np.ndarray,
        f: np.ndarray,
        uf: np.ndarray | None = None,
    ) -> QImage:
        """Run the records; return the screen drawable's image.

        Retained targets survive across calls, so calling execute() again
        with the same doc reuses last frame's content for any Retain
        drawable (feedback).

        ``uf`` is the schedule's third flat buffer of sampled shader
        uniform VALUES; a BLIT's lanes 8/9 give an (offset, count) window
        into it. Passing it lets the executor stash each shaded op's
        uniforms per shader id (introspectable via ``shader_uniforms``);
        the raster reference backend still draws unshaded.
        """
        u = np.ascontiguousarray(u, dtype=np.uint32)
        f = np.ascontiguousarray(f, dtype=np.float32)
        uf = None if uf is None else np.ascontiguousarray(uf, dtype=np.float32)
        # Seed injected content BEFORE the walk so command-less field
        # drawables (never a BEGIN target) carry this frame's live capture
        # for any SRC_DRAWABLE blit that reads them. An id seeded last run
        # but not this one drops its stale target - a command-less field
        # drawable is non-persistent, so an unfed scope reads empty.
        for drawable_id in self._last_injected - self._injected.keys():
            self._targets.pop(drawable_id, None)
        for drawable_id, image in self._injected.items():
            self._targets[drawable_id] = image
        self._last_injected = set(self._injected.keys())
        target_stack: list[int] = []
        painter: QPainter | None = None

        for i in range(u.shape[0]):
            match int(u[i, _U_KIND]):
                case _Op.BEGIN:
                    painter = self._begin(u[i], target_stack, painter)
                case _Op.BLIT:
                    self._blit(u[i], f[i], target_stack, painter, uf)
                case _Op.COPY:
                    self._copy(u[i], target_stack)
                case _Op.END:
                    painter = self._end(target_stack, painter)

        if painter is not None:
            painter.end()
        return self._targets.setdefault(_SCREEN_ID, self._alloc(_SCREEN_ID))

    def _alloc(self, drawable_id: int) -> QImage:
        w, h = self._sizes[drawable_id]
        return _new_image(int(round(w)), int(round(h)))

    def _begin(
        self,
        rec: np.ndarray,
        target_stack: list[int],
        painter: QPainter | None,
    ) -> QPainter:
        if painter is not None:
            painter.end()
        drawable_id = int(rec[_U_A])
        clear = self._clear_override.get(drawable_id, int(rec[_U_B]))
        img = self._targets.get(drawable_id)
        if img is None:
            img = self._alloc(drawable_id)
            self._targets[drawable_id] = img
        if clear == _CLEAR_TRANSPARENT:
            img.fill(0)
        elif clear == _CLEAR_OPAQUE:
            img.fill(QColor(0, 0, 0, 255))
        else:
            # Retain: keep existing content, but fade it by the per-drawable
            # decay factor first (accumulate-with-decay). DestinationIn with a
            # solid src of alpha=factor multiplies every premultiplied channel
            # by factor, so a factor of 1.0 is a no-op (persist forever).
            self._decay_retained(img, self._decay.get(drawable_id, 1.0))

        target_stack.append(drawable_id)
        new_painter = QPainter(img)
        new_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        return new_painter

    def _decay_retained(self, img: QImage, factor: float) -> None:
        """Fade a Retain surface's premultiplied content toward transparent
        by ``factor`` in place (DestinationIn multiplies dest by src alpha).
        1.0 is a no-op, so the common no-decay path pays nothing."""
        if factor >= 1.0:
            return
        fade = QPainter(img)
        fade.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        fade.fillRect(img.rect(), QColor.fromRgbF(0.0, 0.0, 0.0, factor))
        fade.end()

    def _end(
        self,
        target_stack: list[int],
        painter: QPainter | None,
    ) -> QPainter | None:
        if painter is not None:
            painter.end()
        if target_stack:
            target_stack.pop()
        # Re-open the enclosing target so a following op (rare - schedules
        # are flat) lands on it. Flat schedules end with an empty stack.
        if target_stack:
            img = self._targets[target_stack[-1]]
            return QPainter(img)
        return None

    def _copy(self, rec: np.ndarray, target_stack: list[int]) -> None:
        into = int(rec[_U_A])
        if not target_stack:
            return
        source_id = target_stack[-1]
        source = self._targets.get(source_id)
        if source is None:
            return
        # COPY duplicates the CURRENT in-progress target. Deep-copy so the
        # snapshot slot is frozen at this command index and unaffected by
        # later drawing onto the still-open source target.
        self._targets[into] = source.copy()

    def _blit(
        self,
        urec: np.ndarray,
        frec: np.ndarray,
        target_stack: list[int],
        painter: QPainter | None,
        uf: np.ndarray | None,
    ) -> None:
        if painter is None or not target_stack:
            return
        self._stash_uniforms(urec, uf)
        self._log_skipped_lanes(urec, frec)

        src_kind = int(urec[_U_A])
        opacity = float(frec[_F_OPACITY])
        tint = (float(frec[_F_TINT]), float(frec[_F_TINT + 1]), float(frec[_F_TINT + 2]))
        blend = _QT_BLEND.get(int(urec[_U_BLEND]), _QT_BLEND[0])

        painter.save()
        # The clip is in the TARGET's logical space, so apply it under the
        # identity transform (before the op's mat3), then install the op's
        # transform for the source geometry.
        self._apply_clip(painter, urec)
        painter.setTransform(self._qtransform(frec))
        painter.setOpacity(max(0.0, min(1.0, opacity)))
        painter.setCompositionMode(blend)

        match src_kind:
            case n if n == _SRC_FILL:
                self._draw_fill(painter, frec, self._sizes[target_stack[-1]],
                                tint)
            case n if n == _SRC_IMAGE:
                self._draw_image(painter, urec, frec, tint)
            case n if n == _SRC_DRAWABLE:
                self._draw_drawable(painter, urec, frec, tint)
            case n if n == _SRC_LINES:
                self._draw_lines(painter, urec, tint)
            # Mesh sources are geometry-bearing through a projection; the
            # raster reference backend leaves them for a later pass.
        painter.restore()

    def _apply_clip(self, painter: QPainter, urec: np.ndarray) -> None:
        clip_plus_one = int(urec[_U_CLIP])
        if clip_plus_one == 0:
            return
        clip_id = clip_plus_one - 1
        if not 0 <= clip_id < len(self._clip_paths):
            if "clip_missing" not in self._skipped_lanes:
                self._skipped_lanes.add("clip_missing")
                logger.warning("RasterExecutor: clip id %d has no shape, drawn unclipped", clip_id)
            return
        # setClipPath maps through the painter's current transform. The
        # shape is in target logical units, so clip while still at the
        # identity target transform (setTransform for the source has not
        # been called yet at this point).
        painter.setClipPath(self._clip_paths[clip_id])

    def _stash_uniforms(self, urec: np.ndarray, uf: np.ndarray | None) -> None:
        shader_plus_one = int(urec[_U_SHADER])
        count = int(urec[_U_UF_COUNT])
        if shader_plus_one == 0 or count == 0 or uf is None:
            return
        offset = int(urec[_U_UF_OFFSET])
        if offset + count > uf.shape[0]:
            if "uf_range" not in self._skipped_lanes:
                self._skipped_lanes.add("uf_range")
                logger.warning("RasterExecutor: uniform window out of range, ignored")
            return
        self.shader_uniforms[shader_plus_one - 1] = [
            float(v) for v in uf[offset:offset + count]
        ]

    def _qtransform(self, frec: np.ndarray) -> QTransform:
        # mat3 is row-major [m00 m01 m02; m10 m11 m12; 0 0 1]. QTransform's
        # constructor is column-order (m11=xx, m12=xy, m21=yx, m22=yy,
        # dx, dy) with the maps p' = p * M, so QTransform(m00, m10, m01,
        # m11, m02, m12) applies the same affine as the row-major mat3.
        m = frec
        return QTransform(
            float(m[0]), float(m[3]),
            float(m[1]), float(m[4]),
            float(m[2]), float(m[5]),
        )

    def _draw_fill(self, painter: QPainter, frec: np.ndarray, target_size,
                   tint: tuple[float, float, float]) -> None:
        # A fill has no texture to size from, so its natural box is the
        # TARGET's - a field-instance curtain is a screen-sized quad the
        # chart then stretches. An item that DECLARES a size (a storyboard
        # Quad, whose w/h are its zoomto) overrides that. Opacity is already
        # applied via painter.setOpacity, so the color carries only
        # tint * full alpha; premultiplied-safe by construction.
        box = _placed_box(_draw_box(target_size, frec), frec)
        if box is None:
            return
        color = QColor.fromRgbF(
            max(0.0, min(1.0, tint[0])),
            max(0.0, min(1.0, tint[1])),
            max(0.0, min(1.0, tint[2])),
            1.0,
        )
        painter.fillRect(_cropped(box, frec), color)

    def _draw_image(
        self,
        painter: QPainter,
        urec: np.ndarray,
        frec: np.ndarray,
        tint: tuple[float, float, float],
    ) -> None:
        image_id = int(urec[_U_B])
        source = self._images.get(image_id)
        if source is None:
            logger.warning("RasterExecutor: missing image id %d, drew nothing", image_id)
            return
        # An image's logical box IS its natural (pixel) size; the item's
        # transform scales it. Source pixels and source-logical units
        # coincide, so no normalization is needed.
        self._draw_source_image(painter, source, source.width(), source.height(), frec, tint)

    def _draw_drawable(
        self,
        painter: QPainter,
        urec: np.ndarray,
        frec: np.ndarray,
        tint: tuple[float, float, float],
    ) -> None:
        drawable_id = int(urec[_U_B])
        source = self._targets.get(drawable_id)
        if source is None:
            # Not yet composed this run and no retained content: a
            # feedback read of a never-drawn drawable is transparent.
            return
        # SOURCE NORMALIZATION (drawable-ir.md rule 5): a drawable's
        # content covers its LOGICAL box whatever its backing pixel size.
        # The mat3 is authored in the source drawable's logical units, so
        # the quad spans that logical box (not the backing image's pixels,
        # which may be a chart-rect-sized injected capture) and the pixels
        # merely sample across it. Kills the "too zoomed in" bug where a
        # 1280x960 field capture was drawn as if it were 640x480 logical.
        lw, lh = self._sizes[drawable_id]
        self._draw_source_image(painter, source, float(lw), float(lh), frec, tint)

    def _draw_lines(
        self,
        painter: QPainter,
        urec: np.ndarray,
        tint: tuple[float, float, float],
    ) -> None:
        lines_id = int(urec[_U_B])
        verts = self._lines.get(lines_id)
        if verts is None or len(verts) < 2:
            # No vertices bound yet (or a degenerate single point): a
            # travelpath draws nothing until its feed arrives.
            return
        # The op's mat3 is already installed on the painter, so the
        # polyline is stroked in source logical units and rides the same
        # transform as any other source. Tint is the stroke color.
        color = QColor.fromRgbF(
            max(0.0, min(1.0, tint[0])),
            max(0.0, min(1.0, tint[1])),
            max(0.0, min(1.0, tint[2])),
            1.0,
        )
        pen = QPen(color, _LINES_WIDTH)
        pen.setCosmetic(False)
        painter.setPen(pen)
        polyline = QPolygonF([QPointF(float(x), float(y)) for x, y in verts])
        painter.drawPolyline(polyline)

    def _draw_source_image(
        self,
        painter: QPainter,
        source: QImage,
        logical_w: float,
        logical_h: float,
        frec: np.ndarray,
        tint: tuple[float, float, float],
    ) -> None:
        src = source
        if not _is_white(tint):
            src = _tinted(source, tint)

        # `logical_w/h` is the source's logical box (the units the mat3 is
        # authored in); the backing pixels (`pw/ph`) may differ - an
        # injected capture is chart-rect-sized while its drawable is 640x480
        # logical. Quad geometry spans the LOGICAL box; the sampled sub-rect
        # spans the matching fraction of the PIXEL image. Crops are fractions
        # of the logical box, applied to both so they stay aligned.
        pw, ph = source.width(), source.height()
        if logical_w <= 0.0 or logical_h <= 0.0 or pw <= 0 or ph <= 0:
            return
        # The DRAW box, not the logical box: `zoomto`/`setsize` replace the
        # natural basis and scale-to-fit rescales it, and the origin then
        # shifts the quad so the item draws about its own anchor. Reading
        # the logical box straight drew every element top-left-anchored at
        # its natural size, which on the GL side hung each one down-right by
        # half of itself.
        box = _placed_box(_draw_box((logical_w, logical_h), frec), frec)
        if box is None:
            return
        crop_l = float(frec[_F_CROP])
        crop_t = float(frec[_F_CROP + 1])
        vis_fw = max(0.0, 1.0 - crop_l - float(frec[_F_CROP + 2]))
        vis_fh = max(0.0, 1.0 - crop_t - float(frec[_F_CROP + 3]))
        if vis_fw <= 0.0 or vis_fh <= 0.0:
            return
        target = _cropped(box, frec)
        sub = _half_texel_inset(
            QRectF(crop_l * pw, crop_t * ph, vis_fw * pw, vis_fh * ph), pw, ph)
        painter.drawImage(target, src, sub)

    def _log_skipped_lanes(self, urec: np.ndarray, frec: np.ndarray) -> None:
        # Clip, uniform and box lanes are consumed (see _apply_clip /
        # _stash_uniforms / _placed_box). The shader itself is still not
        # applied - the raster reference backend draws unshaded but binds
        # the uniforms - and SetFade needs a per-edge alpha ramp this
        # backend has no equivalent for, so a faded item draws at full
        # opacity. Both are announced rather than silently dropped.
        if int(urec[_U_SHADER]) != 0 and "shader" not in self._skipped_lanes:
            self._skipped_lanes.add("shader")
            logger.warning("RasterExecutor: shader lane not implemented (TODO), drawn unshaded")
        if any(frec[_F_FADE:_F_FADE + 4]) and "fade" not in self._skipped_lanes:
            self._skipped_lanes.add("fade")
            logger.warning("RasterExecutor: fade lanes not implemented (TODO), drawn unfaded")


def _placed_box(size, frec) -> QRectF | None:
    """The item's drawn box in source space, shifted by its origin.

    The origin is a fraction of the item's OWN drawn size, subtracted before
    the transform - SM's `translate(-origin*w, -origin*h)`. None for a
    degenerate box, which is a legitimate zero-size draw (a Quad with
    `zoomto(0, 0)`), not an error."""
    w, h = size
    if w <= 0.0 or h <= 0.0:
        return None
    return QRectF(-float(frec[_F_ORIGIN]) * w, -float(frec[_F_ORIGIN + 1]) * h,
                  w, h)


def _cropped(box: QRectF, frec) -> QRectF:
    """`box` inset by the record's crop fractions. SM crops hide bands of
    the drawn quad and leave the surviving content where it was, so the
    inset is of the box itself rather than a rescale of it."""
    crop_l = float(frec[_F_CROP])
    crop_t = float(frec[_F_CROP + 1])
    vis_fw = max(0.0, 1.0 - crop_l - float(frec[_F_CROP + 2]))
    vis_fh = max(0.0, 1.0 - crop_t - float(frec[_F_CROP + 3]))
    return QRectF(box.x() + crop_l * box.width(),
                  box.y() + crop_t * box.height(),
                  vis_fw * box.width(), vis_fh * box.height())


def _half_texel_inset(sub: QRectF, pw: int, ph: int) -> QRectF:
    """Inset a pixel-space sample rect by half a texel of the backing
    ``pw`` x ``ph`` texture on each edge, so bilinear sampling at a sub-rect
    edge cannot reach the adjacent margin/sidebar texel and bleed it in
    (A5). Guarded: an inset that would invert or collapse the rect (a
    sub-1px sample window) is left untouched - correctness over the
    anti-bleed nicety at that scale."""
    hx = 0.5 if pw > 0 else 0.0
    hy = 0.5 if ph > 0 else 0.0
    if sub.width() <= 2.0 * hx or sub.height() <= 2.0 * hy:
        return sub
    return sub.adjusted(hx, hy, -hx, -hy)


def _clip_path(shape: tuple) -> QPainterPath:
    """Build a QPainterPath from a ClipDesc-shaped tuple in the target's
    logical units: ('rect', l, t, r, b) or ('poly', [(x, y), ...])."""
    path = QPainterPath()
    match shape:
        case ("rect", l, t, r, b):
            path.addRect(QRectF(float(l), float(t), float(r) - float(l), float(b) - float(t)))
        case ("poly", points):
            path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in points]))
        case _:
            raise ValueError(f"unknown clip shape: {shape!r}")
    return path


def _is_white(tint: tuple[float, float, float]) -> bool:
    return tint[0] >= 1.0 and tint[1] >= 1.0 and tint[2] >= 1.0


def _tinted(source: QImage, tint: tuple[float, float, float]) -> QImage:
    """Return a premultiplied-safe tinted copy: RGB multiplied by tint,
    alpha untouched. Correctness over speed - a caching layer can wrap
    this if a hot path ever needs it."""
    src = source.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = src.width(), src.height()
    ptr = src.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).astype(np.float32)
    # Qt ARGB32 is stored 0xAARRGGBB little-endian -> byte order B,G,R,A.
    scale = np.array([tint[2], tint[1], tint[0], 1.0], dtype=np.float32)
    out = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
    result = QImage(out.tobytes(), w, h, QImage.Format.Format_ARGB32)
    return result.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied).copy()
