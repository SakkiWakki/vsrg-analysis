"""Seam-B raster executor: a QPainter/QImage consumer of the
``DrawSchedule`` record stream produced by ``storyboard_native``.

The evaluator (Rust) folds the drawable tree and sampled channels into a
flat, fully-resolved op list crossing Seam B as two fixed-stride SoA
buffers (u32 records + f32 records). This module is the reference raster
backend: it walks those records with painter's-algorithm semantics onto
per-drawable ``QImage`` targets, mapping each target's logical units to
device pixels exactly once. It never sees a timeline, a tree, or a game.

Record layout (frozen; mirrors native/src/evaluate.rs):

  U_STRIDE = 8 u32 lanes per op:
    [kind, a, b, c, blend, shader+1, clip+1, screen_space]
      BEGIN: a = drawable id, b = clear mode
      BLIT:  a = source kind, b = source id, c = frame index
      COPY:  a = destination drawable id (source is the OPEN target)
      END:   a = drawable id
  F_STRIDE = 20 f32 lanes per op:
    mat3 row-major [0..9], opacity [9], tint rgb [10..13],
    crop l,t,r,b [13..17], reserved [17..20]

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
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QTransform

logger = logging.getLogger(__name__)

_SCREEN_ID = 0

# u32 lane offsets.
_U_KIND = 0
_U_A = 1
_U_B = 2
_U_C = 3
_U_BLEND = 4
_U_SHADER = 5
_U_CLIP = 6

# f32 lane offsets.
_F_OPACITY = 9
_F_TINT = 10  # ..13
_F_CROP = 13  # ..17 (l, t, r, b fractions of the SOURCE logical size)

# Op / source / clear codes (kept in sync with storyboard_native; the
# module also exports them, but the executor must run without importing
# it at module load so tests can importorskip cleanly).
class _Op:
    """Op-kind codes as dotted names so `match` can use value patterns."""

    BEGIN, BLIT, COPY, END = 0, 1, 2, 3


_OP_BEGIN, _OP_BLIT, _OP_COPY, _OP_END = _Op.BEGIN, _Op.BLIT, _Op.COPY, _Op.END
_SRC_IMAGE, _SRC_DRAWABLE, _SRC_MESH, _SRC_FILL, _SRC_LINES = 0, 1, 2, 3, 4
_CLEAR_TRANSPARENT, _CLEAR_OPAQUE, _CLEAR_RETAIN = 0, 1, 2

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
    ) -> None:
        self._images = images
        self._sizes = drawable_sizes
        self._targets: dict[int, QImage] = {}
        self._skipped_lanes: set[str] = set()

    def execute(self, u: np.ndarray, f: np.ndarray) -> QImage:
        """Run the records; return the screen drawable's image.

        Retained targets survive across calls, so calling execute() again
        with the same doc reuses last frame's content for any Retain
        drawable (feedback).
        """
        u = np.ascontiguousarray(u, dtype=np.uint32)
        f = np.ascontiguousarray(f, dtype=np.float32)
        target_stack: list[int] = []
        painter: QPainter | None = None

        for i in range(u.shape[0]):
            match int(u[i, _U_KIND]):
                case _Op.BEGIN:
                    painter = self._begin(u[i], target_stack, painter)
                case _Op.BLIT:
                    self._blit(u[i], f[i], target_stack, painter)
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
        clear = int(rec[_U_B])
        img = self._targets.get(drawable_id)
        if img is None:
            img = self._alloc(drawable_id)
            self._targets[drawable_id] = img
        if clear == _CLEAR_TRANSPARENT:
            img.fill(0)
        elif clear == _CLEAR_OPAQUE:
            img.fill(QColor(0, 0, 0, 255))
        # Retain: leave existing content untouched (feedback / snapshots).

        target_stack.append(drawable_id)
        new_painter = QPainter(img)
        new_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        return new_painter

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
    ) -> None:
        if painter is None or not target_stack:
            return
        self._log_skipped_lanes(urec)

        src_kind = int(urec[_U_A])
        opacity = float(frec[_F_OPACITY])
        tint = (float(frec[_F_TINT]), float(frec[_F_TINT + 1]), float(frec[_F_TINT + 2]))
        blend = _QT_BLEND.get(int(urec[_U_BLEND]), _QT_BLEND[0])

        painter.save()
        painter.setTransform(self._qtransform(frec))
        painter.setOpacity(max(0.0, min(1.0, opacity)))
        painter.setCompositionMode(blend)

        if src_kind == _SRC_FILL:
            self._draw_fill(painter, tint)
        elif src_kind == _SRC_IMAGE:
            self._draw_image(painter, urec, frec, tint)
        elif src_kind == _SRC_DRAWABLE:
            self._draw_drawable(painter, urec, frec, tint)
        # Mesh / Lines sources are geometry-bearing; the raster reference
        # backend leaves them for a later pass.
        painter.restore()

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

    def _draw_fill(self, painter: QPainter, tint: tuple[float, float, float]) -> None:
        # A unit rect in source space (0,0)-(1,1) colored tint. Opacity is
        # already applied via painter.setOpacity, so the color carries
        # only tint * full alpha; premultiplied-safe by construction.
        color = QColor.fromRgbF(
            max(0.0, min(1.0, tint[0])),
            max(0.0, min(1.0, tint[1])),
            max(0.0, min(1.0, tint[2])),
            1.0,
        )
        painter.fillRect(QRectF(0.0, 0.0, 1.0, 1.0), color)

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
        self._draw_source_image(painter, source, frec, tint)

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
        self._draw_source_image(painter, source, frec, tint)

    def _draw_source_image(
        self,
        painter: QPainter,
        source: QImage,
        frec: np.ndarray,
        tint: tuple[float, float, float],
    ) -> None:
        src = source
        if not _is_white(tint):
            src = _tinted(source, tint)

        sw, sh = source.width(), source.height()
        crop_l = float(frec[_F_CROP]) * sw
        crop_t = float(frec[_F_CROP + 1]) * sh
        crop_r = float(frec[_F_CROP + 2]) * sw
        crop_b = float(frec[_F_CROP + 3]) * sh
        # Crops are fractions of the source's logical size; inset the
        # sampled source rectangle and place it at the matching logical
        # offset so the geometry stays anchored under the transform.
        vis_w = max(0.0, sw - crop_l - crop_r)
        vis_h = max(0.0, sh - crop_t - crop_b)
        if vis_w <= 0.0 or vis_h <= 0.0:
            return
        target = QRectF(crop_l, crop_t, vis_w, vis_h)
        sub = QRectF(crop_l, crop_t, vis_w, vis_h)
        painter.drawImage(target, src, sub)

    def _log_skipped_lanes(self, urec: np.ndarray) -> None:
        if int(urec[_U_SHADER]) != 0 and "shader" not in self._skipped_lanes:
            self._skipped_lanes.add("shader")
            logger.warning("RasterExecutor: shader lane not implemented (TODO), drawn unshaded")
        if int(urec[_U_CLIP]) != 0 and "clip" not in self._skipped_lanes:
            self._skipped_lanes.add("clip")
            logger.warning("RasterExecutor: clip lane not implemented (TODO), drawn unclipped")


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
