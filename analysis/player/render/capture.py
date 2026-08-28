"""Offscreen capture slots for the field-instance composite.

The renderer captures layer groups into per-purpose offscreen targets
(slots 'field', 'field2', 'screen') and composites them
back as transformed per-instance blits. This module abstracts the
render target + blit + snapshot operations behind one small interface
so the composite step can run on either backend:

- RasterCaptureBackend (here): pooled window-sized QPixmaps and
  QPainter drawPixmap blits. The default; works on any host painter,
  including headless tests under QT_QPA_PLATFORM=offscreen.
- GLCaptureBackend (gl_capture.py): FBO render targets, textured-quad
  blits, glBlitFramebuffer snapshots. Chosen per frame when the host
  painter renders on a GL 3+ context (the QOpenGLWidget canvas).

Handles returned by `close`/`snapshot` are opaque to the renderer: it
stores them (AFT retention dicts, the previous-frame screen capture)
and passes them back to `blit`. Raster handles are QPixmaps; GL
handles are retained textures. `release` returns a snapshot handle the
renderer no longer holds so the GL backend can recycle its texture
(no-op for raster).

Slot lifecycle per frame: open -> paint via the returned QPainter ->
close -> blit/present. `snapshot` may be taken mid-paint (between
open and close) - the AFT node captures the in-progress screen
composite at its draw position.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap


def crop_region(region, crop):
    """`region` inset by (left, top, right, bottom) crop fractions of
    its own size, or unchanged for crop None. SM SetCrop* hides each
    edge's fraction of the actor's texture, and a curtain fill's region
    IS the quad's full extent, so the fractions apply directly. A crop
    that consumes the whole extent yields an invalid rect - callers
    skip the draw."""
    if crop is None:
        return region
    left, top, right, bottom = crop
    w, h = region.width(), region.height()
    return region.adjusted(left * w, top * h, -right * w, -bottom * h)


class RasterCaptureBackend:
    """Pooled-QPixmap capture slots (the CPU raster path)."""

    def __init__(self):
        # slot -> pooled QPixmap, reused across frames while the size
        # holds - a fresh window-sized allocation per capture per frame
        # is a measurable slice of the frame budget.
        self._pool: dict = {}
        self._painters: dict = {}

    def open(self, slot: str, host_painter, w: int, h: int) -> QPainter:
        """An active painter into the named slot's transparent target,
        sized to the window at the host painter's device pixel ratio.
        Pooled targets must never be RETAINED across frames by the
        caller (`snapshot` copies out of the slot)."""
        dpr = float(host_painter.device().devicePixelRatioF())
        size = (int(w * dpr), int(h * dpr))
        pm = self._pool.get(slot)
        if pm is None or (pm.width(), pm.height()) != size:
            pm = QPixmap(*size)
            self._pool[slot] = pm
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self._painters[slot] = painter
        return painter

    def close(self, slot: str):
        """End the slot's painter; returns the slot's drawable handle,
        valid until the slot is next opened."""
        painter = self._painters.pop(slot, None)
        if painter is not None:
            painter.end()
        return self._pool.get(slot)

    def abort(self) -> None:
        """End every open slot painter after a mid-frame exception, so
        the next frame starts from clean paint state."""
        for painter in self._painters.values():
            if painter.isActive():
                painter.end()
        self._painters.clear()

    def snapshot(self, slot: str):
        """An immutable copy of the slot's current pixels. Legal
        mid-paint; the GL backend additionally requires an open blits
        batch targeting the slot (the AFT node capture is always taken
        that way)."""
        pm = self._pool.get(slot)
        return pm.copy() if pm is not None else None

    def retain(self, handle):
        """Take shared ownership of a snapshot handle (the AFT freeze
        dict keeping a capture another slot also holds). Every retained
        reference needs its own `release`; QPixmaps are implicitly
        shared so raster just hands the handle back."""
        return handle

    def release(self, handle) -> None:
        """One reference to a snapshot handle dropped (None accepted);
        QPixmaps just get garbage-collected."""

    def blit(self, painter, handle, region, transform=None, src_box=None,
             opacity=1.0) -> None:
        """One instance blit of `handle` onto `painter` under
        `transform`, clipped only to `src_box` in the handle's own
        source space (the design box - offscreen content never bleeds
        in), at `opacity`. `region` is the curtain-fill geometry, NOT a
        blit clip: clipping blits to the rest-position chart rect
        sliced content that transforms carried across its edge; the
        opaque sidebar fill paints later and owns that area. `blits`
        batches several onto one painter state push."""
        with self.blits(painter, region) as batch:
            batch.blit(handle, transform=transform, src_box=src_box,
                       opacity=opacity)

    def blits(self, painter, region):
        """Context manager for a run of instance blits sharing one
        target painter; yields a batch with `blit(handle, ...)` (same
        semantics as the standalone `blit`) and `fill(rgb, opacity,
        crop)` (a flat color quad covering the `region` rect inset by
        the crop fractions - the AFT-rig curtain at its tree position
        among the blits)."""
        return _RasterBlits(painter, region)

    def present(self, painter, slot: str) -> None:
        """Draw the (closed) slot's full content onto `painter` at the
        origin - the screen composite's final hand-off to the real
        chart target."""
        pm = self._pool.get(slot)
        if pm is not None:
            painter.drawPixmap(0, 0, pm)


class _RasterBlits:
    """One batch of instance blits: each blit saves/restores around its
    own transform + clips so entries stay independent."""

    def __init__(self, painter, region):
        self._painter = painter
        self._region = region

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def blit(self, handle, transform=None, src_box=None,
             opacity=1.0, frag=None, mesh=None) -> None:
        # `frag` is the blit-style payload (shader path, uniforms, tint,
        # additive). The shader/tint are GL-tier effects; the raster
        # fallback draws the positioned blit unshaded/untinted (agreed
        # fallback - no CPU shader emulation) but DOES honour additive
        # blending: `blend('add')` decides what occludes what, and
        # source-over instead of Plus turned the cyriak recursion's
        # dark copies into triangle-hole masks. `mesh` (a Polygon grid
        # drawn through its Vert= shader) is likewise GL-tier; the
        # raster fallback draws the undisplaced quad, which IS the mesh
        # at rest (a flat fullscreen grid).
        painter = self._painter
        painter.save()
        if frag is not None and len(frag) > 3 and frag[3]:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Plus)
        if transform is not None:
            painter.setTransform(transform, True)
        if src_box is not None:
            # Set after the transform: clips the copy's SOURCE (the
            # design box in the capture) so only that region is copied,
            # mapped to the copy's position by the transform.
            painter.setClipRect(src_box, Qt.ClipOperation.IntersectClip)
        painter.setOpacity(min(1.0, opacity))
        painter.drawPixmap(0, 0, handle)
        painter.restore()

    def fill(self, rgb, opacity, crop=None) -> None:
        rect = crop_region(self._region, crop)
        if not rect.isValid():
            return
        painter = self._painter
        painter.save()
        painter.setOpacity(min(1.0, opacity))
        r, g, b = rgb
        painter.fillRect(rect, QColor.fromRgbF(r, g, b))
        painter.restore()
