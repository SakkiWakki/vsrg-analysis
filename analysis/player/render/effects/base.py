"""Effect protocol, per-frame result, and the frame compositor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from PySide6.QtGui import QTransform


@dataclass(frozen=True)
class EffectFrame:
    """One effect's contribution to the current frame.

    - `transform`  affine applied to column-space (composed left-to-
      right across effects, so later effects transform in the space the
      earlier ones set up). None = identity.
    - `draws`      overlay painters as `(z, fn(ctx, painter))`; negative
      z paints below the chart layers (storyboard backgrounds),
      positive above. Drawn outside the column transform.
    - `opacity`    multiplies the chart-layer group opacity (1 = opaque).
    - `shaders`    fullscreen post-process passes as
      `(shader_id, uniforms)` pairs, run in order over the finished
      chart frame by the GL pipeline (see render/shaders/). Ignored
      when no GL pipeline is attached (headless / raster fallback).
    - `scene_transform`  affine applied to the WHOLE scene: below-
      draws, the chart layers, and effect draws under SCENE_TOP_Z
      (fluXis camera; NotITG field mods). Draws at z >= SCENE_TOP_Z
      (pulse ring, foreground flash) stay in screen space, matching
      fluXis's nesting outside its CameraContainer.
    - `fields`     playfield instances as `(transform, opacity)` or
      `(transform, opacity, scope)`. When any effect supplies them, the
      field layers render once into a transparent offscreen buffer and
      blit once per instance (NotITG proxies, fluXis extra playfields);
      `(None, 1.0)` is the untouched original. Empty = single identity
      field, drawn direct. `scope` (default 'field'):
        - 'field'  blits the transparent notefield capture (the shared
          background shows through every copy).
        - 'full'   also blits a backdrop capture (background clear +
          below-draws) under the copy transform, replicating the whole
          chart region incl. background.
        - 'screen' blits the PREVIOUS frame's whole chart-area composite
          (backdrop + field + the copy blits themselves). This is SM's
          ActorFrameTexture: an AFT holds the composed screen as of the
          previous frame, so the copy is one frame delayed and its
          repeated application is the engine's feedback (echo/DelayFrame
          trails). When any 'screen' copy is present the renderer
          composites the chart region offscreen this frame and retains it
          for next frame; on a seek the retention is invalidated.
      Other games omit scope for the zero-cost path.
    """
    transform: QTransform | None = None
    draws: tuple = ()
    opacity: float = 1.0
    shaders: tuple = ()
    scene_transform: QTransform | None = None
    fields: tuple = ()


@runtime_checkable
class Effect(Protocol):
    def at(self, ctx) -> EffectFrame | None:
        """This effect's contribution for `ctx` (frame + t_now), or None
        when inactive so the compositor can skip it cheaply."""
        ...


# Draws at or above this z stay in screen space when a scene
# transform is active (fluXis: pulse + foreground flash sit outside
# the CameraContainer; everything else, storyboard overlay included,
# rides the camera).
SCENE_TOP_Z = 800


@dataclass
class CompositeFrame:
    transform: QTransform | None = None
    below: tuple = ()      # (z, fn) sorted, z < 0
    above: tuple = ()      # (z, fn) sorted, 0 <= z < SCENE_TOP_Z
    top: tuple = ()        # (z, fn) sorted, z >= SCENE_TOP_Z
    opacity: float = 1.0
    shaders: tuple = ()    # (shader_id, uniforms) in effect order
    scene_transform: QTransform | None = None
    fields: tuple = ()     # (transform, opacity) playfield instances

    @property
    def is_identity(self) -> bool:
        return (self.transform is None and self.scene_transform is None
                and not self.below and not self.above and not self.top
                and self.opacity >= 1.0 and not self.shaders
                and not self.fields)


def composite(effects, ctx) -> CompositeFrame:
    """Fold every active effect into one transform + split draw lists.

    Transforms (field and scene) compose in effect order; `draws`
    merge and split at z=0 and SCENE_TOP_Z, each band stable-sorted by
    z so authoring order breaks ties; shader passes concatenate in
    effect order."""
    transform = None
    scene_transform = None
    draws = []
    opacity = 1.0
    shaders = []
    fields = []
    for effect in effects:
        frame = effect.at(ctx)
        if frame is None:
            continue
        if frame.transform is not None:
            transform = (frame.transform if transform is None
                         else frame.transform * transform)
        if frame.scene_transform is not None:
            scene_transform = (
                frame.scene_transform if scene_transform is None
                else frame.scene_transform * scene_transform)
        draws.extend(frame.draws)
        opacity *= frame.opacity
        shaders.extend(frame.shaders)
        fields.extend(frame.fields)

    def band(lo, hi):
        return tuple(sorted((d for d in draws if lo <= d[0] < hi),
                            key=lambda d: d[0]))

    return CompositeFrame(transform=transform,
                          below=band(float('-inf'), 0),
                          above=band(0, SCENE_TOP_Z),
                          top=band(SCENE_TOP_Z, float('inf')),
                          opacity=max(0.0, min(1.0, opacity)),
                          shaders=tuple(shaders),
                          scene_transform=scene_transform,
                          fields=tuple(fields))
