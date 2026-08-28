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
        - 'screen'/'screen_prev' model SM's ActorFrameTexture: the AFT
          node captures the chart area at its draw position each frame
          (backdrop + field blits, never the screen blits made after
          it). 'screen' samplers draw after the node and blit THIS
          frame's capture (identity is a no-op re-draw); 'screen_prev'
          samplers draw before it and blit the previous frame's - their
          own blit lands in the next capture, the one-frame feedback
          that accumulates echo trails. When any is present the renderer
          composites the chart region offscreen, snapshots the capture
          mid-blit, and retains it; on a seek the retention is
          invalidated.
      Other games omit scope for the zero-cost path. A copy may name a
      second field capture with the scope 'field2' (see `second_field`).
    - `second_field`  a second, independently-modded playfield capture
      for dual-player NotITG charts. When present, the renderer renders
      the field layers a SECOND time with an alternate per-player
      note-mod consumer swapped in (its own note positions, receptor
      offsets, reverse baseline), into a separate pixmap. `fields`
      instances with scope 'field2' blit from it; 'field' instances blit
      the primary (player-0) capture as always. None (every other game,
      single-player NotITG) leaves the render path untouched - the second
      capture only happens when a producer supplies this. See
      games/notitg/field_instances.NotitgDualField / SecondFieldSpec.
    """
    transform: QTransform | None = None
    draws: tuple = ()
    opacity: float = 1.0
    shaders: tuple = ()
    scene_transform: QTransform | None = None
    fields: tuple = ()
    second_field: object | None = None


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
    second_field: object | None = None  # dual-player second capture spec

    @property
    def is_identity(self) -> bool:
        return (self.transform is None and self.scene_transform is None
                and not self.below and not self.above and not self.top
                and self.opacity >= 1.0 and not self.shaders
                and not self.fields and self.second_field is None)


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
    field_sources = []
    second_field = None
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
        if frame.fields:
            field_sources.append(frame.fields)
        if frame.second_field is not None:
            second_field = frame.second_field

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
                          fields=_merge_fields(field_sources),
                          second_field=second_field)


def _merge_fields(sources):
    """The frame's field instances, from every effect that supplied any.

    A single source passes through AS THE OBJECT IT IS. Concatenating would
    read it, and a source is allowed to fold its entries lazily - which is
    the whole point when the drawable pipeline draws the instances itself
    and never looks (games/notitg/field_instances._FieldEntries). Two
    effects both supplying fields is not a shape any game produces today;
    if one arises, it concatenates and pays for it."""
    if len(sources) == 1:
        return sources[0]
    return tuple(entry for source in sources for entry in source)
