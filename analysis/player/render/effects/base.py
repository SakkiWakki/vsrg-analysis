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
    """
    transform: QTransform | None = None
    draws: tuple = ()
    opacity: float = 1.0
    shaders: tuple = ()


@runtime_checkable
class Effect(Protocol):
    def at(self, ctx) -> EffectFrame | None:
        """This effect's contribution for `ctx` (frame + t_now), or None
        when inactive so the compositor can skip it cheaply."""
        ...


@dataclass
class CompositeFrame:
    transform: QTransform | None = None
    below: tuple = ()      # (z, fn) sorted, z < 0
    above: tuple = ()      # (z, fn) sorted, z >= 0
    opacity: float = 1.0
    shaders: tuple = ()    # (shader_id, uniforms) in effect order

    @property
    def is_identity(self) -> bool:
        return (self.transform is None and not self.below
                and not self.above and self.opacity >= 1.0
                and not self.shaders)


def composite(effects, ctx) -> CompositeFrame:
    """Fold every active effect into one transform + split draw lists.

    Transforms compose in effect order; `draws` merge and split around
    z=0, each side stable-sorted by z so authoring order breaks ties;
    shader passes concatenate in effect order."""
    transform = None
    draws = []
    opacity = 1.0
    shaders = []
    for effect in effects:
        frame = effect.at(ctx)
        if frame is None:
            continue
        if frame.transform is not None:
            transform = (frame.transform if transform is None
                         else frame.transform * transform)
        draws.extend(frame.draws)
        opacity *= frame.opacity
        shaders.extend(frame.shaders)

    below = tuple(sorted((d for d in draws if d[0] < 0),
                         key=lambda d: d[0]))
    above = tuple(sorted((d for d in draws if d[0] >= 0),
                         key=lambda d: d[0]))
    return CompositeFrame(transform=transform, below=below, above=above,
                          opacity=max(0.0, min(1.0, opacity)),
                          shaders=tuple(shaders))
