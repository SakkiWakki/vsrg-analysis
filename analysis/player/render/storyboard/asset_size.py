"""One place that resolves an image asset's LOGICAL size and frame grid.

A storyboard sprite's on-screen size is `logical_size * zoom`, positioned
by its origin point; the logical size is what `zoom`/`scale` multiply and
what an absolute `zoomto`/`setsize` is expressed against. It is NOT the
texture's raw pixel size: a sheet packs many frames into one image, a
`(doubleres)` texture is authored at twice its logical resolution, and a
manifest or filename hint can state the logical size outright. A consumer
that reads raw `pixmap.width()/height()` as logical size mis-sizes every
such asset (a 3x2 sheet renders 3x too wide, a doubleres texture twice
too big).

`resolve` is the single funnel: raw pixel size + an `AssetSizeSpec` in,
a `LogicalSize` (per-frame logical width/height plus the frame grid) out.
The spec is produced by a game's filename-convention layer (StepMania's
lives in `analysis/games/notitg/sprite_sheet`); the core stays agnostic
so fluXis/osu can feed their own conventions (or none) through the same
funnel.

Precedence, highest first (a higher rule present makes the lower ones
moot):

1. explicit `logical` override - a manifest that states the logical
   frame size directly (`.sprite` redirect target size, an authoring
   tool's record). Used verbatim; the grid still crops frames out of it.
2. filename grid `NxM` - one image is an N-wide x M-high sheet, so the
   logical frame is `(px_w / N, px_h / M)`.
3. `(doubleres)` marker - the texture is authored at 2x its logical
   resolution, so halve (applied to whatever rules 2/1 left).
4. `res WxH` hint - the logical size stated in the filename; used when no
   higher rule fixed it.

The result is in the same design-space units the storyboard model uses
(the caller has already mapped pixels to design units where a game needs
it; StepMania authors 1px = 1 design unit, so no conversion there).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetSizeSpec:
    """A game's decoded size conventions for one asset, all optional.

    `cols`/`rows` are the frame grid (default 1x1, whole image = one
    frame). `doubleres` halves the resolved size. `logical` and `res`
    are explicit logical-frame sizes in design units - `logical` from a
    manifest (rule 1, wins outright), `res` from a filename hint (rule 4,
    a floor used only when the grid did not already fix the size)."""
    cols: int = 1
    rows: int = 1
    doubleres: bool = False
    logical: tuple | None = None
    res: tuple | None = None


@dataclass(frozen=True)
class LogicalSize:
    """The resolution result: one frame's logical size in design units
    plus the frame grid the source rect crops with."""
    logical_w: float
    logical_h: float
    cols: int
    rows: int

    @property
    def natural(self) -> tuple:
        """(w, h) design-space size a plain (unzoomed) draw of this asset
        occupies - one frame."""
        return (self.logical_w, self.logical_h)


def resolve(px_w: float, px_h: float, spec: AssetSizeSpec) -> LogicalSize:
    """Logical frame size + grid for an asset of raw pixel size
    `px_w x px_h` under `spec`, applying the precedence above."""
    cols = max(1, spec.cols)
    rows = max(1, spec.rows)
    if spec.logical is not None:
        logical_w, logical_h = spec.logical
    elif spec.res is not None and cols == 1 and rows == 1:
        logical_w, logical_h = spec.res
    else:
        logical_w = px_w / cols
        logical_h = px_h / rows
    if spec.doubleres:
        logical_w *= 0.5
        logical_h *= 0.5
    return LogicalSize(float(logical_w), float(logical_h), cols, rows)
