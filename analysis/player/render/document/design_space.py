"""The design-space header: a chart's authoring resolution + how it maps
to the on-screen chart region.

Every game authors its visual content (storyboard actors, playfield
transforms, camera moves) in a fixed design resolution and then relies
on the engine to fit that rectangle into whatever viewport the player
runs at. That fit was scattered as constants and per-call kwargs:
`ref_space.REF_W/H` for fluXis effect coordinates, `Storyboard(design_w,
design_h, fit, clip_design_box)` kwargs for every storyboard compiler,
the notitg adapter's hard-coded 640x480 crop box. `DesignSpace` names
that data once so the player maps design->screen in exactly one place
(`render.storyboard.render._design_transform` today) and the compiled
document can carry it as its header.

Fit policy (how the design rect scales into the chart region):

    'min'     uniform scale so the WHOLE design rect fits, letterboxing
              the shorter axis (fluXis DrawSizePreservingFillContainer,
              NotITG's hard-cropped screen box).
    'height'  scale by height only; a viewport wider than the design
              aspect reveals more of the x range instead of letterboxing
              (osu's 480-tall convention, where widescreen extends the
              playfield sideways to 854 rather than adding bars).
    'stretch' scale each axis independently to fill the region (no game
              uses it today; reserved so the enum is complete against a
              non-aspect-preserving authoring space).

`clip` is the hard design-box crop: content painted past the mapped
design edges is clipped away. NotITG presents a fixed 640x480 screen
that hard-crops offscreen actors, and the centered box is where the
notefield centers too; fluXis/osu leave it off because their design
space IS the viewport.

Two design spaces can coexist in one game. fluXis authors EFFECT
coordinates (playfield moves, shakes, camera) at 1366x768 (this
`DesignSpace`) while its .fsb STORYBOARD carries its own per-file
`resolution` (often 1920x1080); the storyboard compiler keeps reading
that per-file value. `design_space()` is the game's primary/effect
authoring space, not an override of every sub-format's own header.
"""
from __future__ import annotations

from dataclasses import dataclass

FIT_MIN = 'min'
FIT_HEIGHT = 'height'
FIT_STRETCH = 'stretch'
_FIT_POLICIES = frozenset((FIT_MIN, FIT_HEIGHT, FIT_STRETCH))


@dataclass(frozen=True)
class DesignSpace:
    width: float
    height: float
    fit: str = FIT_MIN
    clip: bool = False

    def __post_init__(self):
        if self.fit not in _FIT_POLICIES:
            raise ValueError(
                f'unknown fit policy {self.fit!r}; '
                f'expected one of {sorted(_FIT_POLICIES)}')
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError(
                f'design space must be positive, got {self.width}x'
                f'{self.height}')
