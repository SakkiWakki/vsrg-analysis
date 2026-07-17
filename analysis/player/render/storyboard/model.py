"""Storyboard IR: what every game's storyboard format compiles into.

A `Storyboard` is a fixed design-space canvas (`design_w x design_h`)
holding timed `Element`s. Each element carries one `EventTimeline` per
animatable property, so sampling at `t` fully describes its visual
state; times are absolute chart seconds and the per-game compilers
resolve every source quirk (relative animation clocks, use-start
flags, value-string encodings) before the IR exists.

Properties and arities:

    x, y          design-space offset from the anchor point
    scale_x/y     multiplies the element's natural size
    rotation      degrees, clockwise, about the origin point
    alpha         0..1 (element base alpha folded into the rest value)
    hidden        SM's hard visibility bit (0 shown, 1 hidden); gates the
                  draw independently of alpha so a diffusealpha crossfade
                  can ride an actor that a `hidden,1` currently hides
    w, h          design-space size for sized kinds (rect/ellipse/...)
    border        outline stroke width, design px
    color         (r, g, b) 0..1

`anchor` is a fraction of the design rect (where the element hangs);
`origin` is a fraction of the element's own size (what point lands on
the anchor + offset). Kinds: 'sprite', 'frames' (sprite sequence),
'rect', 'ellipse', 'outline_rect', 'outline_ellipse', 'text',
'bitmaptext' (a 'text' whose glyphs come from an SM bitmap-font atlas
carried in `font`), 'group'.

A 'sprite' whose asset is a StepMania NxM grid sheet carries the grid
(`sheet_cols`/`sheet_rows`) and a `sheet_states` list of
`(frame_index, delay_seconds)`: the renderer crops the CURRENT frame's
cell instead of the whole sheet, and the element's natural size is ONE
frame. The frame's LOGICAL size (what `scale_x/y` multiply and a plain
draw occupies) is never the raw pixel size: `size_spec` carries the
grid plus any `(doubleres)`/`res`/manifest conventions, and the renderer
resolves it through `render.storyboard.asset_size` so a sheet, a
doubleres texture, or a res-hinted image all size correctly. With no
`state_pin`, the sheet auto-animates through
`sheet_states` (SM's sprite animation, tied to the effect clock);
`state_pin` is an optional EventTimeline of a frame index over time
(recorded `setstate`/`animate` pokes) that overrides the animation when
present. A plain 1x1 sprite leaves `sheet_cols`/`sheet_rows` at 1 and
draws whole, exactly as before.

A 'group' is an ActorFrame: it draws nothing itself but carries its
own property timelines, and its transform (translate about anchor +
position, rotate/scale about its own origin) composes onto its
`children`, recursively. Rotating a group rotates every descendant.
Flat storyboards (no groups, empty `children`) keep the existing
zero-cost draw path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from analysis.player.render.effects.timeline import EventTimeline

# Absolute on-screen size (SM zoomto/setsize) is UNSET at rest: a
# negative sentinel the renderer reads as "no absolute size, use
# natural*scale". A real zoomto keyframe (>=0) overrides that axis.
_SIZE_UNSET = -1.0

_SCALAR_RESTS = {
    'x': 0.0, 'y': 0.0, 'scale_x': 1.0, 'scale_y': 1.0,
    'rotation': 0.0, 'alpha': 1.0, 'w': 0.0, 'h': 0.0, 'border': 2.0,
    # SM's hard visibility bit, held apart from alpha (0 shown, 1 hidden);
    # an element is drawn only when NOT hidden AND alpha is visible.
    'hidden': 0.0,
    'size_x': _SIZE_UNSET, 'size_y': _SIZE_UNSET,
}


def build_timelines(rests: dict | None = None,
                    keyframes: dict | None = None) -> dict:
    """One `EventTimeline` per property. `rests` overrides the default
    rest state (scalars as floats, 'color' as an (r, g, b) tuple);
    `keyframes` maps property name -> list of Keyframes."""
    rests = {**_SCALAR_RESTS, 'color': (1.0, 1.0, 1.0), **(rests or {})}
    keyframes = keyframes or {}
    out = {}
    for prop, rest in rests.items():
        rest_tuple = rest if isinstance(rest, tuple) else (float(rest),)
        out[prop] = EventTimeline(keyframes.get(prop, []), rest=rest_tuple)
    return out


@dataclass(frozen=True)
class Element:
    kind: str
    z: int                    # effects z-slot (storyboard layer)
    z_index: int              # draw order within the layer
    t_start: float            # seconds; element exists in [t_start, t_end)
    t_end: float
    anchor: tuple             # (ax, ay) fraction of the design rect
    origin: tuple             # (ox, oy) fraction of own size
    timelines: dict           # property -> EventTimeline
    asset: str | None = None  # absolute file path for 'sprite'
    text: str = ''
    font_px: float = 0.0      # 'text' size in design px
    font: object = None       # BitmapFont for 'bitmaptext', else None
    additive: bool = False
    flip_h: bool = False
    flip_v: bool = False
    frames: tuple = ()        # absolute paths for 'frames'
    frame_delay: float = 0.0  # seconds per frame
    loop_forever: bool = True
    children: tuple = ()      # nested Elements for a 'group' (ActorFrame)
    sheet_cols: int = 1       # SM NxM grid: columns of frames in the sheet
    sheet_rows: int = 1       # rows of frames in the sheet
    sheet_states: tuple = ()  # ((frame_index, delay_seconds), ...) auto-anim
    state_pin: object = None  # EventTimeline of a frame index, or None
    # Size conventions (doubleres/res hint/manifest logical override) the
    # renderer feeds to render.storyboard.asset_size.resolve to turn raw
    # pixels into the frame's LOGICAL size. None = a plain asset whose
    # logical size is its pixel size divided by the grid (the default).
    size_spec: object = None  # AssetSizeSpec | None

    def sample(self, prop: str, t: float):
        return self.timelines[prop].sample(t)


@dataclass(frozen=True)
class Storyboard:
    design_w: float
    design_h: float
    # 'min': uniform scale so the whole design rect fits the chart
    # region (fluXis DrawSizePreservingFillContainer). 'height': scale
    # by height only (osu's 480-tall space; wide regions reveal the
    # widescreen x range instead of letterboxing).
    fit: str
    elements: tuple = field(default=())
    # Clip every layer draw to the mapped design rect, not just the chart
    # region. NotITG presents a hard-cropped 640x480 box: actors that run
    # offscreen must crop at the design edges, and the box centered in the
    # chart region is where the notefield centers too. fluXis/osu leave
    # this off (their design space IS the viewport).
    clip_design_box: bool = False

    def __bool__(self):
        return bool(self.elements)
