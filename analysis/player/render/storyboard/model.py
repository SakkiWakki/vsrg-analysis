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
    w, h          design-space size for sized kinds (rect/ellipse/...)
    border        outline stroke width, design px
    color         (r, g, b) 0..1

`anchor` is a fraction of the design rect (where the element hangs);
`origin` is a fraction of the element's own size (what point lands on
the anchor + offset). Kinds: 'sprite', 'frames' (sprite sequence),
'rect', 'ellipse', 'outline_rect', 'outline_ellipse', 'text'.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from analysis.player.render.effects.timeline import EventTimeline

_SCALAR_RESTS = {
    'x': 0.0, 'y': 0.0, 'scale_x': 1.0, 'scale_y': 1.0,
    'rotation': 0.0, 'alpha': 1.0, 'w': 0.0, 'h': 0.0, 'border': 2.0,
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
    additive: bool = False
    flip_h: bool = False
    flip_v: bool = False
    frames: tuple = ()        # absolute paths for 'frames'
    frame_delay: float = 0.0  # seconds per frame
    loop_forever: bool = True

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

    def __bool__(self):
        return bool(self.elements)
