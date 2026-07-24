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
`state_pin` is an optional frame-index sampler (`sample(t) -> (frame,)`,
e.g. a `sprite_sheet.StateAnchors` built from recorded
`setstate`/`animate` pokes) that overrides the animation when present. A plain 1x1 sprite leaves `sheet_cols`/`sheet_rows` at 1 and
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
    # SM crop family (croptop/cropbottom/cropleft/cropright): the fraction
    # (0..1) of the actor hidden from each edge before it draws. Rest 0 =
    # uncropped, so an element never poked with a crop verb draws whole.
    'crop_top': 0.0, 'crop_bottom': 0.0, 'crop_left': 0.0, 'crop_right': 0.0,
    # 3D scene channels: out-of-plane rotation, z position/scale, skew,
    # and the frame's perspective-camera fov (deg). All rest at identity
    # (rot 0, z 0, scale_z 1, skew 0, fov 45 = the LoadMenuPerspective
    # default), so an actor never poked in 3D projects as the exact 2D
    # affine it does today - the flat-chart no-op path.
    'rotation_x': 0.0, 'rotation_y': 0.0, 'z': 0.0, 'scale_z': 1.0,
    'skew_x': 0.0, 'skew_y': 0.0, 'fov': 45.0,
    # SM edge-fade family (SetFadeLeft/Right/Top/Bottom): the fraction
    # (0..1) of the actor over which alpha ramps from 0 at that edge to
    # full at the fade distance (Sprite.cpp:560). Rest 0 = no fade, so an
    # element never poked with a fade verb draws with hard edges.
    'fade_left': 0.0, 'fade_right': 0.0, 'fade_top': 0.0, 'fade_bottom': 0.0,
    # ScaleToCover/ScaleToFitInside(rect): the target rect plus a mode
    # sentinel (0 none, 1 cover, 2 fit-inside). The setter needs the
    # actor's true natural size to pick the uniform zoom, and that size is
    # a render-time fact for a sprite, so the sim records the rect + mode
    # here and the renderer resolves the fitted size. Rest fit_mode 0 =
    # no fit, so an actor never fit draws through natural*scale unchanged.
    'fit_mode': 0.0,
    'fit_left': 0.0, 'fit_top': 0.0, 'fit_right': 0.0, 'fit_bottom': 0.0,
}

# Tuple-valued color rests, merged alongside 'color' in build_timelines.
# The flat single diffuse is 'color' (rgb, rest white). The per-corner
# diffuse channels (SetDiffuseUpperLeft etc., Actor.h:190-197) carry rgba
# and rest at the UNSET sentinel: any component < 0 means "this corner is
# not individually set - use the flat color+alpha", so an element never
# poked with a corner/edge verb draws the exact flat quad it does today
# (the gradient no-op path). glow is an additive overlay color (rgba); its
# rest alpha 0 means no glow pass (Actor.cpp:1008), so an un-glowed actor
# composites identically.
_COLOR_UNSET = (-1.0, -1.0, -1.0, -1.0)
_COLOR_RESTS = {
    'color': (1.0, 1.0, 1.0),
    'color_ul': _COLOR_UNSET, 'color_ur': _COLOR_UNSET,
    'color_ll': _COLOR_UNSET, 'color_lr': _COLOR_UNSET,
    'glow': (1.0, 1.0, 1.0, 0.0),
}

# Non-color tuple/string channels a sprite can carry.
# `texcoord_scroll` is SM's SetTexCoordVelocity recorded as a closed-form
# anchor (t0, offset_u, offset_v, vel_u, vel_v): the drawn UV offset at t
# is offset + vel * (t - t0), wrapping mod 1. Rest = never scrolled.
# `asset_swap` is a runtime Sprite:Load() texture swap recorded as
# (absolute path, sheet cols, sheet rows) - the game frontend decodes
# the grid convention at record time; path '' = never swapped.
_SPRITE_RESTS = {
    'texcoord_scroll': (0.0, 0.0, 0.0, 0.0, 0.0),
    'asset_swap': ('', 1.0, 1.0),
}


def build_timelines(rests: dict | None = None,
                    keyframes: dict | None = None) -> dict:
    """One `EventTimeline` per property. `rests` overrides the default
    rest state (scalars as floats, 'color' as an (r, g, b) tuple);
    `keyframes` maps property name -> list of Keyframes."""
    rests = {**_SCALAR_RESTS, **_COLOR_RESTS, **_SPRITE_RESTS,
             **(rests or {})}
    keyframes = keyframes or {}
    out = {}
    for prop, rest in rests.items():
        rest_tuple = rest if isinstance(rest, tuple) else (float(rest),)
        out[prop] = EventTimeline(keyframes.get(prop, []), rest=rest_tuple)
    return out


class LiveCurve:
    """A `.sample(t)`-protocol curve backed by a LIVE sim instead of baked
    keyframes (lazy replay). `sample(t)` advances the shared sim to `t` and
    reads the actor's engine-current value for `prop`, returning it in the same
    tuple shape a baked `EventTimeline.sample` gives. Duck-typed drop-in: the
    renderer samples it identically, never knowing it is live.

    `sim` exposes `advance_to(t)` and `.env._actors` (a `LiveSim`); `rec_id` is
    the actor's recorder id. Before the actor exists (t < its create time) or a
    missing prop, returns `rest`."""

    __slots__ = ('_sim', '_rec_id', '_prop', '_rest')

    def __init__(self, sim, rec_id, prop, rest):
        self._sim = sim
        self._rec_id = rec_id
        self._prop = prop
        # A tuple rest passes through (color, quat, rotation_order token); a
        # scalar wraps to a 1-tuple. Do NOT force float - a rotation_order rest
        # is a string token.
        self._rest = rest if isinstance(rest, tuple) else (rest,)

    def sample(self, t: float) -> tuple:
        self._sim.advance_to(t)
        actor = self._sim.env._actors.get(self._rec_id)
        if actor is None:
            return self._rest
        # `current` exposes ANY channel (incl the rotation_order token / quat
        # tuple that live outside _current); None -> the prop's rest.
        value = actor.current(self._prop, at_t=t)
        if value is None:
            return self._rest
        return value if isinstance(value, tuple) else (value,)


def build_live_timelines(sim, rec_id, rests: dict | None = None) -> dict:
    """Like `build_timelines`, but each property is a `LiveCurve` reading the
    live sim - the lazy-replay counterpart. Same key set + rest defaults so the
    element samples identically."""
    rests = {**_SCALAR_RESTS, **_COLOR_RESTS, **_SPRITE_RESTS,
             **(rests or {})}
    return {prop: LiveCurve(sim, rec_id, prop, rest)
            for prop, rest in rests.items()}


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
    state_pin: object = None  # frame-index sampler (sample(t)), or None
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
