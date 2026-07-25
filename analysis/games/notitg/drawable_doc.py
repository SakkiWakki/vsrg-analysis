"""NotITG static scene -> channel-backed DrawableDoc in engine tree order.

The Drawable model (`.claude/plans/drawable-ir.md`) folds a game's per-frame
composite into a game-agnostic document (Seam A) a Rust evaluator turns into a
flat op stream each frame (Seam B). This module compiles the STATIC part of a
NotITG chart - the field-instance topology (player fields, proxy re-renders, AFT
samplers, fills, capture snapshots) - into that document ONCE, in engine tree
order, so the static scene no longer rides per-frame feeds.

It is the tree-order sibling of `drawable_bridge` (which routes the same
field-instance stream through per-frame dynamic feeds): where the bridge re-emits
the whole entry stream every frame, this module STATICALLY places each instance
as an `item_link` chain whose per-property channels are exported once at build
time. The Rust evaluator then samples those channels itself - no Python in the
per-frame loop for the static part.

`build_static_doc(compiled)` walks the provider's instance list in order and,
per instance:

  - proxy / player field blits -> `SRC_DRAWABLE` of the per-player field
    drawable, transform = the instance's full link chain via `item_link`;
  - aft samplers -> `SRC_DRAWABLE` of the sampler's freeze-key slot drawable,
    same link chain, with the aft vertical flip on the leaf link;
  - fills -> `SRC_FILL` with the sampled diffuse rgb as tint channels;
  - captures -> a `Snapshot` command into the capture's named slot drawable,
    placed at the instance's tree position (no draw of its own);
  - a `z_group` run of instances is wrapped in a `SortSpan` so the evaluator
    stably re-sorts it by each item's sampled `z` (SetDrawByZPosition).

`export_channel` is THE shared primitive: it converts one of the Python timeline
objects an instance's links carry (an `EventTimeline`, or a lazy
`LiveCurve`/`SegCurve`) into the `(ts, vals, durs, rest, eases)` breakpoint
arrays `DocBuilder.channel` ingests - exactly, by asking a curve that knows its
own closed form (`SegCurve` over the sim's recorded segments, a sheet's frame
cycle) or by translating an `EventTimeline`'s keyframes, and otherwise by
sampling and collapsing. It is documented for later extraction (other games'
static scenes will reuse it verbatim).

The parity harness (`compare_at` / `parity_report`) compares the evaluator's
BLIT stream against `NotitgFieldInstances.at` at N sample times - order, alpha,
and the design-space mat3 - and is the deliverable quality bar.

Scope note (native transform coverage): the Rust `compose_links` path folds the
full 3D link chain - x/y/z, scale incl. scale_z, in-plane rotation AND
rotation_x/y under the chain's Euler order, base-scale, skew, anchor, crop, flip
(`camera::local_matrix`, native/src/transform.rs). `_LINK_PROP_ORDER` forwards
all of it.

(This note used to say the native path was 2D-affine only and that out-of-plane
instances diverged BY DESIGN. That stopped being true when the camera area was
ported, but the prop list was never widened to match - so every 3D field
transform was silently dropped, and the stale note is what made it look
intentional. Kept as a caution: a "by design" limitation is worth re-checking
against the code before trusting it.)

Still not forwarded for field instances: quaternion spin, and the fov
perspective (`item_projection`, which the ELEMENT chain does emit). The parity
harness reports such divergences rather than hiding them.

No Qt import at module load - importable headless.
"""
from __future__ import annotations

import math
import os
from typing import NamedTuple
from pathlib import Path

import numpy as np

from analysis.player.render.effects.timeline import SIMPLIFY_EPS, EventTimeline
from analysis.player.render.storyboard import record as _rec
from analysis.player.render.storyboard.sprite_sheet import (
    frame_at_time, frame_steps)

# The design chart region: SM's fixed 640x480 screen. Sampling the effect over
# this region makes the design map the identity (kx=ky=1, ox=oy=0), so an
# entry's screen QTransform equals its design-space homography - the same space
# the DrawableDoc's natural-640x480 link chains live in, so the two compare
# directly with no conjugation.
_DESIGN_RECT = (0.0, 0.0, 640.0, 480.0)
_SCREEN_W = 640.0
_SCREEN_H = 480.0

# Drawable 0 is always the screen root (minted by the builder).
_SCREEN_ID = 0

# The lazy-curve sampling cadence (LiveCurve / SegCurve have no exposed
# keyframes, so they are densely sampled). 1/30s: the interim approximation the
# spec calls out - an EventTimeline is exported exactly instead.
_DENSE_DT = 1.0 / 30.0

# Linear easing id (effects.easing): the ONLY easing the native channel's
# breakpoint ramp reproduces exactly. Other eases within a keyframe span are
# densified.
_EASE_LINEAR = 0

_REST_EPS = 1e-4

# Storyboard element kinds this wave emits as Image items. Sprites (a single
# cell) and frame-animated sheets both back an SRC_IMAGE draw; every other kind
# (shapes, text, groups-with-no-image, video, compound) is skipped with a
# per-kind count in the report. Backgrounds - the user-reported missing content
# - are BGCHANGES sprites, so they land here.
_IMAGE_KINDS = ('sprite', 'frames')

# Kinds drawn as a solid tinted quad. A Quad's absolute size IS its whole size
# (modfile._fill_size_as_wh mirrors zoomto onto w/h precisely because there is
# no natural basis), so the item's own size lanes carry it and the executor
# needs no texture to size from.
_FILL_KINDS = ('rect',)

# The element property -> native `item` transform-lane mapping. Each pair is
# (element timeline prop, the item() id/rest kwarg stem); every lane is a scalar
# EventTimeline exported once through export_channel. `hidden` is handled apart
# (inverted into the item's visible gate), and the sheet frame is its own
# derived channel - neither appears here.
_ELEMENT_ITEM_LANES = (
    ('x', 'x'), ('y', 'y'),
    ('scale_x', 'sx'), ('scale_y', 'sy'),
    ('rotation', 'rot'),
    ('alpha', 'opacity'),
)

# Element verbs the doc cannot yet express, each counted as a skip so an
# unsupported chart shows up in the report rather than just looking slightly
# wrong once legacy stops drawing elements.
#
# Both rest at a sentinel the flat draw ignores, so "carries the timeline" is
# NOT the test - `build_timelines` mints every property on every element. A
# corner is unset while its red is negative (render._corners); a fade is off
# at 0.
_CORNER_COLOR_PROPS = ('color_ul', 'color_ur', 'color_ll', 'color_lr')
_FADE_PROPS = ('fade_left', 'fade_right', 'fade_top', 'fade_bottom')
_CORNER_UNSET = -1.0
# `render._SIZE_UNSET`: a negative absolute size means "use the natural box".
_SIZE_UNSET = -1.0
_FADE_OFF = 0.0
_GLOW_OFF = 0.0


def _has_motion(timeline) -> bool:
    """Whether a curve ever leaves its rest.

    The two curve families answer this differently, and each one's natural
    probe is WRONG for the other: a `SegCurve` implements `is_static()` but
    has no `__bool__` and so is unconditionally truthy, while an
    `EventTimeline` has no probe and is falsy exactly when it holds no
    keyframes. Asking only one of them mislabels every curve of the other
    kind as moving."""
    probe = getattr(timeline, 'is_static', None)
    return not probe() if probe is not None else bool(timeline)


def _moves_off(timeline, unset: float, prop: int = 0) -> bool:
    """Whether `timeline` ever leaves `unset` on component `prop`.

    A timeline that never moves is untouched when its rest IS the sentinel.
    `_is_static` alone is the WRONG probe: it answers False for a plain
    keyframe-less EventTimeline - the safe direction for export, but here it
    reports everything as poked, which silently built a glow item for every
    element in the chart."""
    return timeline is not None and (
        _has_motion(timeline) or _rest_value(timeline, prop) != unset)


def _is_poked(element, props, unset: float) -> bool:
    """Whether `element` ever moves any of `props` off `unset`."""
    return any(_moves_off(element.timelines.get(prop), unset)
               for prop in props)


# --------------------------------------------------------------------------
# export_channel - the shared timeline -> breakpoint-array primitive
# --------------------------------------------------------------------------

def export_channel(timeline, t0: float, t1: float, prop: int = 0):
    """Convert one Python timeline into the breakpoint arrays
    `DocBuilder.channel` ingests, over the window `[t0, t1]`.

    The native channel (native/src/channels.rs) serves `rest` before the first
    breakpoint, then at each breakpoint either holds its value (dur <= 0) or
    ramps toward the next breakpoint's value over `dur` under that
    breakpoint's ease id. This helper emits breakpoints reproducing the
    timeline's own playback under that model, most structural source first:

    - a curve exposing `breakpoints(t0, t1, prop)` (`SegCurve` over the sim's
      recorded segment lanes): EXACT and free - the closed-form segments are
      translated straight across, at the sim's own resolution.
    - `EventTimeline`: EXACT. Its keyframes are translated directly - an instant
      (duration 0) becomes one hold breakpoint; a linearly-eased ramp becomes a
      (start-value, dur) breakpoint plus a (target, hold) breakpoint, matching
      EventTimeline's ease-from-previous playback; a non-linearly-eased ramp is
      densified across its own span at `_DENSE_DT` (the native ramp is linear,
      so a curved ease cannot be one breakpoint). Rest before the first keyframe
      is carried on the ChannelRef, so no pre-roll breakpoint is emitted.
    - anything else (any `.sample(t)` duck): dense-sampled at `_DENSE_DT`
      across `[t0, t1]` and collapsed to the breakpoints the reconstruction
      actually needs.

    `prop` selects the value index within the timeline's sampled tuple (link
    timelines are scalar, index 0). Returns `(ts, vals, durs, rest, eases)` -
    the five args of `DocBuilder.channel` - as plain Python lists + a float
    rest.

    Examples: called by `_channel_id` for every per-property link channel of a
    field instance, and by the fill-tint export.
    """
    structural = _export_structural(timeline, t0, t1, prop)
    if structural is not None:
        return structural
    if isinstance(timeline, EventTimeline):
        return _export_event_timeline(timeline, prop)
    if _is_static(timeline):
        return [], [], [], _rest_value(timeline, prop), []
    return _export_dense(timeline, t0, t1, prop)


def _export_structural(timeline, t0: float, t1: float, prop: int):
    """The curve's own breakpoints when it can produce them, else None.

    A curve backed by closed-form segments knows its shape exactly; asking it
    replaces a dense sampling walk (one Python call per sample per property)
    with a read of what the writer already recorded. A curve that declines -
    no such method, or a window it cannot answer for - falls through to the
    sampling paths."""
    export = getattr(timeline, 'breakpoints', None)
    if export is None:
        return None
    exported = export(t0, t1, prop)
    if exported is None:
        return None
    ts, vals, durs, eases = exported
    return ts, vals, durs, _rest_value(timeline, prop), eases


def _is_static(timeline) -> bool:
    """True when a curve advertises that it never moves (`is_static`), so the
    dense walk can be skipped for a bare rest channel. A curve that does not
    implement the probe is assumed animated - the safe direction, since a
    wrongly-skipped export would silently freeze real motion."""
    probe = getattr(timeline, 'is_static', None)
    return probe is not None and probe()


def _rest_value(timeline, prop: int) -> float:
    """The timeline's pre-first-keyframe rest value for `prop`. EventTimeline
    exposes it directly; a duck-typed curve is sampled far in the past."""
    rest = getattr(timeline, '_rest', None)
    if rest is not None:
        return float(rest[prop])
    return float(timeline.sample(-1.0e18)[prop])


def _export_event_timeline(timeline: EventTimeline, prop: int):
    """Exact breakpoints for an EventTimeline (see `export_channel`)."""
    keyframes = timeline._kf
    rest = _rest_value(timeline, prop)
    ts: list[float] = []
    vals: list[float] = []
    durs: list[float] = []
    eases: list[int] = []

    def emit(bt: float, value: float, dur: float) -> None:
        # Collapse a redundant hold onto the previous breakpoint of equal value
        # (keeps the channel minimal; the native sampler is unaffected).
        if ts and durs[-1] <= 0.0 and abs(vals[-1] - value) <= _REST_EPS \
                and dur <= 0.0:
            return
        ts.append(bt)
        vals.append(value)
        durs.append(dur)
        eases.append(_EASE_LINEAR)

    prev = rest
    for idx, kf in enumerate(keyframes):
        target = float(kf.values[prop])
        if kf.duration <= 0.0:
            emit(kf.t, target, 0.0)
            prev = target
            continue
        start = float(kf.start[prop]) if kf.start is not None else prev
        if kf.easing == _EASE_LINEAR:
            emit(kf.t, start, kf.duration)
            emit(kf.t + kf.duration, target, 0.0)
        else:
            # A curved ease: the native ramp is linear, so densify the span.
            _densify_span(timeline, prop, kf.t, kf.t + kf.duration,
                          ts, vals, durs, eases)
            emit(kf.t + kf.duration, target, 0.0)
        prev = target

    return ts, vals, durs, rest, eases


def _densify_span(timeline, prop, a: float, b: float, ts, vals, durs,
                  eases) -> None:
    """Append linear-ramp breakpoints tracing `timeline` across `[a, b)` at
    `_DENSE_DT`, so the piecewise-linear reconstruction follows the curve."""
    n = max(1, int(np.ceil((b - a) / _DENSE_DT)))
    step = (b - a) / n
    for k in range(n):
        bt = a + k * step
        ts.append(bt)
        vals.append(float(timeline.sample(bt)[prop]))
        durs.append(step)
        eases.append(_EASE_LINEAR)


def _export_dense(timeline, t0: float, t1: float, prop: int):
    """Dense-sample a duck-typed curve across `[t0, t1]` at `_DENSE_DT`,
    collapsed to the breakpoints the reconstruction needs.

    The fallback for a curve with no structural export: its shape can only be
    discovered by looking. What is emitted, though, is not the walk - a whole
    chart's worth of samples per property is hundreds of megabytes of
    breakpoints, nearly all of them redundant - but the corners
    `_collapse_ramps` keeps."""
    rest = _rest_value(timeline, prop)
    if t1 <= t0:
        return [], [], [], rest, []
    n = max(1, int(np.ceil((t1 - t0) / _DENSE_DT)))
    step = (t1 - t0) / n
    sample = timeline.sample
    ts = [t0 + k * step for k in range(n + 1)]
    vals = [float(sample(bt)[prop]) for bt in ts]
    ts, vals, durs = _collapse_ramps(ts, vals)
    return ts, vals, durs, rest, [_EASE_LINEAR] * len(ts)


# How far a collapsed reconstruction may stray from the samples it replaces.
# `SIMPLIFY_EPS`, the same bound the sim's own instant collapse holds: in
# design pixels / degrees / alpha units it is sub-visible.
_COLLAPSE_EPS = SIMPLIFY_EPS


def _collapse_ramps(ts, vals, eps: float = _COLLAPSE_EPS):
    """`(ts, vals, durs)` keeping only the samples a piecewise-LINEAR
    reconstruction of `(ts, vals)` needs to stay within `eps`.

    The slope corridor `SegmentTimeline.poke` maintains, in batch: each
    accepted point narrows the feasible slope interval from the anchor, and a
    point whose own chord slope falls outside it becomes the next anchor. Every
    kept span is emitted as a RAMP (never a hold), so a smooth run of samples
    reconstructs as motion rather than a staircase."""
    n = len(ts)
    if n == 0:
        return [], [], []
    out_ts, out_vals = [ts[0]], [vals[0]]
    anchor = 0
    lo, hi = -math.inf, math.inf
    j = 1
    while j < n:
        dt = ts[j] - ts[anchor]
        slope = (vals[j] - vals[anchor]) / dt
        if lo <= slope <= hi:
            tol = eps / dt
            lo, hi = max(lo, slope - tol), min(hi, slope + tol)
            j += 1
            continue
        anchor = j - 1
        out_ts.append(ts[anchor])
        out_vals.append(vals[anchor])
        lo, hi = -math.inf, math.inf
    if out_ts[-1] != ts[-1]:
        out_ts.append(ts[-1])
        out_vals.append(vals[-1])
    durs = [out_ts[i + 1] - out_ts[i] for i in range(len(out_ts) - 1)]
    durs.append(0.0)
    return out_ts, out_vals, durs


# --------------------------------------------------------------------------
# Storyboard elements - flatten, band, and (per kind) emit or skip
# --------------------------------------------------------------------------

# A group (ActorFrame) draws nothing itself; its transform composes onto its
# children. Groups are not drawn, but they are NOT discarded: each leaf carries
# its ancestor chain root-first and emits it as an `item_link` chain, so the
# engine's `local @ parent` nesting composes natively (see _flatten_elements).
_GROUP = 'group'


class _FrameCurve:
    """A `.sample(t)`-duck curve tracing a sheet sprite's current frame index
    over time, for `export_channel` to put on the item's frame lane.

    Mirrors `render._sheet_frame`: a recorded `state_pin` sampler wins; else the
    sheet auto-animates through `sheet_states` on the element's own clock. A
    plain 1x1 sprite has no states and no pin, so this rests at frame 0."""

    __slots__ = ('_element', '_rest')

    def __init__(self, element):
        self._element = element
        self._rest = (0.0,)

    def sample(self, t: float) -> tuple:
        element = self._element
        if element.state_pin is not None:
            return (float(element.state_pin.sample(t)[0]),)
        return (float(frame_at_time(element.sheet_states, t - element.t_start)),)

    def breakpoints(self, t0: float, t1: float, prop: int = 0):
        """The frame lane's steps as hold breakpoints, or None to be sampled.

        A frame lane is a step function whose changes are known in closed form
        - the sheet's own animation cycle, re-anchored by whatever
        `setstate`/`animate` the chart recorded - so both the pinned and the
        auto-animating sheet hand them over rather than being sampled for
        them. `limit` bounds the export by what the sampling walk it replaces
        would have cost; a sheet cycling faster than that is sampled instead."""
        element = self._element
        pin = element.state_pin
        limit = int((t1 - t0) / _DENSE_DT)
        if pin is not None:
            export = getattr(pin, 'steps', None)
            steps = None if export is None else export(t0, t1, limit)
        else:
            steps = frame_steps(element.sheet_states, float(element.t_start),
                                t0, t1, limit)
        if steps is None:
            return None
        ts, frames = steps
        return ts, [float(f) for f in frames], [0.0] * len(ts), [0] * len(ts)


class _FillSizeTimeline:
    """A fill's drawn extent on one axis: its absolute size when that is set,
    else its natural `w`/`h`.

    Mirrors `render._draw_size`'s precedence PER FRAME. It cannot be frozen at
    compile time: `_fill_size_as_wh` only mirrors `size_x` onto `w` when the
    size carries keyframes, so a Quad sized by a plain rest has `w = 0` and a
    real `size_x` - and a chart may clear an absolute size by writing a
    negative one, handing the axis back to `w`."""

    def __init__(self, size, natural):
        self._size = size
        self._natural = natural
        # `is not None`, NOT truthiness: EventTimeline.__bool__ is "has
        # keyframes", so a real timeline resting at 640 with no keyframes reads
        # as absent and the axis silently falls back to w.
        self._rest = (self._pick(
            _rest_value(size, 0) if size is not None else _SIZE_UNSET,
            _rest_value(natural, 0) if natural is not None else 0.0),)

    @staticmethod
    def _pick(size: float, natural: float) -> float:
        """The extent, or ZERO when neither source gives a real one.

        A shape has no natural box to fall back to, so "unset" has to mean
        "draws nothing" - which is what legacy's `w > 0 and h > 0` decides.
        Letting a negative through means "keep the natural box" to
        `record.draw_box`, and a fill's natural box is its TARGET: an unsized
        rect then covered the whole screen instead of drawing nothing."""
        extent = size if size >= 0.0 else natural
        return extent if extent >= 0.0 else 0.0

    def is_static(self) -> bool:
        return ((self._size is None or _is_static(self._size))
                and (self._natural is None or _is_static(self._natural)))

    def sample(self, t):
        size = (self._size.sample(t)[0] if self._size is not None
                else _SIZE_UNSET)
        natural = (self._natural.sample(t)[0] if self._natural is not None
                   else 0.0)
        return (self._pick(size, natural),)


class _GlyphCropTimeline:
    """One glyph's share of a bitmaptext run's horizontal crop.

    SM crops the whole TEXT ACTOR, but the doc draws a run as one item per
    glyph, and `compose_links` takes its crop from the LEAF link - so a run
    crop left on the element is dropped entirely. Windfall hides its seizure
    warning with `cropright(1)` and nothing else, which is why it covered the
    screen.

    The share is computed in ADVANCE space: the run spans `[0, total]`, this
    glyph `[start, start + advance]`, and the visible window is
    `[total*left, total*(1 - right)]`. Advance space is not cell space - a
    glyph's drawn cell may be wider than its advance - so a partly-cropped
    glyph is off by that difference. It is exact where it matters (fully in,
    fully out) and the reveal it drives is a wipe, not a measurement.
    """

    def __init__(self, crop, total: float, start: float, advance: float,
                 leading: bool):
        self._crop = crop
        self._total = float(total)
        self._start = float(start)
        self._advance = float(advance)
        self._leading = leading
        self._rest = (self._share(_rest_value(crop, 0)),)

    def _share(self, fraction: float) -> float:
        if self._advance <= 0.0:
            return 1.0 if fraction > 0.0 else 0.0
        if self._leading:
            cut = self._total * fraction - self._start
        else:
            cut = self._start + self._advance - self._total * (1.0 - fraction)
        return min(1.0, max(0.0, cut / self._advance))

    def is_static(self) -> bool:
        return _is_static(self._crop)

    def sample(self, t):
        return (self._share(self._crop.sample(t)[0]),)


class _SpanTimeline:
    """The signed distance between two scalar timelines, as a timeline.

    A fit rect is recorded as four edges but only its EXTENT scales the
    sprite, so the doc sends `right - left` and `bottom - top` rather than
    four lanes the executor would immediately subtract. Static when both
    edges are, which is the usual case (a rect set once), so the common path
    exports one constant rather than a dense walk."""

    def __init__(self, lo, hi):
        self._lo = lo
        self._hi = hi
        self._rest = (_rest_value(hi, 0) - _rest_value(lo, 0),)

    def is_static(self) -> bool:
        return _is_static(self._lo) and _is_static(self._hi)

    def sample(self, t):
        return (self._hi.sample(t)[0] - self._lo.sample(t)[0],)


def _read_shader_source(path):
    """A chart shader's GLSL text, or None when there is no path or it cannot
    be read. Read at COMPILE time (a worker thread) so the doc carries its own
    source and the render thread never touches the filesystem."""
    if not path:
        return None
    try:
        return Path(path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None


def _flatten_elements(elements) -> list:
    """The element tree flattened to a leaf list in tree order, each leaf paired
    with its ANCESTOR CHAIN (root-first, groups only). Returns
    `(pairs, group_count)` where each pair is `(leaf, ancestors)`.

    A group draws nothing itself but carries a transform that composes onto its
    subtree, so dropping it strands its children at their own local placement -
    a background frame that rotates/vibrates its whole subtree leaves the
    subtree still. The chain rides to `item_link`, which composes it
    engine-natively."""
    pairs = []
    groups = 0
    stack = [(element, ()) for element in reversed(list(elements))]
    while stack:
        element, ancestors = stack.pop()
        if element.kind == _GROUP:
            groups += 1
            chain = ancestors + (element,)
            stack.extend((child, chain)
                         for child in reversed(list(element.children)))
            continue
        pairs.append((element, ancestors))
    return pairs, groups


class _Unit(NamedTuple):
    """One thing the doc draws, with the key that orders it.

    `kind` is 'element' (payload `(leaf, ancestors)`), 'instance' (payload the
    instance dict), 'z_run' (payload a list of instances sharing a z_group,
    which must stay contiguous so the SortSpan that wraps them can name its
    own length) or 'base_field' (no payload)."""
    band: float
    z: int
    z_index: int
    tree_index: int
    seq: int
    kind: str
    payload: object


# The band/z/z_index a FIELD INSTANCE sorts at: the notefield's own, which is
# the plain z=0 layer everything unbanded shares.
_FIELD_BAND = (0.0, 0, 0)


def _tree_order_units(elements, instances):
    """Elements and field instances as ONE stream, in the order the engine
    draws them.

    The engine walks its actor tree once, and charts rely on that: an AFT-rig
    curtain has to land between the node that captured the scene and the
    sampler that redraws it, and both sit at z=0. Emitting elements in a band
    before the instance stream and a band after cannot express that, so
    anything in the middle came out at one end.

    Key, outermost first:

    - BAND, because the compiler HOISTS a BGCHANGES subtree to a
      below-the-notes z (`modfile._with_z`) and it must stay behind the field
      wherever its actors sit;
    - the leaf's own Z and Z_INDEX, the storyboard layer sort
      `render.StoryboardRenderer` applies, so an explicit layering still wins;
    - TREE INDEX, document order - the engine's own, and the tiebreak that
      matters, since a NotITG chart leaves nearly everything at z=0;
    - SEQ, so the sort is stable for entries carrying no tree index.

    A z_group run is ONE unit: its members must stay contiguous, because the
    SortSpan the emitter wraps them in names how many commands follow it.

    An instance with no tree index (a synthetic player) inherits the last one
    seen, keeping its place in the instance list instead of sorting to the
    front.
    """
    units: list[_Unit] = []
    seq = 0
    for leaf, ancestors in elements:
        units.append(_Unit(_band_z((leaf, ancestors)), leaf.z, leaf.z_index,
                           leaf.tree_index, seq, 'element', (leaf, ancestors)))
        seq += 1

    # The base field is legacy's prepended `(None, 1.0, 'field')` entry, so it
    # leads the band rather than taking a tree position of its own.
    units.append(_Unit(*_FIELD_BAND, -1, seq, 'base_field', None))
    seq += 1

    index = -1
    run: list = []
    run_group = None

    def close_run():
        nonlocal run, run_group, seq
        if run:
            units.append(_Unit(*_FIELD_BAND, _instance_index(run[0], index),
                               seq, 'z_run', run))
            seq += 1
        run, run_group = [], None

    for inst in instances:
        index = _instance_index(inst, index)
        group = inst.get('z_group')
        if group is not None and group == run_group:
            run.append(inst)
            continue
        close_run()
        if group is not None:
            run, run_group = [inst], group
            continue
        units.append(_Unit(*_FIELD_BAND, index, seq, 'instance', inst))
        seq += 1
    close_run()

    units.sort(key=lambda u: (u.band, u.z, u.z_index, u.tree_index, u.seq))
    return units


def _instance_index(inst, fallback: int) -> int:
    """An instance's document position, or the last one seen."""
    index = inst.get('tree_index')
    return fallback if index is None else index


def _band_z(pair) -> float:
    """The band z for a `(leaf, ancestors)` pair: the TOP-LEVEL element's z.

    The compiler puts the band on the root of a hoisted subtree, not on its
    leaves (modfile `_with_z`: "the band z lives on the TOP-LEVEL element (the
    renderer bands by it)"), so a background subtree hoisted to z=-100 has
    leaves still resting at z=0. Reading the leaf's own z drops the hoist and
    bands the whole background ABOVE the field."""
    leaf, ancestors = pair
    return float(ancestors[0].z if ancestors else leaf.z)


# --------------------------------------------------------------------------
# build_static_doc - the tree-order static-scene compiler
# --------------------------------------------------------------------------

# The item_link parameters, in call order, paired with the link-timeline prop
# name they read. `zoom_*`/`rot` rename `scale_*`/`rotation`; crop lanes rename
# the crop_* edges; halign/valign/base_scale carry through. natural_w/h are
# constants (the 640x480 capture), passed as rests.
#
# The OUT-OF-PLANE props (z, rotation_x/y, scale_z, base_scale_z) are here
# because both ends have always supported them and only this list dropped
# them: `camera::local_matrix` composes [x, y, z] / [rotation_x, rotation_y,
# rot] / scale_z * base_scale_z (native/src/transform.rs:170-180), and the
# legacy `TransformChannel` composes exactly the same (field_compose.py:266-269).
# Omitting them silently discarded every 3D field transform on the drawable
# path - a chart whose whole effect is `a:rotationy(...)` simply lost it.
_LINK_PROP_ORDER = (
    ('x', 'x'), ('y', 'y'), ('z', 'z'),
    ('zoom_x', 'scale_x'), ('zoom_y', 'scale_y'), ('scale_z', 'scale_z'),
    ('rot', 'rotation'),
    ('rotation_x', 'rotation_x'), ('rotation_y', 'rotation_y'),
    ('skew_x', 'skew_x'), ('skew_y', 'skew_y'),
    ('base_scale_x', 'base_scale_x'), ('base_scale_y', 'base_scale_y'),
    ('base_scale_z', 'base_scale_z'),
    ('halign', 'halign'), ('valign', 'valign'),
    ('hidden', 'hidden'), ('awake', 'awake'), ('alpha', 'alpha'),
    ('crop_l', 'crop_left'), ('crop_t', 'crop_top'),
    ('crop_r', 'crop_right'), ('crop_b', 'crop_bottom'),
)


# The `item_link` parameters an ELEMENT chain link carries, paired with the
# element-timeline prop each reads. Mirrors _LINK_PROP_ORDER (the field-instance
# chain) plus the out-of-plane terms elements animate: a group's rotationx /
# rotationy / z / zoomz. Anchor (halign/valign) and crop apply to the LEAF only,
# and the binding's defaults cover every prop an element never sets.
_ELEMENT_LINK_PROPS = (
    ('x', 'x'), ('y', 'y'),
    ('zoom_x', 'scale_x'), ('zoom_y', 'scale_y'), ('scale_z', 'scale_z'),
    ('rot', 'rotation'),
    ('rotation_x', 'rotation_x'), ('rotation_y', 'rotation_y'),
    ('z', 'z'),
    ('skew_x', 'skew_x'), ('skew_y', 'skew_y'),
    ('halign', 'halign'), ('valign', 'valign'),
    ('hidden', 'hidden'), ('awake', 'awake'), ('alpha', 'alpha'),
    ('crop_l', 'crop_left'), ('crop_t', 'crop_top'),
    ('crop_r', 'crop_right'), ('crop_b', 'crop_bottom'),
)

# The SM LoadMenuPerspective default; a chain resting here needs no camera.
_DEFAULT_FOV = 45.0
_FOV_EPS = 1e-4

# Elements and field instances share the record's ONE tag lane. Element tags
# count from 1; an instance tag is this base plus its index, so a reader tells
# the two apart by magnitude and 0 still means "untagged" (a fed note).
_INSTANCE_TAG_BASE = 1 << 20


def notes_inline() -> bool:
    """Whether the base field draws its notes as INLINE ITEMS (note_feed)
    rather than blitting one captured notefield.

    THE DEFAULT for the unified path: an item placed by its own mat3 cannot be
    clipped at a capture box it never should have had, which is what cut
    mod-displaced notes off. `VSRG_DRAWABLE_NOTES=0` reverts to the captured
    notefield for differential testing."""
    return os.environ.get('VSRG_DRAWABLE_NOTES', '1').lower() in ('1', 'true', 'yes')


def elements_in_doc() -> bool:
    """Whether the doc emits the chart's storyboard ELEMENTS as its own image
    items. OFF by default, because today it double-draws them.

    The legacy `StoryboardEffect` paints the same `compiled['tree']`
    unconditionally (it has no idea the pipeline exists), so with this on every
    image element composites TWICE - and the doc's copy is the WORSE of the
    two: it carries no anchor (the compiler stamps `origin=(0.5, 0.5)`, which
    `_ELEMENT_ITEM_LANES` never forwards), no natural size, no tint, no
    zoomto/fit basis and no glow, so it lands displaced, white and mis-sized
    while legacy's lands correctly. Suppressing the wrong copy is a pure
    subtraction that returns element rendering to legacy exactly as it bands
    them; suppressing legacy's instead would keep the wrong one.

    `VSRG_DRAWABLE_ELEMENTS=1` re-enables them for parity work - the element
    lanes have to reach parity against a harness BEFORE this becomes the
    default and the legacy effect can be retired."""
    return os.environ.get('VSRG_DRAWABLE_ELEMENTS', '0').lower() in ('1', 'true', 'yes')


# The link props that take a chain off the z=0 plane, with their rest values.
# Mirrors `transform::is_planar` (native/src/transform.rs): while every one of
# these rests, the design projection is the identity and the camera fold is a
# no-op; the moment one moves, the fold is what turns a rotation into a 3D turn
# rather than a horizontal squash.
_OUT_OF_PLANE_RESTS = (('rotation_x', 0.0), ('rotation_y', 0.0), ('z', 0.0),
                       ('scale_z', 1.0), ('base_scale_z', 1.0))


def _link_is_planar(link) -> bool:
    """Whether one link stays on the z=0 plane for the whole chart.

    A prop counts as off-plane when it is ever poked off its identity value -
    a chain pinned at rotation_y=30 needs the camera just as much as one
    sweeping through it, so `_moves_off` covers both the keyframed and the
    resting case.

    `_is_static` is the WRONG probe here, and using it made this return False
    for EVERY link: a plain keyframe-less EventTimeline answers False to the
    static probe (the safe direction for export, not for a predicate about
    motion). Every field chain therefore carried a perspective camera it never
    needed, and the field parity harness classified every instance as 3D and
    compared nothing at all."""
    return not any(_moves_off(link.get(prop), rest)
                   for prop, rest in _OUT_OF_PLANE_RESTS)


def _rotation_order_of(element) -> str:
    """The element's Euler rotation-order token (SetRotationOrder), or the
    stock RageMatrix 'xyz'. The order is a token rather than an animatable
    scalar, so it reads as the timeline's current value, not a channel."""
    timeline = element.timelines.get('rotation_order')
    if timeline is None:
        return 'xyz'
    (order,) = timeline.sample(0.0)
    return str(order or 'xyz')


class _Ctx:
    """Minimal effect-sampling context: `NotitgFieldInstances.at` reads only
    `t_now` and `chart_rect`."""

    __slots__ = ('t_now', 'chart_rect')

    def __init__(self, t, chart_rect):
        self.t_now = t
        self.chart_rect = chart_rect


def _current_instances(compiled) -> list:
    """The current field-instance list: `field_instances` is a provider
    callable (lazy topology) or a fixed sequence."""
    provider = compiled.get('field_instances')
    if provider is None:
        return []
    return list(provider() if callable(provider) else provider)


def _aft_slot_key(inst) -> str:
    """The freeze/slot key an aft sampler blits (field_instances._extra): its
    isolated upstream capture_source, else its source node, else its own name."""
    return (inst.get('capture_source') or inst.get('aft_node') or inst['name'])


def _horizon(compiled, default: float = 600.0) -> float:
    """The channel export window end. Prefer an explicit compiled horizon,
    then the live sim's own end - the point past which nothing is recorded and
    every channel holds its tail anyway, so exporting beyond it is pure cost.
    Absent both, a generous default."""
    for key in ('duration', 'horizon', 'song_length'):
        value = compiled.get(key)
        if isinstance(value, (int, float)) and value > 0.0:
            return float(value)
    end = getattr(compiled.get('_live_sim'), '_end_seconds', None)
    if isinstance(end, (int, float)) and end > 0.0:
        return float(end)
    return default


class _Builder:
    """Threads the DocBuilder through one static-doc build: mints the field /
    slot drawables, exports channels (de-duplicating identical timelines), and
    emits each instance's commands. Not reused across builds."""

    def __init__(self, compiled, screen_w, screen_h, builder=None):
        import storyboard_native as sn

        self._sn = sn
        self._compiled = compiled
        self._t0 = 0.0
        self._t1 = _horizon(compiled)
        # `builder` injection: a _RecordingBuilder lets the EXPENSIVE part of
        # a build (the channel exports - tens of seconds of pure Python) run
        # on a worker thread; the recorded ops replay onto a real DocBuilder
        # on the thread the unsendable Evaluator must live on (see
        # prepare_static_doc / assemble_static_doc).
        self._builder = builder if builder is not None \
            else sn.DocBuilder(float(screen_w), float(screen_h))
        self._screen = (float(screen_w), float(screen_h))
        self._field_ids: dict[str, int] = {}
        self._slot_ids: dict[str, int] = {}
        # Storyboard image sources: an image id per DISTINCT absolute asset path
        # (loading is the consumer's job). id_maps carries the {id -> path} map.
        self._image_ids: dict[str, int] = {}
        self._image_paths: dict[int, str] = {}
        self._image_grids: dict[int, tuple] = {}
        # Logical size per minted DrawableId (the screen is id 0). Exported
        # so the executor allocates each drawable at its REAL size instead
        # of assuming everything is screen-shaped.
        self._drawable_sizes: dict[int, tuple] = {0: self._screen}
        # Inline note-feed slots by field scope (see notes_inline); empty
        # when fields blit captured notefields instead. One slot per scope,
        # shared by every consumer of that scope's notes - each Feed command
        # composes its own link chain over the same fed items.
        self._notes_slots: dict[str, int] = {}
        # Per-band emitted counts + per-kind skip counts, surfaced in the report.
        self._elem_below = 0
        self._elem_above = 0
        self._elem_skips: dict[str, int] = {}
        # Memoize channel ids by timeline object identity + prop, so a link's
        # shared rest timelines (a whole field of untouched props) collapse to
        # one channel each.
        self._chan_cache: dict[tuple, tuple[int, float]] = {}
        # `id()` only identifies an object while it is ALIVE. Some callers key
        # on a TEMPORARY - `_element_frame_kwarg` builds a `_FrameCurve` inline
        # per element - and CPython recycles the freed address immediately, so
        # the next element would hit this cache and inherit the previous one's
        # channel (every sheet sprite animating on one shared frame lane).
        # Holding a reference for the build's lifetime makes the key honest.
        self._chan_keyed: list = []
        # The shared 'player'-kind visibility gate (see _base_hidden_gate).
        self._base_gate = None
        # The image blits this build emits, in draw order, as
        # `(element, role)` - the element parity harness pairs the record
        # stream against this rather than re-deriving which elements glow
        # (a glowing element emits a SECOND image blit for its glow pass).
        self._element_order: list[tuple] = []
        # The FIELD-INSTANCE commands this build emits, in draw order, as
        # `(instance, command)` - the field parity harness pairs the record
        # stream against this. Snapshots and feeds are recorded alongside items
        # even though only an item can carry a tag, so an instance that emits
        # no comparable quad reads as "not a quad" rather than "gated off".
        self._instance_order: list[tuple] = []
        # `text` elements whose glyphs the consumer rasterises: image id ->
        # (text, pixel size). See `_text_image_id`.
        self._text_images: dict[int, tuple] = {}
        # Asset size CONVENTIONS per image id (see `_record_size_spec`): the
        # consumer needs them to resolve a logical box the way the funnel does.
        self._image_specs: dict[int, tuple] = {}
        # Per-item fragment programs: one id per distinct (frag, vert, names),
        # with the descs exported positionally for GLExecutor.set_shaders.
        self._shader_ids: dict[tuple, int] = {}
        self._shader_descs: list[tuple] = []

    # -- drawable minting -------------------------------------------------

    def _new_drawable(self, persistent: bool, size=None) -> int:
        """Mint one drawable and record its logical size in the exported
        size table. `size` is the drawable's logical box; None = the design
        screen, which is a DELIBERATE choice at each call site, not a
        default shape: an AFT slot is an engine screen capture and a field
        drawable is the design-space capture the link-chain homographies
        map (TransformChannel's 640x480 content contract). A drawable for
        anything else should pass its content size - a plain drawable is
        only as big as what it draws."""
        w, h = size if size is not None else self._screen
        drawable = self._builder.drawable(float(w), float(h),
                                          persistent=persistent,
                                          dynamic=False)
        self._drawable_sizes[drawable] = (float(w), float(h))
        return drawable

    def _field_drawable(self, scope: str) -> int:
        drawable = self._field_ids.get(scope)
        if drawable is None:
            drawable = self._new_drawable(persistent=False)
            self._field_ids[scope] = drawable
        return drawable

    def _slot_drawable(self, name: str) -> int:
        drawable = self._slot_ids.get(name)
        if drawable is None:
            drawable = self._new_drawable(persistent=True)
            self._slot_ids[name] = drawable
        return drawable

    def _notes_slot_for(self, scope: str) -> int:
        slot = self._notes_slots.get(scope)
        if slot is None:
            slot = self._builder.feed_slot()
            # The exported size table is positional by drawable id, so the
            # slot needs an entry even though it never becomes a render
            # target (inline: its items draw into the enclosing drawable).
            self._drawable_sizes[slot] = (0.0, 0.0)
            self._notes_slots[scope] = slot
        return slot

    # -- channel export ---------------------------------------------------

    def _channel(self, timeline, prop: int = 0, window=None) -> tuple[int, float]:
        """The (channel_id, rest) for a timeline+prop, exported once and
        memoized. id < 0 sentinel is never returned here - a real channel is
        always pushed (the rest still rides the ChannelRef).

        `window` narrows the export to a `(t0, t1)` sub-range of the build's
        own window, for a lane nothing reads outside it."""
        t0, t1 = window if window is not None else (self._t0, self._t1)
        key = (id(timeline), prop, t0, t1)
        cached = self._chan_cache.get(key)
        if cached is not None:
            return cached
        ts, vals, durs, rest, eases = export_channel(timeline, t0, t1, prop)
        chan_id = self._builder.channel(ts, vals, durs, float(rest), eases)
        result = (chan_id, float(rest))
        self._chan_cache[key] = result
        self._chan_keyed.append(timeline)
        return result

    def _link_kwargs(self, link, flip_base_y: bool) -> dict:
        """The `item_link` keyword args for one link dict: an (id, rest) channel
        per transform prop, natural size as constant rests, and the leaf flip."""
        kwargs: dict[str, object] = {}
        for param, prop in _LINK_PROP_ORDER:
            timeline = link.get(prop)
            if timeline is None:
                continue
            chan_id, rest = self._channel(timeline)
            kwargs[f'{param}_id'] = chan_id
            kwargs[f'{param}_rest'] = rest
        kwargs['natural_w_rest'] = _SCREEN_W
        kwargs['natural_h_rest'] = _SCREEN_H
        kwargs['flip_base_y'] = flip_base_y
        # The Euler order is a TOKEN, not an animatable scalar, so it reads as
        # the timeline's current value rather than riding a channel - the same
        # treatment `_element_link_kwargs` gives it. It only matters once the
        # out-of-plane rotations above are non-rest.
        order = link.get('rotation_order')
        if order is not None:
            kwargs['rotation_order'] = str(order.sample(self._t0)[0]
                                           or 'xyz')
        return kwargs

    def _emit_links(self, target: int, inst, camera: bool = True) -> None:
        """Attach the instance's full leaf-link chain (root-first) to the item
        most recently pushed onto `target`, plus the chain's perspective camera
        when the chain leaves the z=0 plane. The leaf link carries the aft flip
        (field_compose sets TransformChannel.flip_base_y for aft/stage kinds).

        Legacy ALWAYS folds through `field_projection.design_projection`
        (field_compose.py:216) - the camera is not optional there. Without an
        `item_projection` the native fold takes the z=0 block and skips
        `fold_projection` (evaluate.rs), so a rotated field comes out a flat
        horizontal squash instead of a 3D turn: at gat's t=78 legacy gives a
        trapezoid (near edge 745px tall, far edge 354px) where the unfolded
        block gives a plain rectangle, ~232px of corner error.

        `camera=False` attaches the chain WITHOUT its projection, for a tail
        command that reads the chain as a liveness gate rather than as
        geometry. A capture is the case: `snapshot_live` (evaluate.rs) asks
        only whether the chain composes at all, and folds it with no
        projection, so a camera would be state nothing reads - and `Cmd
        ::Snapshot` accordingly has no projection slot to put it in."""
        links = inst['transform']._links
        flip = getattr(inst['transform'], '_flip_base_y', False)
        leaf = len(links) - 1
        for i, link in enumerate(links):
            self._builder.item_link(
                target, **self._link_kwargs(link, flip and i == leaf))
        chain_camera = self._chain_camera(links) if camera else None
        if chain_camera is not None:
            fov_id, fov_rest = chain_camera
            self._builder.item_projection(
                target, fov_id=fov_id, fov_rest=fov_rest,
                vanish_x_rest=_SCREEN_W / 2.0,
                vanish_y_rest=_SCREEN_H / 2.0)

    def _chain_camera(self, links):
        """`(fov channel id, rest)` for a field chain's camera, or None when
        the chain never leaves the z=0 plane.

        The gate is PLANARITY, not fov - which is where this differs from the
        element chain's `_chain_fov`. At the default fov the projection is the
        identity only ON the z=0 plane; the moment a link carries rotation_x,
        rotation_y, z, scale_z or base_scale_z the fold is required even though
        the fov never moved. gat is exactly that case (its fov is static at the
        default while `rotationy` sweeps), so an fov-based gate would skip the
        camera and change nothing. Mirrors `transform::is_planar`.

        The fov itself still follows the engine: a frame's fov projects its
        whole subtree and the INNERMOST one that set a non-default value wins
        (field_compose.py:203-209)."""
        if all(_link_is_planar(link) for link in links):
            return None
        for link in reversed(links):
            timeline = link.get('fov')
            if timeline is None:
                continue
            chan_id, rest = self._channel(timeline)
            if abs(rest - _DEFAULT_FOV) > _FOV_EPS or not _is_static(timeline):
                return chan_id, rest
        return -1, _DEFAULT_FOV

    # -- instance emission ------------------------------------------------

    def _emit_blit(self, source_kind: int, source_id: int, inst,
                   visible=(-1, 1.0)) -> None:
        z_id, z_rest, has_z = self._z_channel(inst)
        visible_id, visible_rest = visible
        self._builder.item(_SCREEN_ID, source_kind, source_id,
                           z_id=z_id, z_rest=z_rest, has_z=has_z,
                           visible_id=visible_id, visible_rest=visible_rest)
        self._builder.item_tag(_SCREEN_ID, self._tag_instance(inst, 'item'))
        self._emit_tint(_SCREEN_ID, inst)
        self._emit_blend(_SCREEN_ID, inst)
        self._emit_shader(_SCREEN_ID, inst)
        self._emit_links(_SCREEN_ID, inst)

    def _emit_tint(self, target: int, inst) -> None:
        """Attach the instance's diffuse rgb to the item just pushed, when it
        carries one. `inst['color']` is a 3-vector timeline (one channel per
        component); absent it, the item keeps its white rest."""
        color = inst.get('color')
        if color is None:
            return
        chans = [self._channel(color, prop) for prop in range(3)]
        (r_id, r_rest), (g_id, g_rest), (b_id, b_rest) = chans
        self._builder.item_tint(target, r_id=r_id, r_rest=r_rest,
                                g_id=g_id, g_rest=g_rest,
                                b_id=b_id, b_rest=b_rest)

    def _emit_shader(self, target: int, inst) -> None:
        """Attach the instance's `Frag=` program and bind its uniform curves
        to the item just pushed.

        A NotITG `Frag=` on a Sprite is a per-actor program run over that
        actor's own texture - here the source node's AFT capture. Without it
        the sampler blits the capture unshaded, which is not a cosmetic loss:
        a rig whose whole visual IS the shader (horizon, monitor, lumikey)
        degrades to a plain copy of the screen."""
        registered = self._shader_for(inst)
        if registered is None:
            return
        shader_id, names = registered
        self._builder.item_shader(target, shader_id)
        uniforms = inst.get('frag_uniforms') or {}
        for index, name in enumerate(names):
            chan_id, rest = self._channel(uniforms[name])
            self._builder.item_uniform(target, index, ch_id=chan_id,
                                       ch_rest=rest)

    def _shader_for(self, inst):
        """`(shader_id, uniform names)` for an instance carrying a `Frag=`,
        or None. Registered once per (source, vert, names) so samplers that
        share a program share its id - the executor stashes each BLIT's own
        uniform window before resolving, so a shared id stays per-item.

        The uniform NAME ORDER is the contract between `item_uniform`'s index
        and the executor's `uniform_names` lookup, so it is sorted here and
        exported unchanged rather than re-derived downstream."""
        frag_path = inst.get('frag')
        mesh = inst.get('mesh') or {}
        vert_path = mesh.get('vert') or None
        if frag_path is None and vert_path is None:
            return None

        frag_src = _read_shader_source(frag_path)
        vert_src = _read_shader_source(vert_path)
        if frag_src is None and vert_src is None:
            self._elem_skips['shader_unreadable'] = (
                self._elem_skips.get('shader_unreadable', 0) + 1)
            return None

        names = tuple(sorted(inst.get('frag_uniforms') or {}))
        key = (frag_src, vert_src, names)
        registered = self._shader_ids.get(key)
        if registered is None:
            registered = self._builder.shader(frag_src or '', vert=vert_src,
                                              uniform_names=list(names))
            self._shader_ids[key] = registered
            self._shader_descs.append((frag_src, vert_src, list(names)))
        return registered, names

    def _emit_blend(self, target: int, inst) -> None:
        """Attach the instance's additive-blend gate to the item just pushed.

        Every instance kind gets it, not just aft samplers: `blend('add')` is
        an Actor verb a chart can call on anything, at any time, and sampling
        it once at build time bakes whatever the rest happened to be."""
        blend = inst.get('blend_add')
        if blend is None:
            return
        chan_id, rest = self._channel(blend)
        self._builder.item_blend(target, ch_id=chan_id, ch_rest=rest)

    def _z_channel(self, inst):
        """(z_id, z_rest, has_z) for an instance's SortSpan sort key. The
        z_sort timeline is scalar; absent -> no z (pure insertion order)."""
        z_sort = inst.get('z_sort')
        if inst.get('z_group') is None or z_sort is None:
            return -1, 0.0, False
        chan_id, rest = self._channel(z_sort)
        return chan_id, rest, True

    # -- entry point ------------------------------------------------------

    def run(self):
        """Emit ONE draw stream in tree order - storyboard elements and field
        instances interleaved by where their actors sit in the document - and
        finish.

        The engine draws its actor tree in one pass, and a chart relies on
        that: an AFT-rig curtain quad has to land between the node that
        captured the scene and the sampler that redraws it, and both sit at
        z=0. Splitting elements into a band before the instance stream and a
        band after cannot express that, so anything a chart put in the middle
        came out at one end - which is how gat 1's freeze went black.

        The SORT KEY is `(band z, tree index)`. Band z first because a
        BGCHANGES subtree is HOISTED to a below-the-notes z by the compiler
        (`modfile._with_z`) and must stay behind the field wherever its actors
        happen to sit; tree index within a band because that is the engine's
        own order. Instances take band 0, the notefield's own band.
        """
        instances = self._current_instances_ensured()
        elements = (self._owned_elements(instances)
                    if elements_in_doc() else ())

        below = above = 0
        for unit in _tree_order_units(elements, instances):
            match unit.kind:
                case 'element':
                    if self._emit_element(*unit.payload):
                        below += unit.band < 0
                        above += unit.band >= 0
                case 'z_run':
                    self._emit_z_run_units(unit.payload)
                case 'base_field':
                    self._emit_base_field(instances)
                case _:
                    self._emit_instance(unit.payload)
        self._elem_below, self._elem_above = below, above

        evaluator = self._builder.finish()
        id_maps = {'screen': _SCREEN_ID, 'slots': dict(self._slot_ids),
                   'fields': dict(self._field_ids),
                   'images': dict(self._image_paths),
                   'image_grids': dict(self._image_grids),
                   'drawable_sizes': [self._drawable_sizes[i]
                                      for i in sorted(self._drawable_sizes)],
                   'notes_slot': self._notes_slots.get('field'),
                   'note_feeds': dict(self._notes_slots),
                   'shaders': list(self._shader_descs),
                   'text_images': dict(self._text_images),
                   'image_specs': dict(self._image_specs),
                   'element_order': list(self._element_order),
                   'instance_order': list(self._instance_order)}
        return evaluator, id_maps

    def _owned_elements(self, instances):
        """The `(leaf, ancestors)` pairs the DOC draws, from `compiled['tree']`.
        The group count is folded into the skip tally so the report accounts
        for it (a group is not drawn, but it is composed - `_flatten_elements`).

        Leaves already owned by a FIELD INSTANCE are dropped. An AFT-rig
        curtain quad is one actor that both producers claim: the field walk
        emits it as a 'fill', and the element tree compiles the same actor as
        a rect. Drawing both put a second copy of gat 1's blackout curtain
        after the AFT sampler that is supposed to hide behind it, and the
        section went black - the video oracle shows the black background with
        the frozen playfield ON TOP."""
        tree = self._compiled.get('tree') or ()
        leaves, groups = _flatten_elements(tree)
        if groups:
            self._elem_skips['group'] = self._elem_skips.get('group', 0) + groups
        claimed = {inst['tree_index'] for inst in instances
                   if inst.get('tree_index') is not None}
        kept = [pair for pair in leaves if pair[0].tree_index not in claimed]
        dropped = len(leaves) - len(kept)
        if dropped:
            self._elem_skips['owned_by_field'] = (
                self._elem_skips.get('owned_by_field', 0) + dropped)
        return kept

    def _current_instances_ensured(self) -> list:
        instances = _current_instances(self._compiled)
        self._ensure_player_consumers(instances)
        return instances

    def _emit_base_field(self, instances) -> None:
        """The single-player base original: an identity blit of the primary
        'field' capture, drawn FIRST (mirrors NotitgFieldInstances._single_frame
        prepending `(None, 1.0, 'field')`). Gated visible by `1 - base_hidden`,
        so when the chart hides the real field the item drops (matching the
        base-hidden placeholder). Suppressed on the dual-player path, where the
        player instances ARE the originals and no base is prepended."""
        spec = self._compiled.get('player_fields')
        if spec is not None and getattr(spec, 'note_mods', None):
            return
        visible_id, visible_rest = self._visible_from_hidden(
            self._compiled.get('base_field_hidden'))
        if notes_inline():
            # NOTES AS ITEMS: the base field's receptors and notes draw as
            # ordinary screen items, each placed by its own mat3, instead of
            # blitting one captured notefield. Nothing but the real render
            # target bounds them, so a mod that pushes a note outside the
            # playfield no longer clips it at a capture box.
            self._builder.feed_inline(_SCREEN_ID, self._notes_slot_for('field'),
                                      visible_id=visible_id,
                                      visible_rest=visible_rest)
            return
        field = self._field_drawable('field')
        self._builder.item(_SCREEN_ID, self._sn.SRC_DRAWABLE, field,
                           visible_id=visible_id, visible_rest=visible_rest)

    def _visible_from_hidden(self, hidden):
        """(visible_id, visible_rest) inverting a `base_field_hidden` timeline
        into the item's `visible` gate. Absent -> constant visible. A present
        timeline is exported inverted (hidden >= 0.5 -> visible < 0.5), so the
        native visibility gate drops the base while the chart hides the field."""
        if hidden is None:
            return -1, 1.0
        ts, vals, durs, rest, eases = export_channel(hidden, self._t0, self._t1)
        chan_id = self._builder.channel(ts, [1.0 - v for v in vals], durs,
                                        1.0 - float(rest), eases)
        return chan_id, 1.0 - float(rest)

    def _ensure_player_consumers(self, instances) -> None:
        """Mirror NotitgFieldInstances.at: mint per-player consumers before
        scopes resolve, so a proxy of player N > 1 keys its own field{N}."""
        spec = self._compiled.get('player_fields')
        if spec is None:
            return
        spec.ensure({player for inst in instances
                     if inst.get('kind') in ('proxy', 'player')
                     and (player := inst.get('player') or 1) > 1})

    def _emit_z_run_units(self, members) -> None:
        """Emit a SortSpan wrapping one z_group run, then its members.

        `_tree_order_units` keeps a run contiguous precisely so this can
        work: the SortSpan precedes its members and names how many commands
        follow, so anything sorted into the middle would be swept into the
        span's z ordering. The length counts only members that actually emit
        a command - a 'stage' emits none; captures/fills/blits emit one."""
        span_len = sum(1 for inst in members if inst.get('kind') != 'stage')
        self._builder.sort_span(_SCREEN_ID, span_len)
        for inst in members:
            self._emit_instance(inst)

    def _emit_instance(self, inst) -> None:
        sn = self._sn
        kind = inst['kind']
        match kind:
            case 'stage':
                # An isolating AFT node: never a draw of its own (its transform
                # is folded into downstream consumers at sample time). Skipped -
                # the static doc's fold is the Snapshot/slot topology.
                return
            case 'capture':
                # The node's OWN chain gates the slot update: the engine
                # captures only while the node DRAWS, so a hidden or faint
                # node leaves the slot holding its last capture. That retained
                # image IS the freeze a still-frames rig relies on - emitting
                # a bare snapshot re-captured every frame and nothing could
                # ever freeze.
                slot = self._slot_drawable(inst['name'])
                z_id, z_rest, has_z = self._z_channel(inst)
                self._builder.snapshot(_SCREEN_ID, slot, z_id=z_id,
                                       z_rest=z_rest, has_z=has_z)
                self._tag_instance(inst, 'snapshot')
                self._emit_links(_SCREEN_ID, inst, camera=False)
            case 'fill':
                self._emit_blit(sn.SRC_FILL, 0, inst)
            case 'aft':
                slot = self._slot_drawable(_aft_slot_key(inst))
                self._emit_blit(sn.SRC_DRAWABLE, slot, inst)
            case 'player' | 'proxy':
                scope = self._field_scope(inst)
                # A 'player' instance IS the base field, so it disappears
                # while the chart hides it - legacy's
                # `if kind == 'player' and base_hidden: continue`
                # (field_instances.py:289-290). A 'proxy' is a copy and
                # keeps drawing. Statically this is the item's visible gate
                # rather than a per-frame skip.
                visible = (self._base_hidden_gate() if kind == 'player'
                           else (-1, 1.0))
                if notes_inline() and scope == 'field':
                    # RE-RENDER, don't blit (the copy-render rule): the
                    # consumer's chain composes over the shared fed note
                    # items, so a mod-displaced note survives where a
                    # capture-boxed texture would have clipped it. A
                    # per-player 'field{N}' scope keeps the capture blit -
                    # its content differs per player and the feed carries
                    # player 1's items only.
                    z_id, z_rest, has_z = self._z_channel(inst)
                    self._builder.feed_inline(
                        _SCREEN_ID, self._notes_slot_for(scope),
                        visible_id=visible[0], visible_rest=visible[1],
                        z_id=z_id, z_rest=z_rest, has_z=has_z)
                    self._tag_instance(inst, 'feed')
                    self._emit_links(_SCREEN_ID, inst)
                    return
                drawable = self._field_drawable(scope)
                self._emit_blit(sn.SRC_DRAWABLE, drawable, inst,
                                visible=visible)
            case _:
                return

    def _base_hidden_gate(self):
        """(visible_id, visible_rest) that drops a 'player' instance while the
        chart hides the base field. Minted once and shared - every player
        instance gates on the same `base_field_hidden` timeline, so this must
        not push a channel per instance."""
        if self._base_gate is None:
            self._base_gate = self._visible_from_hidden(
                self._compiled.get('base_field_hidden'))
        return self._base_gate

    def _field_scope(self, inst) -> str:
        """The field-capture scope a proxy/player blits, mirroring
        NotitgFieldInstances._scope EXACTLY: the per-player 'field{N}' only when
        player N > 1 has a minted consumer in the player_fields spec, else the
        primary 'field'. A proxy of a player with no consumer falls to 'field'
        (single-player charts proxying a nominal player 2 still draw player 1's
        capture - the effect's own rule)."""
        from analysis.games.notitg.field_instances import _player_scope
        player = inst.get('player') or 1
        if player > 1 and player in self._player_note_mods():
            return _player_scope(player)
        return 'field'

    def _player_note_mods(self) -> dict:
        spec = self._compiled.get('player_fields')
        return getattr(spec, 'note_mods', {}) if spec is not None else {}

    # -- storyboard element emission --------------------------------------

    def _emit_element(self, element, ancestors=()) -> bool:
        """Emit one leaf element as an item, or count it as a per-kind skip.
        Returns True when an item was emitted.

        Image-backed kinds (sprite / frames) with a resolvable asset draw as
        SRC_IMAGE; fill kinds draw as a solid tinted quad. Everything still
        unsupported - text, bitmaptext, video, compound, an image kind with no
        asset - is skipped and tallied by kind (an asset-less image kind counts
        as 'no_asset')."""
        if element.kind in _FILL_KINDS:
            self._sn_element_item(self._sn.SRC_FILL, 0, element, ancestors)
            return True
        if element.kind == 'bitmaptext':
            return self._emit_bitmaptext(element, ancestors)
        if element.kind == 'text':
            image_id = self._text_image_id(element)
            if image_id is None:
                self._count_skip('text')
                return False
            self._sn_element_item(self._sn.SRC_IMAGE, image_id, element,
                                  ancestors)
            return True
        if element.kind not in _IMAGE_KINDS:
            self._count_skip(element.kind)
            return False
        image_id = self._image_id(element)
        if image_id is None:
            self._count_skip('no_asset')
            return False
        self._sn_element_item(self._sn.SRC_IMAGE, image_id, element, ancestors)
        return True

    def _count_skip(self, kind: str) -> None:
        self._elem_skips[kind] = self._elem_skips.get(kind, 0) + 1

    def _image_id(self, element):
        """The image id for an element's asset (a 'frames' element uses its
        first frame path), minted once per distinct absolute path and recorded
        in the image table. A sheet sprite also records its (cols, rows) grid
        so the executor can crop the CURRENT CELL (the `frame` lane indexes
        it) instead of drawing the whole sheet. First grid wins per path (two
        elements sharing a sheet with different grids would conflict; not
        seen in practice). None when the element carries no asset."""
        path = element.asset or (element.frames[0] if element.frames else None)
        if not path:
            return None
        image_id = self._image_ids.get(path)
        if image_id is None:
            image_id = len(self._image_ids)
            self._image_ids[path] = image_id
            self._image_paths[image_id] = path
        cols = int(getattr(element, 'sheet_cols', 1) or 1)
        rows = int(getattr(element, 'sheet_rows', 1) or 1)
        if cols * rows > 1 and image_id not in self._image_grids:
            self._image_grids[image_id] = (cols, rows)
        self._record_size_spec(image_id, element.size_spec)
        return image_id

    def _record_size_spec(self, image_id: int, spec) -> None:
        """Remember an asset's size CONVENTIONS so the consumer resolves its
        logical size the same way the asset-size funnel does.

        Without this the executor takes an image's logical box as its pixel
        box divided by the sheet grid, which silently ignores `(doubleres)` -
        and a doubleres page then draws at DOUBLE size. The compiler cannot
        resolve it here: the funnel needs the asset's pixel dimensions, and
        reading those needs an image decoder this module deliberately does not
        import."""
        if spec is None or image_id in self._image_specs:
            return
        self._image_specs[image_id] = (
            int(getattr(spec, 'cols', 1) or 1),
            int(getattr(spec, 'rows', 1) or 1),
            bool(getattr(spec, 'doubleres', False)),
            getattr(spec, 'logical', None),
            getattr(spec, 'res', None),
        )

    def _sn_element_item(self, source_kind: int, image_id: int, element,
                         ancestors=()) -> None:
        """Push one element item: the element's scalar transform timelines on
        the item's own lanes (export_channel each), the inverted `hidden` gate
        on `visible`, and the sheet-frame channel on `frame`.

        A leaf under one or more groups additionally emits the ANCESTOR CHAIN
        (root-first, then the leaf) as `item_link`s, so the engine's
        `local @ parent` nesting composes natively instead of the leaf drawing
        at its own local placement. When any link in that chain sets a
        non-default fov, the chain's perspective camera rides along too."""
        kwargs = self._element_transform_kwargs(element)
        kwargs.update(self._element_frame_kwarg(element))
        kwargs.update(self._element_visible_kwarg(element))
        self._builder.item(_SCREEN_ID, source_kind, image_id,
                           additive=bool(element.additive), **kwargs)
        self._builder.item_tag(
            _SCREEN_ID, self._tag_element(element, 'content', ancestors))
        self._emit_element_box(element)
        self._emit_element_tint(element)
        if ancestors:
            self._emit_element_links(element, ancestors)
        self._emit_element_glow(source_kind, image_id, element, kwargs,
                                ancestors)

    def _emit_element_glow(self, source_kind: int, image_id: int, element,
                           kwargs, ancestors=()) -> None:
        """Push SM's additive glow pass (Sprite.cpp:536-541) as a SECOND item.

        The glow is the same sprite drawn again over the content, tinted to
        the glow colour and composited additively at the glow alpha - so it
        needs no record lanes of its own, only a duplicate item whose tint and
        opacity read the glow curve. It draws immediately after the content,
        matching the painter's content-then-glow order.

        `glow` rests at alpha 0, so a statically un-glowed element emits
        nothing rather than doubling every element's op count."""
        glow = element.timelines.get('glow')
        if not _moves_off(glow, _GLOW_OFF, prop=3):
            return
        alpha_id, alpha_rest = self._channel(glow, 3)
        glow_kwargs = dict(kwargs)
        glow_kwargs['opacity_id'] = alpha_id
        glow_kwargs['opacity_rest'] = alpha_rest
        self._builder.item(_SCREEN_ID, source_kind, image_id,
                           additive=True, **glow_kwargs)
        self._builder.item_tag(
            _SCREEN_ID, self._tag_element(element, 'glow', ancestors))
        self._emit_element_box(element)
        chans = [self._channel(glow, prop) for prop in range(3)]
        (r_id, r_rest), (g_id, g_rest), (b_id, b_rest) = chans
        self._builder.item_tint(_SCREEN_ID, r_id=r_id, r_rest=r_rest,
                                g_id=g_id, g_rest=g_rest,
                                b_id=b_id, b_rest=b_rest)
        if ancestors:
            self._emit_element_links(element, ancestors)

    def _text_image_id(self, element):
        """An image id for a `text` element's rendered glyphs, or None when it
        carries no text.

        The raster is DEFERRED to the consumer rather than done here: laying
        out a system font needs Qt, and this compiler runs on a worker thread
        and stays Qt-free so it can. The doc records the spec; the image table
        renders it white-on-transparent on first use and the element's own
        tint colours it, which is what legacy's `setPen(color)` does.

        Keyed by (text, size) so a chart repeating a caption uploads once."""
        if not element.text:
            return None
        key = ('text', element.text, float(element.font_px or 0.0))
        image_id = self._image_ids.get(key)
        if image_id is None:
            image_id = len(self._image_ids)
            self._image_ids[key] = image_id
            self._text_images[image_id] = (element.text,
                                           float(element.font_px or 0.0))
        return image_id

    def _emit_bitmaptext(self, element, ancestors=()) -> bool:
        """Emit one item per character: an SM bitmap font is a GRID atlas, so a
        glyph is a sheet cell and needs no machinery a sheet sprite lacks - the
        codepoint IS the cell index on the frame lane.

        Layout needs only the advances, all known at compile time. Legacy's
        destination is `pen + (advance - cell_w)/2` sized `cell_w`, so the
        glyph CENTRE is `pen + advance/2` and the cell size cancels; the same
        cancellation puts every centre on y=0. That matters because the cell's
        logical size depends on the atlas pixels, which only the executor
        knows.

        The run is centred on the actor, which is SM's BitmapText default and
        what the engine draws. It deliberately does NOT reproduce legacy, which
        shifts by `-origin * (text width, cell height)` in `_paint_element` and
        THEN starts its pen at another `-width/2, -cell_h/2` inside
        `_paint_bitmaptext` - a double shift a sprite does not get (a sprite
        draws `QRectF(0, 0, w, h)` in that same shifted space). That is the
        centring bug this compiler just fixed on its own side, so copying it
        would be bug-for-bug against a reference that is not the oracle.
        """
        font = element.font
        if font is None or not element.text:
            self._count_skip('bitmaptext')
            return False
        image_id = self._image_ids.get(font.texture_path)
        if image_id is None:
            image_id = len(self._image_ids)
            self._image_ids[font.texture_path] = image_id
            self._image_paths[image_id] = font.texture_path
        cells = max(1, int(font.cols)) * max(1, int(font.rows))
        self._image_grids.setdefault(image_id, (int(font.cols), int(font.rows)))
        self._record_size_spec(image_id, font.size_spec)

        codepoints = [ord(char) for char in element.text]
        advances = [font.advance(cp) for cp in codepoints]
        total = sum(advances)
        run = 0.0
        emitted = 0
        for codepoint, advance in zip(codepoints, advances):
            centre = run + advance / 2.0 - total / 2.0
            start = run
            run += advance
            if not 0 <= codepoint < cells:
                # Outside the atlas grid: legacy's `font.cell` returns None and
                # draws nothing, but the pen still advanced.
                continue
            self._sn_glyph_item(image_id, codepoint, centre, element, ancestors,
                                self._glyph_crop(element, total, start, advance))
            emitted += 1
        if not emitted:
            self._count_skip('bitmaptext')
        return bool(emitted)

    def _glyph_crop(self, element, total: float, start: float,
                    advance: float) -> dict:
        """`item_link` crop kwargs for one glyph: its share of the run's
        horizontal crop, and the run's vertical crop unchanged (a run is one
        cell tall, so top/bottom apply to every glyph alike)."""
        kwargs: dict = {}
        for param, prop, leading in (('crop_l', 'crop_left', True),
                                     ('crop_r', 'crop_right', False)):
            timeline = element.timelines.get(prop)
            if timeline is None:
                continue
            share = _GlyphCropTimeline(timeline, total, start, advance, leading)
            chan_id, rest = self._channel(share)
            kwargs[f'{param}_id'] = chan_id
            kwargs[f'{param}_rest'] = rest
        for param, prop in (('crop_t', 'crop_top'), ('crop_b', 'crop_bottom')):
            timeline = element.timelines.get(prop)
            if timeline is None:
                continue
            chan_id, rest = self._channel(timeline)
            kwargs[f'{param}_id'] = chan_id
            kwargs[f'{param}_rest'] = rest
        return kwargs

    def _sn_glyph_item(self, image_id: int, codepoint: int, centre: float,
                       element, ancestors, crop=None) -> None:
        """One glyph of a bitmaptext run, placed by a constant offset link so
        the offset rides INSIDE the element's own rotation and scale.

        That offset link is the chain's LEAF, and `compose_links` takes its
        crop from the leaf - so the run's crop share rides here too, or it is
        dropped."""
        kwargs = self._element_visible_kwarg(element)
        kwargs['frame_rest'] = float(codepoint)
        self._builder.item(_SCREEN_ID, self._sn.SRC_IMAGE, image_id,
                           additive=bool(element.additive), **kwargs)
        self._builder.item_tag(
            _SCREEN_ID, self._tag_element(element, 'glyph', ancestors))
        self._builder.item_box(_SCREEN_ID, origin_x=0.5, origin_y=0.5)
        self._emit_element_tint(element)
        for link_element in (*ancestors, element):
            self._builder.item_link(
                _SCREEN_ID, **self._element_link_kwargs(link_element))
        self._builder.item_link(_SCREEN_ID, x_rest=float(centre), y_rest=0.0,
                                natural_w_rest=0.0, natural_h_rest=0.0,
                                **(crop or {}))

    def _tag_element(self, element, role: str, ancestors=()) -> int:
        """Record `(element, role, ancestors)` in emission order and return its
        record tag. Tags count from 1 so 0 stays "untagged" - every item the
        doc does NOT tag (fed notes) reads back as 0."""
        self._element_order.append((element, role, tuple(ancestors)))
        return len(self._element_order)

    def _tag_instance(self, inst, command: str) -> int:
        """Record `(instance, command)` in emission order and return its record
        tag.

        Instances and elements share the record's ONE tag lane, so instance
        tags start past `_INSTANCE_TAG_BASE` and a reader tells them apart by
        magnitude. Snapshots and feeds are recorded here too even though
        `item_tag` can only write an Item: without them the order list would
        skip entries and every later tag would name the wrong instance."""
        self._instance_order.append((inst, command))
        return _INSTANCE_TAG_BASE + len(self._instance_order)

    def _emit_element_box(self, element) -> None:
        """Attach the element's draw-box origin and any absolute size.

        The origin is why an unforwarded element lands displaced: SM draws an
        actor about `origin` (0.5, 0.5 by default) while a bare item quad
        spans (0,0)-(w,h), so every element hung down-right by half its own
        size. `size_x`/`size_y` carry `zoomto`/`setsize`, which REPLACE the
        natural basis - a negative rest keeps the natural size.

        ScaleToCover / ScaleToFitInside rides `_emit_element_fit`, which
        overrides this - the fitted size needs the natural size, which only
        the executor knows."""
        origin_x, origin_y = getattr(element, 'origin', (0.0, 0.0))
        kwargs = {'origin_x': float(origin_x), 'origin_y': float(origin_y)}
        # A fill has no texture to fall back to, so its size lanes carry its
        # WHOLE box - absolute size or natural w/h, resolved per frame. Left
        # unset it would draw the executor's UNIT quad where a zero-size shape
        # must draw nothing, which is what legacy's `w > 0 and h > 0` decides.
        fill = element.kind in _FILL_KINDS
        for axis, size_prop, wh_prop in (('x', 'size_x', 'w'),
                                         ('y', 'size_y', 'h')):
            size = element.timelines.get(size_prop)
            timeline = (_FillSizeTimeline(size, element.timelines.get(wh_prop))
                        if fill else size)
            if timeline is None:
                continue
            chan_id, rest = self._channel(timeline)
            kwargs[f'size_{axis}_id'] = chan_id
            kwargs[f'size_{axis}_rest'] = rest
        self._builder.item_box(_SCREEN_ID, **kwargs)
        self._emit_element_fit(element)
        self._emit_element_fade(element)

    def _emit_element_fade(self, element) -> None:
        """Attach the element's SetFade* edge ramps to the item just pushed.

        Emitted only when the element actually fades: the ramps rest at 0 (a
        hard edge), and a non-resting fade lane routes the blit through a
        second GL program, so a needless one would cost every element two
        extra uniform sets a hard-edged draw does not need."""
        if not _is_poked(element, _FADE_PROPS, _FADE_OFF):
            return
        kwargs = {}
        for param, prop in zip(('l', 'r', 't', 'b'), _FADE_PROPS):
            timeline = element.timelines.get(prop)
            if timeline is None:
                continue
            chan_id, rest = self._channel(timeline)
            kwargs[f'{param}_id'] = chan_id
            kwargs[f'{param}_rest'] = rest
        self._builder.item_fade(_SCREEN_ID, **kwargs)

    def _emit_element_fit(self, element) -> None:
        """Attach ScaleToCover / ScaleToFitInside to the item just pushed.

        Only the recorded rect's EXTENT is sent: the engine's zoom is
        `rect/natural` per axis and then the larger (cover) or smaller
        (fit-inside) of the two, so the rect's position never reaches the
        draw. `_SpanTimeline` turns the recorded edge pairs into those
        extents without a lane each."""
        mode = element.timelines.get('fit_mode')
        if mode is None:
            return
        mode_id, mode_rest = self._channel(mode)
        kwargs = {'mode_id': mode_id, 'mode_rest': mode_rest}
        for param, lo, hi in (('w', 'fit_left', 'fit_right'),
                              ('h', 'fit_top', 'fit_bottom')):
            lo_tl, hi_tl = element.timelines.get(lo), element.timelines.get(hi)
            if lo_tl is None or hi_tl is None:
                continue
            chan_id, rest = self._channel(_SpanTimeline(lo_tl, hi_tl))
            kwargs[f'{param}_id'] = chan_id
            kwargs[f'{param}_rest'] = rest
        self._builder.item_fit(_SCREEN_ID, **kwargs)

    def _emit_element_tint(self, element) -> None:
        """Attach the element's flat diffuse rgb, when it carries one.

        Per-corner gradients (`color_ul`/`ur`/`ll`/`lr`) are NOT handled - one
        item tint colours the whole quad, so a gradient needs per-vertex
        colour. Neither reference chart uses one, so the verb waits for a
        chart that does; an element carrying one is COUNTED, because once
        legacy stops drawing elements an unimplemented verb is invisible
        rather than obviously missing."""
        if _is_poked(element, _CORNER_COLOR_PROPS, _CORNER_UNSET):
            self._count_skip('corner_gradient')
        color = element.timelines.get('color')
        if color is None:
            return
        chans = [self._channel(color, prop) for prop in range(3)]
        (r_id, r_rest), (g_id, g_rest), (b_id, b_rest) = chans
        self._builder.item_tint(_SCREEN_ID, r_id=r_id, r_rest=r_rest,
                                g_id=g_id, g_rest=g_rest,
                                b_id=b_id, b_rest=b_rest)

    def _emit_element_links(self, element, ancestors) -> None:
        """Attach the leaf's ancestor chain (root-first, leaf last) to the item
        just pushed, plus the chain's fov camera when one is set.

        The item's own lanes stay as emitted - `compose_links` REPLACES the TRS
        mat3 when links are present, so the leaf's transform must ride the
        chain's final link rather than the item lanes alone. Alpha and hidden
        are the exception: the item already carries the leaf's, so its link
        must not carry them again (`_element_link_kwargs`).

        The bitmaptext path does NOT take this branch, and must not: its item
        carries no opacity lane at all, so the leaf's alpha rides its chain
        exactly once there."""
        chain = (*ancestors, element)
        for index, link_element in enumerate(chain):
            self._builder.item_link(
                _SCREEN_ID, **self._element_link_kwargs(
                    link_element, leaf=index == len(chain) - 1))
        fov = self._chain_fov(ancestors, element)
        if fov is not None:
            fov_id, fov_rest = fov
            self._builder.item_projection(
                _SCREEN_ID, fov_id=fov_id, fov_rest=fov_rest,
                vanish_x_rest=_SCREEN_W / 2.0,
                vanish_y_rest=_SCREEN_H / 2.0)

    def _chain_fov(self, ancestors, element):
        """The chain's effective perspective camera as `(channel id, rest)`, or
        None when every link rests at the default fov (the flat-chart no-op).

        A frame's fov projects its whole subtree and the INNERMOST frame that
        set one wins (its LoadMenuPerspective replaces the outer), matching
        `TransformChannel.at`. The authoring form is a frame-level `FOV="60"`
        on a background group."""
        for link_element in reversed((*ancestors, element)):
            timeline = link_element.timelines.get('fov')
            if timeline is None:
                continue
            ts, vals, durs, rest, eases = export_channel(
                timeline, self._t0, self._t1)
            # An untouched fov rests at the LoadMenuPerspective default with no
            # breakpoints; minting a channel for it would attach a camera to
            # every chain (a `_channel` id is always valid, so the id alone
            # cannot say whether the fov was ever set).
            if not ts and abs(float(rest) - _DEFAULT_FOV) <= _FOV_EPS:
                continue
            chan_id = self._builder.channel(ts, vals, durs, float(rest), eases)
            return chan_id, float(rest)
        return None

    def _element_link_kwargs(self, element, leaf: bool = False) -> dict:
        """The `item_link` kwargs for one element in a chain: an (id, rest)
        channel per transform prop it carries, plus its Euler rotation order.

        Absent props fall through to the binding's engine-identity defaults, so
        a group that only sets `x`/`y` composes as a pure translation.

        The LEAF link drops `alpha` and `hidden`. The leaf is BOTH the item and
        the chain's last link, and the item already carries its own alpha on
        the opacity lane and its own hidden bit on the visible gate - so
        leaving them on the link multiplied the leaf's alpha in twice. An
        element at diffusealpha 0.5 under opaque parents composited at 0.25,
        and a full-screen rect at 0.05 came out at 0.0025: invisible where the
        engine draws a veil. Placement was exact throughout, which is why this
        outlived the quad harness."""
        kwargs: dict[str, object] = {}
        for param, prop in _ELEMENT_LINK_PROPS:
            if leaf and prop in ('alpha', 'hidden'):
                continue
            timeline = element.timelines.get(prop)
            if timeline is None:
                continue
            chan_id, rest = self._channel(timeline)
            kwargs[f'{param}_id'] = chan_id
            kwargs[f'{param}_rest'] = rest
        kwargs['rotation_order'] = _rotation_order_of(element)
        # No content centering on an element chain. `compose_links` centres the
        # LEAF by -natural/2, and `item_link` defaults that to the 640x480
        # design screen - which shifted every grouped element by (320, 240),
        # the largest error the parity harness found on the real charts. An
        # element's own centering is its `origin`, applied by the executor
        # against the size it actually draws, so the chain must not also do it
        # (and could not do it correctly: the natural size is not known here).
        kwargs['natural_w_rest'] = 0.0
        kwargs['natural_h_rest'] = 0.0
        return kwargs

    def _element_transform_kwargs(self, element) -> dict:
        kwargs: dict[str, object] = {}
        for prop, stem in _ELEMENT_ITEM_LANES:
            timeline = element.timelines.get(prop)
            if timeline is None:
                continue
            chan_id, rest = self._channel(timeline)
            kwargs[f'{stem}_id'] = chan_id
            kwargs[f'{stem}_rest'] = rest
        return kwargs

    def _element_frame_kwarg(self, element) -> dict:
        """A sheet sprite carries a frame lane tracing its current cell; a plain
        1x1 sprite rests at frame 0 (no lane).

        The lane is exported over the element's OWN existence window: outside
        it the item's `visible` gate is shut, so the cell it would show is
        never read - and a sheet cycling every few frames traced across a whole
        chart is a lane of breakpoints nothing looks at."""
        if element.sheet_cols * element.sheet_rows <= 1 and element.state_pin is None:
            return {}
        chan_id, rest = self._channel(_FrameCurve(element),
                                      window=self._element_window(element))
        return {'frame_id': chan_id, 'frame_rest': rest}

    def _element_window(self, element):
        """The element's existence window, clipped to the build's export
        window."""
        return (max(self._t0, float(element.t_start)),
                min(self._t1, float(element.t_end)))

    def _element_visible_kwarg(self, element) -> dict:
        """The element's visible gate: its existence WINDOW [t_start, t_end)
        (the legacy walk draws an element only inside it - render.py's
        `child.t_start <= t < child.t_end`; without this every element drew
        for the whole chart at once) ANDed with the SM `hidden` bit
        inverted. Hidden flips are instant steps, so the merged gate is a
        step channel: the inverted-hidden value inside the window, 0
        outside it."""
        hidden = element.timelines.get('hidden')
        t0 = float(element.t_start)
        t1 = float(element.t_end)
        open_start = t0 <= self._t0 + 1e-9
        open_end = (not math.isfinite(t1)) or t1 >= self._t1 - 1e-9
        if hidden is None and open_start and open_end:
            return {}
        if hidden is None:
            ts, vals = [t0], [1.0]
            if not open_end:
                ts.append(t1)
                vals.append(0.0)
            chan_id = self._builder.channel(
                ts, vals, [0.0] * len(ts), 0.0)
            return {'visible_id': chan_id, 'visible_rest': 0.0}
        hts, hvals, _durs, hrest, _eases = export_channel(
            hidden, self._t0, self._t1)
        inv = [1.0 - float(v) for v in hvals]
        inv_rest = 1.0 - float(hrest)

        def gate_at(t):
            value = inv_rest
            for bt, bv in zip(hts, inv):
                if bt > t:
                    break
                value = bv
            return value

        ts, vals = [t0], [gate_at(t0)]
        for bt, bv in zip(hts, inv):
            if t0 < bt and (open_end or bt < t1):
                ts.append(float(bt))
                vals.append(bv)
        if not open_end:
            ts.append(t1)
            vals.append(0.0)
        chan_id = self._builder.channel(ts, vals, [0.0] * len(ts),
                                        0.0 if not open_start else inv_rest)
        return {'visible_id': chan_id,
                'visible_rest': 0.0 if not open_start else inv_rest}


def build_static_doc(compiled, screen_w: float = _SCREEN_W,
                     screen_h: float = _SCREEN_H):
    """Compile a NotITG chart's static field-instance scene into a
    channel-backed DrawableDoc, in engine tree order.

    Returns `(evaluator, id_maps, report)`:
      - `evaluator`  : a finished `storyboard_native.Evaluator`;
      - `id_maps`    : `{'screen', 'slots': {key -> id}, 'fields': {scope -> id},
                         'images': {image_id -> absolute path}}`;
      - `report`     : `{'instances', 'fields', 'slots', 'captures', 'fills',
                         'aft', 'proxy', 'z_groups', 'images',
                         'elements_below', 'elements_above', 'element_skips'}`.

    The screen root drawable (0) carries the whole static command list; the
    storyboard element tree (`compiled['tree']`) draws around the field-instance
    stream by z band (below-band elements first, then the instances, then
    above-band elements), reproducing today's banding. proxy/player/aft blits
    source per-player field / per-slot drawables (minted lazily as referenced),
    captures emit Snapshot commands, fills emit SRC_FILL items, image elements
    emit SRC_IMAGE items (paths collected in `id_maps['images']`), and z_group
    runs are wrapped in SortSpans.
    """
    b = _Builder(compiled, screen_w, screen_h)
    instances = _current_instances(compiled)
    report = _report(instances)
    evaluator, id_maps = b.run()
    report['fields'] = len(id_maps['fields'])
    report['slots'] = len(id_maps['slots'])
    report['images'] = len(id_maps['images'])
    report['elements_below'] = b._elem_below
    report['elements_above'] = b._elem_above
    report['element_skips'] = dict(b._elem_skips)
    return evaluator, id_maps, report


class _RecordingBuilder:
    """Records DocBuilder calls as (method, args, kwargs) ops, minting the
    SAME sequential ids a real DocBuilder would. The point: `_Builder`'s
    emission is tens of seconds of pure-Python channel export, but the PyO3
    Evaluator is unsendable - so the emission runs against this recorder on
    a WORKER thread, and `assemble_static_doc` replays the cheap FFI calls
    where the evaluator must live (the render thread)."""

    # The DocBuilder methods that RETURN an id, mapped to the counter they
    # draw from. Hand-maintained because the id spaces are not inferable:
    # `feed_slot` shares the drawable counter and the two clip constructors
    # share one clip space. A minting method MISSING from this table records
    # through __getattr__ and hands back None, which reaches replay as a null
    # id - so add an entry when adding an id-returning builder method.
    # `test_recording_builder_mints_the_same_ids_as_a_real_builder` pins it.
    _MINTING = {
        'channel': 'channels',
        'drawable': 'drawables',
        'shader': 'shaders',
        'feed_slot': 'drawables',
        'mesh': 'meshes',
        'clip_rect': 'clips',
        'clip_polygon': 'clips',
    }

    def __init__(self):
        import storyboard_native as sn

        self.ops: list[tuple] = []
        # The screen root owns drawable 0; every other space counts from 0.
        self._next = {'channels': 0, 'drawables': 1, 'shaders': 0,
                      'meshes': 0, 'clips': 0}
        # Resolved once: __getattr__ validates against it on every miss, and
        # the module import is lazy everywhere else in this file.
        self._api = sn.DocBuilder

    def __getattr__(self, name: str):
        """Record any other DocBuilder call generically.

        The alternative - hand-listing every void method - lets a new builder
        method compile, pass every test that builds synchronously, and then
        AttributeError only on the async prepare path at runtime. Validating
        against the real class keeps a typo failing HERE, at the call site,
        instead of at replay on the render thread."""
        if name.startswith('_') or not hasattr(self._api, name):
            raise AttributeError(
                f'{type(self).__name__} has no attribute {name!r} '
                '(and neither does DocBuilder)')
        space = self._MINTING.get(name)

        def record(*args, **kwargs):
            self.ops.append((name, args, kwargs))
            if space is None:
                return None
            minted = self._next[space]
            self._next[space] = minted + 1
            return minted

        # Cache on the instance so the next call resolves normally - a miss
        # is what routes here, and the emitter calls these per item.
        setattr(self, name, record)
        return record

    def finish(self):
        """Not recorded: `assemble_static_doc` finishes the real builder
        itself, after the replay loop."""
        return None


def prepare_static_doc(compiled, screen_w: float = _SCREEN_W,
                       screen_h: float = _SCREEN_H):
    """The worker-safe half of `build_static_doc`: run the full emission
    against a `_RecordingBuilder` and return `(ops, id_maps, report)` - all
    plain Python data, no PyO3 objects, safe to build on any thread. Feed
    the ops to `assemble_static_doc` on the consuming thread."""
    recorder = _RecordingBuilder()
    b = _Builder(compiled, screen_w, screen_h, builder=recorder)
    instances = _current_instances(compiled)
    report = _report(instances)
    _evaluator, id_maps = b.run()
    report['fields'] = len(id_maps['fields'])
    report['slots'] = len(id_maps['slots'])
    report['images'] = len(id_maps['images'])
    report['elements_below'] = b._elem_below
    report['elements_above'] = b._elem_above
    report['element_skips'] = dict(b._elem_skips)
    return recorder.ops, id_maps, report


def assemble_static_doc(ops, screen_w: float = _SCREEN_W,
                        screen_h: float = _SCREEN_H):
    """Replay recorded ops onto a real DocBuilder and finish it - the cheap
    FFI half, run on the thread the Evaluator lives on."""
    import storyboard_native as sn

    builder = sn.DocBuilder(float(screen_w), float(screen_h))
    for method, args, kwargs in ops:
        getattr(builder, method)(*args, **kwargs)
    return builder.finish()


def _report(instances) -> dict:
    kinds = [inst.get('kind') for inst in instances]
    groups = {inst.get('z_group') for inst in instances
              if inst.get('z_group') is not None}
    return {
        'instances': len(instances),
        'captures': kinds.count('capture'),
        'fills': kinds.count('fill'),
        'aft': kinds.count('aft'),
        'proxy': kinds.count('proxy') + kinds.count('player'),
        'z_groups': len(groups),
        'fields': 0,
        'slots': 0,
        'images': 0,
        'elements_below': 0,
        'elements_above': 0,
        'element_skips': {},
    }


# --------------------------------------------------------------------------
# Parity harness - evaluator BLIT stream vs NotitgFieldInstances.at
# --------------------------------------------------------------------------

# Op / source kinds (mirror native/src/evaluate.rs); imported lazily so this
# module stays importable without the extension for the pure helpers above.
_OP_BLIT = 1

# The source kind storyboard element blits carry (mirrors evaluate.rs). Field
# instances only ever blit fills (SRC_FILL) and drawables (SRC_DRAWABLE), so an
# image blit is unambiguously a storyboard element - the parity harness drops
# them to recover the field-instance subsequence.
_SRC_IMAGE = 0


def _blit_stream(evaluator, t):
    """The evaluator's BLIT ops at `t` as `(source_kind, source_id, mat3, alpha)`
    tuples, in draw order. mat3 is the design-space 3x3 (row-major, column-vector
    convention - the BLIT record layout)."""
    u_stride = evaluator.u_stride
    f_stride = evaluator.f_stride
    u_bytes, f_bytes, _uf, n = evaluator.frame(float(t))
    u = np.frombuffer(u_bytes, dtype=np.uint32).reshape(n, u_stride)
    f = np.frombuffer(f_bytes, dtype=np.float32).reshape(n, f_stride)
    out = []
    for i in range(n):
        if u[i, 0] != _OP_BLIT:
            continue
        mat = f[i, 0:9].reshape(3, 3).astype(np.float64)
        alpha = float(f[i, 9])
        out.append((int(u[i, 1]), int(u[i, 2]), mat, alpha))
    return out


def _field_blit_subsequence(evaluator, t):
    """The field-instance SUBSEQUENCE of the blit stream at `t`: the full blit
    stream with storyboard element (SRC_IMAGE) blits dropped. With elements
    banded around the instances, this is exactly the stream the instance-only
    doc produced, in the same order - the parity harness compares against this
    so element inclusion never perturbs field-instance parity."""
    return [blit for blit in _blit_stream(evaluator, t) if blit[0] != _SRC_IMAGE]


def _mat3_from_qtransform(qt):
    """The design-space 3x3 (column-vector convention, matching the BLIT
    record) for a field entry's screen QTransform, or the identity when the
    entry transform is None (a centered original blit)."""
    if qt is None:
        return np.eye(3)
    # Qt row-vector storage: p' = p @ [[m11,m12,m13],[m21,m22,m23],[m31,m32,m33]].
    # The BLIT record is column-vector (M @ p) with the same content transposed.
    return np.array([[qt.m11(), qt.m21(), qt.m31()],
                     [qt.m12(), qt.m22(), qt.m32()],
                     [qt.m13(), qt.m23(), qt.m33()]], dtype=np.float64)


def _mat3_rel_err(a, b) -> float:
    """Relative max-abs error between two 3x3 mats, each first normalized so
    its bottom-right entry is 1 (projective scale-invariance)."""
    a = a / a[2, 2] if abs(a[2, 2]) > 1e-12 else a
    b = b / b[2, 2] if abs(b[2, 2]) > 1e-12 else b
    scale = max(1.0, float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / scale)


def compare_at(evaluator, id_maps, effect, t,
               alpha_atol: float = 1e-3, mat_rtol: float = 1e-2) -> dict:
    """Compare the evaluator's BLIT stream at `t` against the field effect's
    entries, in order. Returns a per-time diff dict:

      `{'t', 'ok', 'n_blit', 'n_entry', 'order_ok', 'diffs': [...],
        'max_alpha_err', 'max_mat_err'}`

    where `diffs` lists `(index, reason, detail)` for each mismatch. An entry
    whose source does not resolve to a doc drawable (a 'capture' snapshot, an
    unresolved slot) is not a BLIT and is skipped on BOTH sides so the streams
    stay aligned. mat3 comparison is projective-relative (both normalized), the
    same space on both sides (design 640x480 via `_DESIGN_RECT`).

    The evaluator side is the field-instance SUBSEQUENCE (storyboard element
    image blits dropped): with elements banded around the instances, that
    subsequence is exactly the instance-only doc's blit stream, so this parity
    is unchanged by element inclusion. `n_blit` counts the field blits only.
    """
    frame = effect.at(_Ctx(float(t), _DESIGN_RECT))
    entries = frame.fields if frame is not None else ()
    expected = _expected_blits(entries, id_maps)
    got = _field_blit_subsequence(evaluator, t)

    diffs = []
    max_alpha_err = 0.0
    max_mat_err = 0.0
    order_ok = len(expected) == len(got)
    for i in range(min(len(expected), len(got))):
        (e_kind, e_id, e_mat, e_alpha) = expected[i]
        (g_kind, g_id, g_mat, g_alpha) = got[i]
        if (e_kind, e_id) != (g_kind, g_id):
            diffs.append((i, 'source', ((e_kind, e_id), (g_kind, g_id))))
            order_ok = False
            continue
        alpha_err = abs(e_alpha - g_alpha)
        mat_err = _mat3_rel_err(g_mat, e_mat)
        max_alpha_err = max(max_alpha_err, alpha_err)
        max_mat_err = max(max_mat_err, mat_err)
        if alpha_err > alpha_atol:
            diffs.append((i, 'alpha', (e_alpha, g_alpha)))
        if mat_err > mat_rtol:
            diffs.append((i, 'mat3', mat_err))
    for i in range(min(len(expected), len(got)), max(len(expected), len(got))):
        diffs.append((i, 'missing', ('entry' if i < len(expected) else 'blit')))

    return {'t': float(t), 'ok': not diffs and order_ok,
            'n_blit': len(got), 'n_entry': len(expected),
            'order_ok': order_ok, 'diffs': diffs,
            'max_alpha_err': max_alpha_err, 'max_mat_err': max_mat_err}


def _expected_blits(entries, id_maps):
    """The field entries that become BLITs, in order, as
    `(source_kind, source_id, mat3, alpha)` - mirroring `_emit_instance`'s
    source resolution so the two streams line up."""
    import storyboard_native as sn

    out = []
    for entry in entries:
        transform = entry[0]
        alpha = float(entry[1]) if len(entry) > 1 else 1.0
        scope = entry[2] if len(entry) > 2 else 'field'
        extra = entry[3] if len(entry) > 3 else None
        # The native emit drops opacity < 1/255 (the base-hidden placeholder,
        # a faded copy); skip those here so the streams stay aligned.
        if alpha < 1.0 / 255.0:
            continue
        source = _resolve_entry_source(sn, scope, extra, id_maps)
        if source is None:
            continue
        out.append((source[0], source[1], _mat3_from_qtransform(transform), alpha))
    return out


def _resolve_entry_source(sn, scope, extra, id_maps):
    """(source_kind, source_id) an entry blits, or None when it is not a BLIT
    (a 'capture' snapshot, or an unresolved slot / field)."""
    match scope:
        case 'fill':
            return (sn.SRC_FILL, 0)
        case 'capture':
            return None
        case 'screen' | 'screen_prev':
            key = extra[0] if isinstance(extra, tuple) and extra else None
            slot = id_maps['slots'].get(key)
            return None if slot is None else (sn.SRC_DRAWABLE, slot)
        case _:
            field = id_maps['fields'].get(scope)
            return None if field is None else (sn.SRC_DRAWABLE, field)


def parity_report(evaluator, id_maps, effect, sample_times,
                  alpha_atol: float = 1e-3, mat_rtol: float = 1e-2) -> dict:
    """`compare_at` over `sample_times`; returns `{'times': [...],
    'all_ok', 'max_alpha_err', 'max_mat_err', 'n_fail'}`."""
    times = [compare_at(evaluator, id_maps, effect, t, alpha_atol, mat_rtol)
             for t in sample_times]
    return {
        'times': times,
        'all_ok': all(r['ok'] for r in times),
        'max_alpha_err': max((r['max_alpha_err'] for r in times), default=0.0),
        'max_mat_err': max((r['max_mat_err'] for r in times), default=0.0),
        'n_fail': sum(0 if r['ok'] else 1 for r in times),
    }


def format_parity_report(report) -> str:
    """A one-line-per-sample-time human summary of a `parity_report`."""
    lines = [
        f"parity: {'OK' if report['all_ok'] else 'FAIL'} "
        f"({report['n_fail']}/{len(report['times'])} times failing) "
        f"max_alpha_err={report['max_alpha_err']:.2e} "
        f"max_mat_err={report['max_mat_err']:.2e}"
    ]
    for r in report['times']:
        detail = '' if r['ok'] else f"  diffs={r['diffs'][:4]}"
        lines.append(
            f"  t={r['t']:8.3f}  blits={r['n_blit']:>3} entries={r['n_entry']:>3} "
            f"order={'ok' if r['order_ok'] else 'BAD'} "
            f"a_err={r['max_alpha_err']:.2e} m_err={r['max_mat_err']:.2e}{detail}")
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Quad parity harnesses - doc BLIT records vs what legacy actually draws
# --------------------------------------------------------------------------
#
# Both harnesses here compare the quantity that decides the pixels: THE FOUR
# CORNERS OF THE DRAWN QUAD, in design space.
#
# Corners rather than the mat3, because the mat3 alone is not the placement:
# origin, the absolute-size override and scale-to-fit all move the box the
# mat3 acts on. `parity_report` above compares mat3 + alpha + order against
# `NotitgFieldInstances.at`, and it passed a curtain that drew ONE DESIGN
# PIXEL wide - a fill's mat3 is identical either way, and the whole
# difference lives in the size lanes it never looks at.
#
# `element_parity_report` covers the storyboard elements (whose legacy path,
# `StoryboardEffect`, paints straight onto a QPainter and exposes no sampled
# transform at all) and `field_parity_report` the field instances.

def _element_bracket(element, t) -> np.ndarray:
    """One element's own painter bracket as a 3x3: translate to its anchored
    position, rotate, scale. Shared by a leaf and by every group above it -
    `render._paint_children` applies the SAME bracket to a group, which is why
    a group needs no size (it has "a zero-size anchor box")."""
    ax, ay = element.anchor
    tx = ax * _SCREEN_W + element.sample('x', t)[0]
    ty = ay * _SCREEN_H + element.sample('y', t)[0]
    theta = math.radians(element.sample('rotation', t)[0])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    sx = element.sample('scale_x', t)[0]
    sy = element.sample('scale_y', t)[0]
    return np.array([
        [cos_t * sx, -sin_t * sy, tx],
        [sin_t * sx, cos_t * sy, ty],
        [0.0, 0.0, 1.0],
    ])


_OUT_OF_PLANE_ELEMENT_PROPS = (('rotation_x', 0.0), ('rotation_y', 0.0),
                               ('z', 0.0), ('skew_x', 0.0), ('skew_y', 0.0),
                               ('scale_z', 1.0))


# How far apart the two axis scales have to be before the transform-ORDER
# difference between the engine and legacy's painter shows up at all. Equal
# scales commute with rotation exactly; a hair apart is sub-pixel.
_SCALE_RATIO_EPS = 1e-3


def _chain_order_diverges(chain, t) -> bool:
    """Whether legacy's ROTATE-THEN-SCALE bracket parts company with the
    engine's scale-after-the-spin on this chain, at `t`.

    `_legacy_element_quad` documents the residual: QPainter rotates then
    scales, so the content scales FIRST, while `Actor::BeginDraw` pushes
    skew -> rotation -> scale -> translate and the scale lands AFTER the
    spin. The two agree exactly when the axis scales are equal (a uniform
    scale commutes with rotation) and diverge in proportion to how far apart
    they are - Bonfire spins a 243:1 bar through 725 degrees and the quads
    end up 1014px apart.

    Not compared, for the same reason as `_chain_is_3d`: the doc follows the
    engine here and legacy does not, so grading one against the other would
    report the doc's correctness as a defect."""
    for link in chain:
        rotation = link.timelines.get('rotation')
        if rotation is None or abs(rotation.sample(t)[0]) <= 1e-9:
            continue
        sx = link.sample('scale_x', t)[0]
        sy = link.sample('scale_y', t)[0]
        if abs(sx - sy) > _SCALE_RATIO_EPS * max(1.0, abs(sx), abs(sy)):
            return True
    return False


def _chain_is_3d(chain, t) -> bool:
    """Whether legacy would paint this chain through its perspective camera.

    Mirrors `render._paint_element`'s `active_3d`: a non-default fov anywhere
    in the chain, or ANY link leaving the z=0 plane, switches the whole subtree
    to the projected path.

    Such a chain is NOT compared here, and the deeper reason is not that the
    2D bracket would measure the wrong renderer - it is that LEGACY'S
    PROJECTED PATH IS NOT THE ORACLE. The engine is. Grading the doc's 3D
    against legacy's 3D would chase legacy's own perspective bugs, which is
    exactly the bug-for-bug outcome this pipeline exists to avoid. 3D
    placement is settled against NotITG itself - the app and the reference
    captures - not against another of our renderers."""
    for link in chain:
        fov = link.timelines.get('fov')
        if fov is not None and abs(fov.sample(t)[0] - _DEFAULT_FOV) > _FOV_EPS:
            return True
        for prop, rest in _OUT_OF_PLANE_ELEMENT_PROPS:
            timeline = link.timelines.get(prop)
            if timeline is not None and abs(timeline.sample(t)[0] - rest) > 1e-6:
                return True
    return False


def _sample_or(element, prop: str, default: float, t: float) -> float:
    timeline = element.timelines.get(prop)
    return default if timeline is None else timeline.sample(t)[0]


def _apply_h(homography, corners) -> list:
    """`corners` through a 3x3 homography, perspective-divided."""
    mat = np.asarray(homography, dtype=float).reshape(3, 3)
    out = []
    for cx, cy in corners:
        p = mat @ np.array([cx, cy, 1.0])
        out.append((p[0] / p[2], p[1] / p[2]) if abs(p[2]) > 1e-12
                   else (p[0], p[1]))
    return out


def _legacy_element_quad(element, t, natural, ancestors=()):
    """The design-space corners the legacy painter draws `element` at.

    Transcribes `render.StoryboardRenderer._draw_element`'s bracket at k=1 with
    no layer offset (design space), composed under every ANCESTOR group's
    bracket - a leaf under a group draws in the group's transformed space
    (`_paint_children`), so leaving the chain out puts the element at its own
    local placement instead of where the chart put it.

    ONLY the 2D bracket. A chain that leaves the z=0 plane or sets an fov
    routes through legacy's projected path, which this does not model -
    `element_parity_report` counts those separately rather than comparing them
    against the wrong renderer. See `_chain_is_3d`.

    KNOWN RESIDUAL, and it is legacy's: this bracket is `R . S` (QPainter
    rotates then scales, so the content scales FIRST), while the engine
    applies an actor's pushes innermost-first as
    `skew -> rotation -> scale -> translate` (openitg Actor::BeginDraw, cited
    in field_compose's header) - scale AFTER the spin. The doc's link chain
    follows the engine; legacy's element painter does not. They agree wherever
    scale is uniform, which is nearly everywhere, and diverge on the
    flip-after-a-spin idiom (negative scale with rotation) that the engine
    model calls a chart staple. A sub-pixel residual on such an element is
    EXPECTED, and is the reference being wrong rather than the doc.

    The quad spans the UNZOOMED draw size `render._draw_size` picks, shifted by
    the origin - that shift is the leaf's alone; a group carries no size."""
    from analysis.player.render.storyboard.render import _draw_size

    chain = (*ancestors, element)
    w, h = _draw_size(element, t, natural)
    world = np.eye(3)
    for link in chain:
        world = world @ _element_bracket(link, t)
    ox, oy = element.origin
    return _apply_h(world, ((-ox * w, -oy * h), (w - ox * w, -oy * h),
                            (w - ox * w, h - oy * h), (-ox * w, h - oy * h)))


def _record_quad(frec, natural):
    """The design-space corners a BLIT record draws, folding the same lanes the
    executor does: the draw box (`_draw_box`), the origin shift, then the
    record mat3."""
    from analysis.player.render.storyboard.gl_executor import _draw_box

    w, h = _draw_box(natural, frec)
    ox = float(frec[_rec.F_ORIGIN]) * w
    oy = float(frec[_rec.F_ORIGIN + 1]) * h
    mat = np.asarray(frec[:9], dtype=float).reshape(3, 3)
    corners = []
    for cx, cy in ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h)):
        p = mat @ np.array([cx - ox, cy - oy, 1.0])
        corners.append((p[0] / p[2], p[1] / p[2]) if abs(p[2]) > 1e-12
                       else (p[0], p[1]))
    return corners


# An f32 opacity lane folded down a chain of multiplies; anything under this
# is rounding, anything over is a compositing difference.
_ALPHA_ATOL = 1e-4


def _legacy_element_alpha(element, ancestors, t) -> float:
    """The alpha the legacy painter composites `element` at.

    A group's alpha multiplies into every child (`render._paint_element`'s
    `inherited_alpha`), so this is the product down the whole chain."""
    alpha = 1.0
    for link in (*ancestors, element):
        alpha *= link.sample('alpha', t)[0]
    return min(1.0, alpha)


def _legacy_element_drawn(element, ancestors, t, role: str = 'content') -> bool:
    """Whether the legacy painter draws `element`'s `role` pass at `t`.

    `render._paint_element`'s own gate: SM's `hidden` bit hard-gates a draw
    independently of alpha, and an ancestor at alpha 0 culls a whole subtree
    the leaf knows nothing about.

    Plus the LIFETIME window the walker applies before the painter is ever
    called (`render._paint_children`: a child outside `[t_start, t_end)` is
    skipped while its siblings draw).

    "Draws" means PUTS PIXELS ON SCREEN, which is why a degenerate chain
    counts as not drawing. The two paths dispose of one differently and
    agree on the result: `compose_links` culls a zero-determinant chain
    outright, while the painter draws a zero-area quad. gat parks an
    ancestor at `zoomy(0)` for minutes at a time, and reading legacy's
    "draws" as literal reported every element under it as MISSING."""
    from analysis.player.render.storyboard.render import _MIN_VISIBLE_ALPHA

    chain = (*ancestors, element)
    if any(not (link.t_start <= t < link.t_end) for link in chain):
        return False
    if any(link.sample('hidden', t)[0] >= 0.5 for link in chain):
        return False
    for prop in ('scale_x', 'scale_y'):
        if any(link.sample(prop, t)[0] == 0.0 for link in chain):
            return False
    if element.kind in _FILL_KINDS and not (element.sample('w', t)[0] > 0.0
                                            and element.sample('h', t)[0] > 0.0):
        # `render._element_size` returns None for a shape whose w/h are not
        # both positive, and the painter returns without drawing - BEFORE
        # `_draw_size` would have let size_x/size_y override the box. A fill
        # primitive's w/h ARE its absolute size (`modfile
        # ._fill_size_timelines`), so this now means a genuinely zero-size
        # shape on both paths.
        return False
    if role == 'glow':
        # The GLOW pass has its own gate: `_paint_glow` returns on a glow
        # alpha at rest, whatever the element's diffuse alpha is doing.
        from analysis.player.render.storyboard.render import _GLOW_MIN_ALPHA
        return element.sample('glow', t)[3] > _GLOW_MIN_ALPHA
    return _legacy_element_alpha(element, ancestors, t) >= _MIN_VISIBLE_ALPHA


def element_parity_report(evaluator, element_order, natural_of, sample_times,
                          atol: float = 1e-3) -> dict:
    """Compare the doc's element blits against the legacy painter at each of
    `sample_times`, in draw order.

    `element_order` is `id_maps['element_order']` - the `(element, role)`
    sequence the build emitted, which the caller must NOT reconstruct: a
    glowing element emits a second blit for its glow pass, so a leaf list
    would misalign the whole stream from the first glow onward.

    `natural_of` maps an element to its `(w, h)` natural box. The executor
    reads that from the uploaded texture, so a caller supplies what the
    texture would have been.

    Returns `{'times': [...], 'all_ok', 'max_corner_err', 'n_fail'}` with
    `max_corner_err` in DESIGN PIXELS - the number worth quoting, because it
    says how far the drawn quad lands from legacy's, not how far off some
    matrix entry is. Only the CONTENT blits are compared: the glow pass draws
    the same quad, so its placement is the content's.
    """
    times = []
    for t in sample_times:
        drawn = {tag: (kind, frec) for kind, _sid, tag, frec
                 in _blit_records(evaluator, t)
                 if 0 < tag < _INSTANCE_TAG_BASE}
        diffs, missing, extra = [], [], []
        worst = 0.0
        n_compared = 0
        n_unsized = 0
        n_projected = 0
        n_unverified = 0
        for index, entry in enumerate(element_order):
            element, role, ancestors = entry
            record = drawn.get(index + 1)
            # WHETHER each side draws it at all, before where. A placement
            # comparison over drawn items alone cannot see a element the doc
            # draws and legacy culls - and a full-screen black rect the doc
            # draws one frame too long blacks out the whole composite while
            # every measured element still reports 0.000px.
            wanted = _legacy_element_drawn(element, ancestors, t, role)
            # A record is not yet a draw. A FILL whose size lanes resolve to
            # zero emits a row that the executors then skip (`_draw_fill`
            # returns on a degenerate box), so counting the row as a draw
            # would report every unsized shape as EXTRA against a reference
            # that is right.
            if record is not None and record[0] == _rec.SRC_FILL:
                w, h = _rec.draw_box((0.0, 0.0), record[1])
                if w <= 0.0 or h <= 0.0:
                    record = None
            if record is None:
                if wanted:
                    missing.append((index, element.kind, role))
                continue
            if not wanted:
                extra.append((index, element.kind, role))
                continue
            if role == 'content':
                # OPACITY, not just placement. The doc applied the leaf's own
                # alpha twice - once on the item lane, once on its own chain
                # link - so every grouped element composited at the square of
                # its alpha while landing at exactly the right pixel.
                want_alpha = _legacy_element_alpha(element, ancestors, t)
                got_alpha = float(record[1][_rec.F_OPACITY])
                if abs(want_alpha - got_alpha) > _ALPHA_ATOL:
                    diffs.append((index, 'opacity',
                                  round(got_alpha - want_alpha, 4)))
            if role != 'content':
                # A glow pass draws the content's own quad, so its placement is
                # already covered. A GLYPH is not: bitmaptext is centred
                # engine-true here and legacy double-shifts it (see
                # `_emit_bitmaptext`), so comparing them would report a
                # systematic offset against a reference that is wrong. Counted
                # so the gap is visible rather than silently uncompared.
                n_unverified += role == 'glyph'
                continue
            kind, frec = record
            # The two sides size a FILL differently and both are right: legacy
            # takes a shape's natural box from its own w/h timelines, while the
            # executor has no texture and scales the unit quad by the size
            # lanes. Handing each its own basis is what makes them comparable.
            if kind == _rec.SRC_FILL:
                want_natural = (element.sample('w', t)[0],
                                element.sample('h', t)[0])
                got_natural = (1.0, 1.0)
            else:
                want_natural = got_natural = natural_of(element)
            if want_natural is None:
                # The caller could not size this asset. That is a gap in the
                # MEASUREMENT, not a placement difference, so it is reported
                # separately - folding it into the diffs would make `all_ok`
                # mean "every element agreed AND every file was readable".
                n_unsized += 1
                continue
            if _chain_order_diverges((*ancestors, element), t):
                # Counted with the 3D chains: another case where the doc
                # follows the engine and legacy does not.
                n_projected += 1
                continue
            if _chain_is_3d((*ancestors, element), t):
                # Counted, not compared - legacy's projected path is not the
                # oracle for this (see `_chain_is_3d`). Reported so the gap is
                # visible: an unmeasured element is exactly the blind spot
                # that hid 118 missing rects behind a 0.001px result.
                n_projected += 1
                continue
            want = _legacy_element_quad(element, t, want_natural, ancestors)
            got = _record_quad(frec, got_natural)
            err = max(max(abs(a - b) for a, b in zip(wc, gc))
                      for wc, gc in zip(want, got))
            worst = max(worst, err)
            n_compared += 1
            if err > atol:
                diffs.append((index, 'corner', round(float(err), 3)))
        times.append({'t': float(t), 'ok': not (diffs or missing or extra),
                      'diffs': diffs, 'missing': missing, 'extra': extra,
                      'n_blit': len(drawn), 'n_compared': n_compared,
                      'n_unsized': n_unsized, 'n_projected': n_projected,
                      'n_unverified': n_unverified, 'max_corner_err': worst})
    return {
        'times': times,
        'all_ok': all(r['ok'] for r in times),
        'max_corner_err': max((r['max_corner_err'] for r in times), default=0.0),
        'n_missing': sum(len(r['missing']) for r in times),
        'n_extra': sum(len(r['extra']) for r in times),
        'n_fail': sum(0 if r['ok'] else 1 for r in times),
    }


def _blit_records(evaluator, t):
    """`(source_kind, source_id, tag, frec)` for every BLIT at `t`, in draw
    order - the whole record row, not just the mat3, so a caller can read the
    box lanes a drawn quad depends on, and the tag that says which item
    produced it. Both quad harnesses read the stream through this."""
    u_bytes, f_bytes, _uf, n = evaluator.frame(float(t))
    u = np.frombuffer(u_bytes, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_bytes, dtype=np.float32).reshape(n, evaluator.f_stride)
    return [(int(u[i, _rec.U_A]), int(u[i, _rec.U_B]), int(u[i, _rec.U_TAG]), f[i])
            for i in range(n) if u[i, _rec.U_KIND] == _rec.OP_BLIT]


# The design box every field-instance blit spans: a field capture and an AFT
# slot are both whole-screen drawables, and legacy's curtain fill covers the
# chart region outright.
_DESIGN_QUAD = ((0.0, 0.0), (_SCREEN_W, 0.0), (_SCREEN_W, _SCREEN_H),
                (0.0, _SCREEN_H))


def _instance_key(inst):
    """A stable identity for one field instance ACROSS provider calls.

    NOT `id()`: the lazy provider rebuilds its instance list whenever the
    topology signature changes, so the list the doc was built from and the one
    the harness samples are different objects describing the same actors. An
    id-keyed lookup then matches nothing - on LINARIA it reported every drawn
    instance as EXTRA and every legacy draw as unaccounted for."""
    return inst['kind'], inst.get('name')


def _legacy_field_draws(instances, base_hidden, t) -> dict:
    """`_instance_key -> (quad, alpha, kind)` for every field instance the
    legacy effect DRAWS at `t`.

    The gate is legacy's own: `transform.at(t)` is the very object
    `NotitgFieldInstances.at` calls, not a transcription of it, and the
    base-hidden rule and stage fold call the same helpers. What IS restated
    here is the loop that walks the instances - which exists only to keep the
    instance IDENTITY that `at()` discards, and which `field_parity_report`
    cross-checks against `at()`'s own entry count so a drift shows up as a
    self-check failure rather than a wrong reference.
    """
    from analysis.games.notitg.field_instances import _fold_stage_chain

    stages: dict = {}
    slots: set = set()
    draws: dict = {}
    for inst in instances:
        kind = inst['kind']
        if kind == 'stage':
            sampled = inst['transform'].at(t)
            if sampled is not None:
                stages[inst['name']] = (*sampled, inst['transform'].crop_at(t),
                                        inst['source'])
            continue
        if kind == 'player' and base_hidden:
            continue
        sampled = inst['transform'].at(t)
        if sampled is None:
            continue
        h, alpha = sampled
        if kind == 'capture':
            slots.add(inst['name'])
        elif kind == 'aft':
            h, alpha, _crop, _extra = _fold_stage_chain(
                stages, slots, inst, h, alpha, inst['transform'].crop_at(t),
                None)
        # A curtain covers the chart region outright - legacy's `batch.fill`
        # takes the colour and the crop and ignores the transform entirely
        # (qt_renderer._blit_field_instance), so its quad is the design box.
        #
        # TRANSPOSED: a field transform is a ROW-vector homography ([x y 1] @ H,
        # the Qt QTransform layout `transform3d.qtransform_from_h` hands
        # straight to Qt), while the record mat3 and `_apply_h` are
        # column-vector. Comparing them untransposed reports every instance as
        # degenerate rather than as a placement difference.
        quad = _DESIGN_QUAD if kind == 'fill' \
            else _apply_h(np.asarray(h, dtype=float).reshape(3, 3).T,
                          _DESIGN_QUAD)
        draws[_instance_key(inst)] = (quad, min(1.0, alpha), kind)
    return draws


def field_parity_report(evaluator, compiled, instance_order, sample_times,
                        atol: float = 1e-2) -> dict:
    """Compare the doc's FIELD-INSTANCE commands against the legacy effect at
    each of `sample_times`, per instance.

    The twin of `element_parity_report` for the other half of the doc. It
    answers the question a corner error cannot: which instances does legacy
    draw that the doc drops, and vice versa. `instance_order` is
    `id_maps['instance_order']`.

    A doc instance that emits a Snapshot or a Feed carries no comparable quad
    (a snapshot draws nothing; a feed re-renders notes as many items), so those
    are counted as `n_uncomparable` and only their PRESENCE is checked - which
    is still the interesting half, since a dropped feed is a field that
    vanished.

    A chain that leaves the z=0 plane is counted as `n_projected` and not
    compared, the same ruling `element_parity_report` makes: legacy's projected
    path is not the oracle for 3D. On a chart like gat that is most of the
    instances, so read `n_compared` before reading the corner error.

    `self_check` is False for a time where this walk disagrees with
    `NotitgFieldInstances.at` about how many entries legacy draws. That means
    the reference is wrong, so the whole row is untrustworthy rather than a
    finding about the doc.

    `atol` is looser than the element harness's: a field quad is the whole
    640x480 design box folded through a chain of f32 record lanes, and lands
    within ~2e-3px of legacy. A tighter bar reports that rounding as a defect.
    """
    from types import SimpleNamespace

    from analysis.games.notitg.field_instances import NotitgFieldInstances

    provider = compiled.get('field_instances')
    base_gate = compiled.get('base_field_hidden')
    effect = NotitgFieldInstances(provider, base_hidden=base_gate,
                                  player_fields=compiled.get('player_fields'))
    ctx = SimpleNamespace(chart_rect=(0.0, 0.0, _SCREEN_W, _SCREEN_H))

    times = []
    for t in sample_times:
        t = float(t)
        instances = provider() if callable(provider) else provider
        base_hidden = (base_gate is not None
                       and base_gate.sample(t)[0] >= 0.5)
        want = _legacy_field_draws(instances or (), base_hidden, t)
        ctx.t_now = t
        frame = effect.at(ctx)
        got = {tag: (kind, sid, frec) for kind, sid, tag, frec
               in _blit_records(evaluator, t) if tag >= _INSTANCE_TAG_BASE}

        missing, extra, diffs = [], [], []
        worst = 0.0
        n_compared = n_uncomparable = n_projected = 0
        # An instance legacy draws that the doc emitted NO command for is
        # invisible to the walk below, which only iterates what the doc
        # emitted. Naming those first is the whole point of the report.
        emitted = {_instance_key(inst) for inst, _cmd in instance_order}
        for key in want.keys() - emitted:
            missing.append((-1, key[1] or key[0], key[0]))
        for index, (inst, command) in enumerate(instance_order):
            record = got.get(_INSTANCE_TAG_BASE + index + 1)
            expected = want.get(_instance_key(inst))
            name = inst.get('name') or inst['kind']
            if expected is None:
                if record is not None:
                    extra.append((index, name, inst['kind']))
                continue
            if command != 'item':
                # Present-or-not is all a snapshot/feed can be checked on, and
                # a snapshot emits no record at all, so only a feed is even
                # observable here. Counted, never silently passed.
                n_uncomparable += 1
                continue
            if record is None:
                missing.append((index, name, inst['kind']))
                continue
            if not all(_link_is_planar(link)
                       for link in inst['transform']._links):
                # Counted, not compared - the same ruling `_chain_is_3d` makes
                # for elements. Legacy folds a field chain through its own
                # projected path, and that path is not the oracle for 3D; the
                # engine is. Comparing anyway would report a real difference
                # against a reference that has no authority.
                n_projected += 1
                continue
            _kind, _sid, frec = record
            got_alpha = float(frec[_rec.F_OPACITY])
            if abs(expected[1] - got_alpha) > _ALPHA_ATOL:
                diffs.append((index, name,
                              round(got_alpha - expected[1], 4)))
            # Every field-instance blit sizes from the design box: a field
            # capture and an AFT slot are whole-screen drawables, and a
            # curtain fill takes its target's box (`_draw_fill`) because it
            # has no texture of its own.
            err = max(max(abs(a - b) for a, b in zip(wc, gc))
                      for wc, gc in zip(expected[0],
                                        _record_quad(frec, (_SCREEN_W,
                                                            _SCREEN_H))))
            worst = max(worst, err)
            n_compared += 1
            if err > atol:
                diffs.append((index, name, round(float(err), 3)))
        n_frame = 0 if frame is None else len(frame.fields)
        times.append({
            't': t, 'n_legacy': len(want), 'n_doc': len(got),
            'n_compared': n_compared, 'n_uncomparable': n_uncomparable,
            'n_projected': n_projected,
            'missing': missing, 'extra': extra, 'diffs': diffs,
            'max_corner_err': worst,
            # `at()` prepends a base-field entry when the base is visible; the
            # walk above only sees real instances, so that one is expected.
            # The second clause catches two instances sharing a
            # `_instance_key`, which would merge them and report the loser as
            # both missing and extra.
            'self_check': (n_frame - (0 if base_hidden else 1) == len(want)
                           and len({_instance_key(i) for i in instances or ()})
                           == len(instances or ())),
            'ok': not (missing or extra or diffs),
        })
    return {
        'times': times,
        'all_ok': all(r['ok'] and r['self_check'] for r in times),
        'max_corner_err': max((r['max_corner_err'] for r in times), default=0.0),
        'n_missing': sum(len(r['missing']) for r in times),
        'n_extra': sum(len(r['extra']) for r in times),
        # Read this BEFORE `all_ok`. A chart whose field rig is entirely 3D
        # (gat is) compares nothing, and an unqualified "OK" over zero
        # comparisons is the most convincing wrong answer this report can give.
        'n_compared': sum(r['n_compared'] for r in times),
        'n_projected': sum(r['n_projected'] for r in times),
        'n_fail': sum(0 if r['ok'] else 1 for r in times),
    }


def format_field_parity_report(report) -> str:
    """A one-line-per-sample-time human summary of `field_parity_report`."""
    verdict = ('OK' if report['all_ok'] else 'FAIL') if report['n_compared'] \
        else 'VACUOUS (nothing planar to compare)'
    lines = [
        f"field parity: {verdict} "
        f"({report['n_fail']}/{len(report['times'])} times failing) "
        f"compared={report['n_compared']} 3d={report['n_projected']} "
        f"missing={report['n_missing']} extra={report['n_extra']} "
        f"max_corner_err={report['max_corner_err']:.3f}px"
    ]
    for r in report['times']:
        detail = ''
        if r['missing']:
            detail += f"  MISSING={[m[1] for m in r['missing'][:5]]}"
        if r['extra']:
            detail += f"  EXTRA={[e[1] for e in r['extra'][:5]]}"
        if r['diffs']:
            detail += f"  diffs={[(d[1], d[2]) for d in r['diffs'][:4]]}"
        if not r['self_check']:
            detail += '  [self-check FAILED: reference untrustworthy]'
        lines.append(f"  t={r['t']:8.3f}  legacy={r['n_legacy']:>3} "
                     f"doc={r['n_doc']:>3} compared={r['n_compared']:>3} "
                     f"3d={r['n_projected']:>3} "
                     f"uncomparable={r['n_uncomparable']:>3} "
                     f"corner_err={r['max_corner_err']:.3f}px{detail}")
    return '\n'.join(lines)


def format_element_parity_report(report) -> str:
    """A one-line-per-sample-time human summary of `element_parity_report`."""
    lines = [
        f"element parity: {'OK' if report['all_ok'] else 'FAIL'} "
        f"({report['n_fail']}/{len(report['times'])} times failing) "
        f"missing={report['n_missing']} extra={report['n_extra']} "
        f"max_corner_err={report['max_corner_err']:.3f}px"
    ]
    for r in report['times']:
        detail = f"  diffs={[d[:2] for d in r['diffs'][:4]]}" if r['diffs'] else ''
        if r['missing']:
            detail += f"  MISSING={r['missing'][:4]}"
        if r['extra']:
            detail += f"  EXTRA={r['extra'][:4]}"
        unsized = f" unsized={r['n_unsized']}" if r['n_unsized'] else ''
        unsized += f" 3d={r['n_projected']}" if r['n_projected'] else ''
        unsized += f" glyphs={r['n_unverified']}" if r['n_unverified'] else ''
        lines.append(f"  t={r['t']:8.3f}  drawn={r['n_blit']:>4} "
                     f"compared={r['n_compared']:>4}{unsized} "
                     f"corner_err={r['max_corner_err']:.3f}px{detail}")
    return '\n'.join(lines)
