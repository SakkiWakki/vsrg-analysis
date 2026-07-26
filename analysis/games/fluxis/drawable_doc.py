"""fluXis storyboard elements -> channel-backed DrawableDoc (Seam A).

The first NON-NotITG producer of the Drawable model (`.claude/plans/drawable-ir.md`),
and the point of the exercise: proving that Seam A - a game-agnostic `DrawableDoc`
a Rust evaluator turns into a flat op stream each frame (Seam B) - is genuinely
game-agnostic. It is the banded-format sibling of `notitg.drawable_doc`: where
NotITG folds a field-instance topology (player fields, proxy re-renders, AFT
samplers) in engine TREE order, fluXis has NO field instances - its storyboard
format IS z-banded (`fsb_storyboard._LAYER_Z`: Background / Foreground / Overlay
layers map to distinct z bands). So the drawable-ir's "compile-time coarse
placement of items (their formats are genuinely banded)" ruling applies directly:
coarse placement = layer order, then `(z, z_index, t_start)` within a layer, in
ONE flat ordered emission onto the screen root drawable.

`build_doc(elements_or_compiled, screen_w, screen_h)` accepts either a fluXis
`Storyboard`, a plain sequence of `model.Element`s, or a `compiled`-style dict
(`{'tree': elements}` / `{'elements': elements}`), flattens groups away (their
image leaves band by their OWN z, an interim until group-transform folding lands),
and emits every image-backed leaf (sprite / frames) as an `SRC_IMAGE` item whose
per-property scalar `EventTimeline`s ride the item's own transform lanes via the
local `export_channel`. Every other kind (rect / ellipse / outline_* / text /
bitmaptext / video / a sprite with no asset) is skipped with a per-kind count in
the report - that gap tally is the deliverable: it enumerates which fluXis element
features do NOT yet fit the Item vocabulary.

The playfield sits between fluXis's Background layer (below) and Foreground layer
(above) in engine z. There are no notefield drawables in this static doc yet, so
a single RESERVED playfield drawable id (`id_maps['playfield']`) is minted and
documented for a later wave to bind the notefield feed onto; the storyboard items
straddle it by band exactly as they will once it draws.

`export_channel` stays local to each game: what a game's curves ARE differs (fluXis
has no segment lanes and no oscillator deltas), so the dispatch differs. What both
copies used to duplicate - translating an `EventTimeline`'s keyframes into
breakpoints - now lives on `EventTimeline.breakpoints` itself, which is where the
keyframes are.

No Qt import at module load - importable headless.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.effects.timeline import EventTimeline
from analysis.player.render.storyboard.sprite_sheet import frame_at_time

# Drawable 0 is always the screen root (minted by the builder).
_SCREEN_ID = 0

# The lazy-curve sampling cadence (LiveCurve / SegCurve expose no keyframes, so
# they are densely sampled). 1/30s: the interim approximation the spec calls out
# - an EventTimeline is exported exactly instead.
_DENSE_DT = 1.0 / 30.0

# The channel export window end when the caller supplies no explicit horizon.
# Generous, since channels hold their tail past it.
_DEFAULT_HORIZON = 600.0

# Storyboard element kinds emitted as Image items. A 'sprite' (single cell) and
# a frame-animated 'frames' sheet both back an SRC_IMAGE draw; every other fluXis
# kind (rect / ellipse / outline_* / text / bitmaptext / video / group-with-no-
# image) is skipped with a per-kind count in the report.
_IMAGE_KINDS = ('sprite', 'frames')

# A group (osu-framework Container / ActorFrame) draws nothing itself; its
# transform composes onto its children. This wave does not yet fold the group
# transform into the leaves (item lanes carry each leaf's OWN timelines only), so
# a group is flattened away and its image leaves emit at their own placement - a
# documented interim (true group composition awaits the transform-fold wave).
_SKIP_GROUP = 'group'

# The element property -> native `item` transform-lane mapping. Each pair is
# (element timeline prop, the item() id/rest kwarg stem); every lane is a scalar
# EventTimeline exported once through export_channel. `hidden` is handled apart
# (inverted into the item's visible gate), and the sheet frame is its own derived
# channel - neither appears here.
_ELEMENT_ITEM_LANES = (
    ('x', 'x'), ('y', 'y'),
    ('scale_x', 'sx'), ('scale_y', 'sy'),
    ('rotation', 'rot'),
    ('alpha', 'opacity'),
)


# --------------------------------------------------------------------------
# export_channel - the shared timeline -> breakpoint-triple primitive
# (DUPLICATED from notitg.drawable_doc; dedup once both waves land)
# --------------------------------------------------------------------------

def export_channel(timeline, t0: float, t1: float, prop: int = 0):
    """Convert one Python timeline into the `(ts, vals, durs, rest)` breakpoint
    quadruple `DocBuilder.channel` ingests, over the window `[t0, t1]`.

    The native channel (native/src/channels.rs) serves `rest` before the first
    breakpoint, then at each breakpoint either holds its value (dur <= 0) or
    ramps LINEARLY toward the next breakpoint's value over `dur`. This helper
    emits breakpoints reproducing the timeline's own playback under that model:

    - `EventTimeline`: EXACT, from the timeline's own `breakpoints` (the
      keyframes ARE the shape). Rest before the first keyframe is carried on
      the ChannelRef, so no pre-roll breakpoint is emitted. Its ease ids are
      dropped: every breakpoint it emits is already a linear ramp.
    - anything else (`LiveCurve` / `SegCurve`, or any `.sample(t)` duck):
      dense-sampled at `_DENSE_DT` across `[t0, t1]` (the documented interim
      approximation - these lazy curves expose no keyframe structure).

    `prop` selects the value index within the timeline's sampled tuple (element
    scalar timelines are index 0). Returns `(ts, vals, durs, rest)` - the four
    args of `DocBuilder.channel` - as plain Python lists + a float rest.
    """
    if isinstance(timeline, EventTimeline):
        ts, vals, durs, _eases = timeline.breakpoints(t0, t1, prop)
        return ts, vals, durs, _rest_value(timeline, prop)
    return _export_dense(timeline, t0, t1, prop)


def _rest_value(timeline, prop: int) -> float:
    """The timeline's pre-first-keyframe rest value for `prop`. EventTimeline
    exposes it directly; a duck-typed curve is sampled far in the past."""
    rest = getattr(timeline, '_rest', None)
    if rest is not None:
        return float(rest[prop])
    return float(timeline.sample(-1.0e18)[prop])


def _export_dense(timeline, t0: float, t1: float, prop: int):
    """Dense-sample a duck-typed curve across `[t0, t1]` at `_DENSE_DT`
    (the interim approximation for LiveCurve / SegCurve)."""
    rest = _rest_value(timeline, prop)
    if t1 <= t0:
        return [], [], [], rest
    n = max(1, int(np.ceil((t1 - t0) / _DENSE_DT)))
    step = (t1 - t0) / n
    ts: list[float] = []
    vals: list[float] = []
    durs: list[float] = []
    for k in range(n + 1):
        bt = t0 + k * step
        ts.append(bt)
        vals.append(float(timeline.sample(bt)[prop]))
        durs.append(step if k < n else 0.0)
    return ts, vals, durs, rest


# --------------------------------------------------------------------------
# Elements - normalize input, flatten groups, band by layer z
# --------------------------------------------------------------------------

def _elements_of(elements_or_compiled) -> tuple:
    """The element sequence from any accepted input shape: a fluXis `Storyboard`
    (`.elements`), a `compiled`-style dict (`{'tree'/'elements': [...]}`), or a
    bare sequence of `model.Element`s."""
    source = elements_or_compiled
    elements = getattr(source, 'elements', None)
    if elements is not None:
        return tuple(elements)
    if isinstance(source, dict):
        return tuple(source.get('tree') or source.get('elements') or ())
    return tuple(source or ())


class _FrameCurve:
    """A `.sample(t)`-duck curve tracing a sheet sprite's current frame index
    over time, so `export_channel` dense-samples it into the item's frame lane.

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


def _flatten_elements(elements) -> tuple:
    """The element tree flattened to a leaf list in tree order, groups replaced
    by their children (see `_SKIP_GROUP`). Returns `(leaves, group_count)`."""
    leaves = []
    groups = 0
    stack = list(reversed(list(elements)))
    while stack:
        element = stack.pop()
        if element.kind == _SKIP_GROUP:
            groups += 1
            stack.extend(reversed(list(element.children)))
            continue
        leaves.append(element)
    return leaves, groups


def _ordered_leaves(leaves) -> list:
    """Flattened leaves in coarse-placement order: layer order (the layer maps
    to a distinct z band) then `(z, z_index, t_start)` within a layer. Because a
    fluXis layer IS its z band, one `(z, z_index, t_start)` sort realises both
    the cross-layer band order and the within-layer draw order - the drawable-ir
    coarse placement for a genuinely banded format."""
    return sorted(leaves, key=lambda e: (e.z, e.z_index, e.t_start))


# --------------------------------------------------------------------------
# build_doc - the flat banded-emission compiler
# --------------------------------------------------------------------------

class _Builder:
    """Threads the DocBuilder through one build: mints the reserved playfield
    drawable, exports channels (de-duplicating identical timelines by object
    identity), and emits each image leaf straddling the playfield by band. Not
    reused across builds."""

    def __init__(self, screen_w, screen_h):
        import storyboard_native as sn

        self._sn = sn
        self._t0 = 0.0
        self._t1 = _DEFAULT_HORIZON
        self._builder = sn.DocBuilder(float(screen_w), float(screen_h))
        self._screen = (float(screen_w), float(screen_h))
        # A reserved, non-drawing playfield drawable a later wave binds the
        # notefield feed onto. Minted so the id is stable and documented now;
        # nothing sources it yet.
        self._playfield_id = self._new_drawable(persistent=False)
        # Storyboard image sources: an image id per DISTINCT absolute asset path
        # (loading is the consumer's job). id_maps carries the {id -> path} map.
        self._image_ids: dict[str, int] = {}
        self._image_paths: dict[int, str] = {}
        # Per-band emitted counts + per-kind skip counts, surfaced in the report.
        self._elem_below = 0
        self._elem_above = 0
        self._elem_skips: dict[str, int] = {}
        # Memoize channel ids by timeline object identity + prop, so shared rest
        # timelines collapse to one channel each.
        self._chan_cache: dict[tuple[int, int], tuple[int, float]] = {}

    def _new_drawable(self, persistent: bool) -> int:
        return self._builder.drawable(self._screen[0], self._screen[1],
                                      persistent=persistent, dynamic=False)

    # -- channel export ---------------------------------------------------

    def _channel(self, timeline, prop: int = 0) -> tuple[int, float]:
        """The (channel_id, rest) for a timeline+prop, exported once and
        memoized. A real channel is always pushed (the rest still rides the
        ChannelRef)."""
        key = (id(timeline), prop)
        cached = self._chan_cache.get(key)
        if cached is not None:
            return cached
        ts, vals, durs, rest = export_channel(timeline, self._t0, self._t1, prop)
        chan_id = self._builder.channel(
            [float(v) for v in ts], [float(v) for v in vals],
            [float(v) for v in durs], float(rest))
        result = (chan_id, float(rest))
        self._chan_cache[key] = result
        return result

    # -- element emission -------------------------------------------------

    def run(self, elements_or_compiled):
        """Emit the below-playfield image leaves, (a later wave draws the
        playfield here), then the above-playfield image leaves, and finish.

        A leaf bands below the playfield when its z is below the playfield band
        (`z < 0` - fluXis's Background layer sits at -900), above otherwise
        (Foreground / Overlay at 400 / 700). Within each band the coarse-placement
        `(z, z_index, t_start)` order holds."""
        raw = _elements_of(elements_or_compiled)
        leaves, groups = _flatten_elements(raw)
        if groups:
            self._elem_skips['group'] = self._elem_skips.get('group', 0) + groups
        ordered = _ordered_leaves(leaves)

        below = [e for e in ordered if e.z < 0]
        above = [e for e in ordered if e.z >= 0]
        self._elem_below = self._emit_band(below)
        # (reserved playfield draws here in a later wave)
        self._elem_above = self._emit_band(above)

        evaluator = self._builder.finish()
        id_maps = {'screen': _SCREEN_ID, 'playfield': self._playfield_id,
                   'images': dict(self._image_paths)}
        return evaluator, id_maps

    def _emit_band(self, leaves) -> int:
        """Emit each leaf in `leaves` as an Image item (or count its skip);
        returns the number of items actually emitted."""
        emitted = 0
        for element in leaves:
            if self._emit_element(element):
                emitted += 1
        return emitted

    def _emit_element(self, element) -> bool:
        """Emit one leaf as an SRC_IMAGE item, or count it as a per-kind skip.
        Returns True when an item was emitted. Only image-backed kinds (sprite /
        frames) with a resolvable asset path draw; everything else - rect,
        ellipse, outline_*, text, bitmaptext, video, an image kind with no asset
        - is skipped and tallied by kind (an asset-less image kind counts as
        'no_asset')."""
        if element.kind not in _IMAGE_KINDS:
            self._count_skip(element.kind)
            return False
        image_id = self._image_id(element)
        if image_id is None:
            self._count_skip('no_asset')
            return False
        self._sn_image_item(image_id, element)
        return True

    def _count_skip(self, kind: str) -> None:
        self._elem_skips[kind] = self._elem_skips.get(kind, 0) + 1

    def _image_id(self, element):
        """The image id for an element's asset (a 'frames' element uses its first
        frame path), minted once per distinct absolute path and recorded in the
        image table. None when the element carries no asset."""
        path = element.asset or (element.frames[0] if element.frames else None)
        if not path:
            return None
        image_id = self._image_ids.get(path)
        if image_id is None:
            image_id = len(self._image_ids)
            self._image_ids[path] = image_id
            self._image_paths[image_id] = path
        return image_id

    def _sn_image_item(self, image_id: int, element) -> None:
        """Push one SRC_IMAGE item: the element's scalar transform timelines on
        the item's own lanes (export_channel each), the inverted `hidden` gate on
        `visible`, and the sheet-frame channel on `frame`. The anchor/origin
        natural-size offset is NOT folded (item lanes are the leaf's own x/y);
        that placement refinement rides the transform-fold wave."""
        kwargs = self._element_transform_kwargs(element)
        kwargs.update(self._element_frame_kwarg(element))
        kwargs.update(self._element_visible_kwarg(element))
        self._builder.item(_SCREEN_ID, self._sn.SRC_IMAGE, image_id,
                           additive=bool(element.additive), **kwargs)

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
        1x1 sprite rests at frame 0 (no lane)."""
        if element.sheet_cols * element.sheet_rows <= 1 and element.state_pin is None:
            return {}
        chan_id, rest = self._channel(_FrameCurve(element))
        return {'frame_id': chan_id, 'frame_rest': rest}

    def _element_visible_kwarg(self, element) -> dict:
        """The element's SM `hidden` bit inverted onto the item's visible gate
        (visible = 1 - hidden). Absent -> constant visible."""
        hidden = element.timelines.get('hidden')
        if hidden is None:
            return {}
        ts, vals, durs, rest = export_channel(hidden, self._t0, self._t1)
        inv_vals = [1.0 - v for v in vals]
        chan_id = self._builder.channel(
            [float(v) for v in ts], inv_vals,
            [float(v) for v in durs], 1.0 - float(rest))
        return {'visible_id': chan_id, 'visible_rest': 1.0 - float(rest)}


def build_doc(elements_or_compiled, screen_w: float, screen_h: float):
    """Compile fluXis storyboard elements into a channel-backed DrawableDoc.

    `elements_or_compiled` may be a fluXis `Storyboard`, a `compiled`-style dict
    (`{'tree'/'elements': [...]}`), or a bare sequence of `model.Element`s.
    `screen_w`/`screen_h` are the design canvas (the storyboard's `design_w`/
    `design_h`).

    Returns `(evaluator, id_maps, report)`:
      - `evaluator` : a finished `storyboard_native.Evaluator`;
      - `id_maps`   : `{'screen': 0, 'playfield': <reserved id>,
                        'images': {image_id -> absolute path}}`;
      - `report`    : `{'elements_below', 'elements_above', 'images',
                        'element_skips': {kind -> count}}`.

    The screen root drawable (0) carries the whole command list. Image-backed
    leaves (sprite / frames) draw around the reserved playfield drawable by z
    band - below-band (Background, z < 0) first, then the playfield (a later
    wave), then above-band (Foreground / Overlay, z >= 0) - with the coarse
    `(z, z_index, t_start)` order within each band. Every non-image kind is
    skipped and tallied by kind in `element_skips`; that tally enumerates which
    fluXis element features do not yet fit the Item vocabulary.
    """
    b = _Builder(screen_w, screen_h)
    evaluator, id_maps = b.run(elements_or_compiled)
    report = {
        'elements_below': b._elem_below,
        'elements_above': b._elem_above,
        'images': len(id_maps['images']),
        'element_skips': dict(b._elem_skips),
    }
    return evaluator, id_maps, report


# --------------------------------------------------------------------------
# Blit-stream inspection (test/consumer helper) - mirrors evaluate.rs
# --------------------------------------------------------------------------

_OP_BLIT = 1
_SRC_IMAGE = 0


def blit_stream(evaluator, t):
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
