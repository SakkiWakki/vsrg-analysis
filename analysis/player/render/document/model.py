"""The compiled-document node model: groups, layers, timelines, leaves.

This is the render-side data shape every game adapter compiles INTO and
the player consumes through one loading path. It is DATA, not behaviour:
frozen dataclasses and the lightest container that fits each field, so
the whole document is a serializable dump (the future native map format)
and a golden-test artifact. A renderer walks this tree; it never mutates
it. See DESIGN_compiled_document.md for the six-axis charter.

The two structural axes are ORTHOGONAL (this is the refinement the
sketch left implicit):

  * GROUP is "transform/effect with me". Every node has one `parent`
    group; a group's property timelines (translate/rotate/scale/opacity/
    effect) compose onto its whole subtree, recursively. Membership is
    over the SUPERSET of map objects -- storyboard sprites, the
    notefield, field copies, text, AND note subsets. A note subset is
    not a distinct node kind: a `notefield` leaf carries a per-note
    `membership` array (parallel to the chart's notes) naming which
    group each note belongs to, generalizing Quaver's `_note_sv_groups`
    from a timing behaviour to arbitrary group membership. So "mirror
    the background and THESE four notes and spin them" is: one group
    node with the background leaf as a child and a notefield leaf whose
    membership array tags those four notes into the group.

  * LAYER is "composite HERE". Independent of the group hierarchy, every
    node names one `layer` slot (a `Timeline`, because NotITG's draworder
    is re-slottable over time -- a node can move between strata as t
    advances). The document's `strata` are the ordered compositing bands
    (background, field, notes, hud, ...); a node's slot value at t picks
    its band, and draws sort within a band by (slot, then tree order).
    CAPTURE RANGES are declared over strata, not bespoke pixmaps: an
    AFT/proxy leaf or a shader pass names the `(low, high)` stratum range
    it samples (see `CaptureRange`), replacing the hand-carved field-vs-
    full pixmap plumbing.

A node therefore carries BOTH: its group parent decides what transforms
it inherits, its layer slot decides where it composites. The two never
collapse into one -- a note can be in a "spin these" group (transform)
while compositing in the notes stratum (layer), and a background sprite
can be in the same spin group while compositing in the background
stratum.

TIMELINES. Every animatable property is a `Timeline` keyed
`(clock_key, curve)` -- the time axis is the SV integral engine
(`analysis.player.sv/`): song time, beat = integral of bpm, scroll =
integral of sv, per-group engine keys. `curve` is any object with
`.sample(t) -> tuple` (the existing `EventTimeline`, `_SpanGatedTimeline`,
`_SumTimeline` all satisfy it); this model does not re-implement
sampling. A node's `properties` dict maps property name -> Timeline.

COMPILED VISIBILITY (first-class, required). Every node carries a
`visibility` Timeline giving an explicit is-rendered answer (>=0.5 ->
rendered) for ALL t -- never a last-value hold. A driven visual rests to
NOT-rendered outside its driver's lifetime; this is the structural form
of the `_SpanGatedTimeline` gate (games/notitg/modfile.py) that fixed
the ghost-receptor bug class. `visibility` is REQUIRED at construction;
`Node.always_visible()` supplies an always-1 timeline and is the ONLY
sanctioned default, reserved for design-time-static scenery whose
lifetime is its whole [t_start, t_end) window. A driven node (its
transform comes from a per-frame integrator) must pass a real gated
visibility, never `always_visible()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from analysis.player.render.effects.timeline import EventTimeline

# A curve is any object exposing `.sample(t_now) -> tuple`. EventTimeline
# is the workhorse; _SpanGatedTimeline / _SumTimeline (notitg/modfile.py)
# are the other in-tree instances. The document stores curves opaquely so
# a renderer samples uniformly and the model owns no sampling logic.
Curve = object

# A note-membership array is `np.ndarray | None` parallel to the chart's
# notes, each entry the group id a note belongs to (None -> default
# group). This is `SvReplayDoc.note_groups` generalized past timing.


@dataclass(frozen=True)
class Timeline:
    """One animatable property over time: a `curve` sampled against a
    named `clock`. `clock_key` selects the integral driving it -- 'song'
    (default, wall-clock seconds), 'beat' (integral of bpm), 'scroll'
    (integral of sv), or a per-group engine key. The document builder
    resolves clock_key -> the concrete SV engine at load; the curve
    itself is authored in whatever clock its keyframes use, so the
    renderer samples `curve.sample(clock_value_at(t))`.

    Phase 1 leaves clock resolution to phase 5: `clock_key='song'` and a
    curve authored in song seconds is the identity path, so every
    existing timeline drops in unchanged."""
    curve: Curve
    clock_key: str = 'song'

    def sample(self, t: float) -> tuple:
        return self.curve.sample(t)


def _always_one() -> Timeline:
    return Timeline(EventTimeline((), rest=(1.0,)))


# --- leaf content variants -----------------------------------------------
# Each is the lightest frozen container carrying only what its draw needs;
# a Node holds exactly one via `content` (None for a pure group).


@dataclass(frozen=True)
class SpriteContent:
    """A textured quad. `asset` is an absolute path (or the virtual name
    'white'); `additive` selects the blend. Size/tint/frame come from the
    node's property timelines, so this holds only the immutable asset
    identity."""
    asset: str
    additive: bool = False


@dataclass(frozen=True)
class TextContent:
    """A text run. `font` is None for a Qt-drawn string (size from the
    node's `font_px` property) or an SM bitmap-font atlas object for
    glyph compositing. `text` is the immutable string; a stream-bound
    content string (phase 5) replaces it with `stream` naming a data
    channel."""
    text: str = ''
    font_px: float = 0.0
    font: object = None
    stream: str | None = None


@dataclass(frozen=True)
class RectContent:
    """A filled or outlined rectangle/ellipse. `outline` selects stroke
    vs fill; `ellipse` selects the shape. Size/color/border come from the
    node's timelines."""
    outline: bool = False
    ellipse: bool = False


@dataclass(frozen=True)
class NotefieldContent:
    """The playfield: notes, holds, receptors, drawn through the note
    pipeline. `membership` is the per-note group array (parallel to the
    chart's notes, each entry a group id or None) that lets a NOTE SUBSET
    join a group -- the Quaver `_note_sv_groups` pattern generalized from
    timing to arbitrary group membership. None -> every note is in the
    node's own group. This is the only leaf whose transform additionally
    samples a note-path curve by scroll offset (axis 5 applied to axis
    3)."""
    membership: object = None  # np.ndarray | None, parallel to notes


@dataclass(frozen=True)
class CaptureRange:
    """A declared range over the document's compositing strata, sampled
    as a texture. `low`/`high` are stratum indices into
    `CompiledDocument.strata` (inclusive). A field proxy declares the
    notes-stratum range (notes + receptors only); a full-screen AFT
    declares [background .. notes]; a fullscreen shader pass declares the
    whole stack. Replaces the bespoke field-vs-full pixmap plumbing with
    one declarative rule."""
    low: int
    high: int

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError(
                f'capture range low {self.low} > high {self.high}')
        if self.low < 0:
            raise ValueError(f'capture range low {self.low} < 0')


@dataclass(frozen=True)
class CaptureContent:
    """A capture-of-node leaf (NotITG AFT/proxy, shader-pass input): it
    draws the pixels of the strata in `capture`, transformed by the
    node's own timelines. The capture range is declared over the
    document's strata (see `CaptureRange`), not a hand-built pixmap, so
    every proxy/AFT/shader source names the compositing band it sees."""
    capture: CaptureRange


_CONTENT_VARIANTS = (SpriteContent, TextContent, RectContent,
                     NotefieldContent, CaptureContent)


@dataclass(frozen=True)
class Node:
    """One entity in the group tree: a group parent, a layer slot, its
    property timelines, its children, and (unless it is a pure group) one
    leaf `content`.

    `parent` is the id of the group this node transforms with (None for a
    root). `children` are the ids of nodes whose transforms this node
    (when a group) composes onto. `layer` is the Timeline whose sampled
    value picks this node's compositing stratum at t (NotITG re-slottable
    draworder). `properties` maps property name -> Timeline (translate/
    rotate/scale/opacity/color/size/...). `visibility` is the REQUIRED
    compiled-visibility timeline (>=0.5 rendered) -- see the module
    docstring; construct via `Node.always_visible(...)` only for static
    scenery.

    A group node has `content=None` and draws nothing itself; a leaf has
    `content` set and no `children`. The design-time window `[t_start,
    t_end)` bounds existence; `visibility` bounds rendering WITHIN it (the
    two together are the full is-rendered answer)."""
    node_id: str
    parent: str | None
    layer: Timeline
    visibility: Timeline
    t_start: float
    t_end: float
    properties: dict = field(default_factory=dict)
    children: tuple = ()
    content: object = None

    def __post_init__(self):
        if self.content is not None and not isinstance(
                self.content, _CONTENT_VARIANTS):
            raise TypeError(
                f'node {self.node_id!r} content is '
                f'{type(self.content).__name__}, not a leaf variant')
        if self.content is not None and self.children:
            raise ValueError(
                f'node {self.node_id!r} is a leaf but has children; '
                'a node is a group (children) or a leaf (content), '
                'never both')

    @classmethod
    def always_visible(cls, node_id, parent, layer, t_start, t_end,
                       properties=None, children=(), content=None):
        """Construct a node whose visibility is always-1 across its whole
        window. The ONLY sanctioned visibility default -- reserved for
        design-time-static scenery. A driven node must pass a real gated
        `visibility` to the primary constructor instead."""
        return cls(node_id=node_id, parent=parent, layer=layer,
                   visibility=_always_one(), t_start=t_start, t_end=t_end,
                   properties=properties or {}, children=children,
                   content=content)

    @property
    def is_group(self) -> bool:
        return self.content is None
