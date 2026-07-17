"""`document_from_player`: wrap a loaded player's per-game outputs into a
`CompiledDocument`, with zero behaviour change.

Phase 1 is a WRAPPER, not a rewrite. The player keeps rendering through
the existing effects pipeline; this builds the skeleton document beside
it so the structure exists and one real consumer -- the storyboard
design-space mapping -- reads design->screen through the document header
instead of a scattered constant. Population is partial: the design space
(always), the storyboard subtree (when the game has one), and the
placeholder clock/stream tables. Effects, note paths, capture ranges,
and streams migrate in across phases 3-5.

The storyboard IR (`render.storyboard.model.Element`) is already a group
tree of leaves, so it maps onto the node model directly: a 'group'
Element -> a group Node with children, every other kind -> a leaf Node
with the matching content variant. Each Element's property timelines
carry over as `Node.properties`; its `[t_start, t_end)` window and the
'hidden' timeline together compile visibility. This wrapping does not
feed the renderer yet -- it exists so the tree can be inspected and
tested, and so the design mapping has one home.
"""
from __future__ import annotations

from analysis.player.render.document.design_space import DesignSpace
from analysis.player.render.document.document import (CompiledDocument,
                                                      DEFAULT_STRATA)
from analysis.player.render.document.model import (Node, RectContent,
                                                   SpriteContent, TextContent,
                                                   Timeline)

# Storyboard z-slots split around the notefield: a negative/zero z draws
# in the background band, a positive z in the hud band (the existing
# StoryboardEffect z convention). Phase 3 replaces this with a real
# re-slottable layer timeline; phase 1 only needs a stable stratum name.
_NOTES_STRATUM = 'notes'


def _element_stratum(z: int) -> str:
    return 'background' if z <= 0 else 'hud'


def _leaf_content(el):
    """The document leaf-content variant for a storyboard Element kind.
    Returns None for a 'group' (a pure group Node has no content)."""
    match el.kind:
        case 'group':
            return None
        case 'sprite' | 'frames':
            return SpriteContent(asset=el.asset or 'white',
                                 additive=el.additive)
        case 'text' | 'bitmaptext':
            return TextContent(text=el.text, font_px=el.font_px, font=el.font)
        case 'rect':
            return RectContent()
        case 'ellipse':
            return RectContent(ellipse=True)
        case 'outline_rect':
            return RectContent(outline=True)
        case 'outline_ellipse':
            return RectContent(outline=True, ellipse=True)
        case _:
            return RectContent()


def _visibility_timeline(el):
    """The compiled-visibility timeline for a storyboard Element: its
    'hidden' channel is SM's hard visibility bit (0 shown, 1 hidden), so
    is-rendered = NOT hidden. The Element's own [t_start, t_end) window
    bounds existence; this bounds rendering within it. Storyboard
    elements carry a real 'hidden' timeline (driven ones via
    _SpanGatedTimeline), so this is never a last-value hold."""
    return Timeline(_HiddenAsVisibility(el.timelines['hidden']))


class _HiddenAsVisibility:
    """Adapt a 'hidden' curve (>=0.5 hidden) to a visibility curve
    (>=0.5 rendered) by inverting it. The document speaks visibility;
    the storyboard IR speaks hidden."""

    def __init__(self, hidden_curve):
        self._hidden = hidden_curve

    def sample(self, t: float) -> tuple:
        (hidden,) = self._hidden.sample(t)
        return (0.0 if hidden >= 0.5 else 1.0,)


def _element_node_id(prefix: str, index: int) -> str:
    return f'{prefix}{index}'


def _wrap_element(el, node_id, parent, layer, index_counter, source):
    """Build a Node from a storyboard Element, recursing into a group's
    children. `index_counter` is a one-element list used as a shared
    mutable counter so child ids stay globally unique across the tree.
    `source` accumulates node_id -> Element: the phase-3 renderer walks
    the node tree for structure/gating and resolves each node's draw
    payload (anchor/origin/flip/sheet/font/... the model has not promoted
    to first-class content yet) from this index. The mapping is emitted in
    the SAME pass that assigns node_ids, so it can never drift from the
    tree."""
    content = _leaf_content(el)
    child_nodes = []
    child_ids = []
    if content is None:
        for child in el.children:
            index_counter[0] += 1
            child_id = _element_node_id('sb', index_counter[0])
            child_nodes.extend(
                _wrap_element(child, child_id, node_id, layer,
                              index_counter, source))
            child_ids.append(child_id)

    node = Node(node_id=node_id, parent=parent, layer=layer,
                visibility=_visibility_timeline(el),
                t_start=el.t_start, t_end=el.t_end,
                properties=dict(el.timelines), children=tuple(child_ids),
                content=content)
    source[node_id] = el
    return [node, *child_nodes]


def _storyboard_nodes(storyboard):
    """Flatten a Storyboard's element tree into (node_table, root_ids,
    element_index). Each top-level Element becomes a root Node; a group's
    children become child Nodes. `element_index` maps node_id -> the
    source Element (see `_wrap_element`). All empty when the game has no
    storyboard."""
    nodes = {}
    roots = []
    source = {}
    counter = [0]
    for el in storyboard.elements:
        node_id = _element_node_id('sb', counter[0])
        layer = Timeline(_ConstCurve(_element_stratum(el.z)))
        for node in _wrap_element(el, node_id, None, layer, counter, source):
            nodes[node.node_id] = node
        roots.append(node_id)
        counter[0] += 1
    return nodes, tuple(roots), source


class _ConstCurve:
    """A layer curve that always samples the same stratum name -- a
    fixed-slot node (phase-1 storyboard z convention). Phase 3 replaces
    it with a real re-slottable draworder timeline."""

    def __init__(self, value):
        self._value = value

    def sample(self, t: float) -> tuple:
        return (self._value,)


def storyboard_document(storyboard, design):
    """Build the (CompiledDocument, element_index) pair for one
    `storyboard` under `design`. The document is the group/layer tree the
    renderer walks; `element_index` (node_id -> Element) is the draw-
    payload lookup the phase-3 storyboard renderer resolves each leaf
    through until the model promotes those fields (phases 4-5). An empty
    storyboard yields an empty tree and index."""
    nodes, roots, index = (_storyboard_nodes(storyboard) if storyboard
                           else ({}, (), {}))
    document = CompiledDocument(design=design, nodes=nodes, roots=roots,
                               strata=DEFAULT_STRATA)
    return document, index


def document_from_player(player) -> CompiledDocument:
    """Wrap `player`'s compiled per-game outputs into a CompiledDocument.
    Reads the adapter's design space and, when present, the storyboard
    subtree. Behaviour-neutral: the player still renders through its
    effects pipeline; this is the skeleton the outputs migrate into."""
    design = player._adapter.design_space()
    storyboard = player._adapter.storyboard(player.replay)
    document, _index = storyboard_document(storyboard, design)
    return document
