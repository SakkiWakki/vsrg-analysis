"""Render a storyboard THROUGH the compiled-document node tree.

`DocumentStoryboardRenderer` is the phase-3 consumer of the group/layer
tree: instead of walking the storyboard's parallel Element list, it walks
`CompiledDocument.nodes` -- group nodes compose their transform onto their
children (the node `parent`/`children` edges), the layer stratum bands the
draws, and each node's REQUIRED `visibility` timeline gates the draw with
an explicit is-rendered answer for all t (never a last-value hold). This
is the render side of the consolidation: the document is the single
source the player will draw from once every channel migrates in.

Behaviour neutrality (the acceptance bar). The document does not yet
promote every draw field to first-class node content -- anchor, origin,
flip, sheet grids, sprite-sequence frames, bitmap fonts, and absolute
size all still live on the source `Element`. So this renderer resolves
each node's DRAW PAYLOAD through the builder's node_id -> Element index
and reuses `StoryboardEffect`'s transform + pixel math unchanged (via the
walker seam `_paint_element` exposes): the pixels are produced by the
exact same code the Element-walk uses, which makes the two paths trivially
equivalent (see tests/test_document_equivalence.py). What differs is only
the WALK -- root banding and the group descent, plus the visibility gate,
come from the node tree, not the Element arrays. As phases 4-5 promote
those fields onto node content and timelines, the Element-payload lookup
shrinks and this becomes a pure node renderer.

Layer banding. A node's layer slot samples a stratum NAME (the coarse
model view: 'background' below the field, 'hud' above); the exact z-slot
integer the effects compositor bands by is a render detail resolved
through the Element index, kept identical to the Element-walk ordering
(z, z_index, t_start).

Gating. The WALK selects a node whenever its window `[t_start, t_end)` is
open -- exactly the Element-walk's window test, so the set of `(z, draw)`
slots the compositor bands is structurally identical (a windowed-but-
hidden root still owns an empty slot, as the Element walk emits). The
REQUIRED visibility timeline then decides is-RENDERED: `_rendered` reads
it (>=0.5). For a storyboard node visibility is the compiled inverse of
SM's hidden bit, which `StoryboardEffect._paint_element` also consults on
the same curve, so gating on visibility here and suppressing on hidden
there give the same pixels. Keeping BOTH gates window-based preserves the
frame's slot structure while the visibility timeline stays the first-class
is-rendered answer the model requires; when phases 4-5 lift the hidden
read out of `_paint_element`, visibility becomes the sole gate.
"""
from __future__ import annotations

from analysis.player.render.effects.base import EffectFrame

_VISIBLE = 0.5


class _NodeWalk:
    """The group-descent source backed by the compiled-document node
    tree. `_paint_children` calls `children(group_el, node_id, t)`; this
    returns each child as `(child_element, child_node_id)`, gated by the
    child node's window and REQUIRED visibility timeline. The paired
    node_id threads back in so the next descent level reads its children
    from the document too -- the whole subtree walk is node-driven while
    the transform/paint math stays in StoryboardEffect."""

    def __init__(self, document, element_index):
        self._nodes = document.nodes
        self._index = element_index

    def children(self, el, node_id, t):
        node = self._nodes[node_id]
        out = []
        for child_id in node.children:
            child = self._nodes[child_id]
            if child.t_start <= t < child.t_end and _rendered(child, t):
                out.append((self._index[child_id], child_id))
        return tuple(out)


def _rendered(node, t) -> bool:
    return node.visibility.sample(t)[0] >= _VISIBLE


class DocumentStoryboardRenderer:
    """Draw a storyboard by walking a `CompiledDocument`'s node tree.

    Constructed from `(document, element_index, storyboard, painter)`: the
    document supplies the node tree (roots, group edges, layer slots,
    visibility); `element_index` (node_id -> Element) and `storyboard`
    supply the draw payload and design mapping the phase-1 model has not
    promoted yet; `painter` is a `StoryboardEffect` whose transform + pixel
    helpers and asset caches this renderer reuses VERBATIM, so the two
    paths cannot diverge -- only the walk differs. Passing the painter in
    (rather than building one) keeps this a plain composition of the
    Element renderer and avoids a construction cycle when StoryboardEffect
    routes through here. The same `EffectFrame` shape comes out, so this is
    a drop-in for `StoryboardEffect`."""

    def __init__(self, document, element_index, storyboard, painter):
        self._doc = document
        self._index = element_index
        self._sb = storyboard
        self._walk = _NodeWalk(document, element_index)
        self._paint = painter
        self._z_slots = self._band_roots(tuple(document.roots))

    def __bool__(self):
        return bool(self._doc.roots)

    def _band_roots(self, roots):
        """Group root node_ids into ordered z-slots, matching the Element-
        walk's `(z, z_index, t_start)` sort exactly. z / z_index come from
        the source Element -- the exact draworder the model has not
        promoted past the coarse stratum name."""
        ordered = sorted(
            roots,
            key=lambda r: (self._index[r].z, self._index[r].z_index,
                           self._index[r].t_start))
        zs = sorted({self._index[r].z for r in roots})
        return tuple(
            (z, tuple(r for r in ordered if self._index[r].z == z))
            for z in zs)

    def at(self, ctx) -> EffectFrame | None:
        # Band roots by WINDOW only, matching the Element walk's `active`
        # test, so the set of z-slots (and their empty-when-all-hidden
        # members) is structurally identical. Per-node is-rendered is
        # decided inside the slot draw, on the visibility timeline.
        t = float(ctx.t_now)
        nodes = self._doc.nodes
        draws = []
        for z, root_ids in self._z_slots:
            live = tuple(r for r in root_ids
                         if nodes[r].t_start <= t < nodes[r].t_end)
            if live:
                draws.append((z, self._slot_draw(live, t)))
        if not draws:
            return None
        return EffectFrame(draws=tuple(draws))

    def _slot_draw(self, root_ids, t):
        from PySide6.QtCore import Qt

        from analysis.player.render.storyboard.render import (
            _design_box_rect, _design_transform)

        sb = self._sb
        paint = self._paint
        walk = self._walk
        index = self._index

        def draw(ctx, painter):
            kx, ky, ox, oy = _design_transform(sb, ctx.chart_rect)
            painter.save()
            if sb.clip_design_box:
                painter.setClipRect(_design_box_rect(sb, kx, ky, ox, oy),
                                    Qt.ClipOperation.IntersectClip)
            painter.translate(ox, oy)
            painter.scale(kx, ky)
            for root_id in root_ids:
                paint._paint_element(
                    painter, index[root_id], t, 1.0, 0.0, 0.0,
                    sb.design_w, sb.design_h, walker=walk, node=root_id)
            painter.restore()
        return draw
