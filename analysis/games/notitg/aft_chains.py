"""Compile-time AFT render-target chain graph.

A NotITG ActorFrameTexture (AFT) is a named render target: it captures
the scene drawn before it, and a sprite that draws `aft:GetTexture()`
blits that capture. gat 1 used a single AFT capturing the whole screen -
field_instances already models that: an 'aft' instance blits the
`screen` composite, keyed by name for preserve-texture freezes.

gat 2 (more_afts.xml / Mdrqnxtagon.xml) CHAINS AFTs: an upstream AFT
captures a proxy, a sprite draws that upstream capture, and a downstream
AFT captures ONLY that sprite - not the whole screen - so its own
consumer blits the isolated upstream content (a post-processed field
copy), not the finished frame. The chart's own comment (Mdrqnxtagon.xml
line 208-213) names the stages:

    gf2_xtag_aft_p<n>          main AFT capturing proxy<n> / whole screen
    gf2_xtag_sprite_p<n>       sprite displaying aft_p<n>
    gf2_xtag_aft_lumikey_p<n>  AFT capturing sprite_p<n>   (the 2nd stage)

This module walks the AFT nodes + their blit sprites in engine draw
order and decides, for each AFT node, WHAT it captures:

- 'screen' (None): the whole composite drawn before it (the gat 1 /
  stage-1 case; also any node whose capture mixes several drawables or a
  live player/proxy field - not a clean single-source isolation).
- an upstream node NAME: the node captures exactly one blit sprite that
  itself draws an upstream AFT, with no other field content in between -
  a chain node isolating that upstream capture. `stage_of` returns the
  captured sprite's rec_id (the render side re-draws that sprite, with
  its own transform, into the node's slot).

The result annotates each 'aft' field instance with `captures` (an
upstream node name or None = 'screen'). None is the gat 1 path exactly,
so a single-AFT chart stays byte-identical.

# Depth and cycles

Chains resolve TRANSITIVELY up to `MAX_CHAIN_DEPTH` links (a defensive
cap - the composed-capture evaluator recurses the chain, and a
degenerate topology must not turn that into unbounded work). A node
whose chain exceeds the cap DEMOTES to a whole-screen capture and is
listed in `depth_capped` for logging - never silently.

A node on a capture CYCLE (a sprite drawing the node's own texture is
captured back into it - gat 2's cyriak recursion - or a longer loop) is
FEEDBACK: previous-frame content re-entering this frame's capture. Pure
same-frame composition cannot express it; those nodes are listed in
`feedback`, demoted to whole-screen until the render side's persistent
ping-pong targets consume the classification.

`unresolved_depth` = depth_capped + feedback: everything the pure
composed-capture path cannot resolve.
"""
from __future__ import annotations

# Maximum chain links resolved to isolated captures. Real charts nest
# 2-3 (gat 2's deepest consumed chain is 3); the cap only guards
# degenerate topologies from turning the evaluator's recursion into
# unbounded per-frame work.
MAX_CHAIN_DEPTH = 6


class AftChainGraph:
    """The resolved capture source of every AFT node in draw order.

    `capture_of(name)` returns the upstream node NAME the AFT isolates,
    or None when the node captures the whole `screen` (stage-1, mixed
    content, depth-capped, or feedback). `stage_of(name)` returns the
    rec_id of the blit sprite captured into an isolating node (None for
    whole-screen nodes). `depth_of(name)` is the node's resolved chain
    depth (0 = whole-screen)."""

    def __init__(self, capture_by_node, stage_by_node, depths,
                 feedback, depth_capped):
        self._capture = dict(capture_by_node)
        self._stage = dict(stage_by_node)
        self._depths = dict(depths)
        self.feedback = frozenset(feedback)
        self.depth_capped = tuple(depth_capped)

    def capture_of(self, node_name):
        """The upstream node name this AFT isolates, or None for a
        whole-screen capture."""
        return self._capture.get(node_name)

    def stage_of(self, node_name):
        """The rec_id of the blit sprite an isolating node captures."""
        return self._stage.get(node_name)

    def depth_of(self, node_name) -> int:
        """Resolved chain depth: 0 = whole-screen, 1 = isolates a
        whole-screen node, ..."""
        return self._depths.get(node_name, 0)

    @property
    def unresolved_depth(self):
        """Nodes the pure composed-capture path cannot resolve: chains
        past MAX_CHAIN_DEPTH plus feedback cycles."""
        return tuple(self.depth_capped) + tuple(sorted(self.feedback))


def build_chain_graph(aft_nodes, blit_sources, draw_order,
                      screen_content_ids) -> AftChainGraph:
    """Resolve the AFT capture graph from draw-order structure.

    - `aft_nodes`: {rec_id: node_name} for every AFT render target.
    - `blit_sources`: {rec_id: upstream_node_name} for every sprite that
      draws an `aft:` texture (its `aft_source`).
    - `draw_order`: rec_ids in engine preorder (the sequence content is
      drawn / captured in).
    - `screen_content_ids`: rec_ids of live player/proxy field draws
      (whole-scene content that makes a following AFT a screen capture).

    An AFT node isolates an upstream node when the run of drawables
    captured into it (since the previous AFT node or scene-content draw)
    is exactly one blit sprite drawing another AFT, and nothing else.
    Otherwise it captures the screen (None)."""
    node_names = frozenset(aft_nodes.values())
    capture_by_node = {}
    stage_by_node = {}
    pending_blits: list = []
    saw_screen_content = False
    for rec_id in draw_order:
        if rec_id in screen_content_ids:
            saw_screen_content = True
            pending_blits = []
            continue
        node_name = aft_nodes.get(rec_id)
        if node_name is not None:
            upstream = _isolated_upstream(
                pending_blits, saw_screen_content, blit_sources, node_names)
            capture_by_node[node_name] = upstream
            if upstream is not None:
                stage_by_node[node_name] = pending_blits[0]
            pending_blits = []
            saw_screen_content = False
            continue
        if rec_id in blit_sources:
            pending_blits.append(rec_id)

    depths, feedback, depth_capped = _resolve(capture_by_node)
    for name in feedback | set(depth_capped):
        capture_by_node[name] = None
        stage_by_node.pop(name, None)
    return AftChainGraph(capture_by_node, stage_by_node, depths,
                         feedback, depth_capped)


def _isolated_upstream(pending_blits, saw_screen_content, blit_sources,
                       node_names):
    """The upstream node an AFT isolates when its capture run is exactly
    one blit of another AFT, else None (a whole-screen capture)."""
    if saw_screen_content or len(pending_blits) != 1:
        return None
    upstream = blit_sources[pending_blits[0]]
    return upstream if upstream in node_names else None


def _resolve(capture_by_node):
    """Transitive chain depths from the one-level capture links, with
    cycle and depth-cap classification.

    Walks each node's upstream chain: a repeat visit marks every node on
    the loop as feedback; a chain longer than MAX_CHAIN_DEPTH marks the
    too-deep tail nodes depth-capped. Both classes demote to
    whole-screen (depth 0) - the callers' pre-chain behavior - so the
    composed evaluator only ever sees pure, bounded chains."""
    depths = {}
    feedback = set()
    depth_capped = []

    def depth(name, trail):
        if name in depths:
            return depths[name]
        if name in trail:
            feedback.update(trail[trail.index(name):])
            return 0
        upstream = capture_by_node.get(name)
        if upstream is None:
            depths[name] = 0
            return 0
        d = depth(upstream, trail + [name])
        if name in feedback:
            return 0
        depths[name] = d + 1
        return depths[name]

    for name in capture_by_node:
        depth(name, [])
    for name in sorted(depths):
        if depths[name] > MAX_CHAIN_DEPTH:
            depth_capped.append(name)
    for name in feedback | set(depth_capped):
        depths[name] = 0
    return depths, feedback, depth_capped


def _fed_by_chain(upstream_name, blit_sources) -> bool:
    """Whether the isolated upstream AFT is itself the target of a blit
    that another chain stage captures. Retained as a structural helper
    for diagnostics; the transitive `_resolve` supersedes it as the
    depth authority."""
    return any(src == upstream_name for src in blit_sources.values())
