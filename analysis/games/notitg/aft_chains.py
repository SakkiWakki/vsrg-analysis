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
  a 2-stage chain node isolating that upstream capture.

The result annotates each 'aft' field instance with `captures` (an
upstream node name or None = 'screen'). None is the gat 1 path exactly,
so a single-AFT chart stays byte-identical.

# Scope

Only the 2-STAGE case is resolved to an isolated named capture (it
covers gat 2's actual consumed chains). A deeper (3+) ping-pong or a
chain feeding a Polygon/crumple.vert vertex target needs the GL executor
(arbitrary N render targets + vertex-stage sampling), out of this
compile graph's reach; those nodes still emit an isolated capture but
their name is recorded in `unresolved_depth` for logging.
"""
from __future__ import annotations


class AftChainGraph:
    """The resolved capture source of every AFT node in draw order.

    `capture_of(name)` returns the upstream node NAME a stage-2 AFT
    isolates, or None when the node captures the whole `screen` (stage-1,
    mixed content). `unresolved_depth` lists chain nodes fed by a further
    upstream chain stage (a 3+ deep ping-pong the compile graph isolates
    only one level of), for logging."""

    def __init__(self, capture_by_node, unresolved_depth):
        self._capture = dict(capture_by_node)
        self.unresolved_depth = tuple(unresolved_depth)

    def capture_of(self, node_name):
        """The upstream node name this AFT isolates, or None for a
        whole-screen capture."""
        return self._capture.get(node_name)


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
    unresolved_depth = []
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
            if upstream is not None and _fed_by_chain(upstream, blit_sources):
                unresolved_depth.append(node_name)
            pending_blits = []
            saw_screen_content = False
            continue
        if rec_id in blit_sources:
            pending_blits.append(rec_id)
    return AftChainGraph(capture_by_node, unresolved_depth)


def _isolated_upstream(pending_blits, saw_screen_content, blit_sources,
                       node_names):
    """The upstream node an AFT isolates when its capture run is exactly
    one blit of another AFT, else None (a whole-screen capture)."""
    if saw_screen_content or len(pending_blits) != 1:
        return None
    upstream = blit_sources[pending_blits[0]]
    return upstream if upstream in node_names else None


def _fed_by_chain(upstream_name, blit_sources) -> bool:
    """Whether the isolated upstream AFT is itself the target of a blit
    that another chain stage captures - a 3+ deep ping-pong. Best-effort
    structural signal for logging, never a correctness gate: the one
    isolated level is still emitted."""
    return any(src == upstream_name for src in blit_sources.values())
