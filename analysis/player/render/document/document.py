"""`CompiledDocument`: one object per (chart, replay) that every game
adapter emits and the player consumes through one loading path.

This is the consolidation target -- the model the six-axis charter
describes (DESIGN_compiled_document.md). Phase 1 builds the SKELETON:
the header (design space), the tables (clocks, streams), the node tree,
and the layer strata, populated partially. Adapters keep their current
outputs; `document_from_player` WRAPS them (storyboard tree + design
space now, the rest as it lands) with zero behaviour change. The player
still renders through the existing effects pipeline; the document is the
structure those outputs will migrate INTO across phases 3-5.

The document is DATA: frozen, lightest containers, serializable. Nothing
here samples or draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from analysis.player.render.document.design_space import DesignSpace

# The default compositing strata, low->high. A node's layer slot samples
# to a stratum name; capture ranges are declared as index pairs into
# this order. The fixed background->field->notes->hud pipeline is the
# default; phase 3 lets a node re-slot over time and phase 4 lets a
# capture name a sub-range.
DEFAULT_STRATA = ('background', 'field', 'notes', 'hud')


@dataclass(frozen=True)
class ClockTable:
    """The named integral clocks a document's timelines key against.
    Each entry maps a `clock_key` ('song'/'beat'/'scroll'/per-group) to
    the SV engine key that evaluates it. Phase 1 is a placeholder: the
    only clock is 'song' (wall-clock seconds = identity), and timelines
    default to it, so the existing song-time curves drop in unchanged.
    Phase 5 populates the beat/scroll/per-group integrals from
    `analysis.player.sv/`."""
    keys: dict = field(default_factory=lambda: {'song': 'song'})


@dataclass(frozen=True)
class StreamTable:
    """Named value/text data channels properties and text content bind
    to: chart-written custom-buffer entries and replay-derived gameplay
    tallies (combo, misses, judgment counts). Phase 1 is a placeholder
    (empty); phase 5 precomputes the gameplay tallies and records
    custom-buffer writes at compile time."""
    values: dict = field(default_factory=dict)  # name -> Timeline
    texts: dict = field(default_factory=dict)    # name -> Timeline


@dataclass(frozen=True)
class CompiledDocument:
    """The whole compiled chart: design-space header, clock table, stream
    table, node tree, layer strata.

    `nodes` is the flat node table keyed by node_id (the tree lives in
    each node's `parent`/`children`); `roots` are the ids with no parent.
    `strata` is the ordered compositing bands a node's layer slot and a
    capture range index into. Population is PARTIAL in phase 1 -- a
    document may carry only its design space + a storyboard subtree + the
    placeholder tables, which is enough for the skeleton to exist and one
    real consumer (the storyboard design mapping) to read through it."""
    design: DesignSpace
    nodes: dict = field(default_factory=dict)   # node_id -> Node
    roots: tuple = ()                            # node_ids with no parent
    strata: tuple = DEFAULT_STRATA
    clocks: ClockTable = field(default_factory=ClockTable)
    streams: StreamTable = field(default_factory=StreamTable)

    def __post_init__(self):
        for node_id, node in self.nodes.items():
            if node.node_id != node_id:
                raise ValueError(
                    f'node table key {node_id!r} disagrees with '
                    f'node_id {node.node_id!r}')
        for node_id in self.roots:
            if node_id not in self.nodes:
                raise ValueError(f'root {node_id!r} not in node table')

    def stratum_index(self, name: str) -> int:
        return self.strata.index(name)
