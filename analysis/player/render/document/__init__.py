"""The compiled-document model: one native modchart object per (chart,
replay) that every game adapter emits and the player consumes through
one loading path.

- design_space.py  DesignSpace: the design-space header (authoring
                   resolution + fit policy + clip), also the
                   `GameAdapter.design_space()` return type.
- model.py         Node + leaf content variants + Timeline + CaptureRange:
                   the group/layer tree (see the module docstring for the
                   refined orthogonal group-vs-layer semantics).
- document.py      CompiledDocument + ClockTable + StreamTable: the
                   header + tables + tree container.
- builder.py       document_from_player: wraps a player's per-game
                   outputs into a CompiledDocument (partial, zero
                   behaviour change).

See DESIGN_compiled_document.md for the six-axis charter and migration
phases.
"""
from analysis.player.render.document.design_space import (
    DesignSpace, FIT_HEIGHT, FIT_MIN, FIT_STRETCH)
from analysis.player.render.document.document import (
    ClockTable, CompiledDocument, DEFAULT_STRATA, StreamTable)
from analysis.player.render.document.model import (
    CaptureContent, CaptureRange, Node, NotefieldContent, RectContent,
    SpriteContent, TextContent, Timeline)

__all__ = [
    'DesignSpace', 'FIT_MIN', 'FIT_HEIGHT', 'FIT_STRETCH',
    'CompiledDocument', 'ClockTable', 'StreamTable', 'DEFAULT_STRATA',
    'Node', 'Timeline', 'CaptureRange', 'CaptureContent',
    'SpriteContent', 'TextContent', 'RectContent', 'NotefieldContent',
]
