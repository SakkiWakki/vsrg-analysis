"""Unified plugin-component API.

Plugins that want to run on multiple surfaces (gui, overlay, ...)
register a single ``(Manifest, draw)`` pair. Each surface
picks up the component through its backend:

    GuiBackend      — draws via QPainter, routes clicks via hitboxes
    OverlayBackend  — emits PAL records, rendered by the active platform

The platform layer (``pal/``) isolates OS-specific overlay plumbing so
Windows and macOS hosts plug in new platforms without touching the
drawing code.
"""
from analysis.components.api import (
    Component,
    Config,
    Context,
    DataAnalysis,
    GameState,
    HudFlags,
    LayerDeclaration,
    LayerPlacement,
    LayerState,
    Manifest,
    ReplayState,
    DataNotAvailable,
    DrawFn,
    GameMemoryState,
    LAYER_AFTER,
    LAYER_BEFORE,
    LAYER_GROUP,
    LAYER_INSIDE,
    LAYER_LEAF,
    REGION_FREE,
    SURFACE_GUI,
    SURFACE_OVERLAY,
    SURFACE_VIZ,
)
from analysis.components.registry import (
    ComponentRegistry,
    bridge_into_gui_registry,
    bridge_into_overlay_registry,
    discover_from_bundles,
)
from analysis.components.viz_backend import (
    VizFields,
    VIZ_ATTACH_LIBRARY_TAB,
    VIZ_ATTACH_WINDOW,
    VIZ_CATEGORY_CHART,
    VIZ_CATEGORY_WIDGET,
)

__all__ = [
    'Component',
    'Context',
    'GameState',
    'Manifest',
    'ComponentRegistry',
    'Config',
    'DataAnalysis',
    'HudFlags',
    'LayerDeclaration',
    'LayerPlacement',
    'LayerState',
    'ReplayState',
    'DataNotAvailable',
    'DrawFn',
    'GameMemoryState',
    'LAYER_AFTER',
    'LAYER_BEFORE',
    'LAYER_GROUP',
    'LAYER_INSIDE',
    'LAYER_LEAF',
    'REGION_FREE',
    'SURFACE_GUI',
    'SURFACE_OVERLAY',
    'SURFACE_VIZ',
    'VizFields',
    'VIZ_ATTACH_LIBRARY_TAB',
    'VIZ_ATTACH_WINDOW',
    'VIZ_CATEGORY_CHART',
    'VIZ_CATEGORY_WIDGET',
    'bridge_into_gui_registry',
    'bridge_into_overlay_registry',
    'discover_from_bundles',
]
