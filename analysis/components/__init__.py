"""Unified plugin-component API.

Plugins that want to run on multiple surfaces (gui, overlay, ...)
register a single ``(ComponentManifest, draw)`` pair. Each surface
picks up the component through its backend:

    GuiBackend      — draws via QPainter, routes clicks via hitboxes
    OverlayBackend  — emits PAL records, rendered by the active platform

The platform layer (``pal/``) isolates OS-specific overlay plumbing so
Windows and macOS hosts plug in new platforms without touching the
drawing code.
"""
from analysis.components.api import (
    Component,
    ComponentContext,
    ComponentDataAnalysis,
    ComponentGameState,
    ComponentManifest,
    ComponentReplayState,
    DataNotAvailable,
    DrawFn,
    GameMemoryState,
    REGION_FREE,
    SURFACE_GUI,
    SURFACE_OVERLAY,
)
from analysis.components.registry import (
    ComponentRegistry,
    bridge_into_gui_registry,
    bridge_into_overlay_registry,
    discover_from_bundles,
)

__all__ = [
    'Component',
    'ComponentContext',
    'ComponentGameState',
    'ComponentManifest',
    'ComponentRegistry',
    'ComponentDataAnalysis',
    'ComponentReplayState',
    'DataNotAvailable',
    'DrawFn',
    'GameMemoryState',
    'REGION_FREE',
    'SURFACE_GUI',
    'SURFACE_OVERLAY',
    'bridge_into_gui_registry',
    'bridge_into_overlay_registry',
    'discover_from_bundles',
]
