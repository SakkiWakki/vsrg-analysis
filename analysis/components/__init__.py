"""Unified plugin-component API.

Plugins that want to run on multiple surfaces (sidebar, overlay, ...)
register a single ``(ComponentManifest, draw)`` pair. Each surface
picks up the component through its backend:

    SidebarBackend  — draws via QPainter, routes clicks via hitboxes
    OverlayBackend  — emits PAL records, rendered by the active platform

The platform layer (``pal/``) isolates OS-specific overlay plumbing so
Windows and macOS hosts plug in new platforms without touching the
drawing code.
"""
from analysis.components.api import (
    Component,
    ComponentContext,
    ComponentDataSource,
    ComponentManifest,
    DataNotAvailable,
    DrawFn,
    SURFACE_OVERLAY,
    SURFACE_SIDEBAR,
)
from analysis.components.registry import (
    ComponentRegistry,
    bridge_into_overlay_registry,
    bridge_into_sidebar_registry,
    discover_from_bundles,
)

__all__ = [
    'Component',
    'ComponentContext',
    'ComponentDataSource',
    'ComponentManifest',
    'ComponentRegistry',
    'DataNotAvailable',
    'DrawFn',
    'SURFACE_OVERLAY',
    'SURFACE_SIDEBAR',
    'bridge_into_overlay_registry',
    'bridge_into_sidebar_registry',
    'discover_from_bundles',
]
