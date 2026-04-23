"""Builtin sidebar plugin.

Defines the layout types and region constants for the sidebar surface.
Components that target the sidebar include a ``SidebarFields`` instance
in their manifest's ``plugin_fields`` dict under the key ``'sidebar'``.

    from plugins.builtin.sidepanel import SidebarFields, REGION_PANEL
    from analysis.components.api import REGION_FREE

    MANIFEST = Manifest(
        key='my:component',
        name='My Component',
        supported_surfaces={SURFACE_GUI},
        plugin_fields={
            'sidebar': SidebarFields(priority=500, draggable=True),
        },
    )

REGION_FREE (floating on the surface, not docked in any panel) is a
core concept defined in analysis.components.api. REGION_PANEL is the
sidebar's own panel column and is sidebar-specific.
"""
from __future__ import annotations

from dataclasses import dataclass

from analysis.components.api import REGION_FREE as REGION_FREE  # noqa: PLC0414


# The sidepanel's own panel region. REGION_FREE is imported from core.
REGION_PANEL = 'sidepanel'

__all__ = ['REGION_FREE', 'REGION_PANEL', 'SidebarFields']


@dataclass(frozen=True)
class SidebarFields:
    """Sidebar-specific layout hints for a component manifest.

    All fields are optional; the sidebar backend substitutes these
    defaults for any component that omits ``plugin_fields['sidebar']``."""
    priority: int = 1000
    pin_bottom: bool = False
    draggable: bool = False
    region: str = REGION_PANEL
    default_free_xy: tuple = (0.5, 0.5)
    default_size: tuple = (210, 120)
