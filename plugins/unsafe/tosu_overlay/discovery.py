"""Discover community tosu overlay directories.

An overlay is any directory that contains an ``index.html`` file.
Scanned locations (in order):
  1. ``<repo>/plugins/overlays/``
  2. ``~/.config/vsrg-analysis/overlays/``
"""
from __future__ import annotations

from pathlib import Path


_BUILTIN_OVERLAYS = Path(__file__).parent.parent.parent / 'overlays'
_USER_OVERLAYS = Path.home() / '.config' / 'vsrg-analysis' / 'overlays'


def find_overlays() -> list[tuple[str, Path]]:
    """Return (display_name, index_html_path) pairs for all available overlays,
    sorted by display name."""
    found: dict[str, Path] = {}
    for root in (_BUILTIN_OVERLAYS, _USER_OVERLAYS):
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            index = entry / 'index.html'
            if index.exists():
                found[entry.name] = index
    return sorted(found.items())
