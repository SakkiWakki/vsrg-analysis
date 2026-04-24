"""Discover community tosu overlay directories.

An overlay is any directory that contains an ``index.html`` file.
Scanned locations (in order):
  1. ``<repo>/plugins/overlays/``
  2. ``~/.config/vsrg-analysis/overlays/``
  3. any directory in ``$TOSU_OVERLAYS_DIRS`` (``:``-separated; dev hook)
    4. ``/tmp/tosu-counters`` (fallback when env var is unset)
"""
from __future__ import annotations

import os
from pathlib import Path


_BUILTIN_OVERLAYS = Path(__file__).parent.parent.parent / 'overlays'
_USER_OVERLAYS = Path.home() / '.config' / 'vsrg-analysis' / 'overlays'
_DEFAULT_DEV_OVERLAYS = Path('/tmp/tosu-counters')


def _extra_dirs() -> list[Path]:
    raw = os.environ.get('TOSU_OVERLAYS_DIRS', '')
    if not raw.strip():
        return ([_DEFAULT_DEV_OVERLAYS]
                if _DEFAULT_DEV_OVERLAYS.is_dir() else [])
    out: list[Path] = []
    for part in raw.split(os.pathsep):
        cleaned = part.strip()
        if not cleaned:
            continue
        out.append(Path(os.path.expandvars(cleaned)).expanduser())
    return out


def discovery_roots() -> list[Path]:
    """Return overlay roots in scan order."""
    return [_BUILTIN_OVERLAYS, _USER_OVERLAYS, *_extra_dirs()]


def find_overlays() -> list[tuple[str, Path]]:
    """Return (display_name, index_html_path) pairs for all available overlays,
    sorted by display name. Earlier roots win on name collision."""
    found: dict[str, Path] = {}
    roots = discovery_roots()
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            index = entry / 'index.html'
            if index.exists():
                found.setdefault(entry.name, index)
    return sorted(found.items())
