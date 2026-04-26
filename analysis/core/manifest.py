"""Per-game GUI manifest. Declarative replacement for the old `GuiAdapter`.

A `GameManifest` is the only thing the GUI needs to know about a game ;
each game package exposes one as `MANIFEST` from its `manifest.py`. The
paths dialog iterates `path_fields` and renders one row per declared
field, so adding a new game is purely additive: drop a new manifest in
`analysis/games/<game>/manifest.py` and the GUI picks it up.

Path-override storage is the "shopkeeper" in `analysis.core.path_overrides` ;
the GUI installs a Qt-backed backend at startup, headless code leaves it
unset and falls back to autodetect. Manifests never touch QSettings
directly ; they declare a `settings_key` and let the shopkeeper persist.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from analysis.core.game import _load_adapters


# A `placeholder` may be a literal string, a per-platform dict keyed by
# `sys.platform`, or a zero-arg callable that returns a string. Resolved
# lazily so platform-specific text is computed once at render time.
Placeholder = str | dict | Callable[[], str]


@dataclass(frozen=True)
class PathField:
    """One configurable path the GUI should expose for a game.

    A field always renders as a labeled text-edit + Browse button. When
    `list_choices` is set, a combo appears below it (used today for the
    osu! profile picker). All callbacks are optional; an absent
    `validate` means "always accept", an absent `autodetect` means "no
    suggested default", an absent `list_choices` means "no combo".

    `settings_key` is the QSettings key the shopkeeper uses to persist
    the override. Keep it stable across versions so saved paths survive.
    """
    key: str                        # field id within the game (e.g. 'root')
    label: str                      # section header text in the dialog
    hint: str                       # multiline help text under the header
    placeholder: Placeholder        # str | {sys.platform: str} | () -> str
    settings_key: str               # QSettings key (e.g. 'paths/etterna_root')
    error_hint: str = ''            # red text shown when validate() fails
    autodetect: Callable[[], str | None] | None = None
    validate: Callable[[str], bool] | None = None
    list_choices: Callable[[str], list[str]] | None = None  # (root) -> options


def resolve_placeholder(p: Placeholder) -> str:
    """Coerce a `Placeholder` to its display string for the current platform."""
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        return p.get(sys.platform, p.get('default', ''))
    return p()


@dataclass(frozen=True)
class GameManifest:
    """Everything the GUI needs to render and integrate one game.

    `path_fields` drives the paths dialog ; `find_dirs` is the autodetect
    used by both the dialog and core path-resolvers ; the rest are the
    library-tab and viz hooks the old `GuiAdapter` exposed."""
    name: str                                          # 'osu' / 'etterna' / ...
    path_fields: list[PathField]                       # rendered top-to-bottom
    find_dirs: Callable[[], dict]                      # () -> autodetect dict
    note_viz_config: Callable[..., dict]               # (replay, judge=, od=)
    needs_enrichment: Callable[[dict], bool] = field(
        default=lambda _entry: False)
    enrich_entry: Callable[[dict], bool] = field(
        default=lambda _entry: False)
    resolve_chart_context: Callable[..., tuple] = field(
        default=lambda _replay, entry=None, progress=None: (None, 0.0, None))


_REGISTRY: dict[str, GameManifest] = {}
_discovered = False


def discover_manifests() -> None:
    """Scan `analysis.games.<pkg>.manifest` for each game package and
    register its `MANIFEST` attribute. Idempotent."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    _REGISTRY.update(_load_adapters('manifest', 'MANIFEST', GameManifest))


def get(name: str) -> GameManifest:
    if not _discovered:
        discover_manifests()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f'unknown game: {name!r}')


def all_manifests() -> dict[str, GameManifest]:
    if not _discovered:
        discover_manifests()
    return dict(_REGISTRY)


def reset_for_tests() -> None:
    """Wipe the discovery cache so tests can swap manifests in/out."""
    global _discovered
    _REGISTRY.clear()
    _discovered = False
