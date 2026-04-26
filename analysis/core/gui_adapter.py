"""Per-game GUI adapter base class + dynamic discovery.

Mirrors `analysis/core/game.py` but isolates surface used only by the GUI
(path discovery, root validation, profile listing, library enrichment, note
viz config). Keeps `GameAdapter` focused on replay/player semantics.

Each game package is expected to expose `GUI_ADAPTER` (an instance of a
`GuiAdapter` subclass) from its `gui_adapter` module.
"""
from __future__ import annotations

from analysis.core.game import _load_adapters


class GuiAdapter:
    name: str = ''

    # --- paths dialog ------------------------------------------------------
    label: str = ''
    hint: str = ''
    placeholder: str = ''
    error_hint: str = ''

    def find_dirs(self) -> dict:
        """Autodetected install dirs. Must include at minimum a 'root' key;
        games may include extras (e.g. 'save_dir', 'songs_dir', 'replays_dir')."""
        return {}

    def default_install_hint(self) -> str | None:
        """Pre-fill value for the paths dialog when no override is saved.
        Etterna uses the install root (parent of Save/); osu uses the root."""
        return self.find_dirs().get('root')

    def validate_root(self, path) -> bool:
        return False

    def list_profiles(self, root) -> list[str]:
        return []

    # --- persisted overrides ----------------------------------------------
    # Games that don't have a profile concept leave the profile hooks as no-ops.
    def get_root_override(self):
        return None

    def set_root_override(self, path):
        pass

    def get_profile_override(self):
        return None

    def set_profile_override(self, name):
        pass

    # --- library enrichment ------------------------------------------------
    def needs_enrichment(self, entry) -> bool:
        """True if this entry is missing metadata that `enrich_entry` would fill."""
        return False

    def enrich_entry(self, entry) -> bool:
        """Fill missing metadata in-place from the chart file. Returns True
        if the entry was mutated."""
        return False

    # --- player / viz ------------------------------------------------------
    def resolve_chart_context(self, replay, entry=None, progress=None):
        """Return (bpms, sm_offset, audio_path). Games without SM-style BPM
        maps return (None, 0.0, audio)."""
        return None, 0.0, None

    def note_viz_config(self, replay, judge=None, od=None) -> dict:
        """Return {'windows', 'unit_label', 'rows_per_ms', 'win'} for
        NoteVizTab's single-window renderer."""
        raise NotImplementedError


_REGISTRY: dict[str, GuiAdapter] = {}
_discovered = False


def discover_games() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True
    _REGISTRY.update(_load_adapters('gui_adapter', 'GUI_ADAPTER', GuiAdapter))


def get(name: str) -> GuiAdapter:
    if not _discovered:
        discover_games()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f'unknown game: {name!r}')


def all_games() -> dict[str, GuiAdapter]:
    if not _discovered:
        discover_games()
    return dict(_REGISTRY)
