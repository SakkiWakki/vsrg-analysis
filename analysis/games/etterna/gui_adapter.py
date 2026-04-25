"""Etterna GUI adapter ; path discovery, root validation, enrichment,
chart-context resolution, and note-viz config. Delegates to the game
package's replay/chart modules; the GUI never imports those directly."""
from __future__ import annotations

from pathlib import Path

from analysis.core.gui_adapter import GuiAdapter


class EtternaGuiAdapter(GuiAdapter):
    name = 'etterna'
    label = 'Etterna install folder'
    hint = ('Point at your Etterna install folder ; the one that contains '
            '`Save/` and `Songs/`. Additional song folders listed in '
            '`Preferences.ini` are picked up automatically.')
    placeholder = 'e.g. ~/.etterna or ~/etterna'
    error_hint = 'folder missing Save/ (or LocalProfiles/Etterna.xml)'

    def find_dirs(self):
        from analysis.games.etterna.replay import find_etterna_dirs
        return find_etterna_dirs()

    def default_install_hint(self):
        save = self.find_dirs().get('save_dir')
        # The override field stores the install root (parent of Save/) so
        # AdditionalSongFolders resolution from Preferences.ini keeps working.
        return str(Path(save).parent) if save else None

    def validate_root(self, path):
        if not path:
            return False
        p = Path(path)
        if not p.is_dir():
            return False
        save = p / 'Save'
        if save.is_dir() and ((save / 'LocalProfiles').is_dir()
                              or (save / 'Etterna.xml').is_file()):
            return True
        # Accept a bare Save/ dir too (back-compat for pre-root overrides).
        return (p / 'LocalProfiles').is_dir() or (p / 'Etterna.xml').is_file()

    def get_root_override(self):
        from analysis.gui.settings import get_etterna_root_override
        return get_etterna_root_override()

    def set_root_override(self, path):
        from analysis.gui.settings import set_etterna_root_override
        set_etterna_root_override(path)

    def needs_enrichment(self, entry):
        # Etterna entries come from Etterna.xml with full metadata already.
        return False

    def enrich_entry(self, entry):
        return False

    def resolve_chart_context(self, replay, entry=None, progress=None):
        from analysis.core import game as game_mod
        return game_mod.get('etterna').resolve_all(
            replay, entry=entry, progress=progress)

    def note_viz_config(self, replay, judge=None, od=None):
        from analysis.viz.note_visualizer import etterna_windows
        j = judge or 'J4'
        return {
            'windows': etterna_windows(j),
            'unit_label': f'noterow  ;  {j}',
            'rows_per_ms': 0.37,
            'win': 2400,
        }


GUI_ADAPTER = EtternaGuiAdapter()
