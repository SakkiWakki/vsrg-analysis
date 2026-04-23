"""osu!mania GUI adapter — path discovery, root validation, profile listing,
enrichment from .osu chart files, and note-viz config."""
from __future__ import annotations

import sys
from pathlib import Path

from analysis.core.gui_adapter import GuiAdapter


_PLACEHOLDER_WIN = r'e.g. %LOCALAPPDATA%\osu!'
_PLACEHOLDER_NIX = 'e.g. ~/.local/share/osu-wine/osu!'


class OsuGuiAdapter(GuiAdapter):
    name = 'osu'
    label = 'osu! install folder'
    hint = ('Point at your osu! install folder — the one that contains '
            "`osu!.<username>.cfg`. The Songs folder is resolved from "
            "the config's BeatmapDirectory setting.")
    placeholder = _PLACEHOLDER_WIN if sys.platform == 'win32' else _PLACEHOLDER_NIX
    error_hint = 'no osu!.<user>.cfg in folder'

    def find_dirs(self):
        from analysis.games.osu.replay import find_osu_dirs
        return find_osu_dirs()

    def validate_root(self, path):
        if not path:
            return False
        p = Path(path)
        if not p.is_dir():
            return False
        try:
            for entry in p.iterdir():
                n = entry.name.lower()
                if n.startswith('osu!.') and n.endswith('.cfg'):
                    return True
        except OSError:
            return False
        return False

    def list_profiles(self, root):
        from analysis.games.osu.replay import list_osu_profiles
        return list_osu_profiles(root) if root else []

    def get_root_override(self):
        from analysis.gui.settings import get_osu_root_override
        return get_osu_root_override()

    def set_root_override(self, path):
        from analysis.gui.settings import set_osu_root_override
        set_osu_root_override(path)

    def get_profile_override(self):
        from analysis.gui.settings import get_osu_profile_override
        return get_osu_profile_override()

    def set_profile_override(self, name):
        from analysis.gui.settings import set_osu_profile_override
        set_osu_profile_override(name)

    def needs_enrichment(self, entry):
        return not entry.get('keycount')

    def enrich_entry(self, entry):
        """Backfill song/version/creator/keycount from the entry's .osu
        chart file. Called both in the library-scan auto-enrich path and
        on-demand from the player when a replay has resolved its chart."""
        chart_path = entry.get('chart_path')
        if not chart_path:
            return False
        needs = (not entry.get('keycount')
                 or (entry.get('song') or '').startswith('['))
        if not needs:
            return False
        try:
            from analysis.games.osu.replay import parse_osu_file
            chart = parse_osu_file(chart_path)
        except Exception:
            return False
        entry['song'] = f"{chart.get('artist','?')} - {chart.get('title','?')}"
        entry['steps'] = chart.get('version', '')
        entry['pack'] = chart.get('creator', entry.get('pack', ''))
        entry['keycount'] = chart.get('keycount')
        entry['chart_path'] = chart_path
        return True

    def resolve_chart_context(self, replay, entry=None, progress=None):
        from analysis.core import game as game_mod
        audio = game_mod.get('osu').resolve_audio(replay)
        return None, 0.0, audio

    def note_viz_config(self, replay, judge=None, od=None):
        from analysis.viz.note_visualizer import (osu_mania_windows,
                                                  effective_osu_od)
        base_od = od if od is not None else float(replay.get('od', 8.0))
        mods = int(replay.get('mods', 0))
        eff_od = effective_osu_od(base_od, mods)
        return {
            'windows': osu_mania_windows(od=eff_od),
            'unit_label': f'time (ms)  —  OD {eff_od:.1f}',
            'rows_per_ms': None,
            'win': 8000,
        }


GUI_ADAPTER = OsuGuiAdapter()
