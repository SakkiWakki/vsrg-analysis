"""Quaver GUI adapter ; install-path discovery, root validation, and
chart-context resolution. Mirrors the osu! adapter's shape so the paths
dialog and library tab can drive Quaver without any game-specific
branching."""
from __future__ import annotations

import sys

from analysis.core.gui_adapter import GuiAdapter
from analysis.games.quaver.paths import find_quaver_dirs, validate_quaver_root


_PLACEHOLDER_WIN = r'e.g. C:\Program Files (x86)\Steam\steamapps\common\Quaver'
_PLACEHOLDER_NIX = 'e.g. ~/.steam/steam/steamapps/common/Quaver'


class QuaverGuiAdapter(GuiAdapter):
    name = 'quaver'
    label = 'Quaver install folder'
    hint = ('Point at your Quaver install folder ; the one that contains '
            '`Songs/` (and usually `Data/Replays/`). Steam libraries on '
            'secondary drives are picked up from `libraryfolders.vdf` '
            'automatically.')
    placeholder = _PLACEHOLDER_WIN if sys.platform == 'win32' else _PLACEHOLDER_NIX
    error_hint = 'folder missing Songs/'

    def find_dirs(self):
        return find_quaver_dirs()

    def validate_root(self, path):
        return validate_quaver_root(path)

    def get_root_override(self):
        from analysis.gui.settings import get_quaver_root_override
        return get_quaver_root_override()

    def set_root_override(self, path):
        from analysis.gui.settings import set_quaver_root_override
        set_quaver_root_override(path)

    def needs_enrichment(self, entry):
        # Library entries built by Quaver's `parse.parse_replay` already
        # carry chart_meta with title/artist/creator/keycount, so the
        # generic enrichment pass has nothing to backfill.
        return not entry.get('keycount')

    def enrich_entry(self, entry):
        chart_path = entry.get('chart_path')
        if not chart_path:
            return False
        try:
            from analysis.games.quaver.qua_chart import parse_qua_file
            chart = parse_qua_file(chart_path)
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
        audio = game_mod.get('quaver').resolve_audio(replay)
        return None, 0.0, audio

    def note_viz_config(self, replay, judge=None, od=None):
        from analysis.games.quaver.judgment import windows_for
        preset = (judge if isinstance(judge, str) and judge
                  else replay.get('judge', 'Standard'))
        windows = [(name, w_s, _VIZ_COLORS.get(name, '#888'))
                   for name, w_s in windows_for(preset)]
        return {
            'windows': windows,
            'unit_label': f'time (ms)  ;  {preset}',
            'rows_per_ms': None,
            'win': 8000,
        }


_VIZ_COLORS = {
    'marv': '#5cf', 'perf': '#5fc', 'great': '#cf5',
    'good': '#fc5', 'okay': '#f5c', 'miss': '#f55',
}


GUI_ADAPTER = QuaverGuiAdapter()
