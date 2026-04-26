"""Quaver game manifest -- the GUI's view of the game."""
from __future__ import annotations

from analysis.core.manifest import GameManifest, PathField


_QUAVER_ROOT_KEY = 'paths/quaver_root'

_VIZ_COLORS = {
    'marv': '#5cf', 'perf': '#5fc', 'great': '#cf5',
    'good': '#fc5', 'okay': '#f5c', 'miss': '#f55',
}


def _find_dirs():
    from analysis.games.quaver.paths import find_quaver_dirs
    return find_quaver_dirs()


def _validate_root(path):
    from analysis.games.quaver.paths import validate_quaver_root
    return validate_quaver_root(path)


def _needs_enrichment(entry):
    # Library entries built by Quaver's `parse.parse_replay` already
    # carry chart_meta with title/artist/creator/keycount, so the generic
    # enrichment pass has nothing to backfill.
    return not entry.get('keycount')


def _enrich_entry(entry):
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


def _resolve_chart_context(replay, entry=None, progress=None):
    from analysis.core import game as game_mod
    audio = game_mod.get('quaver').resolve_audio(replay)
    return None, 0.0, audio


def _note_viz_config(replay, judge=None, od=None):
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


MANIFEST = GameManifest(
    name='quaver',
    path_fields=[
        PathField(
            key='root',
            label='Quaver install folder',
            hint=('Point at your Quaver install folder ; the one that '
                  'contains `Songs/` (and usually `Data/Replays/`). Steam '
                  'libraries on secondary drives are picked up from '
                  '`libraryfolders.vdf` automatically.'),
            placeholder={
                'win32': r'e.g. C:\Program Files (x86)\Steam\steamapps\common\Quaver',
                'linux': 'e.g. ~/.steam/steam/steamapps/common/Quaver',
                'darwin': 'e.g. ~/Library/Application Support/Steam/steamapps/common/Quaver',
                'default': 'e.g. ~/.steam/steam/steamapps/common/Quaver',
            },
            settings_key=_QUAVER_ROOT_KEY,
            error_hint='folder missing Songs/',
            validate=_validate_root,
        ),
    ],
    find_dirs=_find_dirs,
    note_viz_config=_note_viz_config,
    needs_enrichment=_needs_enrichment,
    enrich_entry=_enrich_entry,
    resolve_chart_context=_resolve_chart_context,
)
