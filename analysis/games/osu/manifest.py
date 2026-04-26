"""osu! game manifest -- the GUI's view of the game.

Two path fields: an install root (validated by presence of an
`osu!.<user>.cfg`) and a profile picker that becomes a combo when more
than one profile cfg is found in the chosen root."""
from __future__ import annotations

from pathlib import Path

from analysis.core.manifest import GameManifest, PathField


_OSU_ROOT_KEY = 'paths/osu_root'
_OSU_PROFILE_KEY = 'paths/osu_profile'


def _find_dirs():
    from analysis.games.osu.replay import find_osu_dirs
    return find_osu_dirs()


def _autodetect_root():
    return _find_dirs().get('root')


def _validate_root(path):
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


def _list_profiles(root):
    if not root:
        return []
    from analysis.games.osu.replay import list_osu_profiles
    return list_osu_profiles(root)


def _needs_enrichment(entry):
    return not entry.get('keycount')


def _enrich_entry(entry):
    """Backfill song/version/creator/keycount from the entry's .osu chart
    file. Called both in the library-scan auto-enrich path and on-demand
    from the player when a replay has resolved its chart."""
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


def _resolve_chart_context(replay, entry=None, progress=None):
    from analysis.core import game as game_mod
    audio = game_mod.get('osu').resolve_audio(replay)
    return None, 0.0, audio


def _note_viz_config(replay, judge=None, od=None):
    from analysis.viz.note_visualizer import (osu_mania_windows,
                                              effective_osu_od)
    base_od = od if od is not None else float(replay.get('od', 8.0))
    mods = int(replay.get('mods', 0))
    eff_od = effective_osu_od(base_od, mods)
    return {
        'windows': osu_mania_windows(od=eff_od),
        'unit_label': f'time (ms)  ;  OD {eff_od:.1f}',
        'rows_per_ms': None,
        'win': 8000,
    }


_PLACEHOLDER_ROOT = {
    'win32': r'e.g. %LOCALAPPDATA%\osu!',
    'linux': 'e.g. ~/.local/share/osu-wine/osu!',
    'darwin': 'e.g. ~/Library/Application Support/osu!',
    'default': 'e.g. ~/.local/share/osu-wine/osu!',
}


MANIFEST = GameManifest(
    name='osu',
    path_fields=[
        PathField(
            key='root',
            label='osu! install folder',
            hint=("Point at your osu! install folder ; the one that "
                  "contains `osu!.<username>.cfg`. The Songs folder is "
                  "resolved from the config's BeatmapDirectory setting."),
            placeholder=_PLACEHOLDER_ROOT,
            settings_key=_OSU_ROOT_KEY,
            error_hint='no osu!.<user>.cfg in folder',
            autodetect=_autodetect_root,
            validate=_validate_root,
        ),
        PathField(
            key='profile',
            label='osu! profile',
            hint=('Picked when the install folder has more than one '
                  '`osu!.<user>.cfg`. Stored separately from the root so '
                  'switching root resets the picker.'),
            placeholder='auto-pick newest cfg',
            settings_key=_OSU_PROFILE_KEY,
            list_choices=_list_profiles,
        ),
    ],
    find_dirs=_find_dirs,
    note_viz_config=_note_viz_config,
    needs_enrichment=_needs_enrichment,
    enrich_entry=_enrich_entry,
    resolve_chart_context=_resolve_chart_context,
)
