"""Ratchet against chart-specific code spreading through the codebase.

Harvest code may key on engine STRUCTURE, never on one chart's actor
names. The known gat-named residue (memory item 97) is allowlisted per
file until its de-gatification lands; this test fails the moment a
gat_* name appears anywhere new, and the allowlist may only shrink.
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent / 'analysis'
_PATTERN = re.compile(r'\bgat_[a-z0-9_]+', re.IGNORECASE)

# file (relative to analysis/) -> why it is still allowed. Shrink only.
_ALLOWED = {
    'games/notitg/mod_stubs.py': 'proxy_grid() gat globals (item 97.1)',
    'games/notitg/modfile.py': 'AFT rig message conventions (item 97.2)',
    'games/notitg/aft_drivers.py': 'hand-ported gat driver (item 97.3)',
    # Found by this ratchet, pending item-97 triage (examples-only
    # docstring mentions are fine to keep; logic references are not):
    'games/notitg/field_instances.py': 'triage pending',
    'games/notitg/recording_actor.py': 'triage pending',
    'games/notitg/update_integrator.py': 'triage pending',
}


def test_no_new_chart_specific_names():
    offenders = {}
    for path in _ROOT.rglob('*.py'):
        rel = path.relative_to(_ROOT).as_posix()
        hits = _PATTERN.findall(path.read_text(encoding='utf-8',
                                               errors='replace'))
        if hits and rel not in _ALLOWED:
            offenders[rel] = sorted(set(hits))[:5]
    assert not offenders, (
        'chart-specific names outside the item-97 allowlist (key on '
        f'structure, not names): {offenders}')


def test_allowlist_entries_still_exist():
    stale = [rel for rel in _ALLOWED if not (_ROOT / rel).exists()
             or not _PATTERN.search((_ROOT / rel).read_text(
                 encoding='utf-8', errors='replace'))]
    assert not stale, (
        f'allowlist entries clean or gone - REMOVE them (ratchet): {stale}')
