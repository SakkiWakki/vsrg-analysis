"""One-shot migration of legacy config locations into :class:`ConfigStore`.

Two legacy JSON files live next to the new config:

  * ``player_plugins.json``  ; ``{"disabled": [key, ...]}``
  * ``sidebar_sections.json`` ; same shape

QSettings also holds a few UI-chrome values (paths, first-run flag).
Both are folded into the unified tree, then the legacy files are
deleted so they can't drift. QSettings entries are left in place ;
QSettings is a system-managed store and removing entries from it
races with other Qt code. We just stop reading from it.

Migration is idempotent: a tree that already carries
``_schema_version`` skips the read-legacy step.
"""
from __future__ import annotations

import json
from pathlib import Path

_SCHEMA_VERSION = 1


def migrate_legacy(store) -> None:
    """Fold legacy JSON files into ``store``. Safe to call repeatedly
    ; after the first run, ``_schema_version`` marks the tree and
    subsequent calls are no-ops."""
    if store.get('_schema_version', 0) >= _SCHEMA_VERSION:
        return

    mutated = False
    base = Path.home() / '.config' / 'vsrg-analysis'

    mutated |= _migrate_disabled_list(
        store, base / 'player_plugins.json', 'replay')
    mutated |= _migrate_disabled_list(
        store, base / 'sidebar_sections.json', 'sidebar')
    mutated |= _migrate_qsettings(store)

    store.set('_schema_version', _SCHEMA_VERSION)
    if mutated:
        store.flush()


def _migrate_disabled_list(store, path: Path, kind: str) -> bool:
    """Read a legacy ``{"disabled": [...]}`` file if present, write each
    disabled key as ``plugins.<key>.<kind>_disabled = True`` (the
    plugin-level schema isn't nailed down yet; we use discrete flags
    per role rather than a single ``enabled`` so replay + sidebar
    state for the same bundle can coexist). Deletes the legacy file
    on success."""
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return False
    except OSError as exc:
        print(f'legacy config read failed ({path}): {exc}')
        return False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f'legacy config parse failed ({path}): {exc}')
        return False
    disabled = data.get('disabled') if isinstance(data, dict) else None
    if not isinstance(disabled, list):
        return False

    changed = False
    flag = f'{kind}_disabled'
    for key in disabled:
        if not isinstance(key, str):
            continue
        changed |= store.set(f'plugins.{_escape(key)}.{flag}', True)

    try:
        path.unlink()
    except OSError:
        pass
    return changed


def _migrate_qsettings(store) -> bool:
    """Fold Qt-side paths + first-run flag into the config tree. Only
    reads ; QSettings entries stay where they are. Safe if PySide6
    isn't importable (headless tests)."""
    try:
        from analysis.gui import settings as qs
    except ImportError:
        return False
    except Exception as exc:
        print(f'QSettings read skipped: {exc}')
        return False

    changed = False
    try:
        et = qs.get_etterna_save_override()
        if et:
            changed |= store.set('paths.etterna_save', et)
        osu = qs.get_osu_songs_override()
        if osu:
            changed |= store.set('paths.osu_songs', osu)
        if qs.is_first_run_done():
            changed |= store.set('paths.first_run_done', True)
    except Exception as exc:
        print(f'QSettings migration partial: {exc}')
    return changed


def _escape(key: str) -> str:
    """Plugin keys can contain dots (rare but legal ; bundle authors
    pick them). Dotted path parts are the store's separator, so rewrite
    any dots in a key to underscores at migration time. Colons and
    other characters stay verbatim."""
    return key.replace('.', '_')
