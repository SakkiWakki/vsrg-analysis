"""Read fluXis's realm database via the bundled .NET dumper.

There is no Python Realm reader, and the `.frp`/`.fsc` files carry no
back-references -- every score<->map<->file link lives only inside
`fluxis.realm`. The dumper (realmdump/) prints the relevant tables as
one JSON document.

Two hard-won invariants, discovered against a live database:

  * The dumper must run on a COPY: opening with a newer Realm SDK
    upgrades the file format in place, which would corrupt the DB from
    the game's point of view.
  * The dumper must receive an ABSOLUTE path: Realm resolves relative
    paths against its own default data folder and silently creates a
    fresh empty database there instead of opening the file.

Requires the `dotnet` SDK; the helper is built once on first use and
reused from `realmdump/bin/` afterwards.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

_DUMPER_DIR = Path(__file__).parent / 'realmdump'
_BUILD_TIMEOUT_S = 300
_DUMP_TIMEOUT_S = 120

_dumper_dll_cache: Path | None = None


def dotnet_available() -> bool:
    return shutil.which('dotnet') is not None


def _find_built_dll() -> Path | None:
    for p in sorted(_DUMPER_DIR.glob('bin/Release/*/realmdump.dll')):
        return p
    return None


def _ensure_dumper(progress=None) -> Path | None:
    global _dumper_dll_cache
    if _dumper_dll_cache is not None and _dumper_dll_cache.is_file():
        return _dumper_dll_cache

    dll = _find_built_dll()
    if dll is None:
        if not dotnet_available():
            print('fluxis: dotnet SDK not found; cannot read fluxis.realm '
                  '(install .NET 8+ to enable the fluXis library)')
            return None
        if progress:
            progress('fluxis: building realm reader (first run)…')
        result = subprocess.run(
            ['dotnet', 'build', '-c', 'Release', '--nologo', '-v', 'q'],
            cwd=_DUMPER_DIR, capture_output=True, text=True,
            timeout=_BUILD_TIMEOUT_S)
        if result.returncode != 0:
            print(f'fluxis: realm reader build failed:\n{result.stdout}'
                  f'{result.stderr}')
            return None
        dll = _find_built_dll()
        if dll is None:
            print('fluxis: realm reader build produced no realmdump.dll')
            return None

    _dumper_dll_cache = dll
    return dll


def dump_realm(realm_path, *, progress=None) -> dict | None:
    """Dump `realm_path`'s tables to a dict, or None on any failure.
    Shape: {'schema': {...}, 'RealmScore': [...], 'RealmMap': [...],
    'RealmMapSet': [...]} with object links flattened to `<Prop>ID` and
    embedded objects inlined as `<Prop>.<SubProp>`."""
    dll = _ensure_dumper(progress=progress)
    if dll is None:
        return None

    with tempfile.TemporaryDirectory(prefix='fluxis-realm-') as td:
        copy = Path(td) / 'copy.realm'
        shutil.copyfile(realm_path, copy)
        result = subprocess.run(
            ['dotnet', str(dll), str(copy.resolve())],
            capture_output=True, text=True, timeout=_DUMP_TIMEOUT_S)

    if result.returncode != 0:
        print(f'fluxis: realm dump failed: {result.stderr.strip()[:500]}')
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f'fluxis: realm dump produced invalid JSON: {exc}')
        return None
