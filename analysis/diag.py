"""Tiny diagnostic logger that bypasses stdout.

Every other print in the codebase goes to stdout/stderr; under
gamescope (and under a windowed osu!.exe on Windows) those streams are
buried. ``diag.log(tag, msg)`` writes one timestamped line to a
per-OS cache file so the user can tail it without filtering.

Sink path, in order of preference:
  $VSRG_DIAG_LOG  (override)
  Linux:    $XDG_CACHE_HOME/vsrg-analysis/vsrg-diag.log
            ~/.cache/vsrg-analysis/vsrg-diag.log
  macOS:    ~/Library/Caches/vsrg-analysis/vsrg-diag.log
  Windows:  %LOCALAPPDATA%\\vsrg-analysis\\vsrg-diag.log

Best-effort and lossy by design ; if the file can't be opened we
silently drop the line. Diagnostics must not raise into the caller.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_path: Path | None = None
_resolved = False


def _default_cache_root() -> Path:
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
    elif sys.platform == 'darwin':
        base = str(Path.home() / 'Library' / 'Caches')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
    return Path(base)


def _resolve_path() -> Path | None:
    global _path, _resolved
    if _resolved:
        return _path
    _resolved = True
    override = os.environ.get('VSRG_DIAG_LOG')
    if override:
        _path = Path(override)
    else:
        _path = _default_cache_root() / 'vsrg-analysis' / 'vsrg-diag.log'
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        _path = None
    return _path


def log(tag: str, msg: str) -> None:
    """Append ``[ts] [tag] msg`` to the diag log. Never raises."""
    path = _resolve_path()
    if path is None:
        return
    line = f'[{time.strftime("%H:%M:%S")}] [{tag}] {msg}\n'
    try:
        with _lock:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line)
    except OSError:
        pass


def path() -> Path | None:
    """Return the resolved sink path (or None if unresolvable)."""
    return _resolve_path()
