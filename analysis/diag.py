"""Tiny diagnostic logger that bypasses stdout.

Every other print in the codebase goes to stdout/stderr; under
gamescope, those streams are buried by the compositor's own spam.
``diag.log(tag, msg)`` writes one timestamped line to a dedicated
file so the user can ``tail -f`` it without filtering.

Sink path:
  $VSRG_DIAG_LOG  (override)         else
  $XDG_CACHE_HOME/vsrg-analysis/vsrg-diag.log  else
  ~/.cache/vsrg-analysis/vsrg-diag.log

Best-effort and lossy by design — if the file can't be opened we
silently drop the line. Diagnostics must not raise into the caller.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_path: Path | None = None
_resolved = False


def _resolve_path() -> Path | None:
    global _path, _resolved
    if _resolved:
        return _path
    _resolved = True
    override = os.environ.get('VSRG_DIAG_LOG')
    if override:
        _path = Path(override)
    else:
        cache = os.environ.get('XDG_CACHE_HOME') \
            or str(Path.home() / '.cache')
        _path = Path(cache) / 'vsrg-analysis' / 'vsrg-diag.log'
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
