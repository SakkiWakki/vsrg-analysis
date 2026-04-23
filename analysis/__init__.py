import os
import sys
from pathlib import Path


def cache_dir() -> Path:
    """Per-OS cache directory for the app. Follows the workspace folder
    name so a renamed repo still gets its own bucket.

    - Linux/BSD: ``$XDG_CACHE_HOME`` or ``~/.cache``
    - macOS: ``~/Library/Caches``
    - Windows: ``%LOCALAPPDATA%`` (fallback ``~/AppData/Local``)
    """
    app = Path(__file__).resolve().parents[1].name
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
    elif sys.platform == 'darwin':
        base = str(Path.home() / 'Library' / 'Caches')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
    return Path(base) / app
