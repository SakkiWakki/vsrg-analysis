from pathlib import Path


def cache_dir() -> Path:
    """~/.cache/<repo-folder-name>/ — follows the workspace folder if renamed."""
    return Path.home() / '.cache' / Path(__file__).resolve().parents[1].name
