"""Game-agnostic judgment primitives.

- `JCLR` is the shared color palette the renderer uses for the six
  judgment tiers.
- `judge()` classifies a hit offset against a windows list.

Each game's windows (and the parameter that scales them — OD for osu,
judge-level for Etterna) live on its adapter: see
`GameAdapter.judgement_windows(replay, **overrides)` and the per-game
implementations under analysis/games/*/adapter.py."""
from __future__ import annotations


JCLR = {
    'marv': (255, 255, 255),
    'perf': (255, 213, 79),
    'great': (129, 199, 132),
    'good': (79, 195, 247),
    'bad': (186, 104, 200),
    'miss': (229, 57, 53),
}


def judge(off_s, windows, is_miss):
    """Return the judgment label ('marv'/'perf'/…/'miss') for a hit
    offset in seconds against a [(name, half_window_sec), …] list."""
    if is_miss:
        return 'miss'
    a = abs(off_s)
    for name, w in windows:
        if a <= w:
            return name
    return 'miss'
