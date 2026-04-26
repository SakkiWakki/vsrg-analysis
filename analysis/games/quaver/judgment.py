"""Quaver judge system. Windows are static (no OD analog) ; the judge
parameter is a discrete preset that biases the windows up/down. Defaults
to `Standard` (Quaver's `JudgementWindowsDatabaseCache.Standard`).

Returned shape matches `OsuAdapter.judgement_windows`: `[(name, half_window_sec), ...]`.
"""
from __future__ import annotations


# Standard preset, ms (Quaver/ScoreProcessorKeys.cs::JudgementWindow).
_STANDARD_MS = {
    'marv': 18.0,
    'perf': 43.0,
    'great': 76.0,
    'good': 106.0,
    'okay': 127.0,
    'miss': 164.0,
}


# Built-in presets shipped with Quaver. `Strict`/`Chill` are the mod
# variants ; `Peace` and `Hell` are the JWDB defaults Quaver users can
# pick from in the settings menu.
_PRESETS = {
    'Standard': _STANDARD_MS,
    'Strict': {k: v * 0.85 for k, v in _STANDARD_MS.items()},
    'Chill': {k: v * 1.15 for k, v in _STANDARD_MS.items()},
    'Peace': {k: v * 1.5 for k, v in _STANDARD_MS.items()},
    'Hell': {k: v * 0.5 for k, v in _STANDARD_MS.items()},
}


# Long-note release windows multiply the head windows by 1.5
# (Quaver/ScoreProcessorKeys.cs::WindowReleaseMultiplier).
RELEASE_MULTIPLIER = 1.5


def windows_for(judge='Standard'):
    """Return `[(name, half_window_sec), ...]` for the named preset.
    Unknown presets fall back to Standard so the player keeps rendering."""
    table = _PRESETS.get(judge, _STANDARD_MS)
    # Order matters: the player looks up windows in widening order so a
    # press that lands inside `marv` doesn't first match `miss`.
    return [(name, table[name] / 1000.0)
            for name in ('marv', 'perf', 'great', 'good', 'okay', 'miss')]


def preset_names():
    return list(_PRESETS.keys())
