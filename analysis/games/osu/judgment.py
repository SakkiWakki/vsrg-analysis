"""osu!mania judge system: windows scale with OD (overall difficulty, ms).

Exposed through `OsuAdapter.judgement_windows(replay, od=None)`. The
Player consumes only the `windows_for(od)` result; no OD plumbing needs
to reach outside the adapter.

Formulas match osu!mania stable: hit300g = 16.5 ms (hard-coded), others
= `base - 3 * OD` ms with base = 64 / 97 / 127 / 151."""
from __future__ import annotations


def windows_for(od):
    """Return [(name, half_window_sec), …] for the given OD."""
    return [
        ('marv', 16.5 / 1000.0),
        ('perf', (64 - 3 * od) / 1000.0),
        ('great', (97 - 3 * od) / 1000.0),
        ('good', (127 - 3 * od) / 1000.0),
        ('bad', (151 - 3 * od) / 1000.0),
    ]
