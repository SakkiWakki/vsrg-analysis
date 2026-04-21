"""Replay-time conversion helpers.

The per-game conversion lives on each `GameAdapter.prepare_replay_times`
(osu → ms, Etterna → BPM map). This module just provides the shared bits
a helper would want: keycount inference + a fallback for callers that
don't have an adapter handy (tests, ad-hoc tooling)."""
from __future__ import annotations

import numpy as np


def infer_keycount(replay) -> int:
    kc = replay.get('keycount')
    if kc:
        return int(kc)
    cols = replay.get('columns')
    if cols is not None and len(cols):
        return int(cols.max()) + 1
    return 4


def fallback_times(replay):
    """Game-agnostic noterow→time for callers without a live adapter.
    Etterna-flavored (48 rows/beat @ 120bpm = 96 rows/sec). osu replays
    carry ms in noterows; detect via `chart_path` and divide by 1000."""
    if replay.get('chart_path'):
        return replay['noterows'].astype(np.float64) / 1000.0
    return replay['noterows'].astype(np.float64) / 96.0
