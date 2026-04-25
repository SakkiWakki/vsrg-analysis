"""Etterna `#SCROLLS` displayed-beat cache and its inverse.

`#SCROLLS` is a piecewise-constant velocity in beat-space (a SCROLLS row
`(beat, ratio)` says "from this beat onward, advance displayed-beat at
`ratio` per real beat"). The cache pre-integrates the curve so
`displayed_beat(beat)` is an O(log n) lookup, and the inverse table lets
the cull-space clock smoother project an SV value back to a chart-time
that hits it.

Both the reference `BeatSpaceSVEngine` and the new measure engine consume
this class. Inverse handling follows the reference engine: in scroll<=0
plateaus, displayed-beat is non-monotonic, so we skip those segments and
collapse plateau queries to the segment's start beat.
"""
from __future__ import annotations

import bisect

import numpy as np

from analysis.player.sv.timing import TimingMap


class ScrollsCache:
    """Pre-integrated displayed-beat function for a `#SCROLLS` table.

    cache: list[(beat, displayed_beat, ratio)]. At any real beat b in
    [cache[i].beat, cache[i+1].beat], displayed_beat =
        cache[i].displayed_beat + (b - cache[i].beat) * cache[i].ratio.

    If the first SCROLLS row is not at beat 0, we prepend an implicit
    (0, 0, 1.0) segment matching Etterna's ResetCacheInfo fallthrough.
    """

    def __init__(self, scrolls):
        self._scrolls = list(scrolls or [])

        cache: list[tuple[float, float, float]] = []
        if self._scrolls:
            if self._scrolls[0][0] > 0.0:
                cache.append((0.0, 0.0, 1.0))
            displayed = 0.0
            last_beat = 0.0
            last_ratio = 1.0
            for (b, r) in self._scrolls:
                displayed += (b - last_beat) * last_ratio
                cache.append((b, displayed, r))
                last_beat = b
                last_ratio = r
        self._cache = cache
        self._cache_beats = [c[0] for c in cache]

        self._cache_beats_np = np.asarray(self._cache_beats, dtype=np.float64)
        self._cache_db_np = np.asarray([c[1] for c in cache], dtype=np.float64)
        self._cache_ratio_np = np.asarray([c[2] for c in cache], dtype=np.float64)

        # Sorted-by-displayed-beat table for the inverse. SCROLLS with
        # non-positive ratios make displayed_beat non-monotonic, so we skip
        # those segments during inverse lookup.
        self._dbs_monotonic: list[tuple[float, int]] = []
        last_db = -float('inf')
        for i, c in enumerate(cache):
            db = c[1]
            if db >= last_db:
                self._dbs_monotonic.append((db, i))
                last_db = db
        self._dbs_only = [x[0] for x in self._dbs_monotonic]

    def __bool__(self) -> bool:
        return bool(self._cache)

    def displayed_beat(self, beat: float) -> float:
        if not self._cache:
            return beat
        idx = bisect.bisect_right(self._cache_beats, beat) - 1
        if idx < 0:
            return beat
        b, db, r = self._cache[idx]
        return db + (beat - b) * r

    def displayed_beat_array(self, beats: np.ndarray) -> np.ndarray:
        beats = np.asarray(beats, dtype=np.float64)
        if not beats.size:
            return np.empty(0, dtype=np.float64)
        if self._cache_beats_np.size == 0:
            return beats
        idx = np.searchsorted(self._cache_beats_np, beats, side='right') - 1
        pre_mask = idx < 0
        safe = np.clip(idx, 0, self._cache_beats_np.size - 1)
        out = (self._cache_db_np[safe]
               + (beats - self._cache_beats_np[safe]) * self._cache_ratio_np[safe])
        if pre_mask.any():
            out[pre_mask] = beats[pre_mask]
        return out

    def ratio_at_beat(self, beat: float) -> float:
        if not self._cache:
            return 1.0
        idx = bisect.bisect_right(self._cache_beats, beat) - 1
        if idx < 0:
            return 1.0
        return float(self._cache[idx][2])

    def inverse_displayed_beat(self, db_target: float, timing: TimingMap) -> float:
        """Chart-time t such that displayed_beat(beat(t)) == db_target.

        Returns the chart-time projected back through the timing map.
        Within a scroll<=0 plateau, returns the segment's start time.
        """
        if not self._cache:
            return timing.beat_to_time(db_target)
        if not self._dbs_only:
            return timing.beat_to_time(db_target)
        idx = bisect.bisect_right(self._dbs_only, db_target) - 1
        if idx < 0:
            return timing.beat_to_time(db_target)
        cache_idx = self._dbs_monotonic[idx][1]
        b, db, r = self._cache[cache_idx]
        if r > 0:
            beat = b + (db_target - db) / r
        else:
            beat = b
        return timing.beat_to_time(beat)
