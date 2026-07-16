"""Per-frame consumer: compiled mod channels -> per-note offsets.

`apply(ctx)` runs once per frame after the candidate y arrays exist
(hooked from build_context): it samples the channels at t_now, runs
the vectorized ArrowEffects pipeline over the visible candidates, and
- adds dy to the head/tail/press y arrays in place,
- stashes per-candidate dx / alpha for the note views
  (ctx.candidate_dx / ctx.candidate_alpha, both in our pixel space).

Space conversion: the engine formulas work in ITG pixels (arrow size
64 at a 480-tall field); our lane width is the arrow size, so
`scale = lane_w / 64` converts both directions and `y_offset` is the
signed distance from the judgment line in engine pixels.

Simplifications (documented, revisit with the oracle):
- beat(t) inverts the BPM segments only (stops/warps shift beat_now
  slightly during those regions; note beats come exactly from rows).
- player 0 channels only until per-field routing lands.
- rotation/zoom offsets are computed but not yet consumed by draw
  sites.
"""
from __future__ import annotations

import numpy as np

from analysis.games.etterna.sm_chart import beat_to_time
from analysis.player.render.mods.arrow_effects import ARROW_SIZE, note_offsets

_ACTIVE_EPS = 1e-4


class NotitgNoteMods:
    def __init__(self, channels, bpms):
        self._channels = channels
        segments = []
        for beat, bpm in sorted(bpms):
            if bpm > 0:
                segments.append((beat_to_time(beat, bpms, 0.0), beat,
                                 bpm / 60.0))
        self._segments = segments or [(0.0, 0.0, 2.0)]

    def _beat_at(self, t: float) -> float:
        lo = self._segments[0]
        for seg in self._segments:
            if seg[0] > t:
                break
            lo = seg
        t0, beat0, bps = lo
        return beat0 + max(0.0, t - t0) * bps

    def apply(self, ctx) -> None:
        if not ctx.candidates:
            return
        t = float(ctx.t_now)
        percents = self._channels.values_at(t)
        if all(abs(v) < _ACTIVE_EPS for v in percents.values()):
            return

        p = ctx.player
        idx = np.asarray(ctx.candidates, dtype=np.int64)
        cols = np.asarray(p.columns[idx], dtype=np.int64)
        scale = ctx.lane_w / ARROW_SIZE
        judge_y = float(ctx.judge_y)
        y_offset = (judge_y - ctx.candidate_head_y) / scale

        offs = note_offsets(
            percents, cols, y_offset,
            t_now=t, beat_now=self._beat_at(t), keycount=p.keycount,
            note_beats=p.noterows_array(idx) if hasattr(p, 'noterows_array')
            else np.asarray(p.notes.noterows_list, dtype=np.float64)[idx]
            / 48.0)

        dy = offs.dy * scale
        ctx.candidate_head_y += dy
        ctx.candidate_tail_y += dy
        ctx.candidate_press_y += dy
        ctx.candidate_dx = offs.dx * scale
        ctx.candidate_alpha = offs.alpha_mult
