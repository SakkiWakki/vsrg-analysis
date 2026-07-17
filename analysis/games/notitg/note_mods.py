"""Per-frame consumer: compiled mod channels -> per-note offsets.

`apply(ctx)` runs once per frame after the candidate y arrays exist
(hooked from build_context): it samples the channels at t_now, runs
the vectorized ArrowEffects pipeline over the visible candidates, and
- adds dy to the head/tail/press y arrays in place,
- stashes per-candidate dx / alpha / rotation / zoom for the note views
  (ctx.candidate_dx / _alpha / _rot_deg / _zoom; dx in our pixel space,
  rotation in degrees, zoom a multiplier),
- stashes ctx.receptor_offsets: a dict of numpy arrays keyed
  'dx','dy','rotation_deg','zoom','alpha' (length keycount) in OUR pixel
  space, so the receptor layer displaces the hit marks the same way the
  engine displaces receptors (drunk/tornado shift, confusion spin, ...).

Space conversion: the engine formulas work in ITG pixels (arrow size
64 at a 480-tall field); our lane width is the arrow size, so
`scale = lane_w / 64` converts both directions and `y_offset` is the
signed distance from the judgment line in engine pixels.

Scroll direction and accel:
- ACCEL FAMILY (boost/brake/wave/expand) reshapes the y_offset ->
  position mapping, so it runs FIRST, remapping head/tail/press y_offset
  and rebuilding their positions before any dy contribution.
- REVERSE FAMILY (reverse/split/alternate/cross/centered) slides each
  column's receptor toward the mirrored/centered judge line and flips
  the note distance about it, over the (accel-remapped) candidate y
  arrays. All in pre-flip downscroll space; the global upscroll flip
  wraps the result.

Simplifications (documented, revisit with the oracle):
- beat(t) inverts the BPM segments only (stops/warps shift beat_now
  slightly during those regions; note beats come exactly from rows).
- player 0 channels only until per-field routing lands.
- rotation/zoom offsets are computed but not yet consumed by draw
  sites.
- boomerang deferred (needs the culling peak contract); expand keyed to
  song time not wall clock (scrub-exactness).
"""
from __future__ import annotations

import numpy as np

from analysis.games.etterna.sm_chart import beat_to_time
from analysis.player.render.mods.arrow_effects import (
    ARROW_SIZE, accel_y_offset, note_offsets, receptor_alpha_from_dark,
    receptor_offsets, reverse_fractions)

_ACTIVE_EPS = 1e-4
_EXPAND_RATE = 3.0  # ArrowEffects.cpp:131 cos(g_fExpandSeconds*3)


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

        # ACCEL FAMILY (boost/brake/wave/expand): reshape the raw scroll
        # y_offset -> position mapping BEFORE any dy contribution. Each of
        # head/tail/press is a position on the same scroll axis, so all
        # three are remapped by the same function; positions rebuild as
        # y = judge_y - y_offset' * scale (downscroll, pre-flip space).
        percents = self._with_expand_phase(percents, t)
        head_off = self._remap_accel(percents, ctx.candidate_head_y, judge_y, scale)
        tail_off = self._remap_accel(percents, ctx.candidate_tail_y, judge_y, scale)
        press_off = self._remap_accel(percents, ctx.candidate_press_y, judge_y, scale)
        ctx.candidate_head_y = judge_y - head_off * scale
        ctx.candidate_tail_y = judge_y - tail_off * scale
        ctx.candidate_press_y = judge_y - press_off * scale

        note_beats = (np.asarray(p.notes.noterows_list,
                                 dtype=np.float64)[idx] / 48.0)
        offs = note_offsets(
            percents, cols, head_off,
            t_now=t, beat_now=self._beat_at(t), keycount=p.keycount,
            note_beats=note_beats)

        # REVERSE FAMILY (reverse/split/alternate/cross/centered): per
        # column, the receptor slides toward the mirrored judge line by
        # r_col and the note's distance from it flips sign. Applied over
        # the (already accel-remapped) candidate y arrays, in pre-flip
        # downscroll space; the global upscroll flip wraps the result.
        self._apply_reverse(ctx, percents, cols, judge_y)

        dy = offs.dy * scale
        ctx.candidate_head_y += dy
        ctx.candidate_tail_y += dy
        ctx.candidate_press_y += dy
        ctx.candidate_dx = offs.dx * scale
        ctx.candidate_alpha = offs.alpha_mult
        ctx.candidate_rot_deg = offs.rotation_deg
        ctx.candidate_zoom = offs.zoom
        ctx.receptor_offsets = self._receptor_offsets(
            ctx, percents, p.keycount, scale, t, judge_y)

    def _remap_accel(self, percents, ys, judge_y, scale):
        """Accel-remapped y_offset (engine px) for a candidate y array."""
        y_offset = (judge_y - np.asarray(ys, dtype=np.float64)) / scale
        return accel_y_offset(percents, y_offset)

    def _with_expand_phase(self, percents, t):
        """Inject the expand periodic phase (song time for scrub-exactness;
        the engine uses a wall-clock timer, ArrowEffects.cpp:126-131)."""
        if not percents.get('expand', 0.0):
            return percents
        out = dict(percents)
        out['_expand_phase'] = t * _EXPAND_RATE
        return out

    def _reverse_geom(self, ctx, judge_y):
        """Mirror of the judge line about the chart region's vertical
        center: the position a fully-reversed receptor slides to, in
        pre-flip downscroll pixels."""
        _rx, ry, _w, h = ctx.chart_rect
        return 2.0 * (ry + h / 2.0) - judge_y

    def _apply_reverse(self, ctx, percents, cols, judge_y):
        centered = float(percents.get('centered', 0.0))
        active = centered or any(
            percents.get(m, 0.0)
            for m in ('reverse', 'split', 'alternate', 'cross'))
        if not active:
            return

        r = reverse_fractions(percents, cols, ctx.player.keycount)
        self._reverse_arrays(ctx, r, centered, judge_y)

    def _reverse_arrays(self, ctx, r, centered, judge_y):
        """y' places the receptor at lerp(judge_y, mirror_y, r) and flips
        the note's distance below the receptor by (1 - 2r). centered slides
        the receptor toward field center (mid-screen) by its percent."""
        mirror_y = self._reverse_geom(ctx, judge_y)
        rx, ry, _w, h = ctx.chart_rect
        center_y = ry + h / 2.0

        def remap(ys):
            receptor_y = judge_y + r * (mirror_y - judge_y)
            receptor_y = receptor_y + centered * (center_y - receptor_y)
            return receptor_y + (1.0 - 2.0 * r) * (np.asarray(ys) - judge_y)

        ctx.candidate_head_y = remap(ctx.candidate_head_y)
        ctx.candidate_tail_y = remap(ctx.candidate_tail_y)
        ctx.candidate_press_y = remap(ctx.candidate_press_y)

    def _receptor_offsets(self, ctx, percents, keycount, scale, t, judge_y) -> dict:
        """Per-column receptor mods in OUR pixel space. `receptor_offsets`
        evaluates the pipeline at y_offset = 0 over one note per column;
        dx/dy convert from engine px by `scale`, rotation/zoom/alpha are
        unitless. The reverse family adds a per-column vertical shift (the
        receptor slides to its mirrored/centered position), and dark
        multiplies the receptor mark alpha (note visibility is untouched)."""
        cols = np.arange(keycount, dtype=np.int64)
        offs = receptor_offsets(percents, cols, t_now=t,
                                beat_now=self._beat_at(t), keycount=keycount)
        dy = offs.dy * scale + self._receptor_reverse_dy(ctx, percents, cols, judge_y)
        alpha = offs.alpha_mult * receptor_alpha_from_dark(percents.get('dark', 0.0))
        return {
            'dx': offs.dx * scale, 'dy': dy,
            'rotation_deg': offs.rotation_deg, 'zoom': offs.zoom,
            'alpha': alpha,
        }

    def _receptor_reverse_dy(self, ctx, percents, cols, judge_y):
        """Per-column receptor vertical shift from the reverse family: the
        receptor slides from judge_y to lerp(judge_y, mirror_y, r_col),
        then toward field center by `centered`. Receptors are at y_offset 0
        so their shift is exactly (receptor_y - judge_y)."""
        centered = float(percents.get('centered', 0.0))
        active = centered or any(
            percents.get(m, 0.0)
            for m in ('reverse', 'split', 'alternate', 'cross'))
        if not active:
            return np.zeros(cols.shape[0], dtype=np.float64)

        r = reverse_fractions(percents, cols, ctx.player.keycount)
        mirror_y = self._reverse_geom(ctx, judge_y)
        _rx, ry, _w, h = ctx.chart_rect
        center_y = ry + h / 2.0
        receptor_y = judge_y + r * (mirror_y - judge_y)
        receptor_y = receptor_y + centered * (center_y - receptor_y)
        return receptor_y - judge_y
