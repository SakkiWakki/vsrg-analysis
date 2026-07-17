"""Per-frame consumer: compiled mod channels -> per-note offsets.

`apply(ctx)` runs once per frame after the candidate y arrays exist
(hooked from build_context): it samples the channels at t_now, runs
the vectorized ArrowEffects pipeline over the visible candidates, and
- adds dy to the head/tail/press y arrays in place,
- stashes per-candidate dx / alpha / rotation / zoom for the note views
  (ctx.candidate_dx / _alpha / _rot_deg / _zoom; dx in our pixel space,
  rotation in degrees, zoom a multiplier),
- stashes ctx.hold_body_samples: candidate-position -> (xs, ys) polyline
  arrays (OUR pixel space) tracing each visible hold's body BENT by the
  per-note x-mods (drunk/wave/digital ...) instead of a straight rect;
  the notes layer draws the body through these points. Present only when
  a dx-producing mod is active and a hold is visible,
- stashes ctx.receptor_offsets: a dict of numpy arrays keyed
  'dx','dy','rotation_deg','zoom','alpha' (length keycount) in OUR pixel
  space, so the receptor layer displaces the hit marks the same way the
  engine displaces receptors (drunk/tornado shift, confusion spin, ...).

The pipeline is name-agnostic: it consumes whatever channels values_at
returns, so mods the integrator injects per-frame (confusionxoffset /
confusionyoffset / hallway among them) light up with no change here.
arrow_effects reprojects the out-of-plane confusion tilts into channels
this consumer already carries - confusionx as a per-note zoom, confusiony
and hallway as per-note dx - so they ride the same candidate_zoom /
candidate_dx / hold-body-bend / receptor stashes as the in-plane mods.

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
  arrays. Scroll orientation is owned HERE, with no global flip: the
  native candidate space (receptors at judge_y, notes falling down) is
  engine reverse=1, so the effective per-column fraction is
  1 - r_engine and the remap runs every frame -- zero channels yield
  the engine default (receptors on top, notes scrolling up), and a
  chart pinning reverse=1 (gat) reads back as our native downscroll.

Simplifications (documented, revisit with the oracle):
- beat(t) inverts the BPM segments only (stops/warps shift beat_now
  slightly during those regions; note beats come exactly from rows).
- player 0 channels only until per-field routing lands.
- expand keyed to song time not wall clock (scrub-exactness).
"""
from __future__ import annotations

import numpy as np

from analysis.games.etterna.sm_chart import beat_to_time
from analysis.player.render.mods.arrow_effects import (
    ARROW_SIZE, accel_y_offset, note_offsets, receptor_alpha_from_dark,
    receptor_offsets, reverse_fractions)

_ACTIVE_EPS = 1e-4
_EXPAND_RATE = 3.0  # ArrowEffects.cpp:131 cos(g_fExpandSeconds*3)
_MAX_BODY_SAMPLES = 96  # per-hold cap on the body polyline subdivision
# Body sample spacing, engine px. Must sit well under the shortest
# spatial period of the waveform family (drunk's is ~2*pi*48 px) or a
# long hold's helix aliases into zigzag; 8 px = ~37 samples per period.
_BODY_SAMPLE_SPACING = 8.0
# The sampled span is clamped to the visible window plus this margin
# (fraction of the window height per side): sampling budget goes to the
# on-screen body, and clamped path ends stay far enough off screen that
# a seated tail cap never becomes visible at the clamp edge.
_BODY_WINDOW_PAD = 0.6


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
        t = float(ctx.t_now)
        percents = self._with_expand_phase(self._channels.values_at(t), t)
        scale = ctx.lane_w / ARROW_SIZE
        judge_y = float(ctx.judge_y)

        if ctx.candidates:
            self._apply_to_notes(ctx, percents, scale, judge_y, t)
        # Receptors carry the scroll orientation even on empty frames.
        ctx.receptor_offsets = self._receptor_offsets(
            ctx, percents, ctx.player.keycount, scale, t, judge_y)

    def _apply_to_notes(self, ctx, percents, scale, judge_y, t) -> None:
        p = ctx.player
        idx = np.asarray(ctx.candidates, dtype=np.int64)
        cols = np.asarray(p.columns[idx], dtype=np.int64)
        active = any(abs(v) >= _ACTIVE_EPS for v in percents.values())

        # ACCEL FAMILY (boost/brake/wave/expand): reshape the raw scroll
        # y_offset -> position mapping BEFORE any dy contribution. Each of
        # head/tail/press is a position on the same scroll axis, so all
        # three are remapped by the same function; positions rebuild as
        # y = judge_y - y_offset' * scale (native downscroll space).
        offs = None
        if active:
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
        # r_col and the note's distance from it flips sign, over the
        # (already accel-remapped) candidate y arrays. Runs every frame:
        # the zero-channel baseline is a full mirror to engine-default
        # upscroll (see module doc).
        self._apply_reverse(ctx, percents, cols, judge_y)

        if offs is None:
            return
        dy = offs.dy * scale
        ctx.candidate_head_y += dy
        ctx.candidate_tail_y += dy
        ctx.candidate_press_y += dy
        ctx.candidate_dx = offs.dx * scale
        ctx.candidate_alpha = offs.alpha_mult
        ctx.candidate_rot_deg = offs.rotation_deg
        ctx.candidate_zoom = offs.zoom

        self._stash_hold_body_samples(ctx, percents, cols, idx, head_off,
                                      tail_off, scale, t)

    def _stash_hold_body_samples(self, ctx, percents, cols, idx, head_off,
                                 tail_off, scale, t) -> None:
        """Sample each visible hold's body so the notes layer can draw it
        as a polyline that BENDS under the per-note x/y mods, instead of a
        straight head-to-tail rect (drunk/wave/digital etc. displace every
        strip of an engine-rendered hold body, not just the head).

        A hold's body is subdivided in engine y_offset space between its
        (accel-remapped) head and tail offsets. The x of each sample is
        lane_x(col) + note_offsets(...).dx at that offset; the y is the
        linear interpolation between the FINAL head_y and tail_y (post
        accel + dy + reverse). Both endpoints coincide with the head and
        tail by construction (sample 0 = head_off/head_y, sample -1 =
        tail_off/tail_y), so the bent body stays attached.

        All samples of all visible holds are batched into ONE note_offsets
        call (each sample is just another row: same column, varying
        y_offset, same note_beat), then split back per hold. The result is
        stashed as ctx.hold_body_samples: candidate-position -> (xs, ys)
        arrays in OUR pixel space, consumed by layers/notes.py."""
        p = ctx.player
        lane_x_fn = getattr(ctx, 'lane_x', None)
        ln_tail_times = getattr(p.notes, 'ln_tail_times', None)
        if lane_x_fn is None or ln_tail_times is None:
            return
        tail_off = np.asarray(tail_off, dtype=np.float64)
        is_ln = np.isfinite(np.asarray(ln_tail_times)[idx]) & np.isfinite(tail_off)
        if not is_ln.any():
            return

        head_y = np.asarray(ctx.candidate_head_y, dtype=np.float64)
        tail_y = np.asarray(ctx.candidate_tail_y, dtype=np.float64)
        note_beats = np.asarray(p.notes.noterows_list, dtype=np.float64)[idx] / 48.0
        _rx, ry, _w, h = ctx.chart_rect
        window = (ry - h * _BODY_WINDOW_PAD, ry + h * (1.0 + _BODY_WINDOW_PAD))
        segments = self._build_body_segments(
            np.nonzero(is_ln)[0], cols, head_off, tail_off, head_y, tail_y,
            note_beats, scale, window)
        if not segments['holds']:
            return

        sample = note_offsets(
            percents, segments['cols'], segments['offs'], t_now=t,
            beat_now=self._beat_at(t), keycount=p.keycount,
            note_beats=segments['beats'])

        # A body only needs the polyline when its dx actually VARIES along
        # the body (drunk/wave/digital ...); a constant dx (flip/movex, or
        # reverse-only frames) leaves it a straight strip the rect path
        # already draws. Skip those holds so the rect fallback stays.
        samples = {}
        for pos, screen_ys, start, count in segments['holds']:
            dx = sample.dx[start:start + count]
            if np.ptp(dx) < _ACTIVE_EPS:
                continue
            samples[pos] = (lane_x_fn(int(cols[pos])) + dx * scale, screen_ys)
        if samples:
            ctx.hold_body_samples = samples

    def _build_body_segments(self, ln_positions, cols, head_off, tail_off,
                             head_y, tail_y, note_beats, scale,
                             window) -> dict:
        """Subdivide each hold's body into y samples and pack them into the
        flat arrays one batched `note_offsets` call consumes. Returns the
        concatenated per-sample `cols` / `offs` (engine y_offset) / `beats`
        plus `holds`: (pos, screen_ys, start, count) so the caller can
        split the batched result back per hold.

        The sampled span is the hold's intersection with the padded
        visible `window`: a long hold's body can run many screens, and
        spreading a capped sample count over all of it leaves segments
        longer than the waveform period (a smooth helix aliases into
        zigzag). Fractions interpolate the FULL head..tail range for both
        the engine offset and the FINAL screen y, so clamped samples stay
        exactly on the body's line; a hold entirely outside the window is
        skipped (its rect fallback is clipped away regardless)."""
        win_lo, win_hi = window
        cols_parts, offs_parts, beats_parts, holds = [], [], [], []
        cursor = 0
        for pos in ln_positions:
            span = self._visible_frac_span(head_y[pos], tail_y[pos],
                                           win_lo, win_hi)
            if span is None:
                continue
            f0, f1 = span
            count = self._body_sample_count(
                abs(tail_y[pos] - head_y[pos]) * (f1 - f0), scale)
            frac = np.linspace(f0, f1, count)
            cols_parts.append(np.full(count, cols[pos], dtype=np.int64))
            offs_parts.append(head_off[pos] + frac * (tail_off[pos] - head_off[pos]))
            beats_parts.append(np.full(count, note_beats[pos]))
            screen_ys = head_y[pos] + frac * (tail_y[pos] - head_y[pos])
            holds.append((int(pos), screen_ys, cursor, count))
            cursor += count
        if not holds:
            return {'cols': None, 'offs': None, 'beats': None, 'holds': []}
        return {
            'cols': np.concatenate(cols_parts),
            'offs': np.concatenate(offs_parts),
            'beats': np.concatenate(beats_parts),
            'holds': holds,
        }

    @staticmethod
    def _visible_frac_span(head_y, tail_y, win_lo, win_hi):
        """The [0,1] fraction interval of the head->tail line inside the
        window, or None when the body misses it entirely. Screen y is
        linear in the fraction, so the intersection is a direct clamp."""
        y0, y1 = float(head_y), float(tail_y)
        if y0 == y1:
            return (0.0, 1.0) if win_lo <= y0 <= win_hi else None
        f_a = (win_lo - y0) / (y1 - y0)
        f_b = (win_hi - y0) / (y1 - y0)
        f0 = max(0.0, min(f_a, f_b))
        f1 = min(1.0, max(f_a, f_b))
        if f0 >= f1:
            return None
        return f0, f1

    def _body_sample_count(self, visible_span_px, scale) -> int:
        """Samples for the VISIBLE part of a hold's body, at
        `_BODY_SAMPLE_SPACING` engine px, clamped per hold. At least 2
        (the span endpoints) so every polyline is valid."""
        spacing = _BODY_SAMPLE_SPACING * scale
        return int(np.clip(round(float(visible_span_px) / spacing) + 1,
                           2, _MAX_BODY_SAMPLES))

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
        native downscroll pixels."""
        _rx, ry, _w, h = ctx.chart_rect
        return 2.0 * (ry + h / 2.0) - judge_y

    def _effective_reverse(self, percents, cols, keycount):
        """1 - r_engine: the native candidate space already IS engine
        reverse=1, so engine-default columns (r_engine 0) mirror fully."""
        return 1.0 - reverse_fractions(percents, cols, keycount)

    def _apply_reverse(self, ctx, percents, cols, judge_y):
        centered = float(percents.get('centered', 0.0))
        r = self._effective_reverse(percents, cols, ctx.player.keycount)
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
        so their shift is exactly (receptor_y - judge_y). Runs every frame
        (the zero-channel baseline puts receptors at the mirrored line)."""
        centered = float(percents.get('centered', 0.0))
        r = self._effective_reverse(percents, cols, ctx.player.keycount)
        mirror_y = self._reverse_geom(ctx, judge_y)
        _rx, ry, _w, h = ctx.chart_rect
        center_y = ry + h / 2.0
        receptor_y = judge_y + r * (mirror_y - judge_y)
        receptor_y = receptor_y + centered * (center_y - receptor_y)
        return receptor_y - judge_y
