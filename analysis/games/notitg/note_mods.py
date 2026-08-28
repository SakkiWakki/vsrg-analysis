"""Per-frame consumer: compiled mod channels -> per-note offsets.

`apply(ctx)` runs once per frame after the candidate y arrays exist
(hooked from build_context) and builds THE LANE CURVE for that frame:
`ctx.lane_path`, a `render.lane_path.LanePath` answering
(column, scroll offset) -> where it is / how it is turned / how visible
it is. Everything drawn in a lane is a sample of that one curve, so a
renderer asks it rather than reading a stash shaped for itself
(render/lane_path.py states the layering).

What `apply` then does with it:
- places this frame's candidates on the curve -- replay taps/LNs first,
  then the chart-stream records (mines/lifts/fakes) the renderer appended
  to the same candidate axis, so one batch serves every note kind. Head,
  tail and press are each placed at THEIR OWN offset, and the per-note
  dx / alpha / rotation / zoom / depth / glow land on
  ctx.candidate_dx / _alpha / _rot_deg / _zoom / _z / _rot_x / _rot_y /
  _glow (dx in our pixel space, rotation in degrees, zoom a multiplier),
- stashes ctx.hold_body_samples: candidate-position -> the curve's span
  between that hold's head and tail offsets, so the notes layer draws a
  body BENT by the per-note mods (drunk/wave/digital ...) instead of a
  straight rect. Present only when something varies along the body,
- stashes ctx.receptor_alpha, the receptors' OWN visibility. Their
  position is not stashed at all: a receptor is the curve at offset 0,
  and the engine draws it through the same GetXPos an arrow takes, so the
  drunk/tornado shift and confusion spin come out of the curve. Their
  visibility does not - it is the dark family alone, and the stealth
  gradients an arrow at that point picks up never apply to a receptor,
- stashes ctx.arrowpath_ribbons: the curve over the visible scroll range
  per column, which is what the fork's arrowpath trail IS.

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

Double-apply guard (field-3D vs 2D foreshortening):
- confusionx/confusionxoffset, confusiony/confusionyoffset and hallway are
  the 2D foreshortening APPROXIMATIONS of an out-of-plane field tilt
  (arrow_effects reprojects an X/Y field rotation into per-note zoom / dx).
  When the REAL 3D field projection owns the tilt - the field_3d effect,
  fed by BOTH producers: the recorded actor rotation_x/rotation_y pokes
  AND the scalar confusionx/y mod channels themselves
  (field_projection.FieldTilt) - those same axes must not also be
  approximated in 2D, or the tilt applies twice. `field_tilt_active(t)`,
  when supplied, reports whether the projection owns the X/Y tilt at t;
  while it does, this consumer zeroes the X/Y-tilt confusion channels.
  The 2D kernels remain live exactly where the projection cannot render:
  per-column numbered variants (confusionx0.., per-note content), the
  base-hidden deferral (copies own a flat capture), and fields whose
  instances own the transform (see adapter._field_3d_for). The Z-spin
  confusion (confusion/confusionoffset) is UNTOUCHED - it is an in-plane
  per-note receptor spin, a distinct mechanism that legitimately coexists
  (gat drives it t~58-74 with the actor 3D channels at rest).

Tiny X-spacing (consumer-side):
- tiny's sprite zoom (pow(0.5,tiny)) rides candidate_zoom from
  arrow_effects, but its X-spacing compression (GetXPos :1025 multiplies
  the whole x offset incl. the column term by min(pow(0.5,tiny),1)) is
  applied HERE, in `_tiny_compressed_dx`: arrow_effects.dx carries only
  the mods (the note layer adds lane_x separately), so this consumer folds
  in the column-term compression that pulls columns toward field center.

Simplifications (documented, revisit with the oracle):
- beat(t) inverts the BPM segments only (stops/warps shift beat_now
  slightly during those regions; note beats come exactly from rows).
- one consumer samples ONE player's channels (`player`, default 0). A
  dual-player NotITG chart builds a second consumer at player 1 whose
  field the renderer draws as a separate capture (see field_instances
  NotitgDualField); the two share the chart and candidate set but sample
  disjoint (mod, player) channels, so gat's per-side mods diverge.
- expand keyed to song time not wall clock (scrub-exactness).
"""
from __future__ import annotations

import numpy as np

from analysis.games.etterna.sm_chart import beat_to_time
from analysis.player.render.lane_path import LaneDisplacement, LanePath
from analysis.player.render.mods.arrow_effects import (
    ARROW_SIZE, accel_y_offset, column_offsets, note_offsets,
    receptor_dark_alpha, reverse_fractions, tiny_spacing, waveform_z_zoom)

_ACTIVE_EPS = 1e-4

def _stream_candidates(ctx):
    """This frame's chart-stream candidate indices, or None when the
    frame carries none (narrow test ctxs never set the attribute)."""
    s_idx = getattr(ctx, 'stream_candidates', None)
    if s_idx is None or not len(s_idx):
        return None
    return np.asarray(s_idx, dtype=np.int64)

def _lane_center_of(ctx):
    """The lane curve's undisplaced x per column. A render context knows
    its own lane geometry; a narrow one carrying only a lane width still
    gets a consistent axis, which is all the offsets derived from it
    need."""
    center = getattr(ctx, 'lane_center', None)
    if center is not None:
        return center
    lane_x = getattr(ctx, 'lane_x', None)
    half = float(ctx.lane_w) / 2.0
    if lane_x is not None:
        return lambda col: lane_x(col) + half
    return lambda col: col * float(ctx.lane_w) + half

def _stream_count(ctx) -> int:
    s_idx = _stream_candidates(ctx)
    return 0 if s_idx is None else len(s_idx)

# The 2D per-note foreshortening approximations of an out-of-plane FIELD
# tilt (X/Y rotation). Deferred to the real 3D projection while it drives
# the same axes (see the module doc). confusion/confusionoffset (the Z
# receptor spin) is deliberately absent: it is in-plane and coexists.
_FIELD_TILT_CONFUSION = ('confusionx', 'confusionxoffset',
                         'confusiony', 'confusionyoffset', 'hallway')
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
# Sub-samples box-filtered into each body point. Waveforms whose spatial
# period sits below the sample spacing (negative period companions push
# it to ~2 engine px) cannot be resolved as geometry; unfiltered they
# alias into single-sample reversals that the stroker rounds into lobes.
# The engine draws strips every few px, so at those frequencies its
# visual converges to the local mean band - which is what the filter
# yields.
_BODY_BOX_FILTER = 4
# A waveform kernel is zeroed for span evaluation below this many sample
# spacings of wavelength - safely above Nyquist.
_MIN_RESOLVED_WAVELENGTHS = 3.0
# Arrowpath trail sampling (ReceptorArrowRow::DrawArrowPath @0x53b390:
# the fork draws each column's future note path when the `arrowpath`
# mod is on). Spacing along the scroll axis in engine px, per-column
# sample cap, and the stroke width. The ArrowPathWidth/ArrowPathGirth
# mods exist in the fork but their units are unpinned (COMDAT-folded
# draw site), so the width stays this constant until a reference frame
# fixes it
_ARROWPATH_SAMPLE_SPACING = 16.0
_ARROWPATH_MAX_SAMPLES = 96
_ARROWPATH_WIDTH = 1.0 # Perfect size

# Waveform kernel -> its period companion (wavelength ~ 2*AS*(1+period)).
_WAVEFORM_PERIODS = {
    'digital': 'digitalperiod', 'zigzag': 'zigzagperiod',
    'sawtooth': 'sawtoothperiod', 'square': 'squareperiod',
    'bounce': 'bounceperiod', 'digitalz': 'digitalzperiod',
    'zigzagz': 'zigzagzperiod', 'bouncez': 'bouncezperiod',
    'tandigital': 'tandigitalperiod',
}

def beat_segments(bpms) -> list:
    """(t_start, beat_start, bps) rows for the positive-bpm segments,
    seconds-keyed; an empty/degenerate table falls back to one 120bpm
    segment. Shared beat clock for every seconds-keyed NotITG consumer
    (this module's kernels, field_projection's confusion tilt)."""
    segments = [(beat_to_time(beat, bpms, 0.0), beat, bpm / 60.0)
                for beat, bpm in sorted(bpms) if bpm > 0]
    return segments or [(0.0, 0.0, 2.0)]

def beat_segment_at(segments, t: float) -> tuple:
    """The (t_start, beat_start, bps) segment governing time `t`."""
    lo = segments[0]
    for seg in segments:
        if seg[0] > t:
            break
        lo = seg
    return lo

def beat_at(segments, t: float) -> float:
    """Song beat at time `t` under `beat_segments(bpms)`."""
    t0, beat0, bps = beat_segment_at(segments, t)
    return beat0 + max(0.0, t - t0) * bps

class NotitgNoteMods:
    def __init__(self, channels, bpms, field_tilt_active=None, player=0,
                 note_path=None):
        self._channels = channels
        self._field_tilt_active = field_tilt_active
        self._player = int(player)
        self._segments = beat_segments(bpms)
        # The compiled note-path handle (SetXSpline family): per-column
        # spline displacement every consumer of this pipeline inherits -
        # heads, hold-body strips, receptors, the arrowpath ribbon.
        self._note_path = note_path

    def _beat_at(self, t: float) -> float:
        return beat_at(self._segments, t)

    def _segment_at(self, t: float) -> tuple:
        return beat_segment_at(self._segments, t)

    def _px_per_engine(self, ctx, t: float, fallback: float) -> float:
        """Screen px per engine y_offset px at `t`. The engine maps
        offsets from beats (y_off = beat_diff * 64 * xmod), our display
        maps from time and the effective scroll rate; kernels must see
        ENGINE offsets so mod geometry (waveform wavelengths, accel
        shapes) is chart-faithful and invariant to the user's scroll
        setting. ctx.scroll_speed already contains the chart multiplier
        (effective_scroll_speed), so the ratio is speed / (64*bps*xmod).
        Falls back to the sprite scale when the context carries no
        scroll rate (narrow test ctxs) or the rate degenerates
        (xmod ~ 0 pause sections)."""
        pps = getattr(ctx, 'scroll_speed', None)
        if not pps or pps <= 0.0:
            return fallback
        _t0, _b0, bps = self._segment_at(t)
        timeline = getattr(ctx.player, '_scroll_mult_timeline', None)
        mult = timeline.sample(t)[0] if timeline is not None else 1.0
        engine_rate = ARROW_SIZE * bps * mult
        if engine_rate <= 1e-6:
            return fallback
        return pps / engine_rate

    def apply(self, ctx) -> None:
        t = float(ctx.t_now)
        percents = self._with_expand_phase(
            self._channels.values_at(t, self._player), t)
        percents = self._defer_field_tilt(percents, t)
        scale = ctx.lane_w / ARROW_SIZE
        judge_y = float(ctx.judge_y)

        # The per-column note-path spline (SetXSpline family), or None
        # while inert - the zero-cost fast path. Stashed for the
        # arrowpath ribbon, which traces the same displaced path.
        spline = (self._note_path.sampler_at(t, self._player)
                  if self._note_path is not None else None)
        ctx.note_path_spline = spline

        ppe = self._px_per_engine(ctx, t, scale)
        active = (spline is not None
                  or any(abs(v) >= _ACTIVE_EPS for v in percents.values()))
        path = self._lane_path(ctx, percents, scale, ppe, judge_y, t, spline,
                               active)
        ctx.lane_path = path
        if len(ctx.candidates) or _stream_count(ctx):
            self._apply_to_notes(ctx, path, percents, scale, ppe, judge_y, t,
                                 spline, active)
        # A receptor is the curve at offset 0, so the consumers read it
        # straight off `ctx.lane_path` and only its VISIBILITY - which is
        # not the lane's, see `receptor_dark_alpha` - is stashed here.
        ctx.receptor_alpha = receptor_dark_alpha(
            percents, np.arange(int(ctx.player.keycount), dtype=np.int64))
        self._stash_arrowpath(ctx, path, percents, scale, ppe, t)

    def _lane_path(self, ctx, percents, scale, ppe, judge_y, t, spline,
                   active) -> LanePath:
        """This frame's lane curve - the one place where
        (column, scroll offset) -> where it is / how it is turned / how
        visible it is gets answered, for receptors, heads, hold bodies,
        tail caps and travel paths alike.

        An inert frame builds the same object with NO displacement hook, so
        every consumer's straight-lane fast path (a hold body as one rect
        rather than a ribbon) stays on, and the kernels never run."""
        keycount = int(ctx.player.keycount)
        beat_now = self._beat_at(t)

        def displace(cols, offsets, note_beats, cell) -> LaneDisplacement:
            offs = self._evaluate(percents, cols, offsets, note_beats, cell,
                                  spline, t, beat_now, keycount)
            return self._displacement(percents, cols, offs, scale, keycount)

        return LanePath(
            _lane_center_of(ctx),
            lambda cols, offsets: self._axis_y(ctx, percents, judge_y, ppe,
                                               cols, offsets),
            displace if active else None,
            lambda offsets: accel_y_offset(percents, offsets))

    def _evaluate(self, percents, cols, offsets, note_beats, cell, spline,
                  t, beat_now, keycount):
        """The engine's per-note pipeline at (column, ENGINE y_offset,
        beat). Every consumer's answer comes from this one call, including
        the terms the lane curve does not carry (a note's own glow)."""
        offs = note_offsets(self._band_limited(percents, cell), cols, offsets,
                            t_now=t, beat_now=beat_now, keycount=keycount,
                            note_beats=note_beats, project_3d=True)
        # The note-path spline (SetXSpline family) adds onto the summed
        # mods in engine px, BEFORE tiny's whole-offset compression
        # (GetXPos adds spline and mods into one x the multiplier scales).
        if spline is not None:
            np.add(offs.dx, spline.offsets('x', cols, offsets), out=offs.dx)
            np.add(offs.z, spline.offsets('z', cols, offsets), out=offs.z)
        return offs

    def _displacement(self, percents, cols, offs, scale,
                      keycount) -> LaneDisplacement:
        """The evaluated pipeline as the lane's bend, in OUR pixel space.

        `z` stays in ENGINE px - the field projection's design space, and
        its camera divide is scale-free. A drawer that cannot place a
        sample at a depth takes `flat_zoom`, which is that same push
        reprojected to the center-plane scale the 2D path used to fake."""
        return LaneDisplacement(
            dx=self._tiny_compressed_dx(percents, cols, offs.dx, keycount,
                                        scale),
            dy=offs.dy * scale,
            z=offs.z,
            rotation_deg=offs.rotation_deg,
            rotation_x_deg=offs.rot_x,
            rotation_y_deg=offs.rot_y,
            zoom=offs.zoom,
            flat_zoom=offs.zoom * waveform_z_zoom(offs.z),
            alpha=offs.alpha_mult)

    def _axis_y(self, ctx, percents, judge_y, ppe, cols, offsets):
        """Screen y of the UNDISPLACED scroll axis at accel-remapped
        `offsets`, per column: `ppe` converts engine offset px to screen
        px (see `_px_per_engine`), then the reverse family places each
        column's own receptor line."""
        reverse = self._effective_reverse(percents, cols, ctx.player.keycount)
        return self._reverse_ys(ctx, judge_y - offsets * ppe, reverse,
                                float(percents.get('centered', 0.0)), judge_y)

    def _reverse_ys(self, ctx, ys, r, centered, judge_y):
        """The reverse-family remap: y' places the receptor at
        lerp(judge_y, mirror_y, r) and flips the note's distance below it
        by (1 - 2r). `centered` slides that receptor toward field center
        (mid-screen) by its percent."""
        mirror_y = self._reverse_geom(ctx, judge_y)
        _rx, ry, _w, h = ctx.chart_rect
        center_y = ry + h / 2.0
        receptor_y = judge_y + r * (mirror_y - judge_y)
        receptor_y = receptor_y + centered * (center_y - receptor_y)
        return receptor_y + (1.0 - 2.0 * r) * (np.asarray(ys) - judge_y)

    def _apply_to_notes(self, ctx, path, percents, scale, ppe, judge_y, t,
                        spline, active) -> None:
        """Place this frame's candidates on the lane curve.

        Head, tail and press are three positions on the same axis, so each
        is placed at ITS OWN offset - the tail bends by the displacement
        where the tail is, not by the head's, which is what keeps a hold's
        cap attached to the body span the curve draws between them.

        The whole candidate set goes through ONE evaluation: the pipeline's
        cost is per call, not per row, and the note pipeline needs terms
        the curve does not carry (a note's own glow)."""
        p = ctx.player
        idx = np.asarray(ctx.candidates, dtype=np.int64)
        cols, note_rows = self._candidate_cols_rows(ctx, p, idx)
        self._pin_held_holds(ctx, idx, judge_y, t)

        # ACCEL FAMILY (boost/brake/wave/expand) reshapes the raw scroll
        # y_offset -> position mapping BEFORE any displacement; the REVERSE
        # family then places each column's own receptor line and flips the
        # note's distance from it (`_axis_y`). Reverse runs every frame:
        # the zero-channel baseline is a full mirror to engine-default
        # upscroll (see module doc).
        raw = [self._raw_offsets(getattr(ctx, name), judge_y, ppe)
               for name in ('candidate_head_y', 'candidate_tail_y',
                            'candidate_press_y')]
        offsets = [accel_y_offset(percents, part) for part in raw]
        axis = [self._axis_y(ctx, percents, judge_y, ppe, cols, part)
                for part in offsets]
        if not active:
            (ctx.candidate_head_y, ctx.candidate_tail_y,
             ctx.candidate_press_y) = axis
            return

        n = len(cols)
        rows = np.tile(cols, 3)
        offs = self._evaluate(percents, rows, np.concatenate(offsets),
                              np.tile(note_rows / 48.0, 3), 0.0, spline, t,
                              self._beat_at(t), p.keycount)
        bend = self._displacement(percents, rows, offs, scale, p.keycount)
        head, tail, press = (slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n))
        ctx.candidate_head_y = axis[0] + bend.dy[head]
        ctx.candidate_tail_y = axis[1] + bend.dy[tail]
        ctx.candidate_press_y = axis[2] + bend.dy[press]
        ctx.candidate_dx = bend.dx[head]
        ctx.candidate_alpha = offs.alpha_mult[head]
        ctx.candidate_stream_alpha = (None if offs.stream_alpha is None
                                      else offs.stream_alpha[head])
        ctx.candidate_rot_deg = offs.rotation_deg[head]
        ctx.candidate_zoom = offs.zoom[head]
        # Per-note 3D: real depth (engine px) + out-of-plane tilt (deg).
        # The note draw projects the quad through the field camera; when
        # every note rests flat (z=0, no tilt) the draw keeps its 2D path.
        ctx.candidate_z = offs.z[head]
        ctx.candidate_rot_x = offs.rot_x[head]
        ctx.candidate_rot_y = offs.rot_y[head]
        ctx.candidate_glow = None if offs.glow is None else offs.glow[head]
        ctx.candidate_glow_rgb = (None if offs.glow_rgb is None
                                  else offs.glow_rgb[head])

        self._stash_hold_body_samples(ctx, path, idx, cols, raw[0], raw[1])

    @staticmethod
    def _raw_offsets(ys, judge_y, ppe):
        """The pre-accel scroll offset (engine px) a candidate y array sits
        at - the scroll space the lane curve is asked in."""
        return (judge_y - np.asarray(ys, dtype=np.float64)) / ppe

    def _candidate_cols_rows(self, ctx, p, idx):
        """Columns + beat rows over the FULL candidate axis: replay
        candidates first, then the chart-stream records the renderer
        appended (see qt_renderer._append_stream_candidate_ys). One
        note_offsets batch then serves taps, LNs, and streams -- a mine
        picks up exactly the displacement/visibility its column's taps
        do. Streams without beat rows carry -1 (row-driven kernels are
        NotITG-only, and NotITG streams always have rows)."""
        cols = np.asarray(p.columns, dtype=np.int64)[idx]
        rows = np.asarray(p.notes.noterows_list, dtype=np.float64)[idx]
        s_idx = _stream_candidates(ctx)
        if s_idx is None:
            return cols, rows
        n = p.notes
        cols = np.concatenate([cols, n.stream_cols[s_idx].astype(np.int64)])
        rows = np.concatenate([rows, n.stream_rows[s_idx].astype(np.float64)])
        return cols, rows

    def _pin_held_holds(self, ctx, idx, judge_y, t) -> None:
        """While a hold is held the engine draws its head AT the receptor
        and the body only from there to the tail; the stretch already
        scrolled past is consumed. Clamp the head y (and its press-mark
        twin, which rides the head at autoplay's zero offsets) to the
        judge line in native downscroll space, BEFORE the accel remap:
        the pinned head then sits at y_offset 0 for the whole pipeline,
        which is every accel kernel's fixpoint, lands on the (possibly
        split / centered / animated) receptor after the reverse remap,
        and starts the body sampler's polyline at the receptor instead
        of past it. Missed holds keep falling (NaN offsets and non-LN
        tails drop out of the mask via NaN comparisons)."""
        p = ctx.player
        ln_tail_times = getattr(p.notes, 'ln_tail_times', None)
        if ln_tail_times is None:
            return
        tail_t = np.asarray(ln_tail_times)[idx]
        press_t = p.times[idx] + p.offsets[idx]
        held = ((press_t <= t) & (t <= tail_t)
                & ~np.asarray(p.misses)[idx])
        if not held.any():
            return
        # The candidate axis extends past the replay candidates with
        # chart-stream records, which are never held.
        pad = len(ctx.candidate_head_y) - held.shape[0]
        if pad:
            held = np.concatenate([held, np.zeros(pad, dtype=bool)])
        ctx.candidate_head_y = np.where(
            held, np.minimum(ctx.candidate_head_y, judge_y),
            ctx.candidate_head_y)
        ctx.candidate_press_y = np.where(
            held, np.minimum(ctx.candidate_press_y, judge_y),
            ctx.candidate_press_y)

    def _tiny_compressed_dx(self, percents, cols, dx_engine, keycount, scale):
        """Screen-space per-note dx with tiny's X-spacing compression folded
        in. The engine (GetXPos :1025) multiplies the WHOLE x offset - the
        summed x-mods AND the column's own x-offset from field center - by
        min(pow(0.5,tiny),1). Our `dx` carries only the mods (the note layer
        adds lane_x = field_center + column_offset*scale separately), so we
        emit dx' such that lane_x + dx' reproduces the engine total:
            dx'_engine = spacing*dx + (spacing-1)*column_offset.
        The (spacing-1)*column_offset term pulls the lane toward field center
        (tighter spacing); spacing*dx compresses the mod amplitude with it."""
        dx = np.asarray(dx_engine, dtype=np.float64) * scale
        spacing = tiny_spacing(float(percents.get('tiny', 0.0)))
        if spacing == 1.0:
            return dx
        column = column_offsets(keycount)[cols] * scale
        return spacing * dx + (spacing - 1.0) * column

    def _stash_hold_body_samples(self, ctx, path, idx, cols, head_off,
                                 tail_off) -> None:
        """Sample each visible hold's body as the span of the lane curve
        between its head and tail offsets, so the notes layer draws a
        BENT polyline instead of a straight head-to-tail rect (drunk /
        wave / digital etc. displace every strip of an engine-rendered
        hold body, not just the head).

        Every hold's samples go through ONE `spans` call - each sample is
        just another row of the same evaluation - and the endpoints land
        exactly on the head and tail the candidate pass placed, because
        both are the same curve asked at the same offsets.

        A body only needs its own polyline when something actually VARIES
        along it: a bending x, a per-strip visibility gradient (the engine
        evaluates visibility per drawn part, so a hold's body stays up
        while its head blanks), or a depth push that tilts it out of the
        receptor plane. Anything flat, opaque and in-plane stays on the
        straight rect fast path."""
        p = ctx.player
        ln_tail_times = getattr(p.notes, 'ln_tail_times', None)
        if ln_tail_times is None:
            return
        tail_off = np.asarray(tail_off, dtype=np.float64)
        # Restrict to the replay prefix of the candidate axis: only
        # replay candidates can be LNs (stream span ends ride the same
        # tail array but draw through the stream span path).
        is_ln = (np.isfinite(np.asarray(ln_tail_times)[idx])
                 & np.isfinite(tail_off[:len(idx)]))
        if not is_ln.any():
            return

        head_y = np.asarray(ctx.candidate_head_y, dtype=np.float64)
        tail_y = np.asarray(ctx.candidate_tail_y, dtype=np.float64)
        note_beats = np.asarray(p.notes.noterows_list, dtype=np.float64)[idx] / 48.0
        _rx, ry, _w, h = ctx.chart_rect
        window = (ry - h * _BODY_WINDOW_PAD, ry + h * (1.0 + _BODY_WINDOW_PAD))
        spans = self._body_spans(np.nonzero(is_ln)[0], cols, head_off,
                                 tail_off, head_y, tail_y, note_beats, window)
        if not spans['positions']:
            return

        bodies = path.spans(spans['cols'], spans['starts'], spans['ends'],
                            spans['counts'], spans['beats'],
                            cell=_BODY_SAMPLE_SPACING, taps=_BODY_BOX_FILTER)
        samples = {pos: body for pos, body in zip(spans['positions'], bodies)
                   if not (np.ptp(body.x) < _ACTIVE_EPS
                           and np.ptp(body.alpha) < _ACTIVE_EPS
                           and np.ptp(body.z) < _ACTIVE_EPS)}
        if samples:
            ctx.hold_body_samples = samples

    def _body_spans(self, ln_positions, cols, head_off, tail_off, head_y,
                    tail_y, note_beats, window) -> dict:
        """The visible stretch of each hold's body, as the arrays
        `LanePath.spans` consumes.

        A long hold's body can run many screens, and spreading a capped
        sample count over all of it leaves segments longer than the
        waveform period (a smooth helix aliases into zigzag), so the span
        is clipped to the padded visible window first. The clip is done on
        the head..tail FRACTION, which carries straight back to the engine
        offsets the curve is asked in; a hold entirely outside the window
        is skipped (its rect fallback is clipped away regardless)."""
        win_lo, win_hi = window
        spans = {key: [] for key in
                 ('positions', 'cols', 'starts', 'ends', 'counts', 'beats')}
        for pos in ln_positions:
            visible = self._visible_frac_span(head_y[pos], tail_y[pos],
                                              win_lo, win_hi)
            if visible is None:
                continue
            f0, f1 = visible
            span = tail_off[pos] - head_off[pos]
            spans['positions'].append(int(pos))
            spans['cols'].append(int(cols[pos]))
            spans['starts'].append(head_off[pos] + f0 * span)
            spans['ends'].append(head_off[pos] + f1 * span)
            spans['counts'].append(self._body_sample_count(
                abs(tail_off[pos] - head_off[pos]) * (f1 - f0)))
            spans['beats'].append(note_beats[pos])
        return spans

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

    @staticmethod
    def _body_sample_count(visible_span) -> int:
        """Samples for the VISIBLE part of a hold's body, one per
        `_BODY_SAMPLE_SPACING` engine px, clamped per hold. At least 2
        (the span endpoints) so every polyline is valid."""
        return int(np.clip(
            round(float(visible_span) / _BODY_SAMPLE_SPACING) + 1,
            2, _MAX_BODY_SAMPLES))

    def _band_limited(self, percents, cell):
        """The percents to evaluate at a sample spacing of `cell` engine
        px, with waveform kernels that spacing cannot resolve zeroed.

        A waveform's spatial wavelength is ~2*AS*(1+period); when a
        negative period companion pushes it under the cutoff, point
        sampling aliases (near-resonance leaves low-frequency ghost curves
        no smoothing can remove), while the engine's per-strip rendering
        converges to the straight mean band - exactly what a zeroed kernel
        draws. `cell` 0 is a point query (a head, a receptor), which
        cannot alias and keeps the full percents."""
        if cell <= 0.0:
            return percents
        limited = None
        for mod, period_mod in _WAVEFORM_PERIODS.items():
            if abs(percents.get(mod, 0.0)) < _ACTIVE_EPS:
                continue
            wavelength = 2.0 * ARROW_SIZE * (
                1.0 + percents.get(period_mod, 0.0))
            if wavelength >= _MIN_RESOLVED_WAVELENGTHS * cell:
                continue
            if limited is None:
                limited = dict(percents)
            limited[mod] = 0.0
        return percents if limited is None else limited

    def _defer_field_tilt(self, percents, t):
        """Zero the X/Y-tilt confusion channels while the real 3D field
        projection owns the same axes - from either producer, actor pokes
        or the scalar confusion mods themselves (see the module doc).
        No-op when no field-3D predicate is wired or when it is inactive
        - the common case, so the returned dict is the input untouched
        unless a tilt channel is actually present AND the projection is
        live."""
        if self._field_tilt_active is None or not self._field_tilt_active(t):
            return percents
        zeroed = {mod: 0.0 for mod in _FIELD_TILT_CONFUSION
                  if abs(percents.get(mod, 0.0)) >= _ACTIVE_EPS}
        return {**percents, **zeroed} if zeroed else percents

    def _with_expand_phase(self, percents, t):
        """Inject the expand periodic phase (song time for scrub-exactness;
        the engine uses a wall-clock timer, ArrowEffects.cpp:126-131)."""
        if not percents.get('expand', 0.0):
            return percents
        out = dict(percents)
        out['_expand_phase'] = t * _EXPAND_RATE
        return out

    def _reverse_geom(self, ctx, judge_y):
        """The position a fully-reversed receptor slides to, in native
        downscroll pixels: the adapter's engine mirror line when its
        field geometry is active (the engine's standard/reverse receptor
        rows are NOT symmetric about the screen center), else the judge
        line reflected about the chart region's vertical center."""
        field_geometry = getattr(ctx.player, '_field_geometry', None)
        geom = field_geometry() if field_geometry is not None else None
        if geom is not None:
            return geom[3]
        _rx, ry, _w, h = ctx.chart_rect
        return 2.0 * (ry + h / 2.0) - judge_y

    def _effective_reverse(self, percents, cols, keycount):
        """1 - r_engine: the native candidate space already IS engine
        reverse=1, so engine-default columns (r_engine 0) mirror fully."""
        return 1.0 - reverse_fractions(percents, cols, keycount)

    def _stash_arrowpath(self, ctx, path, percents, scale, ppe,
                         t) -> None:
        """The fork's arrowpath trails: while the `arrowpath` mod is on,
        one polyline per column tracing where that column's notes will
        travel. It is the lane curve over the visible scroll range and
        nothing else, so the trail sits exactly under the arrows by
        construction. Stashed as ctx.arrowpath_ribbons: (xs, ys, gradient
        stops, width px, alpha) per column, in OUR pixel space; the field
        layer strokes them under the receptors. The trail stays a 2D
        stroke - the x displacement carries the visible helix, and notes
        and bodies keep the real depth."""
        amount = min(1.0, abs(float(percents.get('arrowpath', 0.0))))
        if amount < _ACTIVE_EPS:
            ctx.arrowpath_ribbons = None
            return
        keycount = ctx.player.keycount
        _rx, _ry, _w, h = ctx.chart_rect
        max_off = h * (1.0 + 2.0 * _BODY_WINDOW_PAD) / max(ppe, 1e-6)
        count = int(np.clip(round(max_off / _ARROWPATH_SAMPLE_SPACING) + 1,
                            2, _ARROWPATH_MAX_SAMPLES))
        cols = np.arange(keycount, dtype=np.int64)
        trails = path.spans(cols, np.zeros(keycount),
                            np.full(keycount, max_off),
                            np.full(keycount, count),
                            cell=_ARROWPATH_SAMPLE_SPACING)
        note_path = (self._note_path.player(self._player)
                     if self._note_path is not None else None)
        width = _ARROWPATH_WIDTH * scale
        ctx.arrowpath_ribbons = [
            (trail.x, trail.y,
             note_path.gradient_at(t, col) if note_path is not None
             else [(1.0, 1.0, 1.0, 1.0)],
             width, amount)
            for col, trail in enumerate(trails)]

