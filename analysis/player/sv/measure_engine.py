"""Measure-based SV engines: thin wrappers around the integrator that
satisfy the existing `SVEngine` Protocol (engine.py).

Per DESIGN.tex §9.3, both time-space and beat-space engines run the same
integrator on the same SVDocument; they differ only in which measure the
document encodes. Construction helpers here map game-specific inputs to
SVDocument and return a wrapped engine.
"""
from __future__ import annotations

import numpy as np

from analysis.player.sv.document import SVData, SVDocument, TimingMeasure
from analysis.player.sv.integrate import CumulativeIntegrator
from analysis.player.sv.scrolls import ScrollsCache
from analysis.player.sv.speeds import SpeedsEvaluator
from analysis.player.sv.timing import TimingMap


class MeasureSVEngine:
    """Wraps a CumulativeIntegrator with the existing SVEngine surface.

    The engine carries the document and an optional render-time zoom
    function `z(t)`. All cull-space operations go through the integrator;
    `distance` applies z(a) at the playhead per the cumulative decomposition.
    """

    def __init__(self, document: SVDocument, sec_per_base_beat: float = 1.0,
                 inverse_t_from_cum=None,
                 scrolls_cache: 'ScrollsCache | None' = None):
        self._doc = document
        self._integrator = CumulativeIntegrator(document)
        # sec_per_base_beat scales beat-space cumulative into render seconds
        # (beat-space dB has units of beats; multiply by 60/BPM_0 for time
        # units that match the player's px/sec contract).
        self._scale = float(sec_per_base_beat)
        # Optional override for inverse_cumulative_at: beat-space engines
        # need to thread the inverse through the timing map / scroll-cache,
        # which the generic integrator can't do (its own inverse is exact
        # for time-space and good-enough for beat-space without scroll<=0
        # plateaus, but the reference engine's plateau handling is better).
        self._inverse_override = inverse_t_from_cum
        # Beat-space engines hand us their ScrollsCache so project_beats
        # can bypass beat->time->beat round-tripping for chart-stream
        # sprites in old negative-BPM warp aliases.
        self._scrolls_cache = scrolls_cache
        self.enabled = bool(document.enabled)

    # ----- SVEngine Protocol -----

    def cumulative_at(self, t: float) -> float:
        return self._integrator.cumulative_at(t) * self._scale

    def project_times(self, times: np.ndarray) -> np.ndarray:
        return self._integrator.project_times(times) * self._scale

    def __getattr__(self, name):
        # Expose `project_beats` only on beat-space engines (those built
        # with a ScrollsCache). render.py uses hasattr() to decide which
        # path to take; raising AttributeError here keeps the time-space
        # path correct for those engines.
        if name == 'project_beats' and self._scrolls_cache is not None:
            scale = self._scale
            cache = self._scrolls_cache

            def project_beats(beats):
                return cache.displayed_beat_array(beats) * scale
            return project_beats
        raise AttributeError(name)

    def cumulative_velocity_at(self, t: float) -> float:
        return self._integrator.velocity_at(t) * self._scale

    def inverse_cumulative_at(self, sv: float) -> float:
        if self._inverse_override is not None:
            return self._inverse_override(sv / self._scale if self._scale else sv)
        return self._integrator.inverse_at(sv / self._scale if self._scale else sv)

    def distance(self, t_from: float, t_to: float) -> float:
        d = self.cumulative_at(t_to) - self.cumulative_at(t_from)
        z = self._doc.zoom_fn
        return d * (z(t_from) if z is not None else 1.0)

    def render_multiplier_at(self, t: float) -> float:
        z = self._doc.zoom_fn
        return float(z(t)) if z is not None else 1.0

    def as_sections(self) -> list[tuple[float, float]]:
        v = self._doc.data
        if v.empty:
            return []
        return [(float(t), float(m)) for t, m in zip(v.times, v.multipliers)]

    def debug_snapshot_at(self, t: float) -> dict:
        return {
            'engine': 'measure',
            't': float(t),
            'cumulative': self.cumulative_at(t),
            'render_multiplier': self.render_multiplier_at(t),
            'cumulative_velocity': self.cumulative_velocity_at(t),
        }

    def max_visible_t_from(self, song_t: float) -> float:
        return float('inf')


# ---------------------------------------------------------------------------
# Time-space construction (osu!mania, Quaver-without-coupling)
# ---------------------------------------------------------------------------


def time_space_engine(sections, t_start: float = 0.0,
                      t_end: float = 1e9) -> MeasureSVEngine:
    """Build a time-space engine from the legacy [(time, multiplier)] shape.

    Encodes mu = Lebesgue measure on [t_start, t_end], v = the section
    multipliers as a piecewise-constant integrand. Equivalent to
    TimeSpaceSVEngine for purposes of cumulative_at / distance.
    """
    measure = TimingMeasure.lebesgue(t_start, t_end)
    data = SVData.from_sections(sections)
    doc = SVDocument(measure=measure, data=data,
                     zoom_fn=None, enabled=bool(sections))
    return MeasureSVEngine(doc, sec_per_base_beat=1.0)


# ---------------------------------------------------------------------------
# Beat-space construction (Etterna XMOD)
# ---------------------------------------------------------------------------


def beat_space_engine(scrolls, speeds, bpms, sm_offset,
                      stops=None, delays=None, warps=None) -> MeasureSVEngine:
    """Build a beat-space engine matching BeatSpaceSVEngine's contract.

    The measure mu is dB:
      AC density rho(tau) = bpm(tau)/60, with rho = 0 inside STOP/DELAY
      windows where chart beat is frozen.
      Atoms at each warp tau_w with mass Delta_b (the warp's beat extent);
      the integrator multiplies by v(tau_w^+) = SCROLLS ratio at the warp's
      landing beat to produce the displayed-beat jump per DESIGN.tex §5.2.

    The integrand v(tau) = s(B(tau)) is piecewise-constant in chart-time:
    on each [t_i, t_{i+1}) bracketing a SCROLLS row, v is the row's ratio.
    """
    bpms = list(bpms or [(0.0, 120.0)])
    stops = list(stops or [])
    delays = list(delays or [])
    warps = list(warps or [])
    delay_at_beats = {float(b): float(v) for b, v in delays}

    base_bpm = float(bpms[0][1]) if bpms else 120.0
    sec_per_base_beat = 60.0 / base_bpm

    timing = TimingMap(bpms, sm_offset, stops, delays, warps)
    speeds_eval = SpeedsEvaluator(speeds, timing, delay_at_beats=delay_at_beats)
    scrolls_cache = ScrollsCache(scrolls)

    # AC measure: rho(tau) = bps_at_time(tau). Build boundaries as the
    # union of every event's (time_enter, time_exit) so the density is
    # piecewise-constant on each interval -- bps_at_time returns 0 inside
    # STOP/DELAY windows and the active bps elsewhere, exactly the AC part
    # of dB per DESIGN.tex §1.3.
    boundary_set = {0.0}
    for te, tx in zip(timing._time_enter, timing._time_exit):
        boundary_set.add(float(te))
        boundary_set.add(float(tx))
    last_t = max(boundary_set) if boundary_set else 0.0
    far_end = max(last_t + 600.0, 1e6)
    boundary_set.add(far_end)
    boundaries = np.array(sorted(boundary_set), dtype=np.float64)

    if boundaries.size >= 2:
        mids = 0.5 * (boundaries[:-1] + boundaries[1:])
        densities = np.array([timing.bps_at_time(float(m)) for m in mids],
                             dtype=np.float64)
    else:
        densities = np.zeros(0, dtype=np.float64)

    # Atoms at warps. Mass is Delta_b (the warp's beat extent); atom_times
    # are the chart-time at which the warp actually fires -- which on a
    # same-row stop+warp is the POST-stop time, not beat_to_time(beat).
    # Walk the timing map's prewalked events to grab the warp event's own
    # time_enter (== time_exit since warps don't elapse time).
    atoms = []
    warp_beats = {float(b): float(d) for b, d in warps if d > 0}
    for j, kind in enumerate(timing._event_kind):
        if kind != 3:   # 3 == WARP
            continue
        b_enter = timing._beat_enter[j]
        delta_b = warp_beats.get(b_enter)
        if delta_b is None or delta_b <= 0:
            continue
        atoms.append((float(timing._time_enter[j]), delta_b))

    measure = TimingMeasure.from_timing_map(boundaries, densities, atoms)

    # Integrand v(tau) = SCROLLS ratio at beat(tau). Build piecewise-
    # constant records at each scroll segment's chart-time. Etterna's
    # GetDisplayedBeat treats no-scroll-yet as ratio 1; if the first scroll
    # isn't at beat 0, prepend (t=0, 1.0).
    sv_records = []
    if scrolls:
        if scrolls[0][0] > 0.0:
            sv_records.append((0.0, 1.0))
        for (b, r) in scrolls:
            sv_records.append((float(timing.beat_to_time(float(b))), float(r)))
    else:
        sv_records.append((0.0, 1.0))
    sv_records.sort(key=lambda x: x[0])
    data = SVData.from_sections(sv_records)

    # SPEEDS is position-dependent zoom z(t).
    if speeds_eval:
        def zoom_at(t):
            return speeds_eval.percent_at(timing.time_to_beat(float(t)),
                                           float(t))
        zoom_fn = zoom_at
    else:
        zoom_fn = None

    enabled = bool(scrolls or speeds or len(bpms) > 1
                   or stops or delays or warps)
    doc = SVDocument(
        measure=measure,
        data=data,
        zoom_fn=zoom_fn,
        enabled=enabled,
    )

    # Beat-space inverse threads back through the displayed-beat cache and
    # the timing map. ScrollsCache.inverse_displayed_beat handles the
    # scroll<=0 plateau case (collapse to segment start) the integrator
    # can't.
    def inverse_override(displayed_target):
        if not scrolls_cache:
            return float(displayed_target)
        return scrolls_cache.inverse_displayed_beat(displayed_target, timing)

    return MeasureSVEngine(doc,
                           sec_per_base_beat=sec_per_base_beat,
                           inverse_t_from_cum=inverse_override,
                           scrolls_cache=scrolls_cache)
