"""Regression tests for SV engines.

The core invariants enforced here:

1. `cumulative_at(t) == project_times([t])[0]` exactly ; this is what keeps
   the cull bisect in culling.py consistent with what `_time_to_y` draws.

2. `distance(a, b) == cumulative_at(b) - cumulative_at(a)` when no
   position-dependent effect applies (SPEEDS = 1 flat, or no SPEEDS at all).

3. For Etterna charts specifically, `distance` matches the exact formula
   from ArrowEffects.cpp::GetYOffset (XMOD branch) up to floating-point
   error. This is the regression that introduced a silent drift when
   SPEEDS was baked into the culling cache.
"""
import numpy as np
import pytest

from analysis.player.sv.engine import (BeatSpaceSVEngine, IdentitySVEngine,
                                        TimeSpaceSVEngine)


# ---------------------------------------------------------------------------
# Identity engine
# ---------------------------------------------------------------------------

def test_identity_engine_distance():
    e = IdentitySVEngine()
    assert e.distance(1.0, 5.0) == 4.0
    assert e.distance(5.0, 1.0) == -4.0
    assert e.distance(0.0, 0.0) == 0.0


def test_identity_engine_project_is_noop():
    e = IdentitySVEngine()
    arr = np.array([0.0, 1.5, 3.0])
    out = e.project_times(arr)
    assert np.array_equal(out, arr)


def test_identity_engine_enabled_is_false():
    assert IdentitySVEngine().enabled is False


# ---------------------------------------------------------------------------
# TimeSpaceSVEngine (osu!mania)
# ---------------------------------------------------------------------------

def test_time_space_engine_empty_is_identity():
    e = TimeSpaceSVEngine([])
    assert not e.enabled
    assert e.distance(0.0, 5.0) == 5.0


def test_time_space_engine_constant_mult():
    e = TimeSpaceSVEngine([(0.0, 2.0)])
    assert e.distance(0.0, 5.0) == pytest.approx(10.0)
    assert e.cumulative_at(3.0) == pytest.approx(6.0)


def test_time_space_engine_piecewise():
    # 0-10s at 1x, 10-20s at 2x
    e = TimeSpaceSVEngine([(0.0, 1.0), (10.0, 2.0)])
    # 0 to 15 = 10 + 5*2 = 20
    assert e.distance(0.0, 15.0) == pytest.approx(20.0)


def test_time_space_engine_cumulative_matches_project():
    e = TimeSpaceSVEngine([(0.0, 1.0), (5.0, 0.5), (10.0, 2.0)])
    times = np.array([2.0, 5.0, 7.5, 12.0])
    proj = e.project_times(times)
    for t, p in zip(times, proj):
        assert e.cumulative_at(float(t)) == pytest.approx(float(p))


# ---------------------------------------------------------------------------
# BeatSpaceSVEngine (Etterna XMOD)
# ---------------------------------------------------------------------------

_CONSTANT_BPMS = [(0.0, 120.0)]  # 0.5s per beat


def test_beat_space_empty_disabled():
    e = BeatSpaceSVEngine([], [], _CONSTANT_BPMS, 0.0)
    assert not e.enabled


def test_beat_space_scrolls_only():
    # 120 BPM constant: 2 beats per second. SCROLLS = 1 everywhere means
    # displayed_beat == real beat.
    e = BeatSpaceSVEngine([(0.0, 1.0)], [], _CONSTANT_BPMS, 0.0)
    # From beat 0 (t=0) to beat 4 (t=2s): 4 displayed beats, base-BPM=120 so
    # sec_per_base_beat = 0.5, distance = 4 * 0.5 = 2.0
    assert e.distance(0.0, 2.0) == pytest.approx(2.0)


def test_beat_space_scroll_ramp():
    # Scroll 0.5 from beat 0, 1.0 from beat 4. Between beat 0 and beat 4:
    # 4 beats * 0.5 = 2 displayed beats. At 120 BPM -> 1.0 effective sec.
    e = BeatSpaceSVEngine([(0.0, 0.5), (4.0, 1.0)], [], _CONSTANT_BPMS, 0.0)
    assert e.distance(0.0, 2.0) == pytest.approx(1.0)


def test_beat_space_cumulative_consistency():
    """cumulative_at(t) must equal project_times([t])[0] exactly.
    Culling bisect depends on this identity."""
    e = BeatSpaceSVEngine(
        [(0.0, 0.5), (8.0, 1.5), (16.0, 1.0)],
        [(0.0, 1.0), (12.0, 2.0)],  # non-trivial SPEEDS
        _CONSTANT_BPMS, 0.0,
    )
    times = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
    proj = e.project_times(times)
    for t, p in zip(times, proj):
        assert e.cumulative_at(float(t)) == pytest.approx(float(p))


def test_beat_space_distance_matches_etterna_formula():
    """Engine output must match Etterna's ArrowEffects.cpp::GetYOffset
    exactly (up to FP noise). Reimplemented here as the spec."""
    bpms = [(0.0, 140.0), (10.0, 70.0)]  # BPM change mid-chart
    scrolls = [(0.0, 1.0), (5.0, 0.5), (15.0, 2.0)]
    speeds = [(0.0, 1.0), (20.0, 0.0, 0.0, 0), (20.1, 1.5, 0.0, 0)]
    e = BeatSpaceSVEngine(scrolls, speeds, bpms, sm_offset=0.0)
    sec_per_base = 60.0 / 140.0

    def etterna_displayed_beat(beat):
        if not scrolls:
            return beat
        cache = []
        if scrolls[0][0] > 0:
            cache.append((0.0, 0.0, 1.0))
        db, lb, lr = 0.0, 0.0, 1.0
        for b, r in scrolls:
            db += (b - lb) * lr
            cache.append((b, db, r))
            lb, lr = b, r
        idx = -1
        for i, c in enumerate(cache):
            if c[0] <= beat:
                idx = i
            else:
                break
        if idx < 0:
            return beat
        cb, cdb, cr = cache[idx]
        return cdb + (beat - cb) * cr

    def etterna_speed_percent(beat, music_seconds):
        idx = -1
        for i, seg in enumerate(speeds):
            if seg[0] <= beat:
                idx = i
            else:
                break
        if idx < 0:
            return 1.0
        seg_beat, seg_ratio, seg_delay, seg_unit = (
            (speeds[idx][0], speeds[idx][1], speeds[idx][2], speeds[idx][3])
            if len(speeds[idx]) >= 4 else
            (speeds[idx][0], speeds[idx][1], 0.0, 0)
        )
        start_time = e._beat_to_time(seg_beat)
        if seg_unit == 1:
            end_time = start_time + seg_delay
        else:
            end_time = e._beat_to_time(seg_beat + seg_delay)
        first_delay = speeds[0][2] if len(speeds[0]) >= 4 else 0.0
        if idx == 0 and first_delay > 0.0 and music_seconds < start_time:
            return 1.0
        if end_time >= music_seconds and (idx > 0 or first_delay > 0.0):
            prior = 1.0 if idx == 0 else speeds[idx - 1][1]
            duration = end_time - start_time
            ratio_used = 1.0 if duration == 0.0 else (music_seconds - start_time) / duration
            return prior + (seg_ratio - prior) * ratio_used
        return seg_ratio

    def etterna_y_beats(song_beat, note_beat):
        d = etterna_displayed_beat(note_beat) - etterna_displayed_beat(song_beat)
        song_t = e._beat_to_time(song_beat)
        return d * etterna_speed_percent(song_beat, song_t)

    # Test a grid covering: normal region, across scroll change, across BPM
    # change, negative offset, SPEEDS=0 region, SPEEDS non-trivial.
    cases = [
        (0, 3), (0, 10), (3, 8),
        (5, 12),            # straddles scroll change at 5
        (8, 15),            # straddles BPM change at 10
        (15, 25),           # straddles SPEEDS drop at 20
        (20, 25),           # inside SPEEDS=0
        (20.05, 25),        # SPEEDS=1.5 region
        (10, 5),            # going backward (note behind song)
    ]
    for song_b, note_b in cases:
        song_t = e._beat_to_time(song_b)
        note_t = e._beat_to_time(note_b)
        ours = e.distance(song_t, note_t) / sec_per_base
        expected = etterna_y_beats(song_b, note_b)
        assert ours == pytest.approx(expected, abs=1e-9), (
            f'song_beat={song_b}, note_beat={note_b}: expected={expected}, got={ours}'
        )


def test_beat_space_speed_zero_freezes_field():
    """SPEEDS=0 means the field zoom is 0 ; all notes collapse to the
    receptor. Etterna uses this for the stutter-end gimmick in charts
    like Undiscovered Colors."""
    e = BeatSpaceSVEngine(
        [(0.0, 1.0)],
        [(0.0, 1.0), (10.0, 0.0)],
        _CONSTANT_BPMS, 0.0,
    )
    # t=5 corresponds to beat 10, where SPEEDS = 0 kicks in
    t_from = 5.0  # beat 10 at 120 bpm
    t_to = 7.5    # beat 15
    # SPEEDS at beat 10 is 0 -> distance scaled to 0
    assert e.distance(t_from, t_to) == pytest.approx(0.0)


def test_beat_space_speed_transition_beats_interpolates():
    e = BeatSpaceSVEngine(
        [(0.0, 1.0)],
        [(0.0, 2.0, 4.0, 0)],
        _CONSTANT_BPMS, 0.0,
    )
    assert e.render_multiplier_at(0.0) == pytest.approx(1.0)
    assert e.render_multiplier_at(1.0) == pytest.approx(1.5)
    assert e.render_multiplier_at(2.0) == pytest.approx(2.0)


def test_beat_space_speed_transition_seconds_interpolates():
    e = BeatSpaceSVEngine(
        [(0.0, 1.0)],
        [(4.0, 2.0, 2.0, 1)],
        _CONSTANT_BPMS, 0.0,
    )
    assert e.render_multiplier_at(2.0) == pytest.approx(1.0)
    assert e.render_multiplier_at(3.0) == pytest.approx(1.5)
    assert e.render_multiplier_at(4.0) == pytest.approx(2.0)


def test_beat_space_negative_distance_symmetric():
    """distance(a, b) == -distance(b, a) when no SPEEDS asymmetry applies.
    (With SPEEDS this doesn't generally hold because speed_percent is
    sampled at t_from.)"""
    e = BeatSpaceSVEngine([(0.0, 1.0)], [], _CONSTANT_BPMS, 0.0)
    assert e.distance(2.0, 5.0) == pytest.approx(-e.distance(5.0, 2.0))


def test_beat_space_max_visible_caps_lookahead_in_beats():
    """Etterna's engine caps visible lookahead at ~20 beats, matching
    ArrowEffects::FindDisplayedBeats' binary-search convergence. Without
    the cap, a scroll=0 region lets the entire pile through because every
    note in the region has the same SV-cum value."""
    e = BeatSpaceSVEngine([(0.0, 1.0)], [], _CONSTANT_BPMS, 0.0)
    # 120 BPM -> 2 beats per second. 20 beats lookahead = 10s.
    cap_t = e.max_visible_t_from(0.0)
    assert cap_t == pytest.approx(10.0, abs=0.01)
    # Farther into the song: cap scales.
    cap_t = e.max_visible_t_from(5.0)  # beat 10
    assert cap_t == pytest.approx(15.0, abs=0.01)  # 10s + 20-beat lookahead


def test_scroll_zero_region_collapses_to_same_cumulative():
    """All notes inside a scroll=0 region map to the same cull-space
    cumulative ; the whole point of the scroll-stack gimmick. Without
    this check, accidentally applying scroll to beats inside the region
    would spread the pile out."""
    e = BeatSpaceSVEngine([(0.0, 0.0), (26.0, 1.0)], [], [(0.0, 140.0)], 0.0)
    # Beats 0..26 all at displayed_beat=0 -> same cumulative
    ts = [e._beat_to_time(b) for b in [1, 5, 10, 15, 20, 25]]
    cums = [e.cumulative_at(t) for t in ts]
    for c in cums[1:]:
        assert c == pytest.approx(cums[0])


def test_identity_engine_max_visible_is_infinity():
    """Non-Etterna engines don't impose a beat-based cap."""
    from analysis.player.sv.engine import IdentitySVEngine, TimeSpaceSVEngine
    assert IdentitySVEngine().max_visible_t_from(0.0) == float('inf')
    assert TimeSpaceSVEngine([(0.0, 1.0)]).max_visible_t_from(0.0) == float('inf')


def test_beat_space_matches_sm_chart_walker():
    """Pre-walked TimingMap must return bit-identical values to the
    reference beat_to_time walker ; regression against a perf refactor
    that could silently introduce rounding drift on large charts.

    Restricted to BPM + STOP + DELAY cases so the basic timing-boundary
    comparison stays separate from the explicit warp cases below."""
    from analysis.games.etterna.sm_chart import beat_to_time

    bpms = [(0.0, 140.0), (10.0, 70.0), (20.0, 200.0)]
    stops = [(5.0, 0.25), (15.0, 0.1)]
    delays = [(12.0, 0.05)]
    e = BeatSpaceSVEngine([], [], bpms, 0.1,
                           stops=stops, delays=delays)
    for beat in [0.0, 0.5, 4.9, 5.0, 5.1, 10.0, 12.0, 15.0,
                  25.0, 100.0]:
        ref = beat_to_time(beat, bpms, 0.1, stops, delays)
        got = e._beat_to_time(beat)
        assert got == pytest.approx(ref, abs=1e-9), (
            f'beat={beat}: ref={ref}, got={got}'
        )


def test_beat_space_warp_inside_range_collapses_time():
    """Beats inside a WARP region share the time of the warp entry ;
    they're teleported-over in beat space with no time elapsing."""
    bpms = [(0.0, 120.0)]  # 0.5s per beat
    warps = [(10.0, 2.0)]  # warp 2 beats forward at beat 10
    e = BeatSpaceSVEngine([], [], bpms, 0.0, warps=warps)
    # Beat 10 entry: 10 * 0.5 = 5.0s
    t_enter = e._beat_to_time(10.0)
    assert t_enter == pytest.approx(5.0, abs=1e-9)
    # Beats 10..12 all collapse to the same time
    assert e._beat_to_time(11.0) == pytest.approx(5.0, abs=1e-9)
    assert e._beat_to_time(12.0) == pytest.approx(5.0, abs=1e-9)
    # Beat just past warp end: normal advance from beat 12
    assert e._beat_to_time(13.0) == pytest.approx(5.5, abs=1e-9)


def test_beat_space_stop_then_warp_same_row_collapses_after_pause():
    """Etterna processes STOP before WARP at the same row. The precomputed
    timing map used by rendering must preserve that order or notes inside the
    warp render spread out after the stop."""
    e = BeatSpaceSVEngine(
        [], [], [(0.0, 120.0)], 0.0,
        stops=[(10.0, 1.0)], warps=[(10.0, 2.0)],
    )
    assert e._beat_to_time(10.0) == pytest.approx(5.0, abs=1e-9)
    assert e._beat_to_time(11.0) == pytest.approx(6.0, abs=1e-9)
    assert e._beat_to_time(12.0) == pytest.approx(6.0, abs=1e-9)
    assert e._beat_to_time(13.0) == pytest.approx(6.5, abs=1e-9)

    # While the stop is active, chart beat is still pinned to the warp row.
    assert e._time_to_beat(5.5) == pytest.approx(10.0, abs=1e-9)
    # Once the stop ends, the warp has landed at beat 12.
    assert e._time_to_beat(6.0) == pytest.approx(12.0, abs=1e-9)


def test_beat_space_time_to_beat_round_trips():
    """Round-trip beat -> time -> beat must return the original beat
    within float precision (for beats not inside a WARP range)."""
    bpms = [(0.0, 140.0), (10.0, 70.0), (20.0, 200.0)]
    stops = [(5.0, 0.25)]
    e = BeatSpaceSVEngine([], [], bpms, 0.0, stops=stops)
    for beat in [0.5, 3.0, 5.5, 10.0, 12.0, 18.0, 25.0, 50.0]:
        t = e._beat_to_time(beat)
        assert e._time_to_beat(t) == pytest.approx(beat, abs=1e-6)


def test_beat_space_time_to_beat_freezes_inside_stop():
    """Inside a STOP window, chart beat must remain pinned at the stop beat.
    Advancing through the pause then snapping back produces visible stutter
    artifacts in Etterna stop sections."""
    e = BeatSpaceSVEngine([], [], [(0.0, 120.0)], 0.0, stops=[(4.0, 0.5)])
    # Beat 4 occurs at t=2.0s, then freezes until t=2.5s.
    for t in [2.0, 2.1, 2.25, 2.49]:
        assert e._time_to_beat(t) == pytest.approx(4.0, abs=1e-9)


def test_beat_space_time_to_beat_freezes_inside_delay():
    """DELAYS also freeze chart beat at the event beat until the delay ends."""
    e = BeatSpaceSVEngine([], [], [(0.0, 120.0)], 0.0, delays=[(4.0, 0.5)])
    for t in [2.0, 2.1, 2.25, 2.49]:
        assert e._time_to_beat(t) == pytest.approx(4.0, abs=1e-9)
