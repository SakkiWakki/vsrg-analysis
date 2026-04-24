"""Regression tests for SV engines.

The core invariants enforced here:

1. `cumulative_at(t) == project_times([t])[0]` exactly — this is what keeps
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

from analysis.player.sv_engine import (BeatSpaceSVEngine, IdentitySVEngine,
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
    speeds = [(0.0, 1.0), (20.0, 0.0), (20.1, 1.5)]
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

    def etterna_speed_percent(beat):
        v = 1.0
        for b, r in speeds:
            if b <= beat:
                v = r
            else:
                break
        return v

    def etterna_y_beats(song_beat, note_beat):
        d = etterna_displayed_beat(note_beat) - etterna_displayed_beat(song_beat)
        return d * etterna_speed_percent(song_beat)

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
    """SPEEDS=0 means the field zoom is 0 — all notes collapse to the
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


def test_beat_space_negative_distance_symmetric():
    """distance(a, b) == -distance(b, a) when no SPEEDS asymmetry applies.
    (With SPEEDS this doesn't generally hold because speed_percent is
    sampled at t_from.)"""
    e = BeatSpaceSVEngine([(0.0, 1.0)], [], _CONSTANT_BPMS, 0.0)
    assert e.distance(2.0, 5.0) == pytest.approx(-e.distance(5.0, 2.0))
