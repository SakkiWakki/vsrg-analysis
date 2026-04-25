"""Parity tests for the measure-based SV engine port.

The reference engines (TimeSpaceSVEngine, BeatSpaceSVEngine) are the ground
truth -- they're the engines that have been validated against ArrowEffects.cpp
and osu! through extensive testing. This suite asserts that the new
measure-based wrappers (measure_engine.time_space_engine /
beat_space_engine) produce numerically equal output on representative inputs.

Tolerance: 1e-9 absolute on cumulative_at, 1e-9 on distance, exact on
project_times agreement. Beat-space SPEEDS samples may differ by floating-
point summation order in deeply-nested transitions; we use 1e-7 there.
"""
import numpy as np
import pytest

from analysis.player.sv.engine import BeatSpaceSVEngine, TimeSpaceSVEngine
from analysis.player.sv.measure_engine import (beat_space_engine,
                                                time_space_engine)


# ---------------------------------------------------------------------------
# Time-space parity
# ---------------------------------------------------------------------------

_TIME_CASES = [
    ('empty',         []),
    ('constant_2x',   [(0.0, 2.0)]),
    ('two_segments',  [(0.0, 1.0), (10.0, 2.0)]),
    ('three_segments',[(0.0, 1.0), (5.0, 0.5), (10.0, 2.0)]),
    ('descending',    [(0.0, 3.0), (2.0, 2.0), (4.0, 1.0), (6.0, 0.5)]),
    ('zero_segment',  [(0.0, 1.0), (5.0, 0.0), (10.0, 1.0)]),
    ('negative',      [(0.0, 1.0), (5.0, -1.0), (10.0, 1.0)]),
]


@pytest.mark.parametrize('name,sections', _TIME_CASES)
def test_time_space_cumulative_matches(name, sections):
    ref = TimeSpaceSVEngine(sections)
    new = time_space_engine(sections)
    samples = np.linspace(-2.0, 30.0, 200)
    for t in samples:
        assert new.cumulative_at(float(t)) == pytest.approx(
            ref.cumulative_at(float(t)), abs=1e-9), \
            f"{name}: mismatch at t={t}"


@pytest.mark.parametrize('name,sections', _TIME_CASES)
def test_time_space_project_matches(name, sections):
    ref = TimeSpaceSVEngine(sections)
    new = time_space_engine(sections)
    samples = np.linspace(-2.0, 30.0, 200)
    np.testing.assert_allclose(new.project_times(samples),
                                ref.project_times(samples), atol=1e-9)


@pytest.mark.parametrize('name,sections', _TIME_CASES)
def test_time_space_distance_matches(name, sections):
    ref = TimeSpaceSVEngine(sections)
    new = time_space_engine(sections)
    pairs = [(0.0, 5.0), (-1.0, 8.0), (3.0, 3.0), (10.0, 25.0)]
    for a, b in pairs:
        assert new.distance(a, b) == pytest.approx(
            ref.distance(a, b), abs=1e-9), \
            f"{name}: distance({a},{b}) mismatch"


def test_time_space_enabled_flag():
    assert not time_space_engine([]).enabled
    assert time_space_engine([(0.0, 1.5)]).enabled


# ---------------------------------------------------------------------------
# Beat-space parity
# ---------------------------------------------------------------------------

_CONST_BPM = [(0.0, 120.0)]   # 0.5 sec/beat, base 120
_TWO_BPM = [(0.0, 120.0), (16.0, 240.0)]   # half-time after beat 16


def _bs_case(label, **kwargs):
    return pytest.param(kwargs, id=label)


_BEAT_CASES = [
    _bs_case('flat_no_sv',
             scrolls=[], speeds=[], bpms=_CONST_BPM, sm_offset=0.0),
    _bs_case('constant_scroll_2x',
             scrolls=[(0.0, 2.0)], speeds=[], bpms=_CONST_BPM, sm_offset=0.0),
    _bs_case('piecewise_scroll',
             scrolls=[(0.0, 1.0), (8.0, 2.0), (16.0, 0.5)],
             speeds=[], bpms=_CONST_BPM, sm_offset=0.0),
    _bs_case('with_offset',
             scrolls=[(0.0, 1.0), (4.0, 2.0)],
             speeds=[], bpms=_CONST_BPM, sm_offset=0.05),
    _bs_case('bpm_change',
             scrolls=[(0.0, 1.0)], speeds=[], bpms=_TWO_BPM, sm_offset=0.0),
    _bs_case('bpm_change_with_scroll',
             scrolls=[(0.0, 1.0), (12.0, 2.0)],
             speeds=[], bpms=_TWO_BPM, sm_offset=0.0),
]


@pytest.mark.parametrize('kw', _BEAT_CASES)
def test_beat_space_cumulative_matches(kw):
    ref = BeatSpaceSVEngine(**kw)
    new = beat_space_engine(**kw)
    # Sample chart-time over a range that exercises both before and after
    # the BPM change in _TWO_BPM cases (0.5*16 = 8s for first segment).
    samples = np.linspace(0.0, 30.0, 200)
    for t in samples:
        ref_val = ref.cumulative_at(float(t))
        new_val = new.cumulative_at(float(t))
        assert new_val == pytest.approx(ref_val, abs=1e-7, rel=1e-9), \
            f"{kw}: mismatch at t={t}: ref={ref_val} new={new_val}"


@pytest.mark.parametrize('kw', _BEAT_CASES)
def test_beat_space_project_matches(kw):
    ref = BeatSpaceSVEngine(**kw)
    new = beat_space_engine(**kw)
    samples = np.linspace(0.0, 30.0, 200)
    np.testing.assert_allclose(new.project_times(samples),
                                ref.project_times(samples),
                                atol=1e-7, rtol=1e-9)


@pytest.mark.parametrize('kw', _BEAT_CASES)
def test_beat_space_distance_matches_when_speeds_flat(kw):
    # Without SPEEDS, distance == cumulative diff and parity is exact-ish.
    ref = BeatSpaceSVEngine(**kw)
    new = beat_space_engine(**kw)
    pairs = [(0.0, 5.0), (1.0, 3.0), (2.0, 10.0), (5.0, 25.0)]
    for a, b in pairs:
        assert new.distance(a, b) == pytest.approx(
            ref.distance(a, b), abs=1e-7, rel=1e-9), \
            f"{kw}: distance({a},{b}) mismatch"


# ---------------------------------------------------------------------------
# Stops and warps
# ---------------------------------------------------------------------------


def test_beat_space_with_stop():
    # 0.5 second stop at beat 4. Cumulative across the stop should be the
    # same on both sides (notes pinned).
    ref = BeatSpaceSVEngine(
        scrolls=[(0.0, 1.0)],
        speeds=[],
        bpms=_CONST_BPM,
        sm_offset=0.0,
        stops=[(4.0, 0.5)],
    )
    new = beat_space_engine(
        scrolls=[(0.0, 1.0)],
        speeds=[],
        bpms=_CONST_BPM,
        sm_offset=0.0,
        stops=[(4.0, 0.5)],
    )
    # Stop is at beat 4 = 2.0 sec under 120 BPM. Sample around it.
    samples = np.linspace(0.0, 5.0, 100)
    for t in samples:
        assert new.cumulative_at(float(t)) == pytest.approx(
            ref.cumulative_at(float(t)), abs=1e-7, rel=1e-9), \
            f"stop: mismatch at t={t}"


def test_beat_space_with_warp():
    # Warp at beat 4 of length 4 beats. Cumulative jumps by the SCROLLS
    # integral over the warp interior.
    ref = BeatSpaceSVEngine(
        scrolls=[(0.0, 1.0)],
        speeds=[],
        bpms=_CONST_BPM,
        sm_offset=0.0,
        warps=[(4.0, 4.0)],
    )
    new = beat_space_engine(
        scrolls=[(0.0, 1.0)],
        speeds=[],
        bpms=_CONST_BPM,
        sm_offset=0.0,
        warps=[(4.0, 4.0)],
    )
    samples = np.linspace(0.0, 6.0, 200)
    for t in samples:
        assert new.cumulative_at(float(t)) == pytest.approx(
            ref.cumulative_at(float(t)), abs=1e-7, rel=1e-9), \
            f"warp: mismatch at t={t}"


# ---------------------------------------------------------------------------
# Sanity: cumulative_at agrees with project_times scalar-wise
# ---------------------------------------------------------------------------


def test_time_space_cumulative_matches_project_internal():
    # Internal consistency on the new engine alone (the existing tests
    # already check the reference engine).
    e = time_space_engine([(0.0, 1.0), (5.0, 0.5), (10.0, 2.0)])
    times = np.array([2.0, 5.0, 7.5, 12.0])
    proj = e.project_times(times)
    for t, p in zip(times, proj):
        assert e.cumulative_at(float(t)) == pytest.approx(float(p), abs=1e-12)
