"""Tests for eased SV segments (Quaver SSF / fluXis multiplier events).

Linear easing: integrand v(tau) ramps from `multiplier` at the segment's
`time` to `end_multiplier` at `time + duration`, then holds at
`end_multiplier` until the next segment. Step easing (the default) is
piecewise-constant -- the historical case, exercised by every other test
in the SV suite.

These tests verify the trapezoid integrator agrees with closed-form
analytic integrals for linear ramps under both Lebesgue (time-space)
and constant-rho measures, and that step easing reduces to the legacy
piecewise-constant behavior exactly.
"""
import numpy as np
import pytest

from analysis.player.sv.document import (EASING_LINEAR, EASING_STEP, SVData,
                                          SVDocument, SVRecord, TimingMeasure)
from analysis.player.sv.integrate import CumulativeIntegrator


def _doc_with_records(records, t_start=0.0, t_end=100.0):
    """Build a time-space document (mu = Lebesgue) from explicit records."""
    return SVDocument(
        measure=TimingMeasure.lebesgue(t_start, t_end),
        data=SVData.from_records(tuple(records)),
        zoom_fn=None,
        enabled=True,
    )


# ---------------------------------------------------------------------------
# Step easing: must match piecewise-constant exactly
# ---------------------------------------------------------------------------


def test_step_easing_reduces_to_piecewise_constant():
    """A document with only step records must produce identical cumulative
    to one built via from_sections, which is the legacy piecewise-constant
    path."""
    sections = [(0.0, 1.0), (5.0, 2.0), (10.0, 0.5)]
    legacy_doc = SVDocument(
        measure=TimingMeasure.lebesgue(0.0, 20.0),
        data=SVData.from_sections(sections),
        zoom_fn=None, enabled=True,
    )
    eased_doc = _doc_with_records([
        SVRecord(time=t, multiplier=m, easing=EASING_STEP)
        for t, m in sections
    ], t_end=20.0)

    legacy = CumulativeIntegrator(legacy_doc)
    eased = CumulativeIntegrator(eased_doc)

    samples = np.linspace(-1.0, 20.0, 100)
    np.testing.assert_allclose(eased.project_times(samples),
                                legacy.project_times(samples),
                                atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Linear easing: closed-form parity
# ---------------------------------------------------------------------------


def test_linear_ramp_matches_analytic_integral():
    """Single linear segment from m=1 at t=0 to m=3 at t=10 (duration=10).
    Analytic cumulative C(t) on [0, 10] = ∫₀ᵗ (1 + 2*tau/10) dtau
    = t + t²/10. Verify integrator matches at sample points."""
    rec = SVRecord(time=0.0, multiplier=1.0, duration=10.0,
                   easing=EASING_LINEAR, end_multiplier=3.0)
    doc = _doc_with_records([rec], t_end=20.0)
    eng = CumulativeIntegrator(doc)

    for t in [0.0, 1.0, 2.5, 5.0, 7.5, 10.0]:
        expected = t + t * t / 10.0
        actual = eng.cumulative_at(t)
        assert actual == pytest.approx(expected, abs=1e-9), \
            f'at t={t}: expected {expected}, got {actual}'


def test_linear_ramp_holds_after_duration():
    """After the easing window ends, v holds at end_multiplier. The
    cumulative past t=10 should grow linearly at slope = end_multiplier."""
    rec = SVRecord(time=0.0, multiplier=1.0, duration=10.0,
                   easing=EASING_LINEAR, end_multiplier=3.0)
    doc = _doc_with_records([rec], t_end=20.0)
    eng = CumulativeIntegrator(doc)

    # At t=10: C(10) = 10 + 100/10 = 20.
    # At t=15: C(15) = 20 + 5 * 3.0 = 35 (5 sec at slope 3).
    # At t=20: C(20) = 20 + 10 * 3.0 = 50.
    assert eng.cumulative_at(10.0) == pytest.approx(20.0, abs=1e-9)
    assert eng.cumulative_at(15.0) == pytest.approx(35.0, abs=1e-9)
    assert eng.cumulative_at(20.0) == pytest.approx(50.0, abs=1e-9)


def test_linear_velocity_is_lerped_within_easing_window():
    """v(t) for a linear segment ramps from m to end_m over [time,
    time+duration]. velocity_at returns v(t) * rho(t); rho=1 in time-
    space."""
    rec = SVRecord(time=0.0, multiplier=1.0, duration=10.0,
                   easing=EASING_LINEAR, end_multiplier=3.0)
    doc = _doc_with_records([rec], t_end=20.0)
    eng = CumulativeIntegrator(doc)

    # At t=2.5: lerp 1 + 2 * 0.25 = 1.5.
    # At t=5.0: 1 + 2 * 0.5 = 2.0.
    # At t=10.0: 3.0 (end of ramp).
    # At t=15.0: holds at 3.0.
    assert eng.velocity_at(2.5) == pytest.approx(1.5, abs=1e-9)
    assert eng.velocity_at(5.0) == pytest.approx(2.0, abs=1e-9)
    assert eng.velocity_at(15.0) == pytest.approx(3.0, abs=1e-9)


def test_chained_linear_ramps_form_polyline():
    """Two consecutive linear segments. End of segment 1 = start of
    segment 2 (set explicitly via end_multiplier). The cumulative is a
    smooth polyline in v."""
    recs = [
        SVRecord(time=0.0, multiplier=1.0, duration=4.0,
                 easing=EASING_LINEAR, end_multiplier=2.0),
        SVRecord(time=4.0, multiplier=2.0, duration=6.0,
                 easing=EASING_LINEAR, end_multiplier=0.5),
    ]
    doc = _doc_with_records(recs, t_end=20.0)
    eng = CumulativeIntegrator(doc)

    # Segment 1: v ramps 1 -> 2 over [0, 4]. C(4) = ∫₀⁴ (1 + tau/4) dtau
    # = 4 + 8/4 = 4 + 2 = 6.
    assert eng.cumulative_at(4.0) == pytest.approx(6.0, abs=1e-9)

    # Segment 2: v ramps 2 -> 0.5 over [4, 10]. C(10) = C(4) +
    # ∫₀⁶ (2 + (-1.5)*s/6) ds = 6 + 12 - (1.5/6) * 18 = 6 + 12 - 4.5 = 13.5.
    assert eng.cumulative_at(10.0) == pytest.approx(13.5, abs=1e-9)


def test_linear_with_zero_duration_acts_as_step():
    """duration=0 with linear easing should behave like a step at
    `multiplier` (the ramp-fraction clip in _eval_v_array makes this
    fall through to end_multiplier immediately, so the in-window value
    is just end_multiplier)."""
    recs = [
        SVRecord(time=0.0, multiplier=2.0, duration=0.0,
                 easing=EASING_LINEAR, end_multiplier=2.0),
        SVRecord(time=5.0, multiplier=3.0, duration=0.0,
                 easing=EASING_LINEAR, end_multiplier=3.0),
    ]
    doc = _doc_with_records(recs, t_end=20.0)
    eng = CumulativeIntegrator(doc)

    # Effectively step: 0..5 at v=2, 5..10 at v=3.
    assert eng.cumulative_at(5.0) == pytest.approx(10.0, abs=1e-9)
    assert eng.cumulative_at(10.0) == pytest.approx(25.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Project_times vectorized path matches scalar path
# ---------------------------------------------------------------------------


def test_project_times_matches_scalar_for_linear():
    rec = SVRecord(time=0.0, multiplier=1.0, duration=10.0,
                   easing=EASING_LINEAR, end_multiplier=3.0)
    doc = _doc_with_records([rec], t_end=20.0)
    eng = CumulativeIntegrator(doc)

    samples = np.linspace(-1.0, 20.0, 100)
    vec = eng.project_times(samples)
    for t, v in zip(samples, vec):
        assert eng.cumulative_at(float(t)) == pytest.approx(float(v), abs=1e-9)
