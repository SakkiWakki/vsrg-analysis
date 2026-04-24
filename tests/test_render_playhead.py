"""Regression tests for the per-frame RenderPlayhead."""

import pytest

from analysis.player.render_playhead import RenderPlayhead


class _IdentityEngine:
    enabled = True

    def cumulative_at(self, t):
        return float(t)

    def render_multiplier_at(self, t):
        del t
        return 1.0


class _PiecewiseEngine:
    enabled = True

    def cumulative_at(self, t):
        t = float(t)
        if t <= 1.0:
            return t
        return 1.0 + (t - 1.0) * 10.0

    def render_multiplier_at(self, t):
        del t
        return 1.0


def test_identity_frame_matches_raw_time():
    p = RenderPlayhead(_IdentityEngine())
    p.reset(0.0)
    frame = p.frame(raw_t=0.25, scroll_speed=500.0, use_sv=True)
    assert frame.visual_cum_now == pytest.approx(0.25)
    assert frame.render_multiplier == pytest.approx(1.0)
    assert frame.px_per_cum == pytest.approx(500.0)


def test_piecewise_engine_follows_exact_integrated_delta():
    p = RenderPlayhead(_PiecewiseEngine())
    p.reset(0.9)
    f1 = p.frame(raw_t=1.0, scroll_speed=400.0, use_sv=True)
    f2 = p.frame(raw_t=1.1, scroll_speed=400.0, use_sv=True)
    # Crossing the section boundary should follow the exact cumulative
    # integral, not a local slope estimate from either side.
    assert f1.visual_cum_now == pytest.approx(1.0)
    assert f2.visual_cum_now == pytest.approx(2.0)
