"""Regression tests for ChartClock.

The core invariants this singleton promises:

1. While paused, `now()` doesn't advance no matter what the wall-clock does.
2. Unpausing resumes from wherever `now()` was (no jump).
3. Attaching/detaching an audio source doesn't move the playhead.
4. Seeking updates the clock and doesn't get overwritten by the next read.
5. Rate changes take effect going forward without rebasing the visible `now()`.
"""
import time

import pytest

from analysis.player.scroll.chart_clock import ChartClock


def test_paused_clock_does_not_advance():
    c = ChartClock()
    # Default: paused at t=0.0
    t0 = c.now()
    time.sleep(0.05)
    assert c.now() == t0


def test_unpause_resumes_from_current_t():
    c = ChartClock(initial=42.0)
    c.seek(42.0)
    assert c.now() == 42.0
    c.set_paused(False)
    t0 = c.now()
    time.sleep(0.05)
    t1 = c.now()
    assert t1 - t0 == pytest.approx(0.05, abs=0.02)
    assert t0 == pytest.approx(42.0, abs=0.01)


def test_seek_takes_effect():
    c = ChartClock()
    c.seek(10.0)
    assert c.now() == 10.0
    c.seek(-5.0)
    # Default t_min is -2.0 so seek clamps
    assert c.now() == -2.0


def test_rate_change_scales_future_advance():
    c = ChartClock()
    c.set_paused(False)
    c.set_rate(2.0)
    t0 = c.now()
    time.sleep(0.05)
    t1 = c.now()
    # Rate 2 -> roughly 0.1s over 0.05s real
    assert t1 - t0 == pytest.approx(0.1, abs=0.03)


def test_audio_source_overrides_wall_clock():
    c = ChartClock()
    c.set_paused(False)
    source = [100.0]
    c.set_audio_source(lambda: source[0])
    assert c.now() == pytest.approx(100.0)
    source[0] = 200.0
    assert c.now() == pytest.approx(200.0)


def test_detach_preserves_current_t():
    """Bug we hit: after detaching the audio source, `now()` must not jump
    backward to whatever wall-clock would have computed without it."""
    c = ChartClock()
    c.set_paused(False)
    source = [75.0]
    c.set_audio_source(lambda: source[0])
    assert c.now() == pytest.approx(75.0)
    c.set_audio_source(None)
    # Immediately after detach, t should still be 75 (wall-clock rebased)
    assert c.now() == pytest.approx(75.0, abs=0.01)


def test_audio_getter_exception_falls_back_to_wall_clock():
    """If the audio reader throws, keep rendering from wall-clock."""
    c = ChartClock()
    c.set_paused(False)
    c.set_audio_source(lambda: (_ for _ in ()).throw(RuntimeError('boom')))
    t0 = c.now()  # Should fall through to wall-clock, not raise
    time.sleep(0.05)
    t1 = c.now()
    assert t1 > t0


def test_paused_audio_source_still_returns_frozen_time():
    """When paused, ignore the audio source even if it's installed ; the
    user might have hit pause while audio was still producing samples."""
    c = ChartClock()
    source = [50.0]
    c.set_audio_source(lambda: source[0])
    c.seek(30.0)
    assert c.paused is True
    assert c.now() == pytest.approx(30.0)
    source[0] = 999.0
    # Still paused, still 30
    assert c.now() == pytest.approx(30.0)


def test_bounds_clamp_seeks():
    c = ChartClock(t_min=0.0, t_max=100.0)
    c.seek(200.0)
    assert c.now() == 100.0
    c.seek(-5.0)
    assert c.now() == 0.0


def test_set_bounds_updates_clamp():
    c = ChartClock(t_max=10.0)
    c.set_bounds(t_min=-5.0, t_max=500.0)
    c.seek(250.0)
    assert c.now() == 250.0
