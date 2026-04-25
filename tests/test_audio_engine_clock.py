"""Regression tests for AudioEngine's hardware-timed chart clock."""

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.player.audio.engine import AudioEngine


class _FakeStream:
    def __init__(self, t: float = 0.0) -> None:
        self.time = float(t)


def _engine_with_stream(t: float = 0.0) -> tuple[AudioEngine, _FakeStream]:
    engine = AudioEngine(None)
    stream = _FakeStream(t)
    engine._stream = stream
    engine._playing = True
    engine._rate = 1.0
    return engine, stream


def test_current_chart_time_uses_stream_clock_not_monotonic():
    engine, stream = _engine_with_stream(100.000)
    # Callback queued a block whose audible end lands at stream time 100.010.
    engine._hw_pos = 5.010
    engine._hw_wall = 100.010
    engine._dac_anchor_valid = True

    # Halfway through that block, the audible playhead should be halfway
    # through the chart interval too, not pinned to the callback boundary.
    stream.time = 100.005
    assert engine.current_chart_time() == pytest.approx(5.005, abs=1e-9)


def test_current_chart_time_is_continuous_across_callback_anchors():
    engine, stream = _engine_with_stream(10.000)
    block = 512 / 44100.0

    # Old callback anchor: end of the previous block.
    engine._hw_pos = 1.000
    engine._hw_wall = 10.000
    engine._dac_anchor_valid = True
    stream.time = 10.000
    before = engine.current_chart_time()

    # New callback anchor: end of the next block. Querying at the new end
    # should not introduce a jump relative to the old extrapolation.
    engine._hw_pos = 1.000 + block
    engine._hw_wall = 10.000 + block
    after = engine.current_chart_time()

    assert before == pytest.approx(1.000, abs=1e-9)
    assert after == pytest.approx(1.000, abs=1e-9)


def test_current_chart_time_scales_with_play_rate():
    engine, stream = _engine_with_stream(20.000)
    engine._rate = 1.5
    engine._hw_pos = 30.150
    engine._hw_wall = 20.100
    engine._dac_anchor_valid = True

    stream.time = 20.080
    assert engine.current_chart_time() == pytest.approx(30.120, abs=1e-9)


def test_current_chart_time_clamps_backstep_jitter():
    """Stream-clock jitter that pulls the reading backward must not
    propagate into the engine's chart time. The cull-space predictor
    smooths sub-ms jitter on the render side; the audio layer's job is
    to expose a strictly monotone non-decreasing chart time.
    """
    engine, stream = _engine_with_stream(50.000)
    engine._hw_pos = 10.000
    engine._hw_wall = 50.000
    engine._dac_anchor_valid = True

    stream.time = 50.010
    forward = engine.current_chart_time()
    assert forward == pytest.approx(10.010, abs=1e-9)

    # Backward jitter on the stream clock holds the reading flat.
    stream.time = 50.008
    back = engine.current_chart_time()
    assert back == pytest.approx(10.010, abs=1e-9)
    assert back >= forward


def test_first_callback_does_not_backstep_before_audible_time_catches_up():
    """Before the first callback-backed anchor is valid, the engine returns
    the explicit transport time. Once the callback lands, the DAC-timed
    extrapolation is clamped so the visible chart time never snaps backward.

    This keeps the audio engine on one consistent DAC-clock contract without a
    provisional moving pre-callback timeline.
    """
    engine, stream = _engine_with_stream(100.000)
    block = 512 / 44100.0
    output_latency = 0.056

    class _FakePV:
        def generate(self, frames: int):
            return np.zeros((frames, 1), dtype=np.float32), True

    engine._pv = _FakePV()
    engine._scheduled_chart_pos = 0.0
    engine._chart_time = 0.524

    stream.time = 100.524
    before = engine.current_chart_time()

    info = SimpleNamespace(
        currentTime=100.524,
        outputBufferDacTime=100.524 + output_latency,
    )
    out = np.zeros((512, 1), dtype=np.float32)
    engine._callback(out, 512, info, None)

    stream.time = 100.524
    after = engine.current_chart_time()

    assert before == pytest.approx(0.524, abs=1e-9)
    assert after == pytest.approx(0.524, abs=1e-9)
    assert after >= before
