"""Live oscillator delta channels: seeded per-frame vibrate (the
receptor mirage), analytic sines, open-span extension, no sample-cap
freeze."""
import math

import pytest

from analysis.games.notitg import modfile
from analysis.games.notitg.recording_actor import _OscSpan


def _clock():
    return modfile._OscillatorClock(lambda beat: float(beat), (0.0, 64.0))


def _span(kind, start, end, mag, period=1.0, explicit=True):
    span = _OscSpan(kind, start, period, 0.0, 'bgm')
    span.end = end
    span.explicit_end = explicit
    span.set_magnitude(start, mag)
    return span


def _context(end_seconds=None):
    return modfile._OscContext({}, _clock(), None, seed=1234,
                               end_seconds=end_seconds)


def _vibrate_channel(span, end_seconds=None, zoom=None):
    return modfile.OscDeltaChannel([span], 'x', _clock(), seed=1234,
                                   end_seconds=end_seconds, zoom=zoom)


def test_vibrate_rerolls_per_frame_cell_within_bounds():
    span = _span('vibrate', 0.0, 10.0, (10.0, 10.0, 10.0))
    ch = _vibrate_channel(span)
    cells = [ch.sample(i / 60.0 + 0.001)[0] for i in range(60)]
    assert all(abs(v) <= 10.0 for v in cells)
    assert len({round(v, 6) for v in cells}) > 30   # re-rolls, not held
    # Deterministic: the same time samples the same offset.
    assert ch.sample(0.5)[0] == ch.sample(0.5)[0]
    # Within one 60Hz cell the offset holds (one teleport per frame).
    assert ch.sample(0.5001)[0] == ch.sample(0.5122)[0]


def test_vibrate_is_zero_outside_span_and_at_zero_magnitude():
    span = _span('vibrate', 1.0, 2.0, (10.0, 10.0, 10.0))
    ch = _vibrate_channel(span)
    assert ch.sample(0.5) == (0.0,)
    assert ch.sample(2.5) == (0.0,)

    zeroed = _span('vibrate', 0.0, 10.0, (0.0, 0.0, 0.0))
    assert _vibrate_channel(zeroed).sample(5.0) == (0.0,)


def test_vibrate_never_freezes_on_long_open_spans():
    """The baked path capped at ~50s of samples and then held one stuck
    random offset; the live channel keeps re-rolling for the whole
    extended span (the beat-917.5 proxy mirage sits ~150s after its
    span opens)."""
    span = _span('vibrate', 0.0, 0.0, (10.0, 10.0, 10.0), explicit=False)
    ch = _vibrate_channel(span, end_seconds=400.0)
    late = [ch.sample(290.0 + i / 60.0 + 0.001)[0] for i in range(30)]
    assert len({round(v, 6) for v in late}) > 15


def test_vibrate_amplitude_scales_with_zoom():
    from analysis.player.render.effects.timeline import EventTimeline, Keyframe
    span = _span('vibrate', 0.0, 10.0, (10.0, 10.0, 10.0))
    half = EventTimeline([Keyframe(0.0, (0.5,), 0.0, 0)], rest=(1.0,))
    full = _vibrate_channel(span)
    zoomed = _vibrate_channel(span, zoom=half)
    t = 0.201
    assert zoomed.sample(t)[0] == pytest.approx(full.sample(t)[0] * 0.5)


def test_sine_channel_matches_engine_waveform():
    span = _span('bob', 0.0, 4.0, (0.0, -100.0, 0.0), period=2.0)
    ch = modfile.OscDeltaChannel([span], 'y', _clock(), seed=0)
    for t in (0.25, 0.5, 1.3):
        expected = -100.0 * math.sin((t % 2.0) / 2.0 * 2.0 * math.pi)
        assert ch.sample(t)[0] == pytest.approx(expected)


def test_open_span_extends_to_compile_end_explicit_does_not():
    mag = (0.0, 0.0, 20.0)
    open_span = _span('wag', 0.0, 1.0, mag, period=2.0, explicit=False)
    closed = _span('wag', 0.0, 1.0, mag, period=2.0, explicit=True)
    kw = dict(prop='rotation', clock=_clock(), seed=0, end_seconds=100.0)
    assert modfile.OscDeltaChannel([open_span], **kw).sample(50.3)[0] != 0.0
    assert modfile.OscDeltaChannel([closed], **kw).sample(50.3)[0] == 0.0


def test_delta_channels_map_kinds_to_props():
    spans = [_span('vibrate', 0.0, 1.0, (10.0, 10.0, 10.0)),
             _span('wag', 2.0, 3.0, (0.0, 0.0, 20.0))]
    channels = modfile.oscillator_delta_channels(spans, _context(), seed=7)
    assert set(channels) == {'x', 'y', 'rotation'}
    # A kind with no 2D mapping yields no channels at all.
    assert modfile.oscillator_delta_channels(
        [_span('rainbow', 0.0, 1.0, (0.0, 0.0, 0.0))], _context(),
        seed=7) is None
