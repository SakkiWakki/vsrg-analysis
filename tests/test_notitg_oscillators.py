"""Live oscillator delta channels: seeded per-frame vibrate (the
receptor mirage), analytic sines, open-span extension, no sample-cap
freeze."""
from bisect import bisect_right
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


def _bake(span, base_keyframes=None, end_seconds=None):
    """Bake one span into keyframes via the compile path (the synthesis
    the pulse/pulseramp zoom oscillators use), returning the merged dict."""
    return modfile.compile_oscillator_keyframes(
        [span], dict(base_keyframes or {}), _clock(), rng=None,
        end_seconds=end_seconds)


def _sample(keyframes, prop, t, rest):
    from analysis.player.render.effects.timeline import EventTimeline
    return EventTimeline(keyframes.get(prop, []), rest=(rest,)).sample(t)[0]


def test_pulse_zoom_factor_matches_engine_scale():
    """pulse multiplies scale by SCALE(sin(pct*pi), 0,1, min, max)
    (Actor.cpp:360-366). With no base zoom the baked scale IS that factor."""
    span = _span('pulse', 0.0, 4.0, (0.5, 1.0, 0.0), period=2.0)
    kf = _bake(span)
    assert set(kf) == {'scale_x', 'scale_y'}
    for t in (0.5, 1.0, 1.7):
        frac = math.sin((t % 2.0) / 2.0 * math.pi)
        expected = 0.5 + (1.0 - 0.5) * frac
        assert _sample(kf, 'scale_x', t, 1.0) == pytest.approx(expected)
        assert _sample(kf, 'scale_y', t, 1.0) == pytest.approx(expected)


def test_pulse_multiplies_the_tweened_base_zoom():
    """The factor rides the actor's base scale (scale *= factor), not
    replaces it: a base zoom of 2.0 doubles every pulse sample."""
    from analysis.player.render.effects.timeline import Keyframe
    base = {'scale_x': [Keyframe(0.0, (2.0,), 0.0, 0)]}
    span = _span('pulse', 0.0, 4.0, (0.5, 1.0, 0.0), period=2.0)
    kf = _bake(span, base)
    t = 1.0   # sin(pi/2)=1 -> factor 1.0; base 2.0 -> 2.0
    frac = math.sin((t % 2.0) / 2.0 * math.pi)
    assert _sample(kf, 'scale_x', t, 1.0) == pytest.approx(2.0 * (0.5 + 0.5 * frac))


def test_pulseramp_is_the_sawtooth_sibling():
    """pulseramp feeds the raw percent-through (a sawtooth) into SCALE,
    not the abs-sine, mirroring diffuse_ramp vs diffuse_shift."""
    span = _span('pulseramp', 0.0, 4.0, (0.5, 1.0, 0.0), period=2.0)
    kf = _bake(span)
    for t in (0.5, 1.0, 1.9):
        pct = (t % 2.0) / 2.0
        expected = 0.5 + (1.0 - 0.5) * pct
        assert _sample(kf, 'scale_x', t, 1.0) == pytest.approx(expected)


def test_pulse_rests_at_identity_after_the_span():
    """The trailing rest returns scale to 1.0 (multiplicative identity),
    not 0.0, so a stopped pulse hands zoom back to the base untouched."""
    span = _span('pulse', 0.0, 2.0, (0.5, 1.0, 0.0), period=2.0)
    kf = _bake(span)
    assert _sample(kf, 'scale_x', 5.0, 1.0) == pytest.approx(1.0)


def test_pulse_absent_from_the_additive_field_channel():
    """The live field-instance channel sums additive deltas; a zoom
    oscillator does not belong to it, so a pulse-only actor yields no
    live channel (its synthesis is the bake path only)."""
    assert modfile.oscillator_delta_channels(
        [_span('pulse', 0.0, 1.0, (0.5, 1.0, 0.0))], _context(),
        seed=7) is None


# --------------------------------------------------------------------------
# Channel export: what a piecewise linear-ramp consumer is handed
# --------------------------------------------------------------------------

def _replay(exported, rest, t):
    """The consumer model `breakpoints` targets: rest before the first
    breakpoint, then hold (dur <= 0) or ramp toward the next value. Ties in
    time go to the LATER breakpoint, as the consumer's bisect does."""
    ts, vals, durs, _eases = exported
    index = bisect_right(ts, t) - 1
    if index < 0:
        return rest
    if durs[index] <= 0.0 or index + 1 >= len(ts):
        return vals[index]
    return vals[index] + (vals[index + 1] - vals[index]) * min(
        1.0, (t - ts[index]) / durs[index])


def _probe_times(t0, t1, n=997):
    """Times spread over the window, deliberately off any 60Hz grid."""
    return [t0 + (t1 - t0) * (i + 0.31415) / n for i in range(n)]


def test_vibrate_exports_as_one_hold_per_cell():
    span = _span('vibrate', 1.0, 4.0, (10.0, 10.0, 10.0))
    ch = _vibrate_channel(span)
    exported = ch.breakpoints(0.0, 6.0)
    ts, _vals, durs, _eases = exported
    assert all(d == 0.0 for d in durs)          # steps, never ramps
    assert ts == sorted(ts)
    for t in _probe_times(0.0, 6.0):
        assert _replay(exported, 0.0, t) == pytest.approx(ch.sample(t)[0])


def test_no_cells_where_the_magnitude_envelope_is_zero():
    """An effect the chart never stops runs to the compile end, and gat 2
    holds the magnitude at zero for most of that - 29K breakpoints of which
    3K moved. A cell at zero magnitude describes a flat line at rest."""
    span = _span('vibrate', 0.0, 30.0, (10.0, 10.0, 10.0))
    span.set_magnitude(5.0, (0.0, 0.0, 0.0))
    span.set_magnitude(25.0, (10.0, 10.0, 10.0))
    ch = _vibrate_channel(span)

    exported = ch.breakpoints(0.0, 30.0)
    ts = exported[0]
    assert not [t for t in ts if 5.0 < t < 25.0]
    assert len([t for t in ts if t < 5.0]) > 250        # 60Hz over 5s
    assert len([t for t in ts if t > 25.0]) > 250
    for t in _probe_times(0.0, 30.0):
        assert _replay(exported, 0.0, t) == pytest.approx(ch.sample(t)[0])


def test_a_zeroed_axis_lays_no_cells_while_the_other_still_shakes():
    """effectmagnitude is per axis: a vibrate driving y alone must not put a
    grid on x, which multiplies the same waveform by zero."""
    span = _span('vibrate', 0.0, 10.0, (0.0, 10.0, 10.0))
    on_x = modfile.OscDeltaChannel([span], 'x', _clock(), seed=7)
    on_y = modfile.OscDeltaChannel([span], 'y', _clock(), seed=7)

    assert on_x.breakpoints(0.0, 10.0)[0] == []
    assert len(on_y.breakpoints(0.0, 10.0)[0]) > 500


def test_a_continuous_kind_exports_as_ramps():
    span = _span('bob', 0.0, 4.0, (20.0, 20.0, 20.0), period=1.0)
    ch = modfile.OscDeltaChannel([span], 'y', _clock(), seed=99)
    exported = ch.breakpoints(0.0, 6.0)
    _ts, _vals, durs, _eases = exported
    assert any(d > 0.0 for d in durs)
    for t in _probe_times(0.0, 6.0):
        assert _replay(exported, 0.0, t) == pytest.approx(ch.sample(t)[0],
                                                          abs=0.05)


def test_the_delta_sum_exports_the_teleport_the_grid_would_alias():
    """A dense grid cannot see a 60Hz staircase - the field came out 43px
    off LINARIA's legacy placement that way. The union export replays the
    sum of the base and the delta exactly."""
    from analysis.games.notitg.field_compose import _SumTimeline
    from analysis.player.render.effects.timeline import EventTimeline, Keyframe

    base = EventTimeline([Keyframe(0.0, (100.0,), 0.0, 0),
                          Keyframe(2.0, (-40.0,), 0.0, 0)], rest=(0.0,))
    delta = _vibrate_channel(_span('vibrate', 1.0, 4.0, (30.0, 30.0, 30.0)))
    total = _SumTimeline((base, delta))

    exported = total.breakpoints(0.0, 6.0)
    assert exported is not None
    ts, _vals, _durs, _eases = exported
    assert ts == sorted(ts)
    for t in _probe_times(0.0, 6.0):
        assert _replay(exported, 0.0, t) == pytest.approx(total.sample(t)[0])


def test_the_delta_sum_subdivides_a_part_that_curves():
    """The union export reads the sum back as straight lines between the
    parts' own breakpoints, which a CURVED span is not - so the curve
    subdivides its own span into the union. A recorded segment lane carries
    its ease id this way (disperagioia's field was 16px off when the export
    gave up on one instead)."""
    from analysis.games.notitg.field_compose import _SumTimeline

    class _CurvedLane:
        def sample(self, t):
            return (float(t) ** 2,)

        def breakpoints(self, t0, t1, index=0):
            return [t0], [t0 * t0], [t1 - t0], [3]

    delta = _vibrate_channel(_span('vibrate', 1.0, 4.0, (30.0, 30.0, 30.0)))
    total = _SumTimeline((_CurvedLane(), delta))
    exported = total.breakpoints(0.0, 6.0)
    assert exported is not None
    for t in _probe_times(0.0, 6.0):
        assert _replay(exported, 0.0, t) == pytest.approx(total.sample(t)[0],
                                                          abs=1e-3)


def test_a_sum_of_ducks_with_no_shape_of_their_own_declines():
    from analysis.games.notitg.field_compose import _SumTimeline

    class _Duck:
        def sample(self, t):
            return (math.sin(t),)

    assert _SumTimeline((_Duck(), _Duck())).breakpoints(0.0, 1.0) is None
