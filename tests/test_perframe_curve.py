"""Per-frame beat-keyed mod curves (the low-fps fix).

A per-frame driver re-fires a mod densely, its value tracking a function
of the song beat (Crazy Shuffle: alt = 30*sin(beat*pi)). The raw fires
ARE the intended continuous curve; running them through the time-keyed
approach chase distorts a dense fire sequence into a jagged staircase.
These pin: dense pure-in-beat streams become beat curves sampled at
beat_now, sparse streams and accumulators keep the chase, and a mod
lives in exactly one representation."""
import numpy as np
import pytest

from analysis.games.notitg.sim import record
from analysis.player.render.mods.channels import ModChannels, ModEvent


def _applied(rows):
    """`(t, beat, modstring, player)` rows in the sim's applied-mods
    shape. t == beat here (1 beat/sec) so frame-resolution keeps every
    distinct-beat fire."""
    return [(float(b), float(b), s, p) for b, s, p in rows]


# parse_modstring converts a percent to a fraction (50% -> 0.5), so a
# fire of `50*sin(beat)` percent lands as `0.5*sin(beat)` in the curve.
def _sine_percent(beat):
    return 50.0 * np.sin(beat)


def _sine_fraction(beat):
    return _sine_percent(beat) / 100.0


def _sine_fires(mod, n=64, beat0=0.0, step=0.2):
    """A dense per-frame driver painting 50*sin(beat) percent: one snap
    fire per body tick, the shape the mpf bodies produce."""
    rows = []
    for i in range(n):
        beat = beat0 + i * step
        rows.append((beat, f'*10000 {_sine_percent(beat):g} {mod}', None))
    return _applied(rows)


def test_dense_sine_becomes_beat_curve():
    curves = record.perframe_curves(_sine_fires('drunk'))
    assert ('drunk', 0) in curves and ('drunk', 1) in curves
    beats, values = curves[('drunk', 0)]
    assert len(beats) >= record._CURVE_MIN_FIRES
    # strictly increasing beat axis (interp precondition)
    assert all(b1 > b0 for b0, b1 in zip(beats, beats[1:]))


def test_beat_curve_is_smooth_where_the_chase_was_jagged():
    """Sampling the beat curve at display rate tracks the smooth sine;
    the 2nd difference stays tiny (the chase produced ~0.2)."""
    curves = record.perframe_curves(_sine_fires('drunk'))
    channels = ModChannels.compile([], beat_curves={
        k: v for k, v in curves.items() if k[1] == 0})
    fine = np.arange(1.0, 10.0, 1.0 / 240.0)
    sampled = np.array([channels.value('drunk', t, 0, beat_now=t)
                        for t in fine])
    assert np.abs(np.diff(sampled, 2)).max() < 0.01


def test_beat_curve_reproduces_every_fire_value():
    """The curve passes exactly through the per-frame fire values (the
    engine ground truth); interpolation only fills between them."""
    fires = _sine_fires('drunk', n=40)
    curves = record.perframe_curves(fires)
    channels = ModChannels.compile([], beat_curves={('drunk', 0):
                                                    curves[('drunk', 0)]})
    for _t, beat, modstring, _p in fires:
        # the fire percent is formatted %g (6 sig figs), so compare to the
        # value the modstring actually parsed to, within that rounding.
        assert channels.value('drunk', beat, 0, beat_now=beat) == \
            pytest.approx(_sine_fraction(beat), abs=1e-6)


def test_beat_curve_rests_at_zero_outside_its_span():
    """Before the first fire and after the last, the mod RESTS at 0 - the
    driver only paints its perframe window. Holding the endpoint flat
    would leave a z-push/zoom mod fully on for the whole song before its
    window, exploding note size (the "notes spawn from the camera" bug)."""
    curves = record.perframe_curves(_sine_fires('bounce', n=40,
                                                beat0=60.0, step=0.2))
    channels = ModChannels.compile([], beat_curves={('bounce', 0):
                                                    curves[('bounce', 0)]})
    assert channels.value('bounce', 0.0, 0, beat_now=0.0) == 0.0
    assert channels.value('bounce', 10.0, 0, beat_now=10.0) == 0.0
    assert channels.value('bounce', 999.0, 0, beat_now=999.0) == 0.0
    # inside the span it tracks the curve (nonzero somewhere)
    mid = channels.value('bounce', 0.0, 0, beat_now=63.0)
    assert mid != 0.0


def test_sparse_stream_stays_on_the_chase():
    """A handful of sparse windows is a step sequence the chase renders
    exactly - not a per-frame curve."""
    sparse = _applied([(0.0, '*1 50 drunk', None),
                       (8.0, '*1 0 drunk', None)])
    assert record.perframe_curves(sparse) == {}


def test_same_beat_disagreement_is_not_a_curve():
    """The purity guard: fires that revisit the same beat with different
    values are not a function of beat, so `_is_perframe_curve` rejects
    them (an accumulator signature). Built directly at the fires level
    since frame-resolution collapses same-FRAME calls to last-wins."""
    fires = [(i * 0.1, float(i)) for i in range(20)]
    fires += [(1.0, 999.0)]  # revisit beat 1.0 with a clashing value
    fires.sort()
    assert not record._is_perframe_curve(fires)


def test_curve_mod_excluded_from_chase_windows():
    """A mod handled as a beat curve is dropped from the chased windows
    (one representation): compile_mod_channels must not also build a
    time-chase channel for it."""
    from analysis.games.notitg.mod_channels import compile_mod_channels
    events = [{'t_start': b * 0.2, 't_end': b * 0.2 + 0.017,
               'modstring': f'*10000 {50.0 * np.sin(b * 0.2):g} drunk',
               'player': None}
              for b in range(40)]
    curves = record.perframe_curves(_sine_fires('drunk', n=40))
    channels = compile_mod_channels(events, beat_curves={
        k: v for k, v in curves.items()})
    assert ('drunk', 0) in channels._beat_curves
    assert ('drunk', 0) not in channels._channels


def test_values_at_merges_curve_and_chase():
    """values_at returns beat-curve mods (at beat_now) alongside chase
    mods (at t) in one percents dict."""
    curves = record.perframe_curves(_sine_fires('drunk', n=40))
    channels = ModChannels.compile(
        [ModEvent(0.0, 0.5, 1.0, 'mini', 0)],
        beat_curves={('drunk', 0): curves[('drunk', 0)]})
    percents = channels.values_at(5.0, 0, beat_now=5.0)
    assert 'drunk' in percents and 'mini' in percents
    assert percents['drunk'] == pytest.approx(_sine_fraction(5.0), abs=1e-6)
