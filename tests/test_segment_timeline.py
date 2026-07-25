"""Property harness for the append-only segment timeline.

The load-bearing property: a timeline built by ONLINE pokes must
reproduce the same playback the batch pipeline gives (raw Keyframes ->
`simplify_instants` -> `EventTimeline`), because the online corridor is
the same collapse run incrementally. Both paths promise the poked
values back within SIMPLIFY_EPS, so they may differ from each other by
at most 2 * SIMPLIFY_EPS at any poked time.
"""
import bisect
import math
import random

import pytest

from analysis.player.render.effects.easing import ease
from analysis.player.render.effects.timeline import (
    EventTimeline, Keyframe, SIMPLIFY_EPS, simplify_instants)
from analysis.player.render.segment_timeline import (
    Cursor, OSC_SIN, OSC_TRIANGLE, SegmentTimeline)


def _batch_timeline(pokes, rest):
    frames = [Keyframe(t, (v,), 0.0, 0) for t, v in pokes]
    return EventTimeline(simplify_instants(frames), (rest,))


def _online_timeline(pokes, rest):
    tl = SegmentTimeline(rest=rest)
    for t, v in pokes:
        tl.poke(t, v)
    tl.finish()
    return tl


def _poke_stream(rng, n=400, step=1.0 / 60.0):
    """A sim-shaped stream: alternating windows of constants, linear
    ramps with sub-eps noise, and value jumps, poked on a tick grid."""
    pokes = []
    t, v = 0.0, rng.uniform(-100.0, 100.0)
    while len(pokes) < n:
        mode = rng.randrange(3)
        span = rng.randrange(5, 40)
        slope = rng.uniform(-50.0, 50.0) if mode == 1 else 0.0
        if mode == 2:
            v = rng.uniform(-100.0, 100.0)
        for _ in range(span):
            noise = rng.uniform(-0.4, 0.4) * SIMPLIFY_EPS
            pokes.append((t, v + noise))
            t += step
            v += slope * step
    return pokes[:n]


@pytest.mark.parametrize('seed', range(20))
def test_online_matches_batch_at_poked_times(seed):
    rng = random.Random(seed)
    rest = rng.uniform(-10.0, 10.0)
    pokes = _poke_stream(rng)

    online = _online_timeline(pokes, rest)
    batch = _batch_timeline(pokes, rest)
    for t, v in pokes:
        got = online.sample(t)
        assert got == pytest.approx(batch.sample(t)[0], abs=2 * SIMPLIFY_EPS)
        assert got == pytest.approx(v, abs=SIMPLIFY_EPS + 1e-9)


@pytest.mark.parametrize('seed', range(20))
def test_online_matches_batch_between_pokes(seed):
    rng = random.Random(seed)
    pokes = _poke_stream(rng)

    online = _online_timeline(pokes, 0.0)
    batch = _batch_timeline(pokes, 0.0)
    t_end = pokes[-1][0]
    for i in range(1000):
        t = rng.uniform(-1.0, t_end + 1.0)
        assert online.sample(t) == pytest.approx(
            batch.sample(t)[0], abs=2 * SIMPLIFY_EPS)


def test_collinear_run_collapses_to_one_segment():
    tl = SegmentTimeline()
    for i in range(10_000):
        tl.poke(i / 60.0, 3.0 + 2.0 * i / 60.0)
    tl.finish()
    assert len(tl) == 1


def test_constant_run_collapses_to_head_hold():
    tl = SegmentTimeline()
    for i in range(10_000):
        tl.poke(i / 60.0, 7.0)
    tl.finish()
    assert len(tl) == 1
    assert tl.sample(83.0) == 7.0


def test_compression_on_piecewise_content():
    rng = random.Random(1)
    pokes = _poke_stream(rng, n=4000)
    tl = _online_timeline(pokes, 0.0)
    assert len(tl) < len(pokes) / 10


def test_ramp_matches_event_timeline_exactly():
    rng = random.Random(2)
    tl = SegmentTimeline(rest=5.0)
    frames = []
    t = 0.0
    for _ in range(50):
        dur = rng.uniform(0.1, 2.0)
        v0, v1 = rng.uniform(-50, 50), rng.uniform(-50, 50)
        ease_id = rng.randrange(0, 12)
        tl.add_ramp(t, t + dur, v0, v1, ease_id)
        frames.append(Keyframe(t, (v1,), dur, ease_id, start=(v0,)))
        t += dur + rng.uniform(0.0, 1.0)
    tl.finish()
    batch = EventTimeline(frames, (5.0,))
    # The timeline stores endpoints (t0, t1) where Keyframe stores
    # (t, duration); the one-ulp difference in the eased fraction's
    # denominator bounds the achievable parity.
    for i in range(2000):
        q = rng.uniform(-1.0, t + 1.0)
        assert tl.sample(q) == pytest.approx(batch.sample(q)[0], abs=1e-6)


def test_finger_matches_bisect_under_mixed_access():
    rng = random.Random(3)
    pokes = _poke_stream(rng, n=2000)
    tl = _online_timeline(pokes, 0.0)
    t_end = pokes[-1][0]

    cursor = Cursor()
    t = 0.0
    for _ in range(5000):
        if rng.random() < 0.9:
            t += rng.uniform(0.0, 0.05)
        else:
            t = rng.uniform(-1.0, t_end + 1.0)
        assert tl.sample(t, cursor) == tl.sample(t)


def test_before_first_content_returns_rest():
    tl = SegmentTimeline(rest=42.0)
    tl.poke(10.0, 1.0)
    tl.finish()
    assert tl.sample(9.999) == 42.0
    assert tl.sample(10.0) == 1.0


def test_frontier_clamps_reads():
    tl = SegmentTimeline(rest=0.0)
    tl.poke(0.0, 1.0)
    tl.poke(1.0, 1.0)
    tl.poke(2.0, 5.0)
    tl.publish(1.0)
    assert tl.sample(50.0) == tl.sample(1.0)
    tl.finish()
    assert tl.sample(50.0) == 5.0


def test_open_run_reads_are_live_and_converge():
    tl = SegmentTimeline()
    slope_points = [(i / 60.0, 2.0 * i / 60.0) for i in range(120)]
    for t, v in slope_points:
        tl.poke(t, v)
        tl.publish(t)
        assert tl.sample(t) == pytest.approx(v, abs=SIMPLIFY_EPS)

    mid_t = 1.0
    provisional = tl.sample(mid_t)
    tl.finish()
    assert tl.sample(mid_t) == pytest.approx(provisional, abs=2 * SIMPLIFY_EPS)


def test_structural_hold_breaks_runs_like_batch():
    # Batch keeps non-plain instants verbatim and restarts collapse
    # runs after them; a poke run following a structural hold must
    # anchor at its own head, not drag the pre-hold corridor along.
    tl = SegmentTimeline()
    tl.poke(0.0, 0.0)
    tl.add_hold(5.0, 0.0)
    tl.poke(14.0, 0.1)
    tl.poke(15.0, 0.2)
    tl.poke(16.0, 0.3)
    tl.finish()
    assert tl.sample(14.5) == pytest.approx(0.15, abs=2 * SIMPLIFY_EPS)


def test_open_two_point_run_steps_instead_of_bridging():
    tl = SegmentTimeline(rest=0.0)
    tl.poke(0.0, 1.0)
    tl.poke(30.0, 0.0)
    tl.publish(60.0)
    assert tl.sample(15.0) == 1.0
    assert tl.sample(30.0) == 0.0
    tl.finish()
    assert tl.sample(15.0) == 1.0
    assert tl.sample(45.0) == 0.0


def test_osc_evaluation_and_hold_after():
    tl = SegmentTimeline()
    tl.add_osc(0.0, 4.0, base=10.0, mag=3.0, period=2.0)
    tl.finish()
    assert tl.sample(0.0) == pytest.approx(10.0)
    assert tl.sample(0.5) == pytest.approx(13.0)
    assert tl.sample(1.5) == pytest.approx(7.0)
    assert tl.sample(4.0) == pytest.approx(10.0)
    assert tl.sample(100.0) == pytest.approx(10.0)


def test_triangle_osc_shape():
    tl = SegmentTimeline()
    tl.add_osc(0.0, 8.0, base=0.0, mag=1.0, period=4.0,
               shape_id=OSC_TRIANGLE)
    tl.finish()
    assert tl.sample(1.0) == pytest.approx(1.0)
    assert tl.sample(3.0) == pytest.approx(-1.0)


def test_slab_interpolates_and_holds():
    tl = SegmentTimeline()
    tl.add_slab(1.0, hz=10.0, samples=[0.0, 1.0, 4.0, 9.0])
    tl.finish()
    assert tl.sample(1.0) == pytest.approx(0.0)
    assert tl.sample(1.05) == pytest.approx(0.5)
    assert tl.sample(1.25) == pytest.approx(6.5)
    assert tl.sample(1.3) == pytest.approx(9.0)
    assert tl.sample(9.0) == pytest.approx(9.0)


def test_mixed_kinds_in_time_order():
    tl = SegmentTimeline(rest=0.0)
    tl.poke(0.0, 1.0)
    tl.poke(0.5, 1.0)
    tl.add_ramp(1.0, 2.0, 1.0, 10.0)
    tl.add_osc(3.0, 5.0, base=10.0, mag=2.0, period=1.0)
    tl.add_slab(6.0, hz=2.0, samples=[10.0, 20.0, 30.0])
    tl.finish()

    assert tl.sample(0.25) == 1.0
    assert tl.sample(1.5) == pytest.approx(5.5)
    assert tl.sample(2.5) == 10.0
    assert tl.sample(3.25) == pytest.approx(12.0)
    assert tl.sample(5.5) == 10.0
    assert tl.sample(6.5) == pytest.approx(20.0)
    assert tl.sample(99.0) == 30.0


def test_same_time_pokes_break_runs_without_loss():
    tl = SegmentTimeline()
    tl.poke(1.0, 5.0)
    tl.poke(1.0, 9.0)
    tl.poke(2.0, 9.0)
    tl.finish()
    assert tl.sample(1.0) == 9.0
    assert tl.sample(1.5) == 9.0


def test_same_time_pair_at_run_tail_keeps_step_semantics():
    """A duplicate write at the run TAIL (playcommand handlers on one
    actor often re-set a property at the same clock) is a zero-tween
    chain step: it must NOT count as a third corridor sample, or a
    sparse two-point step pair collapses into a ramp bridging the whole
    gap (a `hidden` flip smeared over 8 seconds)."""
    pokes = [(2.0, 0.0), (10.0, 1.0), (10.0, 1.0)]

    online = _online_timeline(pokes, 0.0)
    for t in (3.0, 6.0, 9.9):
        assert online.sample(t) == 0.0
    assert online.sample(10.0) == 1.0
    assert online.sample(11.0) == 1.0

    batch = _batch_timeline(pokes, 0.0)
    for t in (3.0, 6.0, 9.9):
        assert batch.sample(t)[0] == 0.0
    assert batch.sample(11.0)[0] == 1.0


def test_same_time_pair_at_run_tail_last_value_wins():
    pokes = [(2.0, 0.0), (10.0, 1.0), (10.0, 0.0), (12.0, 0.0)]

    online = _online_timeline(pokes, 0.0)
    batch = _batch_timeline(pokes, 0.0)
    for t in (6.0, 10.0, 11.0, 13.0):
        assert online.sample(t) == batch.sample(t)[0]
    assert online.sample(6.0) == 0.0
    assert online.sample(11.0) == 0.0


def test_out_of_order_segment_start_asserts():
    tl = SegmentTimeline()
    tl.add_ramp(5.0, 6.0, 0.0, 1.0)
    with pytest.raises(AssertionError):
        tl.add_ramp(1.0, 2.0, 0.0, 1.0)


# ── breakpoint export ----------------------------------------------------

def _replay(ts, vals, durs, eases, rest, t):
    """Sample the breakpoint arrays the way a consumer channel does
    (storyboard_native channels.rs): rest before the first, hold when the
    duration is non-positive, else ease toward the next value."""
    i = bisect.bisect_right(ts, t) - 1
    if i < 0:
        return rest
    if durs[i] <= 0.0 or i + 1 >= len(vals):
        return vals[i]
    u = min(1.0, max(0.0, (t - ts[i]) / durs[i]))
    return vals[i] + (vals[i + 1] - vals[i]) * ease(eases[i], u)


def _replays_the_timeline(tl, lo, hi, samples=2001):
    """Max deviation between the exported breakpoints' replay and the
    timeline's own `sample` across [lo, hi]."""
    exported = tl.breakpoints(hi + 1.0)
    assert exported is not None
    ts, vals, durs, eases = exported
    return max(abs(_replay(ts, vals, durs, eases, tl._rest, t) - tl.sample(t))
               for t in (lo + (hi - lo) * i / samples for i in range(samples + 1)))


def test_breakpoints_replay_holds_ramps_and_slabs_exactly():
    tl = SegmentTimeline(rest=5.0)
    tl.add_hold(1.0, 10.0)
    tl.add_ramp(2.0, 3.0, 10.0, 30.0)
    tl.add_ramp(4.0, 6.0, 30.0, -10.0, -3)   # an SM tween curve, not linear
    tl.add_slab(7.0, 10.0, [1.0, 2.0, 5.0, 4.0])
    tl.finish()
    assert _replays_the_timeline(tl, 0.0, 12.0) < 1e-6


def test_breakpoints_replay_a_collapsed_poke_run_exactly():
    tl = SegmentTimeline(rest=0.0)
    for k in range(200):
        tl.poke(k / 20.0, math.sin(k / 20.0))
    tl.finish()
    assert _replays_the_timeline(tl, 0.0, 12.0) < 1e-6


def test_breakpoints_track_an_open_poke_run():
    # No finish(): the run is still open, and `sample` gives it priority over
    # the sealed directory, so the export must mirror `_run_value`.
    tl = SegmentTimeline(rest=0.0)
    tl.frontier = math.inf
    tl.add_hold(0.0, -1.0)
    for k in range(20):
        tl.poke(1.0 + k / 10.0, 2.0 * k)
    assert _replays_the_timeline(tl, 0.0, 6.0) < 1e-6


def test_breakpoints_approximate_an_oscillator_within_a_pixel():
    tl = SegmentTimeline(rest=0.0)
    tl.add_osc(0.0, 4.0, 100.0, 20.0, 0.5, shape_id=OSC_SIN)
    tl.finish()
    # The one kind with no exact breakpoint form (the consumer ramps
    # linearly); traced fine enough to stay sub-visible in design pixels.
    assert _replays_the_timeline(tl, 0.0, 6.0) < 0.5


def test_breakpoints_stop_at_the_requested_end():
    tl = SegmentTimeline(rest=0.0)
    for t in (1.0, 2.0, 3.0, 40.0, 50.0):
        tl.add_hold(t, t)
    tl.finish()
    ts, _vals, _durs, _eases = tl.breakpoints(3.5)
    assert ts == [1.0, 2.0, 3.0]


def test_breakpoints_stay_ordered_when_a_ramp_is_interrupted():
    # A segment ends where the NEXT one begins, whatever span its own row
    # claims: `sample` picks by bisect over the directory, so a later segment
    # takes over and truncates whatever was still ramping. Exporting the full
    # span pushed breakpoints PAST the interrupting segment, and since the
    # consumer binary-searches a non-monotonic array it then read arbitrary
    # values - 1979 of Bonfire's 5447 alpha breakpoints went backwards and an
    # element at alpha 0.99 composited at 0.0009.
    tl = SegmentTimeline(rest=0.0)
    tl.add_ramp(0.0, 10.0, 0.0, 100.0)   # a long ramp ...
    tl.add_hold(1.0, 7.0)                # ... cut off after one second
    tl.add_ramp(2.0, 9.0, 7.0, 70.0)     # and itself cut off
    tl.add_hold(3.0, 42.0)
    tl.finish()

    ts, _vals, _durs, _eases = tl.breakpoints(20.0)
    assert ts == sorted(ts), f'breakpoints went backwards: {ts}'
    assert _replays_the_timeline(tl, 0.0, 12.0) < 1e-6


def test_an_interrupted_ramp_replays_its_partial_value():
    # The truncation has to keep the interrupted span's OWN shape, not ramp
    # straight to whatever the next segment holds: at the cut the curve is
    # partway up its original ramp.
    tl = SegmentTimeline(rest=0.0)
    tl.add_ramp(0.0, 10.0, 0.0, 100.0)
    tl.add_hold(2.0, 0.0)
    tl.finish()

    ts, vals, durs, eases = tl.breakpoints(20.0)
    # 20% into a 0->100 ramp when the hold takes over.
    assert _replay(ts, vals, durs, eases, tl._rest, 1.999) == pytest.approx(
        20.0, abs=0.1)
    assert _replay(ts, vals, durs, eases, tl._rest, 2.001) == 0.0


def test_an_interrupted_curved_ease_keeps_its_shape():
    # Shortening the DURATION of an eased span replays the whole ease curve
    # over the surviving part: right at both ends, wrong through the middle.
    # Corrupted's alpha came out 13% low that way.
    tl = SegmentTimeline(rest=0.0)
    tl.add_ramp(0.0, 10.0, 0.0, 100.0, -3)   # an SM tween curve
    tl.add_hold(2.0, 0.0)
    tl.finish()
    assert _replays_the_timeline(tl, 0.0, 4.0) < 0.5
