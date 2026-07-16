"""Parent-side AudioProcessClient clock reads, without a child process.

`current_chart_time` and `seek_to_chart_time` only touch the shared
status array and plain attributes, so these tests build a client via
`object.__new__` and drive the fields directly instead of spawning the
audio child.
"""
import time

from analysis.player.audio import audio_process as ap


def _client():
    c = object.__new__(ap.AudioProcessClient)
    c._status = [0.0] * ap._NUM_STATUS_FIELDS
    c._chart_time_floor = -float('inf')
    c._seek_gen_sent = 0
    c._last_seek_target = 0.0
    c._send = lambda op, payload: None
    return c


def _anchor(c, pos, mono, rate=1.0):
    c._status[ap._F_HW_POS] = pos
    c._status[ap._F_HW_MONO] = mono
    c._status[ap._F_HW_RATE] = rate
    c._status[ap._F_DAC_ANCHOR_VALID] = 1.0


def test_extrapolates_between_anchors():
    c = _client()
    _anchor(c, pos=10.0, mono=time.monotonic() - 0.010)
    t = c.current_chart_time()
    assert 10.005 < t < 10.1


def test_extrapolation_scales_with_rate():
    c = _client()
    _anchor(c, pos=10.0, mono=time.monotonic() - 0.010, rate=2.0)
    t = c.current_chart_time()
    assert 10.015 < t < 10.2


def test_consecutive_reads_advance():
    c = _client()
    _anchor(c, pos=10.0, mono=time.monotonic() - 0.010)
    t1 = c.current_chart_time()
    time.sleep(0.002)
    t2 = c.current_chart_time()
    assert t2 > t1


def test_monotone_floor_absorbs_anchor_regression():
    c = _client()
    _anchor(c, pos=10.0, mono=time.monotonic() - 0.010)
    t1 = c.current_chart_time()
    # A torn read can pair a fresh pos with a not-yet-written mono,
    # yielding a value below the previous read. The floor must hold.
    _anchor(c, pos=10.0, mono=time.monotonic() + 0.050)
    t2 = c.current_chart_time()
    assert t2 == t1


def test_lead_in_subtracted_from_extrapolation():
    c = _client()
    _anchor(c, pos=10.0, mono=time.monotonic() - 0.010)
    c._status[ap._F_LEAD_IN_SECONDS] = 2.0
    t = c.current_chart_time()
    assert 8.005 < t < 8.1


def test_lead_in_before_first_anchor():
    c = _client()
    c._status[ap._F_LEAD_IN_SECONDS] = 2.5
    assert c.current_chart_time() == -2.5


def test_seek_gates_stale_anchor_until_child_ack():
    c = _client()
    _anchor(c, pos=100.0, mono=time.monotonic())
    c.current_chart_time()

    c.seek_to_chart_time(5.0)
    # Child hasn't acknowledged: the anchor fields still describe the
    # pre-seek playhead and must not be read.
    assert c.current_chart_time() == 5.0

    # Child acknowledged the seek but no callback has re-anchored yet.
    c._status[ap._F_SEEK_GEN] = 1.0
    c._status[ap._F_DAC_ANCHOR_VALID] = 0.0
    assert c.current_chart_time() == 5.0

    # First post-seek callback: reads resume from the new position
    # instead of being clamped to the pre-seek floor.
    _anchor(c, pos=5.0, mono=time.monotonic() - 0.005)
    t = c.current_chart_time()
    assert 5.0 < t < 5.1
