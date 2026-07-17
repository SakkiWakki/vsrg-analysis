"""Lifetime-linter heuristics on synthetic timelines.

Each test builds a timeline by hand (a real `EventTimeline`, or a tiny
gate stub) whose visible spans and position curve are computable by
inspection, then asserts the finding it should or should not produce.
The point is to pin every heuristic's threshold behaviour and its known
false-positive guards (data-holder quads, mirror flips, transform-only
groups, momentary overshoot) without compiling a chart.
"""
import pytest

from analysis.player.render.effects.timeline import EventTimeline, Keyframe
from tools import lifetime_lint as ll


def _timeline(pairs, rest=0.0):
    """An EventTimeline of instantaneous keyframes: `pairs` is a list of
    (t, value), each a zero-duration step so sampling returns the value
    held from that t. Rest before the first keyframe is `rest`."""
    keyframes = [Keyframe(t=t, values=(v,), duration=0.0, easing=0)
                 for t, v in pairs]
    return EventTimeline(keyframes, rest=(rest,))


def _burst(t0, count, dt=1.0 / 60.0, value=0.0):
    """A per-frame driving burst: `count` keyframes at `dt` spacing from
    `t0` - the integrator-poked signature `is_burst` should recognise."""
    return _timeline([(t0 + i * dt, value) for i in range(count)])


def _const(value):
    return _timeline([], rest=value)


# --- keyframe probing + burst detection -----------------------------

def test_keyframe_times_recurses_wrappers():
    inner = _timeline([(1.0, 0.0), (2.0, 0.0)])

    class _Gate:
        def __init__(self, child):
            self._child = child

    class _Sum:
        def __init__(self, *tls):
            self._timelines = tls

    assert ll.keyframe_times(_Gate(inner)) == [1.0, 2.0]
    assert ll.keyframe_times(_Sum(inner, _timeline([(3.0, 0.0)]))) == \
        [1.0, 2.0, 3.0]


def test_is_burst_true_at_60hz():
    assert ll.is_burst([i / 60.0 for i in range(40)])


def test_is_burst_false_for_sparse_tweens():
    assert not ll.is_burst([0.0, 1.0, 2.0, 3.0])


def test_is_burst_false_below_min_keyframes():
    assert not ll.is_burst([i / 60.0 for i in range(10)])


# --- visible-span derivation ----------------------------------------

def test_hidden_bit_gates_visibility():
    hidden = _timeline([(0.0, 0.0), (2.0, 1.0)])
    entity = ll._Entity('e', 'sprite', hidden=hidden)
    spans = ll.visible_spans(entity, chart_end=4.0)
    assert len(spans) == 1
    start, end = spans[0]
    assert start == pytest.approx(0.0)
    assert end == pytest.approx(2.0, abs=ll._SAMPLE_DT)


def test_alpha_below_threshold_is_invisible():
    alpha = _const(0.0)
    entity = ll._Entity('e', 'sprite', alpha=alpha)
    assert ll.visible_spans(entity, chart_end=2.0) == []


def test_span_gate_bounds_visibility():
    class _SpanGate:
        def sample(self, t):
            return (0.0,) if 1.0 <= t <= 2.0 else (1.0,)

    entity = ll._Entity('e', 'field_copy', span_gate=_SpanGate())
    spans = ll.visible_spans(entity, chart_end=4.0)
    assert len(spans) == 1
    assert spans[0][0] == pytest.approx(1.0, abs=ll._SAMPLE_DT)


# --- HOLD-FOREVER ---------------------------------------------------

def test_hold_forever_flags_ghost_after_burst():
    # Visible 0..chart_end; a burst drives x for the first second then
    # stops while visibility runs on - the ghost-copy signature.
    chart_end = 30.0
    entity = ll._Entity(
        'ghost', 'field_copy',
        hidden=_const(0.0),
        driving={'x': _burst(0.0, 60)})
    findings = ll.lint_entity(entity, chart_end)
    holds = [f for f in findings if f.heuristic == 'HOLD-FOREVER']
    assert holds
    assert holds[0].evidence['burst_end'] == pytest.approx(59 / 60.0, abs=0.1)


def test_no_hold_forever_without_burst():
    # Visible to the end but driven only by sparse tweens: legitimate
    # static-to-end scenery, not a ghost.
    entity = ll._Entity(
        'scenery', 'sprite',
        hidden=_const(0.0),
        driving={'x': _timeline([(0.0, 0.0), (10.0, 5.0)])})
    findings = ll.lint_entity(entity, chart_end=30.0)
    assert not [f for f in findings if f.heuristic == 'HOLD-FOREVER']


# --- RUNAWAY --------------------------------------------------------

def test_runaway_flags_sustained_offscreen():
    # y ramps to 4000 (>2x the 480 box) and stays for seconds.
    entity = ll._Entity(
        'runaway', 'field_copy',
        hidden=_const(0.0),
        position={'y': _timeline([(0.0, 0.0), (0.5, 4000.0)])})
    findings = ll.lint_entity(entity, chart_end=10.0)
    runs = [f for f in findings if f.heuristic == 'RUNAWAY']
    assert runs
    assert runs[0].evidence['peak_y'] == pytest.approx(4000.0)


def test_mirror_flip_is_not_a_runaway():
    # base_scale_y = -1 is SM's basezoomy mirror; magnitude 1, not a
    # shrink-to-zero, so no RUNAWAY.
    entity = ll._Entity(
        'mirror', 'field_copy',
        hidden=_const(0.0),
        position={'base_scale_y': _const(-1.0)})
    findings = ll.lint_entity(entity, chart_end=10.0)
    assert not [f for f in findings if f.heuristic == 'RUNAWAY']


def test_momentary_overshoot_is_not_a_runaway():
    # y spikes offscreen for a single keyframe instant then returns; the
    # out-of-bounds dwell is far below the sustained-duration floor.
    entity = ll._Entity(
        'spike', 'field_copy',
        hidden=_const(0.0),
        position={'y': _timeline([(0.0, 0.0), (5.0, 4000.0),
                                  (5.0 + ll._SAMPLE_DT, 0.0)])})
    findings = ll.lint_entity(entity, chart_end=10.0)
    assert not [f for f in findings if f.heuristic == 'RUNAWAY']


# --- STUCK-VISIBLE-STATIC -------------------------------------------

def test_stuck_visible_ranks_burst_born_over_scenery():
    scenery = ll._Entity(
        'bg', 'sprite', hidden=_const(0.0),
        driving={'x': _const(0.0)})
    # A burst runs from t=4 to ~5s (driving its spawn); visibility then
    # opens at t=5 and the entity sits static (no keyframes in the span)
    # for the rest of the chart - static-but-born-via-burst.
    burst_born = ll._Entity(
        'suspect', 'field_copy',
        hidden=_timeline([(0.0, 1.0), (5.0, 0.0)]),
        driving={'x': _burst(4.0, 40)})
    chart_end = 50.0
    s_find = [f for f in ll.lint_entity(scenery, chart_end)
              if f.heuristic == 'STUCK-VISIBLE-STATIC']
    b_find = [f for f in ll.lint_entity(burst_born, chart_end)
              if f.heuristic == 'STUCK-VISIBLE-STATIC']
    assert s_find and b_find
    assert b_find[0].evidence['started_via_burst'] is True
    assert s_find[0].evidence['started_via_burst'] is False
    assert b_find[0].score > s_find[0].score


def test_short_visibility_is_not_stuck():
    entity = ll._Entity('brief', 'sprite',
                        hidden=_timeline([(0.0, 0.0), (5.0, 1.0)]))
    findings = ll.lint_entity(entity, chart_end=50.0)
    assert not [f for f in findings if f.heuristic == 'STUCK-VISIBLE-STATIC']


# --- FLASH-ORPHAN ---------------------------------------------------

def test_flash_orphan_flags_zero_length_window():
    # Hidden flips off then on within one sample step: a blink window.
    hidden = _timeline([(0.0, 1.0), (5.0, 0.0),
                        (5.0 + ll._SAMPLE_DT / 2, 1.0)])
    entity = ll._Entity('blink', 'sprite', hidden=hidden)
    findings = ll.lint_entity(entity, chart_end=10.0)
    assert [f for f in findings if f.heuristic == 'FLASH-ORPHAN']


def test_flash_orphan_flags_dead_alpha():
    # An authored alpha channel (keyframes exist) that never rises to
    # visible: a dead element. It produces no span but is reported as
    # noise regardless.
    entity = ll._Entity(
        'dead', 'sprite',
        alpha=_timeline([(0.0, 0.0), (5.0, 0.0)]))
    assert ll.visible_spans(entity, chart_end=10.0) == []
    findings = ll.lint_entity(entity, chart_end=10.0)
    dead = [f for f in findings if f.heuristic == 'FLASH-ORPHAN']
    assert dead
    assert 'alpha never' in dead[0].evidence['reason']


def test_rest_only_alpha_is_not_dead():
    # A default opaque alpha (no keyframes, rest 1.0) gated off by hidden
    # is not a dead element - it just is not showing right now.
    entity = ll._Entity('opaque', 'sprite',
                        alpha=_const(1.0), hidden=_const(1.0))
    findings = ll.lint_entity(entity, chart_end=10.0)
    assert not [f for f in findings
                if f.evidence.get('reason', '').startswith('alpha never')]


# --- PREMATURE-END --------------------------------------------------

def test_premature_end_flags_future_animation():
    # Visible a real 5s, then hidden; position keyframes scheduled at 20s.
    entity = ll._Entity(
        'early', 'sprite',
        hidden=_timeline([(0.0, 0.0), (5.0, 1.0)]),
        position={'x': _timeline([(0.0, 0.0), (20.0, 100.0)])})
    findings = ll.lint_entity(entity, chart_end=30.0)
    ends = [f for f in findings if f.heuristic == 'PREMATURE-END']
    assert ends
    assert ends[0].evidence['last_scheduled_kf'] == pytest.approx(20.0)


def test_data_holder_quad_is_not_premature_end():
    # Visible only a blink (<1s) before going permanently hidden, with
    # timelines scheduled far later: an invisible controller, not a
    # cut-short render.
    entity = ll._Entity(
        'holder', 'rect',
        hidden=_timeline([(0.0, 0.0), (0.2, 1.0)]),
        position={'x': _timeline([(0.0, 0.0), (400.0, 100.0)])})
    findings = ll.lint_entity(entity, chart_end=500.0)
    assert not [f for f in findings if f.heuristic == 'PREMATURE-END']


# --- non-drawing groups ---------------------------------------------

def test_group_transform_is_not_linted():
    # A transform-only group is never a wrong DRAW; its children carry the
    # heuristics. Even a runaway transform on the group yields nothing.
    entity = ll._Entity(
        'grp', 'group', hidden=_const(0.0), draws=False,
        position={'y': _timeline([(0.0, 0.0), (0.5, 4000.0)])})
    assert ll.lint_entity(entity, chart_end=10.0) == []


# --- ranking + sort -------------------------------------------------

def test_findings_rank_runaway_first_and_flash_last():
    entities = [
        ll._Entity('r', 'field_copy', hidden=_const(0.0),
                   position={'y': _timeline([(0.0, 0.0), (0.5, 4000.0)])}),
        ll._Entity('b', 'sprite',
                   hidden=_timeline([(0.0, 1.0), (5.0, 0.0),
                                     (5.0 + ll._SAMPLE_DT / 2, 1.0)])),
    ]
    findings = ll.lint_entities(entities, chart_end=10.0)
    assert findings[0].heuristic == 'RUNAWAY'
    assert findings[-1].heuristic == 'FLASH-ORPHAN'
