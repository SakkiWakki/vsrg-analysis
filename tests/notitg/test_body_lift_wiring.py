"""Phase 3 wiring spec: the frame-IR closed-form split lifts a pure
f(clock) body poke into baked keyframes over its guard window. gat's
own body pokes are derived-actor reads the router (correctly) refuses;
this pins the pipe itself so router expansions land against a green
baseline."""
import math

import pytest

from analysis.games.notitg.frame_compile import compile_update
from analysis.player.render.expr.surface import UNRESOLVED


class _BeatClock:
    def __call__(self, t):
        return t * 2.0


class _DriverSurface:
    def symbol(self, name):
        return UNRESOLVED

    def method(self, *args):
        return UNRESOLVED

    def clock_reader(self, name):
        return _BeatClock() if name == 'beat' else None


BODY = """
if beat >= 8 and beat < 16 then
    P1:x(100 + 50*math.sin(beat))
end
"""


def test_pure_clock_poke_bakes_over_its_window():
    plan = compile_update(BODY, _DriverSurface(), lambda b: b * 0.5)
    assert plan.closed == 1
    kfs = plan.closed_keyframes[('P1', 'x')]

    assert kfs[0].t == pytest.approx(4.0)
    assert kfs[-1].t <= 8.0
    for kf in kfs[:20]:
        beat = kf.t * 2.0
        assert kf.values[0] == pytest.approx(
            100 + 50 * math.sin(beat), abs=1e-6)


def test_derived_actor_read_stays_evaluated():
    body = """
if beat >= 8 and beat < 16 then
    P1:x(other:GetX() * 2)
end
"""
    plan = compile_update(body, _DriverSurface(), lambda b: b * 0.5)
    assert plan.closed == 0
    assert plan.evaluated >= 1
    assert plan.evaluated_windows


class _Lane:
    """A stub preview lane: y = 3t."""

    def sample(self, t):
        return 3.0 * t


def test_derived_read_lifts_through_preview_lanes():
    body = """
if beat >= 8 and beat < 16 then
    P1:x(other:GetX() * 2)
end
"""
    plan = compile_update(body, _DriverSurface(), lambda b: b * 0.5,
                          preview_lanes={'other': {'x': [_Lane()]}})
    assert plan.closed == 1
    kfs = plan.closed_keyframes[('P1', 'x')]
    for kf in kfs[:20]:
        assert kf.values[0] == pytest.approx(6.0 * kf.t, abs=1e-6)


def test_derived_read_refused_when_body_writes_the_source():
    body = """
if beat >= 8 and beat < 16 then
    other:x(beat)
    P1:x(other:GetX() * 2)
end
"""
    plan = compile_update(body, _DriverSurface(), lambda b: b * 0.5,
                          preview_lanes={'other': {'x': [_Lane()]}})
    assert ('P1', 'x') not in plan.closed_keyframes


def test_local_indirection_substitutes_and_bakes():
    body = """
if beat >= 8 and beat < 16 then
    local amp = 50
    local wob = math.sin(beat) * amp
    P1:x(100 + wob)
end
"""
    plan = compile_update(body, _DriverSurface(), lambda b: b * 0.5)
    kfs = plan.closed_keyframes.get(('P1', 'x'))
    assert kfs, 'local-routed poke should bake'
    for kf in kfs[:20]:
        assert kf.values[0] == pytest.approx(
            100 + 50 * math.sin(kf.t * 2.0), abs=1e-6)


def test_local_holding_derived_read_lifts_with_lanes():
    body = """
if beat >= 8 and beat < 16 then
    local wagx = other:GetX()
    P1:x(wagx * 2)
end
"""
    plan = compile_update(body, _DriverSurface(), lambda b: b * 0.5,
                          preview_lanes={'other': {'x': [_Lane()]}})
    kfs = plan.closed_keyframes.get(('P1', 'x'))
    assert kfs, 'derived read through a local should bake'
    for kf in kfs[:20]:
        assert kf.values[0] == pytest.approx(6.0 * kf.t, abs=1e-6)


def test_unbound_local_stays_evaluated():
    body = """
if beat >= 8 and beat < 16 then
    P1:x(mystery * 2)
end
"""
    plan = compile_update(body, _DriverSurface(), lambda b: b * 0.5)
    assert ('P1', 'x') not in plan.closed_keyframes
