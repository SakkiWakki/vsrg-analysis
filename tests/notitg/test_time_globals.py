"""Spec: the DateTime.cpp wall-clock globals (OpenITG DateTime.cpp:286-293).

Government Knows renders a clock+date overlay from Hour()/Minute()/Year()/
MonthOfYear()/DayOfMonth(); before these existed every av-update tick
faulted (535 faults) and the overlay stayed blank. The globals anchor at
env creation and advance with SIM time, so the on-screen clock ticks with
the song and a compile is deterministic given its start moment.
"""
import time

import pytest

pytest.importorskip('lupa')

import analysis.games.notitg.sim.env as env_mod
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.xml_actors import parse_actor_xml

_CHART = '<ActorFrame><children><Quad Name="A"/></children></ActorFrame>'

# 2021-01-09 13:37:42 local time (the chart's release vintage): month 1
# proves MonthOfYear is 1-12 (tm_mon+1), not C's 0-11.
_EPOCH = time.mktime((2021, 1, 9, 13, 37, 42, 0, 0, -1))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(env_mod.time, 'time', lambda: _EPOCH)
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(_CHART).root)
    return env


def _lua(env, expr):
    return env._host.run(f'return {expr}')


def test_openitg_datetime_semantics(env):
    assert _lua(env, 'Year()') == 2021
    assert _lua(env, 'MonthOfYear()') == 1
    assert _lua(env, 'DayOfMonth()') == 9
    assert _lua(env, 'Hour()') == 13
    assert _lua(env, 'Minute()') == 37
    assert _lua(env, 'Second()') == 42
    # 2021-01-09 is a Saturday: C tm_wday (Sunday=0) says 6.
    assert _lua(env, 'Weekday()') == 6
    # tm_yday is 0-based: Jan 9 is day 8.
    assert _lua(env, 'DayOfYear()') == 8


def test_clock_advances_with_sim_time(env):
    env._now = 61.0
    assert _lua(env, 'Minute()') == 38
    assert _lua(env, 'Second()') == 43


def test_the_charts_own_format_strings(env):
    time_text = _lua(env, "string.format('%02d:%02d', Hour(), Minute())")
    date_text = _lua(env, "string.format('%04d/%02d/%02d', Year(), "
                          'MonthOfYear(), DayOfMonth())')
    assert time_text == '13:37'
    assert date_text == '2021/01/09'
