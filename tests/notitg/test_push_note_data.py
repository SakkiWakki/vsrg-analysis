"""PushNoteData (@0x0052dc60), corpus semantics: fill the chart-NAMED
Lua global with the player's (beat, column) note rows in a beat range.
Government Knows initializes `govParappaNotedata = {}`, calls
`PushNoteData('govParappaNotedata', 420, 452)`, and its parappa dancer
reads `{beat, column}` rows back."""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.sim.loop import chart_note_rows
from analysis.games.notitg.xml_actors import parse_actor_xml


def test_chart_note_rows_parses_measures_and_heads():
    # Measure 0: quarter notes; measure 1: eighths - beats scale by line
    # count. '1' tap, '2' hold head, '4' roll head all count; '3' (hold
    # tail), 'M' (mine) and '0' do not.
    notedata = '''
1000
0200
0040
000M
,
10003000
'''
    rows = chart_note_rows({'notedata': notedata})
    assert rows == ((0.0, 0), (1.0, 1), (2.0, 2), (4.0, 0))


def test_push_note_data_fills_the_named_global():
    chart = ('<ActorFrame><children><Quad Name="Player"/>'
             '</children></ActorFrame>')
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(chart).root)
    env.note_rows = ((419.0, 0), (420.0, 2), (430.5, 1), (452.0, 3))
    player = next(rec for rec, label in env._labels.items()
                  if label.endswith('Player'))
    env._host.run('parappa = {}')
    env._actor_poke(player, 'PushNoteData', 'parappa', 420.0, 452.0)
    assert env._host.run('return #parappa') == 2
    assert env._host.run('return parappa[1][1]') == 420.0
    assert env._host.run('return parappa[1][2]') == 2.0
    assert env._host.run('return parappa[2][1]') == 430.5


def test_unresolvable_call_leaves_the_charts_table():
    chart = '<ActorFrame><children><Quad Name="P"/></children></ActorFrame>'
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(chart).root)
    player = next(rec for rec, label in env._labels.items()
                  if label.endswith('P'))
    env._host.run('t = {}')
    env._actor_poke(player, 'PushNoteData', 't')       # missing range
    assert env._host.run('return #t') == 0
