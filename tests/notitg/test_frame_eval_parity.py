"""The interpreter drives a real SimEnvironment to the SAME actor pokes the
Lua path does. This is GATE A in miniature: `frame_eval` + `NotitgGuardSurface`
poking a real actor must produce keyframes byte-identical (by played-back
value) to `run_update_body` running the same body as Lua. Faithful charts,
different code.
"""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.guard_surface import NotitgGuardSurface
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.sim.keyframe_diff import diff_runs, sample_grid
from analysis.games.notitg.xml_actors import parse_actor_xml, _strip_lua_wrapper
from analysis.player.render.expr.frame_eval import Interpreter
from analysis.player.render.expr.parser import parse_body

_TICKS = [0.0, 2.0, 4.0, 6.0]


def _quad_id(env):
    return [rid for rid, label in env._labels.items() if label == 'Q'][0]


def _run_lua(update_wrapped: str):
    body = _strip_lua_wrapper(update_wrapped)
    xml = '<ActorFrame><children><Quad Name="Q"/></children></ActorFrame>'
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(xml).root)
    rec = _quad_id(env)
    env._host.env['self'] = env._tables[rec]
    for beat in _TICKS:
        env.set_time(beat * 0.5, beat)
        env._host.env['beat'] = beat        # the Lua body reads `beat` global
        env.run_update_body(body, rec_id=rec)
    return env


def _run_interp(update_wrapped: str):
    body = _strip_lua_wrapper(update_wrapped)
    xml = '<ActorFrame><children><Quad Name="Q"/></children></ActorFrame>'
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(xml).root)
    rec = _quad_id(env)
    interp = Interpreter(NotitgGuardSurface(env))
    stmts, _ = parse_body(body)
    for beat in _TICKS:
        env.set_time(beat * 0.5, beat)     # the interpreter reads `beat` off
        interp.root.set_local('self', env._tables[rec])   # the live surface
        interp.run(stmts)
    return env


def _assert_parity(update_wrapped: str):
    lua = _run_lua(update_wrapped)
    interp = _run_interp(update_wrapped)
    divs = diff_runs(lua, interp, sample_grid(0.0, 3.0))
    assert divs == [], f'interpreter diverged from Lua: {divs[:3]}'


def test_beat_gated_setters_match_lua():
    _assert_parity(
        '%function(self) local b = beat '
        'if b >= 4 then self:x(200) self:zoom(2) '
        'else self:x(50) self:zoom(1) end end')


def test_arithmetic_setter_matches_lua():
    # a pure-curve poke: x driven by an expression over beat.
    _assert_parity(
        '%function(self) self:x(beat * 10 + 5) end')


def test_nested_if_and_multiple_props_match_lua():
    _assert_parity(
        '%function(self) local b = beat '
        'if b >= 2 then '
        '  if b >= 4 then self:rotationz(90) else self:rotationz(45) end '
        '  self:diffusealpha(1) '
        'else self:diffusealpha(0) end end')
