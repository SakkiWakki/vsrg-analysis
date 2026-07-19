"""Phase 3: the compiled Update-body path (frame_eval driving a real
SimEnvironment via the use_compiled_body flag) produces the same actor pokes
the Lua path does, and the engine bridges the interpreter needs - singleton
methods, screen-child seeding, math library, persistent globals - are wired.
"""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.guard_surface import NotitgGuardSurface
from analysis.games.notitg.sim.compiled_body import CompiledBody, _LuaEnvStore
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.sim.keyframe_diff import diff_runs, sample_grid
from analysis.games.notitg.xml_actors import parse_actor_xml, _strip_lua_wrapper
from analysis.player.render.expr.frame_eval import Interpreter
from analysis.player.render.expr.parser import parse_body


def _quad_id(env):
    return [rid for rid, label in env._labels.items() if label == 'Q'][0]


def _run(update_wrapped, compiled, beats=(0.0, 2.0, 4.0, 6.0)):
    body = _strip_lua_wrapper(update_wrapped)
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(
        '<ActorFrame><children><Quad Name="Q"/></children></ActorFrame>').root)
    env.use_compiled_body = compiled
    rec = _quad_id(env)
    env._host.env['self'] = env._tables[rec]
    for beat in beats:
        env.set_time(beat * 0.5, beat)
        env._host.env['beat'] = beat
        env.run_update_body(body, rec_id=rec)
    return env


def test_flag_selects_compiled_path_with_parity():
    body = ('%function(self) local b = beat '
            'if b >= 4 then self:x(200) self:zoom(2) '
            'else self:x(50) self:zoom(1) end end')
    lua = _run(body, compiled=False)
    comp = _run(body, compiled=True)
    assert diff_runs(lua, comp, sample_grid(0.0, 3.0)) == []


def test_math_library_resolves_in_compiled_body():
    # A pure-curve poke through the math library must match Lua's math.
    body = '%function(self) self:x(320 + 40 * math.sin(beat)) end'
    lua = _run(body, compiled=False)
    comp = _run(body, compiled=True)
    assert diff_runs(lua, comp, sample_grid(0.0, 3.0)) == []


def _run_with_globals(setup, update_wrapped, compiled, beats=(0.0, 2.0, 4.0)):
    """Like `_run` but runs `setup` (a Lua chunk) in the host env FIRST - so a
    body can rely on a global host function or a load-populated host table, the
    way a real chart's InitCommand seeds one."""
    body = _strip_lua_wrapper(update_wrapped)
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(
        '<ActorFrame><children><Quad Name="Q"/></children></ActorFrame>').root)
    env._host.compile(setup)()
    env.use_compiled_body = compiled
    rec = _quad_id(env)
    env._host.env['self'] = env._tables[rec]
    for beat in beats:
        env.set_time(beat * 0.5, beat)
        env._host.env['beat'] = beat
        env.run_update_body(body, rec_id=rec)
    return env


def test_global_host_function_call_resolves_in_compiled_body():
    # A body calling a GLOBAL host function (`SecondsToClock`, defined at load
    # in the host env, not a body local) must invoke it - not fall to the
    # surface's free-call path and return nil (the Private Caller regression).
    setup = "function dub(n) return n * 2 end"
    body = '%function(self) self:x(dub(beat) + 100) end'
    lua = _run_with_globals(setup, body, compiled=False)
    comp = _run_with_globals(setup, body, compiled=True)
    assert diff_runs(lua, comp, sample_grid(0.0, 3.0)) == []
    # and it actually moved (not a nil-arg no-op parity)
    rec = _quad_id(comp)
    assert 'x' in comp.actor_keyframes().get(rec, {})


def test_host_table_write_lands_in_compiled_body():
    # `t[i] = expr` on a LOAD-POPULATED host table must write through the
    # surface (the interpreter's own LuaTable path skips a host table). The
    # Private Caller `pc_strinkku[i] = string.sub(...)` scratch-state idiom.
    setup = "scratch = {}"
    body = ('%function(self) scratch[1] = beat * 3 '
            'self:x(scratch[1] + 10) end')
    lua = _run_with_globals(setup, body, compiled=False)
    comp = _run_with_globals(setup, body, compiled=True)
    assert diff_runs(lua, comp, sample_grid(0.0, 3.0)) == []


def test_type_of_host_function_and_table_dispatch_in_compiled_body():
    # `type(v)` over host values drives action dispatch (Machine Wave's
    # `type(action) == 'function'` gate). A host function is 'function', a host
    # table is 'table' - both must match Lua so the same branch fires.
    setup = "helper = function() return 5 end\ndata = {1, 2}"
    body = ('%function(self) '
            "if type(helper) == 'function' then self:x(helper() * beat) end "
            "if type(data) == 'table' then self:y(200) end end")
    lua = _run_with_globals(setup, body, compiled=False)
    comp = _run_with_globals(setup, body, compiled=True)
    assert diff_runs(lua, comp, sample_grid(0.0, 3.0)) == []
    rec = _quad_id(comp)
    assert 'y' in comp.actor_keyframes().get(rec, {})  # the 'table' gate fired


def test_lua_env_store_shares_globals_with_the_host():
    # A global the interpreter writes is visible in the Lua env (one
    # namespace), so a guard or another body reads what the body wrote.
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    store = _LuaEnvStore(env._host.env)
    interp = Interpreter(NotitgGuardSurface(env), store=store)
    interp.run(parse_body('counter = 0')[0])
    step, _ = parse_body('counter = counter + 1')
    for _ in range(3):
        interp.run(step)
    assert env._host.env['counter'] == 3
    assert store.get('counter') == 3


def test_singleton_method_resolves_through_the_surface():
    # GAMESTATE:GetSongBeat() is a singleton method (not an actor); the
    # surface routes it to the Lua singleton so the interpreter reads it.
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml('<ActorFrame/>').root)
    env.set_time(3.0, 6.0)
    surface = NotitgGuardSurface(env)
    gamestate = env._host.env['GAMESTATE']
    assert surface.method(gamestate, 'GetSongBeat', []) == 6.0


def test_getshader_uniform_chains_onto_the_owning_actor():
    # `self:GetShader():uniform1f('timer', beat)` - GetShader chains the actor
    # back, so the uniform poke lands on the frag-owning actor's uniform:timer
    # channel (the gat2 shader-driver idiom). Compiled must match Lua.
    body = ("%function(self) if beat >= 2 then "
            "self:GetShader():uniform1f('timer', beat) end end")
    lua = _run(body, compiled=False)
    comp = _run(body, compiled=True)
    assert diff_runs(lua, comp, sample_grid(0.0, 3.0)) == []
    # and it actually recorded the uniform (not a no-op parity)
    rec = _quad_id(comp)
    assert 'uniform:timer' in comp.actor_keyframes().get(rec, {})


def test_screen_getchild_seeds_player_start_position():
    # The top screen's GetChild seeds the player at its engine start X (the
    # Lua metatable path) - not a plain synthetic child at rest.
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml('<ActorFrame/>').root)
    surface = NotitgGuardSurface(env)
    top = surface.method(env._host.env['SCREENMAN'], 'GetTopScreen', [])
    child = surface.method(top, 'GetChild', ['PlayerP1'])
    assert child is not None
    assert env.player_actor('PlayerP1').read('GetX') == 160.0
