"""The Rust-native frame interpreter (`notitg_frame_native`) driving the live
sim via the `NativeFrontier` bridge must match the Python interpreter
byte-for-byte (keyframe_diff). The native path is the residue tick loop ported
to Rust; the Python `CompiledBody` is the reference oracle (itself frozen
against Lua). Skipped when the native wheel is not installed (a source checkout
that has not run `make frame-native`)."""
import pytest

pytest.importorskip('lupa')
pytest.importorskip('notitg_frame_native')

from analysis.games.notitg.native_frontier import NativeFrontier
from analysis.games.notitg.guard_surface import NotitgGuardSurface
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.xml_actors import parse_actor_xml, _strip_lua_wrapper
from analysis.player.render.effects.timeline import EventTimeline
from analysis.player.render.expr.parser import parse_body
from analysis.player.render.expr.surface import UNRESOLVED

import notitg_frame_native as native


def _env():
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(
        '<ActorFrame><children><Quad Name="foo"/></children></ActorFrame>').root)
    rec = [rid for rid, l in env._labels.items() if l == 'foo'][0]
    env._host.env['foo'] = env._tables[rec]
    return env, rec


def _native_values(body, prop, beats, init=None):
    env, rec = _env()
    for k, v in (init or {}).items():
        env._host.env[k] = v
    bridge = NativeFrontier(NotitgGuardSurface(env), env._host.env)
    interp = native.NativeInterpreter()
    stmts, _ = parse_body(_strip_lua_wrapper(body))
    interp.compile_body(stmts)   # marshal + snapshot-analysis once (the sim path)
    out = []
    for beat in beats:
        env.set_time(beat * 0.5, beat)
        env._host.env['beat'] = beat
        bridge.set_self(env._tables[rec])
        interp.run_compiled_frontier(bridge, UNRESOLVED)
        frames = env.actor_keyframes().get(rec, {}).get(prop, [])
        out.append(EventTimeline(frames, (0.0,)).sample(beat * 0.5)[0])
    return out


_BEATS = [1.0, 2.0, 3.0, 4.0, 5.0]


def test_native_backend_imports():
    assert native.backend_name() == 'notitg_frame_native'


def test_native_sin_curve_matches_expected():
    import math
    body = '%function(self) foo:x(320 + 40 * math.sin(beat)) end'
    got = _native_values(body, 'x', _BEATS)
    for beat, value in zip(_BEATS, got):
        assert abs(value - (320 + 40 * math.sin(beat))) < 1e-6


def test_native_accumulator_persists_across_ticks():
    # `acc` grows one per tick through the frontier-backed global store.
    body = '%function(self) acc = acc + 1 foo:rotationz(acc) end'
    got = _native_values(body, 'rotation', _BEATS, init={'acc': 0.0})
    assert got == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_native_type_and_math_mod_via_host():
    # `type` is a native builtin; `math.mod` is a HOST function (LuaJIT dropped
    # it, gat defines a shim) reached through the frontier - both must resolve
    # so `math.mod(gat_frame, 2)` gates a branch.
    body = ('%function(self) '
            'if type(foo) == "table" then foo:x(100) end '
            'foo:y(50 + math.floor(beat)) end')
    x = _native_values(body, 'x', _BEATS)
    y = _native_values(body, 'y', _BEATS)
    import math
    assert all(v == 100.0 for v in x)         # the type()=='table' gate fired
    for beat, value in zip(_BEATS, y):
        assert value == 50.0 + math.floor(beat)


def test_native_live_method_read_drives_position():
    # A poke reading the SAME actor's live position (self:GetX) each tick - the
    # frontier method crossing. Seed x, then read it back and offset.
    body = '%function(self) foo:x(beat * 10) end'
    got = _native_values(body, 'x', _BEATS)
    assert got == [10.0, 20.0, 30.0, 40.0, 50.0]


def _lua_table(env, name, spec):
    """Bind a load-populated host (lupa) DATA TABLE global from a Lua literal."""
    env._host.env[name] = env._host.compile(f'return {spec}')()


def test_native_read_only_data_table_is_snapshotted():
    # A nested host DATA TABLE the body only READS is snapshotted native, so
    # `v[i][j]` reads never cross the frontier - and the value is identical.
    env, rec = _env()
    _lua_table(env, 'vtab', '{{10, 11}, {20, 21}, {30, 31}}')
    bridge = NativeFrontier(NotitgGuardSurface(env), env._host.env)
    interp = native.NativeInterpreter()
    stmts, _ = parse_body(_strip_lua_wrapper(
        '%function(self) foo:x(vtab[2][1] + vtab[3][2]) end'))
    interp.compile_body(stmts)
    for beat in [1.0, 2.0]:
        env.set_time(beat * 0.5, beat)
        env._host.env['beat'] = beat
        bridge.set_self(env._tables[rec])
        interp.run_compiled_frontier(bridge, UNRESOLVED)
    frames = env.actor_keyframes().get(rec, {}).get('x', [])
    assert EventTimeline(frames, (0.0,)).sample(1.0)[0] == 51.0  # 20 + 31


def test_native_written_data_table_is_not_snapshotted():
    # A host table the body WRITES must NOT be snapshotted (a frozen copy would
    # diverge). `scratch[1] = beat` then read it back - the write must land AND
    # be read back each tick (proving the live table, not a snapshot, is used).
    env, rec = _env()
    _lua_table(env, 'scratch', '{0, 0}')
    bridge = NativeFrontier(NotitgGuardSurface(env), env._host.env)
    interp = native.NativeInterpreter()
    stmts, _ = parse_body(_strip_lua_wrapper(
        '%function(self) scratch[1] = beat * 5 foo:x(scratch[1]) end'))
    interp.compile_body(stmts)
    out = []
    for beat in [1.0, 2.0, 3.0]:
        env.set_time(beat * 0.5, beat)
        env._host.env['beat'] = beat
        bridge.set_self(env._tables[rec])
        interp.run_compiled_frontier(bridge, UNRESOLVED)
        frames = env.actor_keyframes().get(rec, {}).get('x', [])
        out.append(EventTimeline(frames, (0.0,)).sample(beat * 0.5)[0])
    assert out == [5.0, 10.0, 15.0]   # the live write is seen each tick
