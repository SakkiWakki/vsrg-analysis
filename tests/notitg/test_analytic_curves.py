"""The analytic curve compiler (`expr/requests.compile_curves`): a per-frame
setter poke whose arg is a pure function of clock+math+consts lowers to ONE
`CurveRequest` - a live Channel - instead of ticking. The gate is the contract's:
the analytic curve must play back the SAME value the residue interpreter records
(`frame_eval` is the oracle). A poke reading live state stays residue.
"""
import math

import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.frame_compile import _DriverOnlySurface
from analysis.games.notitg.guard_surface import NotitgGuardSurface
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.xml_actors import parse_actor_xml, _strip_lua_wrapper
from analysis.player.render.effects.timeline import EventTimeline
from analysis.player.render.expr.requests import CurveRequest, compile_curves


def _env(actors='<Quad Name="foo"/>'):
    # `foo` is bound as a GLOBAL pointing at its recorder, so a poke names it
    # statically (`foo:x(...)`) - what the analytic compiler keys its target on
    # (a `self:` body pokes a dynamic actor, which has no compile-time name).
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(
        f'<ActorFrame><children>{actors}</children></ActorFrame>').root)
    rec = [rid for rid, l in env._labels.items() if l == 'foo'][0]
    env._host.env['foo'] = env._tables[rec]
    return env


def _driver_surface(env):
    return _DriverOnlySurface(NotitgGuardSurface(env, to_beat=lambda t: t / 0.5))


def _interp_values(env, body, prop, beats, init=None):
    """Run `body` through the residue interpreter tick by tick; return the
    RECORDED value of `prop` at each beat - the oracle the analytic curve must
    match."""
    rec = [rid for rid, l in env._labels.items() if l == 'foo'][0]
    for name, value in (init or {}).items():
        env._host.env[name] = value
    env.use_compiled_body = True
    inner = _strip_lua_wrapper(body)
    out = []
    for beat in beats:
        env.set_time(beat * 0.5, beat)
        env._host.env['beat'] = beat
        env.run_update_body(inner, rec_id=rec)
        frames = env.actor_keyframes().get(rec, {}).get(prop, [])
        out.append(EventTimeline(frames, (0.0,)).sample(beat * 0.5)[0])
    return out


_BEATS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.2]


def test_sin_oscillator_curve_matches_interpreter():
    # The contract's canonical example: p:x(320 + 40*math.sin(beat)).
    body = '%function(self) foo:x(320 + 40 * math.sin(beat)) end'
    curves, residue = compile_curves(body, _driver_surface(_env()))
    assert residue == []
    assert len(curves) == 1 and curves[0].target == 'foo' \
        and curves[0].prop == 'x' and curves[0].analytic
    got = _interp_values(_env(), body, 'x', _BEATS)
    for beat, oracle in zip(_BEATS, got):
        assert abs(curves[0].channel.at(beat * 0.5) - oracle) < 1e-6


def test_arithmetic_curve_matches_interpreter():
    body = '%function(self) foo:x(100 + beat * 2 - beat * beat) end'
    curves, _residue = compile_curves(body, _driver_surface(_env()))
    got = _interp_values(_env(), body, 'x', _BEATS)
    for beat, oracle in zip(_BEATS, got):
        assert abs(curves[0].channel.at(beat * 0.5) - oracle) < 1e-6


def test_nested_math_curve_matches_interpreter():
    body = ('%function(self) '
            'foo:y(240 + 100 * math.cos(beat * 2) + math.floor(beat)) end')
    curves, _residue = compile_curves(body, _driver_surface(_env()))
    assert len(curves) == 1 and curves[0].prop == 'y'
    got = _interp_values(_env(), body, 'y', _BEATS)
    for beat, oracle in zip(_BEATS, got):
        assert abs(curves[0].channel.at(beat * 0.5) - oracle) < 1e-6


def test_multi_writer_prop_stays_residue():
    # A (target, prop) written by two pokes composes last-writer-wins per tick,
    # which one standalone curve cannot reproduce - so neither becomes analytic.
    body = '%function(self) foo:x(100 + beat) foo:x(200 + beat * 2) end'
    curves, residue = compile_curves(body, _driver_surface(_env()))
    assert curves == []
    assert [u.name for u, _f in residue] == ['foo.x', 'foo.x']


def test_live_read_poke_stays_residue():
    # A poke reading another actor's live position cannot be a clock-only
    # curve - it must fall to residue (the sampler), never a guessed curve.
    body = '%function(self) foo:x(bar_x + 10) end'
    env = _env()
    curves, residue = compile_curves(body, _driver_surface(env))
    assert curves == []
    assert [u.name for u, _f in residue] == ['foo.x']


def test_accumulator_poke_stays_residue():
    # `acc = acc + 1; foo:rotationz(acc)` - the value is a recurrence over its
    # own past, not a clock function, so rotationz stays residue.
    body = ('%function(self) acc = acc + 1 foo:rotationz(acc) end')
    curves, residue = compile_curves(body, _driver_surface(_env()))
    assert all(c.prop != 'rotation' for c in curves)
    assert any(u.name == 'foo.rotationz' for u, _f in residue)


def test_math_pi_constant_folds_into_curve():
    body = '%function(self) foo:x(math.sin(beat * math.pi) * 50) end'
    curves, _residue = compile_curves(body, _driver_surface(_env()))
    assert len(curves) == 1
    got = _interp_values(_env(), body, 'x', _BEATS)
    for beat, oracle in zip(_BEATS, got):
        assert abs(curves[0].channel.at(beat * 0.5) - oracle) < 1e-6


# -- residue path (EMIT_SAMPLED) ---------------------------------------------

def _tick_body(env, body, beats, init=None):
    """Run `body` through the residue interpreter over `beats`; return
    `{'foo': {prop: [Keyframe]}}` - the recorded streams the sampler wraps."""
    rec = [rid for rid, l in env._labels.items() if l == 'foo'][0]
    for name, value in (init or {}).items():
        env._host.env[name] = value
    env.use_compiled_body = True
    inner = _strip_lua_wrapper(body)
    for beat in beats:
        env.set_time(beat * 0.5, beat)
        env._host.env['beat'] = beat
        env.run_update_body(inner, rec_id=rec)
    return {'foo': env.actor_keyframes().get(rec, {})}


def _residue_props(residue):
    return {p for u, _f in residue
            if (p := _split_actor_prop_safe(u.name)) is not None}


def _split_actor_prop_safe(name):
    from analysis.player.render.expr.requests import _split_actor_prop
    return _split_actor_prop(name)


def test_accumulator_residue_samples_the_ticked_stream():
    # An accumulator poke (rotationz = growing acc) is residue; the sampler
    # emits a Channel that plays back exactly what the tick loop recorded.
    from analysis.player.render.expr.requests import compile_sampled
    from analysis.player.render.scheduler import SongTimeClock

    body = '%function(self) acc = acc + 1 foo:rotationz(acc) end'
    _curves, residue = compile_curves(body, _driver_surface(_env()))
    beats = [1.0, 2.0, 3.0, 4.0, 5.0]
    recorded = _tick_body(_env(), body, beats, init={'acc': 0.0})
    sampled = compile_sampled(recorded, _residue_props(residue),
                              SongTimeClock())
    assert len(sampled) == 1 and sampled[0].prop == 'rotation'
    # the accumulator advances one per tick -> rotation = 1, 2, 3, 4, 5
    for i, beat in enumerate(beats, start=1):
        assert sampled[0].channel.at(beat * 0.5) == float(i)


def test_analytic_and_residue_compose_to_the_full_body():
    # The contract's reconstruction invariant: a body with BOTH an analytic
    # poke (foo:x, a sin curve) and a residue poke (foo:rotationz, accumulator)
    # -> the analytic Channel + the sampled Channel together reproduce every
    # value the whole-body interpreter run records. No (target, prop) is
    # dropped or double-emitted.
    from analysis.player.render.expr.requests import compile_sampled
    from analysis.player.render.scheduler import SongTimeClock

    body = ('%function(self) acc = acc + 1 '
            'foo:x(320 + 40 * math.sin(beat)) foo:rotationz(acc) end')
    curves, residue = compile_curves(body, _driver_surface(_env()))
    assert {c.prop for c in curves} == {'x'}                 # x is analytic
    beats = [1.0, 2.0, 3.0, 4.0, 5.0]
    recorded = _tick_body(_env(), body, beats, init={'acc': 0.0})
    sampled = compile_sampled(recorded, _residue_props(residue),
                              SongTimeClock())
    assert {s.prop for s in sampled} == {'rotation'}         # rotation sampled

    # every emitted (prop -> Channel), analytic or sampled, matches the oracle
    channels = {c.prop: c.channel for c in curves}
    channels.update({s.prop: s.channel for s in sampled})
    for prop in ('x', 'rotation'):
        oracle = _interp_values(_env(), body, prop, beats, init={'acc': 0.0})
        for beat, want in zip(beats, oracle):
            assert abs(channels[prop].at(beat * 0.5) - want) < 1e-6


def test_analytic_prop_not_re_emitted_as_sampled():
    # A (target, prop) the analytic path emitted as a curve is NOT in the
    # residue set, so compile_sampled skips it (analytic wins - it is exact).
    from analysis.player.render.expr.requests import compile_sampled
    from analysis.player.render.scheduler import SongTimeClock

    body = '%function(self) foo:x(100 + beat * 5) end'
    curves, residue = compile_curves(body, _driver_surface(_env()))
    assert len(curves) == 1
    recorded = _tick_body(_env(), body, [1.0, 2.0], init={})
    sampled = compile_sampled(recorded, _residue_props(residue),
                              SongTimeClock())
    assert sampled == []          # x was analytic; not re-sampled
