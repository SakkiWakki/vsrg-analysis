"""Specs for `t.f(args)` lowering to ONE crossing instead of two.

Dispatching through a table of functions (`gf2_mod_readers.mods(beat, 1)`) used
to cost a FIELD crossing to fetch the callable and a CALL_VALUE crossing to
invoke it, and the value in between existed for nothing else. `CALL_FIELD` does
both in one, so these specs pin that the fusion did not change what the call
DOES: the same function, the same args, the same answer for a field that holds
no function at all - and that the free calls with their own lowering
(`math.floor`, `table.insert`) did not get swallowed by the new branch.
"""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.guard_surface import NotitgGuardSurface
from analysis.games.notitg.sim.compiled_body import _LuaEnvStore
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.xml_actors import parse_actor_xml
from analysis.player.render.expr.frame_eval import Interpreter
from analysis.player.render.expr.native_c import opstream
from analysis.player.render.expr.native_c.cbody import CompiledBodyC
from analysis.player.render.expr.parser import parse_body

_READERS = """
readers = {
    add = function(a, b) return a + b end,
    count = 0,
}
function readers.bump(n) readers.count = readers.count + n end
"""


def _env_with_actor():
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(
        '<ActorFrame><children><Quad Name="Q"/></children></ActorFrame>').root)
    rec_id = [r for r, label in env._labels.items() if label == 'Q'][0]
    return env, rec_id


def _run(source, env, rec_id):
    stmts, _sink = parse_body(source)
    program = opstream.compile_body_ops(stmts)
    surface = NotitgGuardSurface(env)
    store = _LuaEnvStore(env._host)
    interp = Interpreter(surface, store=store)
    body = CompiledBodyC(program, surface, store, program.nodes,
                         lambda node: None, interp=interp)
    body.run(env._tables[rec_id])
    return program


def _ops(source, opcode):
    stmts, _sink = parse_body(source)
    program = opstream.compile_body_ops(stmts)
    return [(op, a, b) for op, a, b, _c in program.ops if op == opcode]


def test_a_table_function_call_lowers_to_one_op():
    """The premise: without CALL_FIELD emitted, everything below is vacuous."""
    assert _ops('out = readers.add(2, 3)', opstream.Op.CALL_FIELD)
    assert not _ops('out = readers.add(2, 3)', opstream.Op.CALL_VALUE)


def test_it_calls_the_function_with_its_arguments():
    env, rec_id = _env_with_actor()
    env._host.run(_READERS)

    _run('out = readers.add(2, 3)', env, rec_id)

    assert env._host.env['out'] == 5.0


def test_the_call_still_reaches_the_host_for_its_effects():
    """A dispatched reader is called for what it DOES, not only what it
    returns - the fusion must not turn an effect into a value read."""
    env, rec_id = _env_with_actor()
    env._host.run(_READERS)

    _run('readers.bump(4)\nreaders.bump(3)', env, rec_id)

    assert env._host.env['readers']['count'] == 7.0


def test_a_field_holding_no_function_answers_as_the_split_path_did():
    """`call_value` returned UNRESOLVED for a non-callable, and a body that
    calls a missing reader must still finish the tick rather than fault."""
    env, rec_id = _env_with_actor()
    env._host.run(_READERS)

    _run('missing = readers.nosuch(1)\n'
         'notafn = readers.count(1)\n'
         'after = 9', env, rec_id)

    assert env._host.env['after'] == 9.0


def test_an_absent_table_does_not_reach_the_host_at_all():
    """An UNRESOLVED receiver has no field to read and nothing to call."""
    env, rec_id = _env_with_actor()

    _run('out = nothing_here.reader(1)\nafter = 2', env, rec_id)

    assert env._host.env['after'] == 2.0


def test_the_natively_lowered_free_calls_are_untouched():
    """`math.*` and `table.*` are Field calls too, and they have their own ops
    precisely so they never cross. The new branch sits AFTER them."""
    assert not _ops('x = math.floor(1.5)', opstream.Op.CALL_FIELD)
    assert _ops('x = math.floor(1.5)', opstream.Op.CALL_MATH)
    assert not _ops('table.insert(t, 1)', opstream.Op.CALL_FIELD)
    assert _ops('n = table.getn(t)', opstream.Op.CALL_FIELD) == []


def test_a_method_call_is_not_a_field_call():
    """`t:f(x)` passes a receiver and pokes; `t.f(x)` does neither. The colon
    form must keep its own lowering."""
    assert not _ops('self:zoom(2)', opstream.Op.CALL_FIELD)
