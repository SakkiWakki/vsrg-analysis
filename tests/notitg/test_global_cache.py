"""Specs for the op-stream executor's GLOBAL cache.

`LOAD_GLOBAL` answers from a per-name cache in the executor instead of crossing
to the Lua namespace. The cache is not an assumption that globals hold still -
it is sound because the sandbox REPORTS every write by name (a proxied
`__newindex`, `rawset` included), so an entry dies the moment its value can have
changed. `STORE_GLOBAL` refills it with what it just wrote, since the store
already knows the new value and a read would have to cross to learn it.

Three ways that can silently break, one spec each: a host write that the cache
outlives, a body write whose own later read sees the pre-store value, and a
name the write observer never reaches because the writer computed it. The
assertions are against what the BODY reads back, not against cache bookkeeping.
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


def _env_with_actor():
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(
        '<ActorFrame><children><Quad Name="Q"/></children></ActorFrame>').root)
    rec_id = [r for r, label in env._labels.items() if label == 'Q'][0]
    return env, rec_id


def _build(source, env):
    """One compiled body over the live env, wired as the sim wires it."""
    stmts, _sink = parse_body(source)
    program = opstream.compile_body_ops(stmts)
    surface = NotitgGuardSurface(env)
    store = _LuaEnvStore(env._host)
    interp = Interpreter(surface, store=store)
    body = CompiledBodyC(program, surface, store, program.nodes,
                         lambda node: None, interp=interp)
    return body, program


# `total` is written here, so it compiles to LOAD_GLOBAL / STORE_GLOBAL rather
# than to the bare-symbol path the stable cache serves. It is SEEDED by each
# spec: an accumulator the chart never initialised reads as an absent operand,
# which poisons the arithmetic for reasons that have nothing to do with caching.
_ACCUMULATE = 'total = total + 1\nseen = total'


def _accumulator(env, start=0.0):
    env._host.env['total'] = start
    return _build(_ACCUMULATE, env)


def test_a_body_written_global_carries_across_ticks():
    """The accumulator contract: what a tick stores is what the next tick
    loads. Write-through must publish the stored value, not a stale one."""
    env, rec_id = _env_with_actor()
    body, _program = _accumulator(env)

    for tick in range(1, 5):
        body.run(env._tables[rec_id])
        assert env._host.env['seen'] == float(tick)


def test_a_host_write_drops_the_cached_value():
    """Chart Lua rebinding the global between ticks must be visible to the very
    next read - this is the invalidation the whole cache rests on."""
    env, rec_id = _env_with_actor()
    body, _program = _accumulator(env)

    body.run(env._tables[rec_id])
    assert env._host.env['seen'] == 1.0

    env._host.env['total'] = 40.0
    body.run(env._tables[rec_id])
    assert env._host.env['seen'] == 41.0, 'the cache outlived a host write'


def test_a_write_through_a_computed_name_is_seen():
    """`_G['tot'..'al'] = x` reaches no name any static analysis could predict,
    and is exactly why invalidation is by REPORT rather than by inference."""
    env, rec_id = _env_with_actor()
    body, _program = _accumulator(env)

    body.run(env._tables[rec_id])
    env._host.run("_G['tot' .. 'al'] = 100")
    body.run(env._tables[rec_id])

    assert env._host.env['seen'] == 101.0


def test_a_write_through_rawset_is_seen():
    """`rawset` skips `__newindex`, so the sandbox publishes a notifying one.
    If that ever regressed, the cache would serve a value the chart replaced."""
    env, rec_id = _env_with_actor()
    body, _program = _accumulator(env)

    body.run(env._tables[rec_id])
    env._host.run("rawset(_G, 'total', 7)")
    body.run(env._tables[rec_id])

    assert env._host.env['seen'] == 8.0


def test_a_write_from_a_command_body_mid_tick_is_seen():
    """The write that is hardest to catch lands INSIDE a tick: a poke runs a
    command body whose Lua rebinds the global the body reads two ops later. The
    report fires as the write happens, which is why the cache is dropped then
    rather than at the tick boundary."""
    env, rec_id = _env_with_actor()
    env._host.env['total'] = 0.0
    env._named_commands.setdefault(rec_id, {})['Bump'] = '%function(self) total = 5 end'
    # `total` is assigned here so its reads compile to LOAD_GLOBAL; read-only,
    # they would take the bare-symbol path and pin a different cache.
    body, _program = _build('before = total\n'
                            'self:playcommand("Bump")\n'
                            'after = total\n'
                            'total = after', env)

    body.run(env._tables[rec_id])

    assert env._host.env['before'] == 0.0
    assert env._host.env['after'] == 5.0, (
        'the cache held a value across a mid-tick command write')


def test_a_cached_miss_is_not_sticky():
    """A global that is absent on the first tick and written before the second
    must read as written. Caching the MISS is what makes an accumulator cheap
    on tick one; keeping it is what would make the chart wrong on tick two."""
    env, rec_id = _env_with_actor()
    body, _program = _build('seen = later\nlater = seen', env)

    body.run(env._tables[rec_id])
    env._host.env['later'] = 3.0
    body.run(env._tables[rec_id])

    assert env._host.env['seen'] == 3.0
