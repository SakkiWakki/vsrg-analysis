"""Specs for the op-stream executor's cross-run STABLE symbol cache.

`LOAD_SYMBOL` tests `stable[id] == 2` BEFORE the epoch check, so a stable entry
is immune to every invalidation the executor has. That is only safe while the
body cannot write the name - and it can: `collect_global_writes` subtracts every
name that is EVER a `local` anywhere, so a name that is `local` in one scope and
an implicit global in another compiles to LOAD_SYMBOL at its reads while
STORE_GLOBAL writes it. These specs pin that STORE_GLOBAL evicts the name.

Chart Lua is arbitrary user content, so the rule has to hold by construction,
not because no sampled chart happens to hit it.
"""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.guard_surface import NotitgGuardSurface
from analysis.games.notitg.sim.compiled_body import _LuaEnvStore
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.player.render.expr.frame_eval import Interpreter
from analysis.player.render.expr.native_c import opstream
from analysis.player.render.expr.native_c.cbody import CompiledBodyC
from analysis.player.render.expr.parser import parse_body
from analysis.games.notitg.xml_actors import parse_actor_xml

# Reads compile to LOAD_SYMBOL (X is in `locals_seen` via the loop-body local,
# so it is subtracted from `_global_writes`), while the assignment at top level
# resolves to no slot and compiles to STORE_GLOBAL. That pairing is the hole.
_WRITE_HOLE_BODY = """
for i = 1, 0 do local X = 1 end
before = X
X = 5
after = X
"""


def _env_with_actor():
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(
        '<ActorFrame><children><Quad Name="Q"/></children></ActorFrame>').root)
    rec_id = [r for r, label in env._labels.items() if label == 'Q'][0]
    return env, rec_id


def _build(source, env):
    stmts, _sink = parse_body(source)
    program = opstream.compile_body_ops(stmts)
    surface = NotitgGuardSurface(env)
    store = _LuaEnvStore(env._host)
    interp = Interpreter(surface, store=store)
    body = CompiledBodyC(program, surface, store, program.nodes,
                         lambda node: None, interp=interp)
    return body, program


def _mark_stable(body, program, name):
    """Force the host-side stable mark the arena-snapshot path would set. The
    executor promotes 1 -> 2 on the next resolution of a non-handle value."""
    name_id = program.names.index(name)
    body._lib.cbody_mark_stable(body._b, name_id)


def test_the_write_hole_shape_still_compiles_that_way():
    """If this stops holding the spec below is vacuous, so assert the premise:
    X reads as a SYMBOL and writes as a GLOBAL in the same body."""
    stmts, _sink = parse_body(_WRITE_HOLE_BODY)
    program = opstream.compile_body_ops(stmts)
    emitted = {(opstream.Op.LOAD_SYMBOL, 'X'): False,
               (opstream.Op.STORE_GLOBAL, 'X'): False}
    for op, a, _b, _c in program.ops:
        key = (op, program.names[a] if a < len(program.names) else None)
        if key in emitted:
            emitted[key] = True
    assert all(emitted.values()), emitted
    assert 'X' in program.symbol_reads


def test_store_global_evicts_a_stable_symbol():
    """The spec: after the body writes X, a later bare-symbol read of X must
    see the written value, not the cached pre-store one."""
    env, rec_id = _env_with_actor()
    env._host.env['X'] = 7.0
    body, program = _build(_WRITE_HOLE_BODY, env)
    _mark_stable(body, program, 'X')

    body.run(env._tables[rec_id])

    assert env._host.env['before'] == 7.0
    assert env._host.env['after'] == 5.0, (
        'a stable entry survived its own STORE_GLOBAL')


def test_eviction_persists_across_ticks():
    """Eviction must be permanent, not per-run: the host probes a name for
    snapshotting only once, so a re-promoted entry would never be re-checked."""
    env, rec_id = _env_with_actor()
    env._host.env['X'] = 7.0
    body, program = _build(_WRITE_HOLE_BODY, env)
    _mark_stable(body, program, 'X')

    for tick in range(4):
        env._host.env['X'] = 100.0 + tick   # a rebind the cache must not hide
        body.run(env._tables[rec_id])
        assert env._host.env['before'] == 100.0 + tick
        assert env._host.env['after'] == 5.0


def test_unwritten_symbol_still_caches():
    """The eviction must be scoped to the written name - a symbol the body only
    READS is exactly what the stable cache exists for, and must keep serving
    from cache once marked."""
    env, rec_id = _env_with_actor()
    env._host.env['K'] = 11.0
    body, program = _build('for i = 1, 0 do local K = 1 end\nseen = K', env)
    _mark_stable(body, program, 'K')

    body.run(env._tables[rec_id])
    assert env._host.env['seen'] == 11.0

    # Rebound behind the executor's back: a STABLE symbol is permitted to keep
    # serving the cached value. This pins the CURRENT contract, and is exactly
    # the staleness the eligibility rule (which names may be marked at all) has
    # to prevent upstream - it is not a licence to mark arbitrary names.
    env._host.env['K'] = 99.0
    body.run(env._tables[rec_id])
    assert env._host.env['seen'] == 11.0
