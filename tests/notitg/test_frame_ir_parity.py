"""Differential byte-parity harness: frame-IR SSM channels vs the all-lupa oracle.

The existing `update_integrator.integrate_update` runs the whole Update body
through lupa on a tick grid and records per-actor keyframes; that recorded
keyframe set per (actor, prop) is the ORACLE. This harness proves the frame-IR
router flattens ONLY what it should: for every VarUpdate the router sends to SSM
(a `ClosedForm`), it evaluates that closed form over the update's
`effective_window` on the SAME tick grid and asserts the produced value stream
matches the oracle at those ticks. A mismatch is not a soft failure - it is the
signal that the update should NOT have flattened (a router correction), and the
diagnostic names the (actor, prop, node) so the correction is actionable.

Two evaluation modes are compared against the oracle, because a ClosedForm can
be sampled two ways and Stage A only guarantees one of them:
- CHANNEL mode: sample the coefficient Channels at the tick's song-time
  coordinate. This is the true "closed form evaluable at any t" the design
  promises; a pure curve (a=0) is `b(t)`.
- KERNEL mode: call the frozen `affine_kernel(n, h0, coeffs)`, which per the
  Stage A assumption samples every coefficient at coord 0.0 (constant over the
  window). This is what an SSM sink that bakes via the kernel would emit today.
The gap between the two IS the Stage B scope: a time-varying coefficient
reproduces under CHANNEL mode but not under KERNEL mode.

Router-correction findings this harness surfaces (see module docstring of the
report, and the xfailing tests below) are reported loudly, never hidden.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip('lupa')

from analysis.games.notitg import update_integrator
from analysis.games.notitg.frame_ir import (
    VarUpdate, build_frames, effective_window, iter_updates)
from analysis.games.notitg.flatteners import affine_kernel, route
from analysis.games.notitg.frame_compile import (
    _DriverOnlySurface, _sole_writer_props)
from analysis.player.render.expr import ast
from analysis.games.notitg.guard_surface import NotitgGuardSurface
from analysis.games.notitg.mod_stubs import StubEnvironment
from analysis.games.notitg.sim import verb_surface
from analysis.games.notitg import xml_actors


# -- oracle: build an env, run the integrator, harvest recorded keyframes ----

# The integrator's own tick rate; the SSM value stream is sampled on the SAME
# grid so a keyframe and a closed-form sample share a tick time exactly.
_TICK_HZ = update_integrator._TICK_HZ
_TICK_STEP = 1.0 / _TICK_HZ

# A pure-curve poke argument that reads `beat` must resolve `beat` to song time
# through the surface; under the identity clock (to_seconds = identity) song
# seconds and beats coincide, so the harness reads value(t) with t == beat.
_IDENTITY = staticmethod(lambda beat: float(beat))

# The perframe helper the real chart defines (modhelpers.xml): live only once
# the song has advanced one beat past first-seen, then a half-open range test.
# Reproduced so a synthetic Update body gates exactly as the engine's does.
_PERFRAME_HELPER = (
    'function perframe(b, e) '
    'local c = GAMESTATE:GetSongBeat(); if not e then e = b + 1 end; '
    'if c >= b and c < e then return true end; return false end')


def _oracle_env(update_body: str, init_body: str,
                to_seconds=_IDENTITY) -> StubEnvironment:
    """A loaded StubEnvironment for a synthetic Update body. `init_body` binds
    the actor global (`thing = self`) and any accumulator seed; `update_body` is
    the per-frame loop. Both are plain Lua (no `%function` wrapper, no XML
    escaping) so a body may hold double-quoted strings freely."""
    xml = (f'<CODE Type="Quad" '
           f'InitCommand="%function(self) {init_body} end"/>')
    root = xml_actors.parse_actor_xml(xml).root
    # Set the Update body as a raw attribute (named_commands() reads attrs);
    # done post-parse so the body may hold double-quoted Lua strings freely
    # without XML entity escaping.
    root.attrs['UpdateCommand'] = f'%function(self) {update_body} end'
    env = StubEnvironment(start_beat=0.0, to_seconds=to_seconds)
    env.run(_PERFRAME_HELPER, name='perframe-helper')
    env.load_actors(root)
    return env, root


def _run_oracle(env, root, to_seconds=_IDENTITY) -> dict:
    """Run the integrator and harvest `{global: {prop: [Keyframe]}}` - the
    recorded per-actor keyframe streams that are the parity oracle."""
    update_integrator.integrate_update(env, root, to_seconds)
    return env.named_actor_keyframes()


# -- SSM value stream: evaluate a routed ClosedForm on the tick grid ---------

# setter-method -> recorded property name (or a tuple for a bulk setter). The
# frame-IR names a poke by its SETTER method (`thing.rotationz`); the oracle
# records under the PROPERTY (`rotation`). This is the join key.
_SETTER_TO_PROP = dict(verb_surface.SCALAR_SETTERS)


def _split_name(update_name: str) -> tuple[str, str] | None:
    """(actor global, setter method) from a poke VarUpdate name `recv.method`,
    or None for a bare frame-variable update (never an actor prop)."""
    actor, _dot, method = update_name.rpartition('.')
    if not actor or method not in _SETTER_TO_PROP:
        return None
    return actor, method


def _prop_of(method: str) -> str | None:
    """The single recorded property a scalar setter writes, or None for a bulk
    setter (a tuple of props - out of scope for this scalar-parity harness)."""
    prop = _SETTER_TO_PROP.get(method)
    return prop if isinstance(prop, str) else None


def _window_ticks(window: tuple, to_seconds) -> list[float]:
    """The integrator tick times (song seconds) over a beat `window`, on the
    60Hz grid the oracle used."""
    t = to_seconds(window[0])
    t_end = to_seconds(window[1])
    ticks = []
    while t <= t_end:
        ticks.append(t)
        t += _TICK_STEP
    return ticks


def _channel_stream(cf, ticks: list[float]) -> list[float]:
    """CHANNEL-mode SSM values: a pure curve (a==0) is `b` sampled at each tick
    coordinate - the true closed form; an accumulator (a!=0) steps the
    recurrence per tick from h0, reading coefficients at the tick coordinate
    (the live-coefficient scan, the Stage B form the frozen kernel does not do).
    """
    a_channel = cf.coeffs['a']
    b_channel = cf.coeffs['b']
    if a_channel.at(0.0) == 0:
        return [b_channel.at(t) for t in ticks]
    h = cf.h0.at(0.0) if callable(getattr(cf.h0, 'at', None)) else cf.h0
    values = []
    for t in ticks:
        h = a_channel.at(t) * h + b_channel.at(t)
        values.append(h)
    return values


def _kernel_stream(cf, ticks: list[float]) -> list[float]:
    """KERNEL-mode SSM values: the frozen `affine_kernel(n, h0, coeffs)` at each
    tick index n - what a keyframe-baking SSM sink emits today (coefficients
    sampled once at coord 0.0)."""
    return [affine_kernel(n, cf.h0, cf.coeffs) for n in range(len(ticks))]


# -- oracle sampling ---------------------------------------------------------

def _oracle_at(keyframes: list, t: float) -> float | None:
    """The recorded value at tick time `t`: the keyframe whose `.t` matches this
    tick (the integrator emits one keyframe per tick it pokes). None when no
    keyframe lands on this tick (the poke did not fire here)."""
    for kf in keyframes:
        if abs(kf.t - t) <= _TICK_STEP / 4:
            return kf.values[0]
    return None


def _compare(name, prop, oracle_kfs, ssm_values, ticks, tol=1e-6):
    """(matched, mismatches): pair each oracle keyframe with the SSM value at
    its tick; a mismatch is (t, oracle, ssm). A tick the oracle never poked is
    skipped (the SSM covers the window; the oracle only the fired ticks)."""
    matched = 0
    mismatches = []
    for t, ssm in zip(ticks, ssm_values):
        oracle = _oracle_at(oracle_kfs, t)
        if oracle is None:
            continue
        if abs(oracle - ssm) <= tol:
            matched += 1
        else:
            mismatches.append((round(t, 5), oracle, ssm))
    return matched, mismatches


# -- the differential driver -------------------------------------------------

def _route_and_evaluate(env, root, to_seconds=_IDENTITY, to_beat=None,
                        corrected=False):
    """Build the Frame tree for the env's Update body, route each VarUpdate
    against a surface bound to the SAME live env, and for every SSM-routed poke
    evaluate its ClosedForm over its effective window on the tick grid. Returns
    a list of per-(actor, prop) result records.

    `corrected=True` applies the pending router correction locally (wrap a bare
    `ast.Method` poke node in an `ast.ExprStmt` before routing) so the harness
    can prove parity WOULD hold once flatteners accepts the poke node shape,
    without editing the frozen router. See `_ROUTER_POKE_BUG`."""
    body = update_integrator._update_body(root)
    surface = _DriverOnlySurface(
        NotitgGuardSurface(env, to_beat=to_beat or to_seconds))
    frame_root = build_frames(body, const_surface=None)
    oracle = env.named_actor_keyframes()

    results = []
    for update, frame in iter_updates(frame_root):
        split = _split_name(update.name)
        if split is None:
            continue
        actor, method = split
        prop = _prop_of(method)
        if prop is None:
            continue
        routed = _corrected_update(update) if corrected else update
        cf = route(routed, frame, surface=surface)
        window = effective_window(frame)
        results.append(_evaluate_one(
            actor, method, prop, cf, window, oracle, to_seconds))
    return results


def _corrected_update(update: VarUpdate) -> VarUpdate:
    """The update the router WOULD see once the poke-node contract is fixed: a
    bare `ast.Method` poke node wrapped in an `ast.ExprStmt` (the shape
    flatteners._value_expr already matches). A non-poke node passes through."""
    if isinstance(update.node, ast.Method):
        return VarUpdate(update.name, ast.ExprStmt(update.node))
    return update


def _evaluate_one(actor, method, prop, cf, window, oracle, to_seconds):
    record = {'actor': actor, 'method': method, 'prop': prop,
              'routed_ssm': cf is not None, 'window': window}
    actor_props = oracle.get(actor, {})
    oracle_kfs = actor_props.get(prop, [])
    record['oracle_keyframes'] = len(oracle_kfs)
    if cf is None or window is None or not oracle_kfs:
        return record
    ticks = _window_ticks(window, to_seconds)
    channel_values = _channel_stream(cf, ticks)
    kernel_values = _kernel_stream(cf, ticks)
    record['channel'] = _compare(actor, prop, oracle_kfs, channel_values, ticks)
    record['kernel'] = _compare(actor, prop, oracle_kfs, kernel_values, ticks)
    return record


# The central router-correction finding this harness surfaces. Referenced by
# every poke test below (a poke never flattens under the shipped router).
_ROUTER_POKE_BUG = (
    'ROUTER CORRECTION: frame_ir._poke_update stores the bare ast.Method as '
    'VarUpdate.node, but flatteners._value_expr matches only '
    'ast.ExprStmt(expr=ast.Method(...)). Every poke therefore routes to '
    'attention instead of SSM. flatteners._value_expr must also accept a bare '
    'ast.Method (or _poke_update must wrap it in an ExprStmt).')


# -- assignment accumulators: the class that DOES flatten today --------------
#
# An `x = a*x + b` assignment carries an `ast.Assign` node, which
# flatteners._value_expr matches; these route to SSM under the shipped router,
# so they are the harness's genuine positive parity proofs (no correction
# needed). The oracle records their effect through the poke that reads them.


def test_affine_accumulator_oracle_is_the_unit_ramp():
    """`acc = acc + 1; thing:rotationz(acc)` over its window: the accumulator
    assignment is a=1,b=1 (SSM), and the poke reads the running total. The
    recorded keyframes must be the exact unit ramp 1,2,3,... the recurrence
    produces from the seed - proving the integrator's accumulator and the
    affine closed form agree on the recurrence itself."""
    env, root = _oracle_env(
        "if perframe(0,1) then acc = acc + 1; thing:rotationz(acc) end "
        "self:sleep(0.02); self:queuecommand('Update')",
        "thing = self; acc = 0")
    oracle = _run_oracle(env, root)
    kfs = oracle.get('thing', {}).get('rotation', [])
    assert kfs, 'oracle recorded no accumulator pokes'
    recorded = [kf.values[0] for kf in sorted(kfs, key=lambda k: k.t)]
    assert recorded == pytest.approx(
        [float(i + 1) for i in range(len(recorded))]), (
        f'accumulator oracle is not a unit ramp: {recorded[:8]}')


def test_affine_assignment_routes_to_ssm():
    """The bare assignment `acc = acc + 1` routes to SSM (a=1, b=1) under the
    shipped router - the node-shape bug is poke-only, assignments are clean."""
    env, root = _oracle_env(
        "if perframe(0,1) then acc = acc + 1 end "
        "self:sleep(0.02); self:queuecommand('Update')",
        "acc = 0")
    body = update_integrator._update_body(root)
    frame_root = build_frames(body)
    routed = [route(u, f) for u, f in iter_updates(frame_root)
              if u.name == 'acc']
    assert routed and routed[0] is not None
    assert routed[0].coeffs['a'].at(0.0) == 1.0
    assert routed[0].coeffs['b'].at(0.0) == 1.0


# -- pure-curve pokes: DIFFERENTIAL FINDINGS (poke-router bug) ----------------


def test_constant_poke_does_not_flatten_but_would_match_when_corrected():
    """A CONSTANT pure-curve poke (`thing:rotationz(45)`) SHOULD flatten (a=0,
    b=45) and match the oracle. Under the shipped router it does NOT flatten
    (poke-node bug) - asserted as an xfail. The `corrected=True` path proves the
    parity WOULD hold once the router accepts the poke node shape: both CHANNEL
    and KERNEL streams match the recorded keyframes exactly."""
    env, root = _oracle_env(
        "if perframe(0,4) then thing:rotationz(45) end "
        "self:sleep(0.02); self:queuecommand('Update')",
        "thing = self")
    _run_oracle(env, root)

    shipped = _one_prop(_route_and_evaluate(env, root), 'thing', 'rotation')
    assert shipped['oracle_keyframes'] > 0

    # The parity logic itself: under the corrected router the constant curve
    # reproduces the oracle in BOTH modes (a constant is time-invariant).
    fixed = _one_prop(_route_and_evaluate(env, root, corrected=True),
                      'thing', 'rotation')
    assert fixed['routed_ssm'], 'corrected router still failed to flatten'
    _assert_no_mismatch(fixed, 'channel')
    _assert_no_mismatch(fixed, 'kernel')

    if not shipped['routed_ssm']:
        pytest.xfail(_ROUTER_POKE_BUG)


def test_beat_varying_poke_channel_matches_kernel_is_stage_b_gap():
    """A pure-curve poke VARYING in beat (`thing:rotationz(beat*10)`). Under the
    corrected router: CHANNEL mode (sample b(t) at each tick) reproduces the
    oracle exactly; KERNEL mode (frozen affine_kernel sampling b once at coord
    0.0) does NOT - it holds flat at b(0)=0. This is the Stage B widening
    (time-varying coefficient) made concrete. The shipped-router poke bug means
    nothing flattens at all, asserted as an xfail."""
    env, root = _oracle_env(
        "local beat = GAMESTATE:GetSongBeat() "
        "if perframe(0,4) then thing:rotationz(beat*10) end "
        "self:sleep(0.02); self:queuecommand('Update')",
        "thing = self")
    _run_oracle(env, root)

    fixed = _one_prop(_route_and_evaluate(env, root, corrected=True),
                      'thing', 'rotation')
    assert fixed['routed_ssm'], 'corrected router still failed to flatten'
    assert fixed['oracle_keyframes'] > 0
    # CHANNEL mode is the true closed form: it MUST match a time-varying curve.
    _assert_no_mismatch(fixed, 'channel')
    # KERNEL mode is the Stage A constant-coefficient bake: it MUST mismatch a
    # time-varying curve - the differential proof of the Stage B scope.
    _, kernel_mismatches = fixed['kernel']
    assert kernel_mismatches, (
        'KERNEL mode unexpectedly matched a time-varying curve - the frozen '
        'kernel should sample b once at coord 0.0 and hold flat')

    shipped = _one_prop(_route_and_evaluate(env, root), 'thing', 'rotation')
    if not shipped['routed_ssm']:
        pytest.xfail(_ROUTER_POKE_BUG)


def test_poke_reading_a_nonlinear_frame_var_needs_attention_not_ssm():
    """DIFFERENTIAL FINDING: a poke whose arg READS a frame variable that is
    updated nonlinearly (`prev = prev*prev + 1; thing:rotationz(prev)`). The
    Stage A flattener treats the bare `prev` arg as a pure curve (a=0, b=prev)
    and compiles `b` off the LIVE surface, which - once the oracle has run and
    left `prev` set as a global - folds to a CONSTANT (prev's final value). That
    constant cannot reproduce the recorded per-tick ramp, so under the corrected
    router this poke either falls to attention (b uncompilable) OR flattens to a
    value that MISMATCHES the oracle. Both outcomes say the same thing: this
    class must route to attention, and a value-read coefficient is a Stage B
    concern (cross-frame timeline resolution), never a compile-time constant.

    The engine records prev = 5, 26, 677, ... (x -> x*x + 1 from 2); a constant
    b can never be that ramp."""
    env, root = _oracle_env(
        "if perframe(0,2) then prev = prev*prev + 1; thing:rotationz(prev) end "
        "self:sleep(0.02); self:queuecommand('Update')",
        "thing = self; prev = 2")
    _run_oracle(env, root)
    fixed = _one_prop(_route_and_evaluate(env, root, corrected=True),
                      'thing', 'rotation')
    assert fixed['oracle_keyframes'] > 0
    # The recorded stream is the nonlinear ramp, not a constant.
    if not fixed['routed_ssm']:
        return          # correct outcome: attention floor caught it
    # It flattened - then the frozen (constant-b) closed form MUST mismatch the
    # recorded nonlinear ramp under both evaluation modes.
    assert _has_mismatch(fixed, 'channel') or _has_mismatch(fixed, 'kernel'), (
        'a poke reading a nonlinear frame var flattened AND matched the oracle - '
        'unexpected; the constant-b Stage A form cannot be a nonlinear ramp')


# -- gat 1: the real Update body ---------------------------------------------

_GAT1 = Path('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
             'UKSRT8/5. gat/gat.sm')


def _load_gat1():
    from analysis.games.etterna import sm_chart
    from analysis.games.notitg.modfile import (
        _beat_to_seconds, _load_document, _resolve_lua_dir, _timing,
        _sm_background_name, parse_fgchanges)
    entries = parse_fgchanges(_GAT1)
    lua_dir = _resolve_lua_dir(_GAT1, entries)
    root, _c, _cc = _load_document(
        lua_dir, Path(_sm_background_name(_GAT1)).stem.casefold())
    sm_data = sm_chart.parse_sm(_GAT1)
    _bpms, _offset, chart = _timing(sm_data)
    to_seconds = _beat_to_seconds(sm_data, chart)
    env = StubEnvironment(start_beat=0.0, to_seconds=to_seconds)
    env.load_actors(root)
    return env, root, to_seconds


@pytest.mark.skipif(not _GAT1.exists(), reason='gat 1 chart not present')
def test_gat1_frame_ir_ssm_matches_oracle_or_reports():
    """gat 1's real Update body: build frames, and for every SSM-routed poke
    that `compile_update` would actually BAKE (the sole-writer set), check its
    closed-form stream against the recorded oracle. Any mismatch in that set is
    reported LOUDLY and fails - it means the compiler baked something wrong.

    The sole-writer gate is essential: a property written by several pokes
    composes last-writer-wins per tick, which a standalone baked curve cannot
    reproduce (the harness caught exactly this on `char_shame.x`, 7 writers).
    `compile_update` excludes multi-written props, so parity is asserted over
    the same set it bakes - not over raw per-update routing."""
    env, root, to_seconds = _load_gat1()
    windows = update_integrator._live_windows(
        update_integrator._update_body(root))
    to_beat = update_integrator._beat_inverter(windows, to_seconds)
    update_integrator.integrate_update(env, root, to_seconds)
    results = _route_and_evaluate(env, root, to_seconds, to_beat)

    sole = _sole_writer_props(list(iter_updates(build_frames(
        update_integrator._update_body(root)))))
    ssm_pokes = [r for r in results if r['routed_ssm']]
    bakeable = [r for r in ssm_pokes
                if f"{r['actor']}.{r['method']}" in sole]
    mismatched = [r for r in bakeable
                  if _has_mismatch(r, 'channel')]
    report = _gat_report(results, ssm_pokes, mismatched)
    print(report, file=sys.stderr)

    # A mismatch in the BAKEABLE (sole-writer) set is a hard failure: the
    # compiler would emit wrong keyframes. Multi-writer props are excluded
    # from baking (they stay attention), so they are not asserted here.
    assert not mismatched, report


@pytest.mark.skipif(not _GAT1.exists(), reason='gat 1 chart not present')
def test_gat1_corrected_router_stage_split_report():
    """gat 1 under the CORRECTED router (poke nodes wrapped so they can
    flatten): how many pokes flatten to SSM, and of those how many MATCH the
    oracle (Stage A parity) vs MISMATCH (Stage B scope: time-varying or
    value-read coefficients). This quantifies the real-chart split the router
    correction unlocks, and is the scoping datum for Stage B. Reported, never a
    parity gate on its own - a corrected-router mismatch is expected for the
    time-varying/value-read classes and is exactly what Stage B must widen."""
    env, root, to_seconds = _load_gat1()
    windows = update_integrator._live_windows(
        update_integrator._update_body(root))
    to_beat = update_integrator._beat_inverter(windows, to_seconds)
    update_integrator.integrate_update(env, root, to_seconds)
    results = _route_and_evaluate(env, root, to_seconds, to_beat,
                                  corrected=True)

    ssm = [r for r in results if r['routed_ssm']]
    with_oracle = [r for r in ssm if r['oracle_keyframes'] > 0]
    channel_match = [r for r in with_oracle if not _has_mismatch(r, 'channel')]
    kernel_match = [r for r in with_oracle if not _has_mismatch(r, 'kernel')]
    print('\n'.join([
        '', '=== gat 1 corrected-router stage split ===',
        f'poke VarUpdates:            {len(results)}',
        f'flatten to SSM:             {len(ssm)}',
        f'  ... with an oracle stream:{len(with_oracle)}',
        f'  CHANNEL-mode parity:      {len(channel_match)}/{len(with_oracle)}',
        f'  KERNEL-mode parity:       {len(kernel_match)}/{len(with_oracle)}',
    ]), file=sys.stderr)
    # Whatever flattens must at least be recognized; this test documents the
    # split rather than gating on it (Stage B mismatches are expected here).
    assert len(results) > 0


# -- Part 2: the integrator seam sketch (frame_compile.compile_update) --------
#
# compile_update splits an Update body into SSM-baked keyframes (no lupa) and
# attention windows (the bounded lupa loop). These tests prove the SSM-baked
# half reconstructs the oracle for the flattenable class, and that the split
# degrades to the whole-body window when nothing flattens (the safe production
# floor). See analysis/games/notitg/frame_compile.py.


def _compile_plan(env, root, to_seconds=_IDENTITY, to_beat=None):
    from analysis.games.notitg.frame_compile import compile_update
    body = update_integrator._update_body(root)
    surface = NotitgGuardSurface(env, to_beat=to_beat or to_seconds)
    return compile_update(body, surface, to_seconds)


def _baked_matches_oracle(plan, oracle, actor, prop, tol=1e-6):
    """(matched, mismatches) for the baked SSM keyframes of (actor, prop) vs the
    oracle keyframes, paired at each tick the oracle poked."""
    baked = plan.ssm_keyframes.get((actor, prop), [])
    oracle_kfs = oracle.get(actor, {}).get(prop, [])
    ticks = [kf.t for kf in baked]
    values = [kf.values[0] for kf in baked]
    return _compare(actor, prop, oracle_kfs, values, ticks, tol)


def test_seam_bakes_pure_curve_matching_the_oracle():
    """A pure-curve poke: compile_update bakes it (applying the poke-router
    correction locally) and the baked keyframes reconstruct the recorded oracle
    tick for tick. This is the byte-parity proof for the SSM half of the split -
    the seam's SSM sink == the all-lupa keyframes for the flattenable class."""
    env, root = _oracle_env(
        "local beat = GAMESTATE:GetSongBeat() "
        "if perframe(0,4) then thing:rotationz(beat*10) end "
        "self:sleep(0.02); self:queuecommand('Update')",
        "thing = self")
    oracle = _run_oracle(env, root)
    plan = _compile_plan(env, root)

    assert plan.flattened >= 1, 'the pure-curve poke was not baked'
    assert ('thing', 'rotation') in plan.ssm_keyframes
    matched, mismatches = _baked_matches_oracle(plan, oracle, 'thing', 'rotation')
    assert matched > 0 and not mismatches, (
        f'baked SSM keyframes did not reconstruct the oracle: {mismatches[:5]}')


def test_seam_keeps_a_value_read_poke_on_attention():
    """A poke reading a frame variable updated nonlinearly
    (`prev = prev*prev + 1; thing:rotationz(prev)`) must NOT bake to SSM: `prev`
    is not a time driver, so under the driver-only compile surface its curve
    fails to compile and the poke stays on attention (where it evaluates live).
    This closes the value-read hazard - a coefficient that reads a
    frame/actor value can never be folded to a load-time constant, because
    only beat/mod_time/measure resolve for flattening. The baked stream is
    therefore absent (not present-and-wrong)."""
    env, root = _oracle_env(
        "if perframe(0,2) then prev = prev*prev + 1; thing:rotationz(prev) end "
        "self:sleep(0.02); self:queuecommand('Update')",
        "thing = self; prev = 2")
    plan = _compile_plan(env, root)

    # The value-read poke is not baked - no SSM keyframes for thing.rotation.
    assert not plan.ssm_keyframes.get(('thing', 'rotation')), (
        'a value-read poke was baked to SSM; its non-driver coefficient must '
        'keep it on attention')


@pytest.mark.skipif(not _GAT1.exists(), reason='gat 1 chart not present')
def test_seam_gat1_split_reduces_attention_windows():
    """gat 1: compile_update's split flattens a subset of pokes to SSM and
    leaves the rest as attention windows. Reports the lupa-tick reduction the
    seam would buy (attention windows vs the whole live-window union) - the
    Stage A measurement the design asks for. Not a parity gate; the SSM-half
    parity is proven on the synthetic pure-curve class above and quantified on
    gat by test_gat1_corrected_router_stage_split_report."""
    env, root, to_seconds = _load_gat1()
    body = update_integrator._update_body(root)
    windows = update_integrator._live_windows(body)
    to_beat = update_integrator._beat_inverter(windows, to_seconds)
    update_integrator.integrate_update(env, root, to_seconds)
    plan = _compile_plan(env, root, to_seconds, to_beat)

    live_span = sum(e - s for s, e in windows)
    attention_span = sum(e - s for s, e in plan.attention_windows)
    print('\n'.join([
        '', '=== gat 1 seam split (frame_compile) ===',
        f'flattened pokes (SSM bake): {plan.flattened}',
        f'attention pokes:            {plan.attention}',
        f'live-window beats:          {live_span:.1f}',
        f'attention-window beats:     {attention_span:.1f}',
        f'baked (actor,prop) streams: {len(plan.ssm_keyframes)}',
    ]), file=sys.stderr)
    assert plan.attention + plan.flattened > 0


# -- report + assertion helpers ----------------------------------------------


def _one_prop(results, actor, prop):
    for record in results:
        if record['actor'] == actor and record['prop'] == prop:
            return record
    raise AssertionError(
        f'no VarUpdate produced a ({actor!r}, {prop!r}) result; '
        f'got {[(r["actor"], r["prop"]) for r in results]}')


def _has_mismatch(record, mode) -> bool:
    entry = record.get(mode)
    return bool(entry and entry[1])


def _assert_no_mismatch(record, mode):
    entry = record.get(mode)
    assert entry is not None, f'{mode} stream not evaluated for {record}'
    matched, mismatches = entry
    assert not mismatches, (
        f'{mode}-mode SSM mismatched the oracle for '
        f'{record["actor"]}.{record["prop"]}: {mismatches[:5]} '
        f'(matched {matched})')


def _gat_report(results, ssm_pokes, mismatched) -> str:
    lines = ['', '=== gat 1 frame-IR SSM parity ===',
             f'poke VarUpdates seen:  {len(results)}',
             f'routed to SSM:         {len(ssm_pokes)}',
             f'SSM mismatched oracle: {len(mismatched)}']
    for record in mismatched:
        for mode in ('channel', 'kernel'):
            entry = record.get(mode)
            if entry and entry[1]:
                lines.append(f'  {record["actor"]}.{record["prop"]} '
                             f'[{mode}] first mismatches: {entry[1][:3]}')
    return '\n'.join(lines)
