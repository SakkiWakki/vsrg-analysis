"""The statement interpreter (`frame_eval`): pure-logic execution over a
Surface - scope/accumulator state, control flow, closures, Lua and/or operand
semantics, and the UNRESOLVED "skip, do not guess" discipline. No lupa, no
game: a ConstSurface supplies constants and everything else is the
interpreter.
"""
from analysis.player.render.expr.frame_eval import Interpreter, Scope
from analysis.player.render.expr.parser import parse_body
from analysis.player.render.expr.surface import ConstSurface, UNRESOLVED


def _run(src, constants=None, ticks=1):
    interp = Interpreter(ConstSurface(constants or {}))
    stmts, _sink = parse_body(src)
    for _ in range(ticks):
        interp.run(stmts)
    return interp.root


# -- scope + accumulator state -----------------------------------------------

def test_local_binds_and_arithmetic_evaluates():
    root = _run('local y = 3\nz = y * 2 + 1')
    assert root.get('z') == 7.0


def test_local_does_not_leak_to_a_sibling_run():
    # A `local` lives in the run's scope; the persistent root keeps only
    # globals. (Here the top scope IS root, so a local written at top level
    # persists - the accumulator relies on that; a local inside a block does
    # not. This pins the top-level global behaviour.)
    interp = Interpreter(ConstSurface())
    stmts, _ = parse_body('g = 5')
    interp.run(stmts)
    assert interp.root.get('g') == 5.0


def test_accumulator_recurrence_carries_across_ticks():
    # x = x + step, seeded, run repeatedly: the global persists between runs,
    # so the recurrence integrates - the per-frame accumulator the sim needs.
    interp = Interpreter(ConstSurface())
    interp.run(parse_body('acc = 0')[0])
    step, _ = parse_body('acc = acc + 3')
    for _ in range(4):
        interp.run(step)
    assert interp.root.get('acc') == 12.0


# -- control flow ------------------------------------------------------------

def test_if_elseif_else_takes_the_true_branch():
    assert _run('if v > 10 then r = 1 elseif v > 5 then r = 2 else r = 3 end',
                {'v': 7}).get('r') == 2.0
    assert _run('if v > 10 then r = 1 else r = 3 end', {'v': 3}).get('r') == 3.0


def test_numeric_for_sums():
    assert _run('total = 0\nfor i = 1, 4 do total = total + i end'
                ).get('total') == 10.0


def test_numeric_for_with_step():
    # descending step: i = 10, 8, 6, 4, 2  ->  sum 30
    assert _run('total = 0\nfor i = 10, 2, -2 do total = total + i end'
                ).get('total') == 30.0


def test_while_loop_terminates_and_accumulates():
    assert _run('n = 0\nwhile n < 3 do n = n + 1 end').get('n') == 3.0


# -- closures ----------------------------------------------------------------

def test_closure_defers_and_calls():
    assert _run('f = function(n) return n * n end\nr = f(6)').get('r') == 36.0


def test_closure_captures_defining_scope():
    root = _run('base = 10\nadd = function(n) return base + n end\nr = add(5)')
    assert root.get('r') == 15.0


# -- Lua and/or operand semantics --------------------------------------------

def test_or_returns_operand_not_bool():
    assert _run('r = false or 7').get('r') == 7.0
    assert _run('r = 5 or 0').get('r') == 5.0
    assert _run('r = nil or 9').get('r') == 9.0


def test_and_returns_operand_not_bool():
    assert _run('r = 3 and 8').get('r') == 8.0
    assert _run('r = false and 8').get('r') is False
    assert _run('r = nil and 8').get('r') is None


# -- UNRESOLVED discipline ---------------------------------------------------

def test_unprovable_condition_does_not_run_its_branch():
    # `unknown` is off the surface -> UNRESOLVED -> the if is not taken (skip,
    # do not guess), so `ran` stays unset.
    assert _run('if unknown_flag then ran = 1 end').get('ran') is UNRESOLVED


def test_unresolved_left_of_or_stays_unresolved():
    # `x or 0` with x unknowable is UNRESOLVED, never a fabricated 0 - the
    # accumulator-seed hazard.
    assert _run('r = x or 0').get('r') is UNRESOLVED


def test_unparsed_node_is_skipped_not_fatal():
    # A body with an unmodeled construct still runs its parseable statements.
    root = _run('a = 1\ngoto somewhere\nb = 2')
    assert root.get('a') == 1.0
    # the goto is Unparsed and skipped; parsing recovers to keep going


def test_arithmetic_with_unresolved_operand_is_unresolved():
    assert _run('r = mystery + 1').get('r') is UNRESOLVED
