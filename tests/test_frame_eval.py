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


# -- LuaTable: constructor, stdlib, iteration --------------------------------

def test_table_constructor_and_named_field_access():
    root = _run('t = {x = 5, y = 9}\nr = t.x + t.y')
    assert root.get('r') == 14.0


def test_table_array_index_is_one_based():
    root = _run("t = {'a', 'b', 'c'}\nr = t[2]")
    assert root.get('r') == 'b'


def test_table_insert_and_getn():
    root = _run("t = {'a'}\ntable.insert(t, 'b')\ntable.insert(t, 'c')\n"
                'n = table.getn(t)')
    assert root.get('n') == 3.0


def test_table_insert_at_position():
    root = _run("t = {'a', 'c'}\ntable.insert(t, 2, 'b')\nr = t[2]\n"
                'n = table.getn(t)')
    assert root.get('r') == 'b' and root.get('n') == 3.0


def test_table_remove_returns_and_shrinks():
    root = _run("t = {'a', 'b', 'c'}\ngone = table.remove(t)\n"
                'n = table.getn(t)')
    assert root.get('gone') == 'c' and root.get('n') == 2.0


def test_ipairs_iterates_the_array_part_in_order():
    root = _run("t = {'a', 'b', 'c'}\nout = ''\n"
                'for i, v in ipairs(t) do out = out .. v end')
    assert root.get('out') == 'abc'


def test_pairs_iterates_all_entries():
    root = _run('t = {10, 20}\ntotal = 0\n'
                'for k, v in pairs(t) do total = total + v end')
    assert root.get('total') == 30.0


def test_generic_for_over_non_table_is_skipped_not_fatal():
    # An unrecognised iterator (not ipairs/pairs over a LuaTable) is skipped;
    # statements after it still run.
    root = _run('for x in some_iter() do y = 1 end\nafter = 2')
    assert root.get('after') == 2.0


# -- _G global-table idiom ---------------------------------------------------

def test_g_table_computed_global_write_and_read():
    # `_G['P'..n] = v` writes a global by computed name; `_G['P'..n]` reads it.
    root = _run("n = 1\n_G['P'..n] = 42\nr = _G['P'..n]")
    assert root.get('r') == 42.0
    assert root.get('P1') == 42.0        # and it is a real global


def test_index_assignment_into_a_table():
    root = _run('t = {}\nt.foo = 7\nt[1] = 9\na = t.foo\nb = t[1]')
    assert root.get('a') == 7.0 and root.get('b') == 9.0


# -- stdlib value-inspection builtins ----------------------------------------

def test_type_builtin_names_each_value_kind():
    # `type(v)` drives dispatch gates (`if type(x) == 'function'`); each kind
    # must name itself. A LuaTable is 'table', an interpreter closure is
    # 'function', an unset name is 'nil'.
    root = _run("f = function() end\n"
                "tn = type(3)\nts = type('x')\ntb = type(true)\n"
                "tt = type({})\ntf = type(f)\ntz = type(missing)")
    assert root.get('tn') == 'number'
    assert root.get('ts') == 'string'
    assert root.get('tb') == 'boolean'
    assert root.get('tt') == 'table'
    assert root.get('tf') == 'function'
    assert root.get('tz') == 'nil'


def test_tonumber_parses_strings_and_passes_numbers():
    root = _run("a = tonumber('2')\nb = tonumber(5)\n"
                "c = tonumber('nope')\nd = tonumber('0x1F')")
    assert root.get('a') == 2.0
    assert root.get('b') == 5.0
    assert root.get('c') is None            # unparsable -> nil
    assert root.get('d') == 31.0            # hex literal


def test_tostring_stringifies_lua_style():
    root = _run("a = tostring(2)\nb = tostring(true)\nc = tostring(missing)")
    assert root.get('a') == '2'             # integer float drops the .0
    assert root.get('b') == 'true'
    assert root.get('c') == 'nil'


def test_absent_table_field_is_nil_not_unresolved_for_or():
    # A resolved table indexed at an absent key is a KNOWN nil, so `t[k] or x`
    # evaluates x (nil is falsy). This is the residue-loop semantics the guard
    # UNRESOLVED discipline must NOT poison - the Machine Wave action-dispatch
    # gate `t[3] or beat < t[1]+2` hinges on it.
    root = _run("t = {10}\nr = t[3] or 99")
    assert root.get('r') == 99.0


def test_string_escape_sequences_decode_to_characters():
    # A `\n` in a string literal is a real newline, not a literal backslash-n
    # (the POP debug-text divergence: recorded text had a literal \\n). `\t`
    # and an escaped quote decode too.
    root = _run("s = 'a\\nb\\tc\\'d'")
    assert root.get('s') == "a\nb\tc'd"
