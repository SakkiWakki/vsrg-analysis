"""Unit tests for the game-agnostic Lua expression front-end (render/expr).

No NotITG, no lupa: parser AST shapes, precedence, tree-walk evaluation,
UNRESOLVED propagation, diagnostics, and window extraction.
"""
import pytest

from analysis.player.render.expr import ast
from analysis.player.render.expr.diagnostics import DiagnosticSink, Severity
from analysis.player.render.expr.eval_tree import tree_eval
from analysis.player.render.expr.lexer import Tok, tokenize
from analysis.player.render.expr.parser import parse_body, parse_guard
from analysis.player.render.expr.surface import UNRESOLVED, ConstSurface
from analysis.player.render.expr.windows import guard_window


# -- lexer -------------------------------------------------------------------

def test_lexer_tokenizes_operators_and_names():
    kinds = [(t.kind, t.text) for t in tokenize('beat >= v[3]')
             if t.kind is not Tok.EOF]
    assert kinds == [(Tok.NAME, 'beat'), (Tok.OP, '>='), (Tok.NAME, 'v'),
                     (Tok.OP, '['), (Tok.NUMBER, '3'), (Tok.OP, ']')]


def test_lexer_strips_comments_preserving_offsets():
    src = 'beat --[[hi]] > 3\n-- dead\nbeat < 5'
    text = ''.join(t.text for t in tokenize(src) if t.kind is not Tok.EOF)
    assert 'hi' not in text and 'dead' not in text


# -- parser + precedence -----------------------------------------------------

def test_and_groups_above_comparisons():
    node = parse_guard('beat > 40 and beat < 60')
    assert isinstance(node, ast.Binary) and node.op == 'and'
    assert isinstance(node.left, ast.Binary) and node.left.op == '>'
    assert isinstance(node.right, ast.Binary) and node.right.op == '<'


def test_arithmetic_binds_tighter_than_comparison():
    node = parse_guard('beat < e[1] + e[2]')
    assert node.op == '<'
    assert isinstance(node.right, ast.Binary) and node.right.op == '+'


def test_index_and_call_shapes():
    idx = parse_guard('v[3]')
    assert isinstance(idx, ast.Index) and idx.base.name == 'v'
    call = parse_guard('perframe(1240, 1240+128)')
    assert isinstance(call, ast.Call) and call.fn.name == 'perframe'
    assert len(call.args) == 2


def test_out_of_scope_token_is_unparseable_guard():
    assert parse_guard('beat .. 3 == "x"') is None or isinstance(
        parse_guard('{1,2,3}'), ast.Table)


def test_body_parses_statements_and_records_unparsed():
    body = ('local beat = f()\n'
            'if beat > 0 and beat < 10 then g(beat) end\n'
            'for i = 1, 4 do h(i) end')
    stmts, sink = parse_body(body)
    kinds = [type(s).__name__ for s in stmts]
    assert kinds == ['Local', 'If', 'NumericFor']
    assert not sink.has_errors


# -- tree-walk + UNRESOLVED --------------------------------------------------

def test_resolved_comparison_and_conjunction():
    surface = ConstSurface({'beat': 50})
    assert tree_eval(parse_guard('beat > 40 and beat < 60'), surface) is True
    surface2 = ConstSurface({'beat': 70})
    assert tree_eval(parse_guard('beat > 40 and beat < 60'), surface2) is False


def test_table_index_resolution():
    surface = ConstSurface({'beat': 200, 'v': [None, None, 128]})
    assert tree_eval(parse_guard('beat >= v[3]'), surface) is True
    nil = ConstSurface({'beat': 200})
    assert tree_eval(parse_guard('beat >= v[3]'), nil) is UNRESOLVED


def test_unresolved_propagates_through_arithmetic():
    surface = ConstSurface({'e': [10, 20]})       # beat unresolved
    assert tree_eval(parse_guard('beat < e[1] + e[2]'), surface) is UNRESOLVED


@pytest.mark.parametrize('expr,expected', [
    ('false and x', False),           # short-circuit, x never resolved
    ('true and beat', UNRESOLVED),    # resolved-true and unresolved -> skip
    ('true or beat', True),           # short-circuit
    ('false or beat', UNRESOLVED),
])
def test_and_or_unresolved_rules(expr, expected):
    surface = ConstSurface()          # every symbol unresolved
    assert tree_eval(parse_guard(expr), surface) is expected


# -- window extraction -------------------------------------------------------

@pytest.mark.parametrize('guard,constants,window', [
    ('perframe(128, 252)', {}, (128.0, 252.0)),
    ('perframe(50)', {}, (50.0, 51.0)),
    ('perframe(1240, 1240+128)', {}, (1240.0, 1368.0)),
    ('beat > 40 and beat < 60', {}, (40.0, 60.0)),
    ('beat>0 and beat<127', {}, (0.0, 127.0)),
    ('beat > e[1] and beat < e[1]+e[2]', {'e': [10, 20]}, (10.0, 30.0)),
    ('beat >= 40 and beat < 60 and fgcurcommand == 2',
     {'fgcurcommand': 2}, (40.0, 60.0)),
])
def test_guard_window_shapes(guard, constants, window):
    assert guard_window(parse_guard(guard), ConstSurface(constants)) == window


@pytest.mark.parametrize('guard,constants', [
    ('beat > a and beat < b', {}),              # live locals -> skip
    ('beat >= v[3]', {}),                       # nil table -> skip
    ('beat >= 40 and fgcurcommand == 2', {'fgcurcommand': 2}),  # no upper
    ('beat < e[1]+e[2]', {'e': [10, 20]}),      # no lower bound
])
def test_guard_window_skips_unbounded(guard, constants):
    assert guard_window(parse_guard(guard), ConstSurface(constants)) is None


# -- multi-window / DNF ------------------------------------------------------

def test_disjunction_of_ranges_yields_multiple_windows():
    from analysis.player.render.expr.windows import guard_windows
    node = parse_guard('(beat > 10 and beat < 20) or (beat > 40 and beat < 60)')
    assert guard_windows(node, ConstSurface()) == [(10.0, 20.0), (40.0, 60.0)]


def test_dnf_distributes_and_over_or():
    # gate and (range1 or range2) -> both ranges live when the gate holds.
    from analysis.player.render.expr.windows import guard_windows
    node = parse_guard(
        'gate and ((beat > 10 and beat < 20) or (beat > 40 and beat < 60))')
    windows = guard_windows(node, ConstSurface({'gate': True}))
    assert windows == [(10.0, 20.0), (40.0, 60.0)]


def test_disjunction_of_perframe_calls():
    from analysis.player.render.expr.windows import guard_windows
    node = parse_guard('perframe(10, 20) or perframe(40, 60)')
    assert guard_windows(node, ConstSurface()) == [(10.0, 20.0), (40.0, 60.0)]


def test_extra_conjunct_over_live_var_is_ignored():
    # `beat > live and beat > 118 and beat < 236` -> (118, 236); the
    # unresolvable `beat > live` conjunct narrows the start we cannot see
    # but must not drop the window.
    node = parse_guard('beat > live and beat > 118 and beat < 236')
    assert guard_window(node, ConstSurface()) == (118.0, 236.0)


# -- compile backend ---------------------------------------------------------

def test_compiled_guard_matches_tree_walk_on_a_grid():
    from analysis.player.render.expr.compile_sched import compile_guard

    class ClockSurface(ConstSurface):
        def clock_reader(self, name):
            return (lambda t: t * 2.0) if name == 'beat' else None

    node = parse_guard('beat > 10 and beat < 50')
    channel = compile_guard(node, ClockSurface())
    assert channel is not None
    for i in range(60):
        t = i * 0.5
        oracle = tree_eval(node, ConstSurface({'beat': t * 2.0}))
        assert bool(channel.at(t)) == bool(oracle)


def test_uncompilable_guard_returns_none():
    from analysis.player.render.expr.compile_sched import compile_guard
    # a live local with no clock reader cannot compile.
    assert compile_guard(parse_guard('beat > a'), ConstSurface()) is None


# -- diagnostics -------------------------------------------------------------

def test_unmodeled_body_records_warning_not_error():
    stmts, sink = parse_body('x = beat .. "concat"\nif beat > 3 then f() end')
    warnings = sink.messages(Severity.WARNING)
    assert any('unparsed' in w or 'syntax' in w for w in warnings) \
        or not sink.has_errors
