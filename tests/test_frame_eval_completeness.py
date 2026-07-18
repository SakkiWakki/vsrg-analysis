"""Grammar-completeness firewall for the interpreter (map-generality gate).

The interpreter is map-general iff it covers the LANGUAGE - every AST node the
parser can emit - not any one chart's idioms. This ratchets that: for every
non-`Unparsed` node type in `expr.ast`, the interpreter must handle it (execute
or deliberately, traceably skip) WITHOUT raising and WITHOUT silently falling
to an unaccounted generic default. A new node type added to the grammar fails
this test until the interpreter accounts for it - the same discipline
VERB_REGISTRY enforces on the verb surface.
"""
from dataclasses import fields

from analysis.player.render.expr import ast
from analysis.player.render.expr.frame_eval import Interpreter, Scope, _Return
from analysis.player.render.expr.surface import ConstSurface


def _all_node_types():
    return {name: obj for name in dir(ast)
            if isinstance(obj := getattr(ast, name), type)
            and issubclass(obj, ast.Node) and obj is not ast.Node}


# Every node type the parser can emit, split by how the interpreter must treat
# it. STATEMENT nodes are executed by `_exec`; EXPRESSION nodes are evaluated
# by `_eval`. `Unparsed` is the skip-floor. This mapping IS the completeness
# claim: a node absent from both sets (and not Unparsed) is an unaccounted gap.
_STATEMENT_NODES = {
    'Local', 'Assign', 'ExprStmt', 'If', 'NumericFor', 'GenericFor',
    'While', 'FuncDef', 'Return',
}
_EXPRESSION_NODES = {
    'Num', 'Str', 'Bool', 'Nil', 'Sym', 'Index', 'Field', 'Unary', 'Binary',
    'Call', 'Method', 'FuncExpr', 'Table',
}
_SKIP_FLOOR = {'Unparsed'}


def test_every_ast_node_is_accounted_for():
    covered = _STATEMENT_NODES | _EXPRESSION_NODES | _SKIP_FLOOR
    actual = set(_all_node_types())
    missing = actual - covered
    stale = covered - actual
    assert not missing, (
        f'AST node types the interpreter does not account for (add to '
        f'_exec/_eval and this map): {sorted(missing)}')
    assert not stale, (
        f'accounted node types that no longer exist (prune the map): '
        f'{sorted(stale)}')


def _minimal(name: str) -> ast.Node:
    """A minimal well-typed instance of node `name` for a does-not-raise
    smoke. Only the fields the constructor requires are filled, with inert
    leaf children."""
    leaf = ast.Num(1.0)
    kwargs = {}
    for field in fields(getattr(ast, name)):
        if field.name == 'span':
            continue
        kwargs[field.name] = _field_value(field.name, leaf)
    return getattr(ast, name)(**kwargs)


def _field_value(field_name: str, leaf):
    match field_name:
        case 'value':
            return 1.0
        case 'name':
            return 'x'
        case 'op':
            return '+'
        case 'is_local':
            return False
        case 'raw':
            return 'raw'
        case 'targets':
            return (ast.Sym('x'),)
        case 'names' | 'params':
            return ('x',)
        case 'args' | 'body' | 'array' | 'fields' | 'values' | 'exprs' \
                | 'elifs' | 'orelse':
            return ()
        case 'step':
            return None
        case _:
            # recv / fn / cond / base / key / start / stop / operand / left /
            # right - all Node-typed children.
            return leaf


def test_no_node_type_raises_the_interpreter():
    # Each statement node executes and each expression node evaluates without
    # raising, over an empty ConstSurface - the total-over-the-grammar floor.
    interp = Interpreter(ConstSurface())
    for name in _STATEMENT_NODES | _SKIP_FLOOR:
        try:
            interp._exec(_minimal(name), Scope(), 0)
        except _Return:
            # `return` unwinds via _Return by design (caught at the call
            # boundary); that is handling, not a fault.
            pass
    for name in _EXPRESSION_NODES:
        interp._eval(_minimal(name), Scope(), 0)
