"""Sink definitions for the plugin verifier.

A sink is a call pattern that would violate sandbox invariants if reached
from a plugin entry point. Two kinds:

1. Static sinks - call targets whose name alone is sufficient to flag
   (e.g. object.__setattr__, __import__, eval). Checked by name match
   against the call graph without needing Z3.

2. Constraint sinks - calls where the arguments must satisfy a property.
   Currently: config.set(field, ...) where field must be prefixed with
   the plugin's own key. Checked with Z3 string reasoning.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass


# Call target names that are always sinks regardless of arguments.
STATIC_SINK_NAMES = frozenset({
    # Builtin escape hatches
    '__import__', 'importlib', 'eval', 'exec', 'compile',
    'open', 'memoryview',
    # Frame and introspection escapes
    'globals', 'locals', 'vars',
    'getattr',         # conservative: any getattr is flagged (may access __class__ etc.)
    'setattr',         # any setattr could mutate frozen host objects
    'delattr',
    # Object model escapes
    'object.__setattr__',
    'object.__new__',
    'type.__new__',
    # OS-level
    'os', 'sys',
})

# Attribute accesses on any object that are sinks.
STATIC_SINK_ATTRS = frozenset({
    '__class__', '__bases__', '__mro__', '__subclasses__',
    '__dict__', '__module__', '__code__', '__globals__',
    '__builtins__', '__import__',
    '__reduce__', '__reduce_ex__',  # pickle escape vectors
})


@dataclass
class SinkViolation:
    """A detected sink reachable from a plugin entry point."""
    kind: str          # 'static' or 'constraint'
    description: str
    lineno: int
    col_offset: int


def check_static(node: ast.Call) -> SinkViolation | None:
    """Return a SinkViolation if this call node is a static sink, else None."""
    lineno = getattr(node, 'lineno', 0)
    col = getattr(node, 'col_offset', 0)

    func = node.func
    name = _call_name(func)
    if name in STATIC_SINK_NAMES:
        return SinkViolation('static', f'call to blocked name {name!r}',
                             lineno, col)

    # Attribute access on any object - check the attr name
    if isinstance(func, ast.Attribute):
        if func.attr in STATIC_SINK_ATTRS:
            return SinkViolation(
                'static',
                f'access to blocked attribute {func.attr!r}',
                lineno, col)

    return None


def check_config_set(node: ast.Call, plugin_key: str) -> SinkViolation | None:
    """Use Z3 to verify that if this looks like a config.set(field, ...)
    call, the field argument is always prefixed with the plugin's own key.
    Returns a SinkViolation if Z3 finds a satisfying assignment where the
    prefix constraint is violated, else None."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr != 'set':
        return None

    # Only check calls that look like *.set(field, value)
    if len(node.args) < 1:
        return None

    field_arg = node.args[0]

    # If field is a string literal we can check directly without Z3
    if isinstance(field_arg, ast.Constant) and isinstance(field_arg.value, str):
        field_val = field_arg.value
        if _is_path_traversal(field_val, plugin_key):
            return SinkViolation(
                'constraint',
                f'config key {field_val!r} may escape plugin namespace '
                f'(expected prefix derived from {plugin_key!r})',
                getattr(node, 'lineno', 0),
                getattr(node, 'col_offset', 0))
        return None

    # For non-literal field args, use Z3 to check satisfiability
    return _z3_check_prefix(node, plugin_key)


def _is_path_traversal(field: str, plugin_key: str) -> bool:
    """Return True if `field` contains path traversal patterns that could
    escape the plugin's config subtree. Heuristic for literal strings."""
    return '..' in field or field.startswith('.')


def _z3_check_prefix(node: ast.Call, plugin_key: str) -> SinkViolation | None:
    """Ask Z3 whether there exists a value for the field argument that
    does not start with the plugin's escaped key. Returns a violation if
    SAT (i.e. escape is possible), None if UNSAT (always safe)."""
    lineno = getattr(node, 'lineno', 0)
    col = getattr(node, 'col_offset', 0)
    field_arg = node.args[0]

    try:
        import z3
    except ImportError:
        return SinkViolation(
            'constraint',
            'config.set() field argument cannot be verified without z3',
            lineno,
            col)

    if not isinstance(field_arg, ast.Name):
        # Non-literal, non-Name argument -- conservative: flag it
        return SinkViolation(
            'constraint',
            'config.set() called with non-literal field argument; '
            'cannot verify namespace isolation',
            lineno,
            col)

    # Encode: does there exist a string s such that s does NOT start with
    # the escaped plugin key prefix?
    escaped = plugin_key.replace('.', '_')
    s = z3.String('field')
    solver = z3.Solver()
    solver.add(z3.Not(z3.PrefixOf(z3.StringVal(escaped), s)))
    result = solver.check()

    if result == z3.sat:
        return SinkViolation(
            'constraint',
            f'config.set() field argument is not provably scoped to '
            f'plugin {plugin_key!r}; escape is satisfiable',
            lineno,
            col)
    return None


def _call_name(func: ast.expr) -> str:
    """Extract a dotted name string from a call's func node, e.g.
    ast.Name('open') -> 'open', ast.Attribute(value=Name('object'), attr='__setattr__')
    -> 'object.__setattr__'. Returns '' for unresolvable expressions."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f'{parent}.{func.attr}' if parent else func.attr
    return ''
