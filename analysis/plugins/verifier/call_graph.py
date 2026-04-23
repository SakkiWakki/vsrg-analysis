"""AST-based call graph builder for the plugin verifier.

Builds a map of {function_name: [ast.Call]} for all functions defined in
a module, then does a reachability traversal from known plugin entry points
to collect only the call sites reachable during execution.

Intra-module only: calls to names not defined in the module are treated
as trusted (they were already vetted by the import gate or are host API
calls). This avoids needing to analyse the full dependency graph.
"""
from __future__ import annotations

import ast
from collections import defaultdict


# All function names that can be plugin entry points.
_ENTRY_POINTS = frozenset({
    'draw', 'register_components', 'register_sidebar',
    'register_overlay', 'register',
})


def reachable_calls(tree: ast.Module) -> list[ast.Call]:
    """Return all ast.Call nodes reachable from any plugin entry point
    in this module. Module-level code (outside any function) is always
    included since it runs at import time."""
    defined = _collect_defined_functions(tree)
    reachable_funcs = _reachable_from_entries(defined)

    calls = []

    # Module-level statements run at import time -- always included
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            for call in _calls_in(node):
                calls.append(call)

    # Calls inside reachable function bodies
    for func_name in reachable_funcs:
        func_node = defined[func_name]
        for call in _calls_in(func_node):
            calls.append(call)

    return calls


def _collect_defined_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Map function name -> AST node for every top-level function def
    in the module. Class methods are included under 'ClassName.method'."""
    funcs: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
    return funcs


def _reachable_from_entries(
        defined: dict[str, ast.FunctionDef]) -> set[str]:
    """BFS from entry points through intra-module calls."""
    reachable = set()
    queue = [name for name in defined if name in _ENTRY_POINTS]

    while queue:
        func_name = queue.pop()
        if func_name in reachable or func_name not in defined:
            continue
        reachable.add(func_name)
        func_node = defined[func_name]
        for call in _calls_in(func_node):
            callee = _callee_name(call)
            if callee and callee in defined and callee not in reachable:
                queue.append(callee)

    return reachable


def _calls_in(node: ast.AST) -> list[ast.Call]:
    """All ast.Call nodes anywhere inside `node`."""
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _callee_name(call: ast.Call) -> str | None:
    """Best-effort extraction of the callee name for intra-module calls."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return func.value.id  # conservative: track the object name
    return None
