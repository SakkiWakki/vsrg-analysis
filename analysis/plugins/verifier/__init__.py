"""Plugin symbolic verifier.

Analyses plugin source ASTs before execution to verify sandbox invariants
that the import allow-list alone cannot catch:

  - No reachable call to a statically-blocked name (alias chains,
    getattr escapes, object model bypasses).
  - No mutation of frozen host objects via setattr / object.__setattr__.
  - config.set() field arguments are always scoped to the plugin's own
    namespace (checked with Z3 string reasoning).

Called from analysis/plugins/__init__.py between prepare_sandboxed_module
and exec_module. Verification failure raises VerificationError, which the
loader catches and surfaces in bundle.load_errors -- the plugin is not
loaded.

Timeouts
--------
Each phase has a hard wall-clock timeout. Exceeding it is treated as a
hard block identical to a detected violation. A plugin that causes the
verifier to time out is refused -- whether the cause is legitimate
complexity or a deliberate DoS attempt doesn't matter; the result is the
same.

  CALL_GRAPH_TIMEOUT_MS  -- AST parse + reachable-call traversal.
                            A 1000-line plugin takes well under 10ms;
                            200ms is generous headroom.
  Z3_TOTAL_TIMEOUT_MS    -- total budget for all Z3 solver calls across
                            the entire plugin. Kept separate from the
                            call-graph budget so one phase can't starve
                            the other.

Intra-module only: cross-module calls are treated as trusted since the
import gate already vets all imports. Module-level code (executed at
import time) is always included regardless of entry-point reachability.
"""
from __future__ import annotations

import ast
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path

from analysis.plugins.verifier.call_graph import reachable_calls
from analysis.plugins.verifier.sinks import (
    SinkViolation,
    check_config_set,
    check_static,
)


CALL_GRAPH_TIMEOUT_MS = 200
Z3_TOTAL_TIMEOUT_MS = 500


@dataclass
class VerificationError(Exception):
    """Raised when a plugin fails symbolic verification."""
    plugin_key: str
    violations: list[SinkViolation] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f'plugin {self.plugin_key!r} failed verification:']
        for v in self.violations:
            lines.append(
                f'  line {v.lineno}: [{v.kind}] {v.description}')
        return '\n'.join(lines)


class VerificationTimeout(VerificationError):
    """Raised when a verification phase exceeds its time budget."""

    def __str__(self) -> str:
        return (f'plugin {self.plugin_key!r} refused: '
                f'verification timed out (possible DoS or extreme complexity)')


def verify(file_path: Path, plugin_key: str) -> None:
    """Parse and verify a plugin source file. Raises VerificationError
    (or VerificationTimeout) if any sandbox invariant is violated or
    a phase exceeds its time budget. No-op when the file cannot be
    parsed (parse errors are caught by exec_module instead)."""
    try:
        source = file_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(file_path))
    except (OSError, SyntaxError):
        return

    calls = _call_graph_phase(tree, plugin_key)
    _z3_phase(calls, plugin_key)


def _call_graph_phase(tree: ast.Module,
                      plugin_key: str) -> list[ast.Call]:
    """Run the AST call graph traversal with a timeout. Returns the list
    of reachable call nodes on success."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(reachable_calls, tree)
        try:
            return future.result(timeout=CALL_GRAPH_TIMEOUT_MS / 1000.0)
        except concurrent.futures.TimeoutError:
            raise VerificationTimeout(plugin_key=plugin_key)


def _z3_phase(calls: list[ast.Call], plugin_key: str) -> None:
    """Run all static + Z3 sink checks within the Z3 time budget.
    The budget is shared across all call sites -- if the total time
    is exhausted mid-check the plugin is refused."""
    def _check_all():
        violations = []
        for call in calls:
            static = check_static(call)
            if static:
                violations.append(static)
                continue
            config = check_config_set(call, plugin_key)
            if config:
                violations.append(config)
        return violations

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_check_all)
        try:
            violations = future.result(timeout=Z3_TOTAL_TIMEOUT_MS / 1000.0)
        except concurrent.futures.TimeoutError:
            raise VerificationTimeout(plugin_key=plugin_key)

    if violations:
        raise VerificationError(plugin_key=plugin_key, violations=violations)
