"""Diagnostics for the Lua front-end: errors and warnings, not silent skips.

The regexes this replaces failed silently - `_beat_arg` returning None just
dropped a window with no trace. The parser/compiler instead COLLECTS a
`Diagnostic` for anything unimplemented or unresolvable, so a chart that hits
an unmodeled construct still compiles the rest and the skipped part is
reportable (span + kind), not swallowed.

The exception hierarchy is a base `ExprError` with syntax/name/eval
subclasses. Exceptions are RAISED only where recovery is impossible; the
normal path records a `Diagnostic` into a `DiagnosticSink` and continues
(the parser leaves an `Unparsed` node behind).

Severity split:
- INFO   - an operand is off the value surface (a nil `v[]` in a guard):
           expected, the window is skipped, no problem.
- WARNING - an unimplemented method/verb (a verb_surface DEFERRED name, or a
           name absent from every mechanism table): the poke is dropped.
- ERROR  - malformed Lua the lexer/parser cannot tokenize/parse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from analysis.player.render.expr.ast import Span


class ExprError(Exception):
    """Base for Lua front-end errors. Never raised directly; a subclass is.
    Most problems are COLLECTED as `Diagnostic`s instead of raised - these
    are for the unrecoverable cases (or a caller that opts into strictness)."""


class ExprSyntaxError(ExprError):
    """A malformed token or expression the lexer/parser cannot form a node
    from."""


class ExprNameError(ExprError):
    """A name (operand, method, verb) not on the value surface / not in the
    verb registry."""


class ExprEvalError(ExprError):
    """An evaluation-time fault: wrong arity, a type mismatch."""


class Severity(Enum):
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'


@dataclass(frozen=True)
class Diagnostic:
    """One collected problem. `kind` is a short slug ('unresolved-operand',
    'unimplemented-verb', 'syntax'); `span` locates it in the source body."""
    severity: Severity
    kind: str
    message: str
    span: Span


@dataclass
class DiagnosticSink:
    """Collects diagnostics during a parse/compile pass instead of raising,
    so one bad construct never aborts the whole body. Callers read `items`
    (or the severity filters) afterward and fold them into the compile
    `warnings` list / `fault_messages`."""
    items: list[Diagnostic] = field(default_factory=list)

    def add(self, severity: Severity, kind: str, message: str,
            span: Span) -> None:
        self.items.append(Diagnostic(severity, kind, message, span))

    def info(self, kind: str, message: str, span: Span) -> None:
        self.add(Severity.INFO, kind, message, span)

    def warn(self, kind: str, message: str, span: Span) -> None:
        self.add(Severity.WARNING, kind, message, span)

    def error(self, kind: str, message: str, span: Span) -> None:
        self.add(Severity.ERROR, kind, message, span)

    def messages(self, min_severity: Severity = Severity.WARNING) -> list[str]:
        """Human-readable lines at or above `min_severity`, for the compile
        warnings list. INFO notices stay out by default (expected skips)."""
        order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2}
        floor = order[min_severity]
        return [f'{d.severity.value}: {d.kind}: {d.message}'
                for d in self.items if order[d.severity] >= floor]

    @property
    def has_errors(self) -> bool:
        return any(d.severity is Severity.ERROR for d in self.items)
