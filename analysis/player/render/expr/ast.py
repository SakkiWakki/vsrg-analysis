"""AST for the NotITG chart-Lua subset - the parse-time port boundary.

A chart's per-frame `%function(self) ... end` body (and every other Lua
body we used to scrape with regexes) parses into these nodes. They carry
NO evaluation logic: a tree-walk backend (`eval_tree`) and a
compile-to-scheduler backend (`compile_sched`) both dispatch over them,
and a future rust evaluator mirrors the same node set. Keeping the nodes
inert is what makes them a stable cross-language contract.

Only the Lua subset real charts use is modeled: numeric/string/bool/nil
literals, identifiers, table index (`v[i]`) and field (`a.b`) access,
unary/binary operators, function calls (`perframe(a,b)`) and method calls
(`self:zoom(x)`), and the statement forms the Update bodies use (assign,
local, if/elif/else, numeric for, while, function def, return, bare call).
Anything outside the subset parses to `Unparsed(raw)` so a body still
yields a partial tree and a consumer falls back for that node alone -
never a hard failure (the "skip, do not guess" discipline the regexes had,
kept but now traceable via diagnostics).

`span` on every node is the (start, end) character offset in the source
body, so a diagnostic can point at the exact text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Span = tuple[int, int]

_NO_SPAN: Span = (-1, -1)


@dataclass(frozen=True)
class Node:
    """Base for every AST node. `span` locates the node in the source."""
    span: Span = field(default=_NO_SPAN, kw_only=True)


# -- expressions -------------------------------------------------------------

@dataclass(frozen=True)
class Num(Node):
    value: float


@dataclass(frozen=True)
class Str(Node):
    value: str


@dataclass(frozen=True)
class Bool(Node):
    value: bool


@dataclass(frozen=True)
class Nil(Node):
    pass


@dataclass(frozen=True)
class Sym(Node):
    """A bare identifier: `beat`, `mod_time`, `fgcurcommand`, a global."""
    name: str


@dataclass(frozen=True)
class Index(Node):
    """Table index `base[key]` (`v[3]`, `mods[i]`). `base` is a Node so
    `t[i][j]` nests; the surface resolves a constant table + int key."""
    base: Node
    key: Node


@dataclass(frozen=True)
class Field(Node):
    """Field access `base.name` (`math.pi`, `self.x`)."""
    base: Node
    name: str


@dataclass(frozen=True)
class Unary(Node):
    """`op` in {'-', 'not'}."""
    op: str
    operand: Node


@dataclass(frozen=True)
class Binary(Node):
    """`op` in arithmetic {'+','-','*','/','%','^'}, comparison
    {'>','>=','<','<=','==','~='}, logical {'and','or'}, or concat '..'."""
    op: str
    left: Node
    right: Node


@dataclass(frozen=True)
class Call(Node):
    """Free call `fn(args)` (`perframe(a,b)`). `fn` is a Node (usually a
    `Sym`, occasionally a `Field` like `math.sin`)."""
    fn: Node
    args: tuple[Node, ...]


@dataclass(frozen=True)
class Method(Node):
    """Method call `recv:name(args)` (`self:zoom(x)`); the NotITG verb
    surface resolves `name` against verb_surface's mechanism tables."""
    recv: Node
    name: str
    args: tuple[Node, ...]


@dataclass(frozen=True)
class Table(Node):
    """Table constructor `{a, b, k = v}`. `array` are positional entries;
    `fields` are (key, value) named entries."""
    array: tuple[Node, ...] = ()
    fields: tuple[tuple[str, Node], ...] = ()


# -- statements --------------------------------------------------------------

@dataclass(frozen=True)
class Assign(Node):
    """`targets = values` (global/field/index assignment). Parallel Lua
    assignment keeps both as tuples (`a, b = 1, 2`)."""
    targets: tuple[Node, ...]
    values: tuple[Node, ...]


@dataclass(frozen=True)
class Local(Node):
    """`local names = values` (a `local beat = GetSongBeat()`)."""
    names: tuple[str, ...]
    values: tuple[Node, ...]


@dataclass(frozen=True)
class If(Node):
    """`if cond then body [elseif...] [else...] end`. `elifs` is a tuple
    of (cond, body); `orelse` is the else body (empty if absent)."""
    cond: Node
    body: tuple[Node, ...]
    elifs: tuple[tuple[Node, tuple[Node, ...]], ...] = ()
    orelse: tuple[Node, ...] = ()


@dataclass(frozen=True)
class NumericFor(Node):
    """`for var = start, stop[, step] do body end`. `step` is None for the
    Lua default of 1 (a literal-0 step is a no-op loop the consumer drops,
    replacing _ZERO_STEP_FOR_RE)."""
    var: str
    start: Node
    stop: Node
    step: Node | None
    body: tuple[Node, ...]


@dataclass(frozen=True)
class GenericFor(Node):
    """`for names in exprs do body end` (`for i, v in ipairs(t) do`). The
    iteration is opaque to windowing, but its body is walked for guards."""
    names: tuple[str, ...]
    exprs: tuple[Node, ...]
    body: tuple[Node, ...]


@dataclass(frozen=True)
class While(Node):
    cond: Node
    body: tuple[Node, ...]


@dataclass(frozen=True)
class FuncDef(Node):
    """`function name(params) body end` (or `local function`)."""
    name: str
    params: tuple[str, ...]
    body: tuple[Node, ...]
    is_local: bool = False


@dataclass(frozen=True)
class Return(Node):
    values: tuple[Node, ...] = ()


@dataclass(frozen=True)
class ExprStmt(Node):
    """A bare expression used as a statement - almost always a call
    (`self:queuecommand('Update')`, `update_proxies()`)."""
    expr: Node


@dataclass(frozen=True)
class Unparsed(Node):
    """A source span outside the modeled subset. Carries the raw text so a
    consumer can fall back (or a diagnostic can quote it) without aborting
    the whole parse."""
    raw: str
