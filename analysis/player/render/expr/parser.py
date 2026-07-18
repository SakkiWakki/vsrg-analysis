"""Parser for the chart-Lua subset: tokens -> AST.

Two layers over the token stream:

- `parse_expr` is a precedence-climbing (Pratt) expression parser. The
  binding-power ladder (low -> high) is: `or` < `and` < comparison
  < `..` < `+ -` < `* / %` < unary `- not #` < `^` < postfix
  (index `[]`, field `.`, call `()`, method `:`) < atom/paren. So
  `beat > A and beat < B` groups as `and(>(...), <(...))` and
  `e[1] + e[2]` groups `+` under a surrounding comparison - exactly the
  shapes the window extractor needs.

- `parse_body` parses a statement sequence (an Update body): assignments,
  `local`, `if/elseif/else`, numeric `for`, `while`, `function`, `return`,
  and bare-call statements. A construct outside the subset (or a token the
  lexer flagged ERROR) is captured as an `Unparsed` node spanning to a safe
  recovery point and recorded as a diagnostic, so the rest of the body
  still parses.

`parse_guard` is the narrow entry the window extractor uses: parse a single
expression (the text between `if` and `then`, or a `perframe(...)` call) and
return the AST or None when it is unparseable.
"""
from __future__ import annotations

from analysis.player.render.expr import ast
from analysis.player.render.expr.diagnostics import DiagnosticSink, Severity
from analysis.player.render.expr.lexer import Tok, Token, tokenize

# Binary operator -> (left binding power, right binding power). Left < right
# gives left-associativity; `^` and `..` are right-associative (right < left).
_BINARY_BP = {
    'or': (1, 2), 'and': (3, 4),
    '<': (5, 6), '>': (5, 6), '<=': (5, 6), '>=': (5, 6),
    '==': (5, 6), '~=': (5, 6),
    '..': (8, 7),
    '+': (9, 10), '-': (9, 10),
    '*': (11, 12), '/': (11, 12), '%': (11, 12),
    '^': (16, 15),
}
_UNARY_BP = 13          # unary `- not #` bind tighter than */ , looser than ^
_POSTFIX_BP = 17        # index / field / call / method - tightest


class _Parser:
    def __init__(self, tokens: list[Token], source: str,
                 sink: DiagnosticSink):
        self._toks = tokens
        self._src = source
        self._sink = sink
        self._i = 0

    # -- token cursor --------------------------------------------------------

    def _peek(self) -> Token:
        return self._toks[self._i]

    def _next(self) -> Token:
        tok = self._toks[self._i]
        if tok.kind is not Tok.EOF:
            self._i += 1
        return tok

    def _at_op(self, text: str) -> bool:
        tok = self._peek()
        return tok.kind is Tok.OP and tok.text == text

    def _at_kw(self, *words: str) -> bool:
        tok = self._peek()
        return tok.kind is Tok.KEYWORD and tok.text in words

    def _eat_op(self, text: str) -> bool:
        if self._at_op(text):
            self._next()
            return True
        return False

    def _eat_kw(self, word: str) -> bool:
        if self._at_kw(word):
            self._next()
            return True
        return False

    # -- expressions ---------------------------------------------------------

    def parse_expr(self, min_bp: int = 0) -> ast.Node | None:
        left = self._parse_prefix()
        if left is None:
            return None
        while True:
            left = self._parse_postfix(left)
            tok = self._peek()
            if tok.kind not in (Tok.OP, Tok.KEYWORD):
                break
            bp = _BINARY_BP.get(tok.text)
            if bp is None or bp[0] < min_bp:
                break
            op = self._next().text
            right = self.parse_expr(bp[1])
            if right is None:
                return None
            left = ast.Binary(op, left, right,
                              span=(left.span[0], right.span[1]))
        return left

    def _parse_prefix(self) -> ast.Node | None:
        tok = self._peek()
        if tok.kind is Tok.OP and tok.text in ('-', '#'):
            self._next()
            operand = self.parse_expr(_UNARY_BP)
            if operand is None:
                return None
            return ast.Unary(tok.text, operand,
                             span=(tok.span[0], operand.span[1]))
        if self._at_kw('not'):
            self._next()
            operand = self.parse_expr(_UNARY_BP)
            if operand is None:
                return None
            return ast.Unary('not', operand,
                             span=(tok.span[0], operand.span[1]))
        return self._parse_atom()

    def _parse_atom(self) -> ast.Node | None:
        tok = self._peek()
        match tok.kind:
            case Tok.NUMBER:
                self._next()
                return ast.Num(_to_number(tok.text), span=tok.span)
            case Tok.STRING:
                self._next()
                return ast.Str(tok.text, span=tok.span)
            case Tok.NAME:
                self._next()
                return ast.Sym(tok.text, span=tok.span)
            case Tok.KEYWORD if tok.text in ('true', 'false'):
                self._next()
                return ast.Bool(tok.text == 'true', span=tok.span)
            case Tok.KEYWORD if tok.text == 'nil':
                self._next()
                return ast.Nil(span=tok.span)
        if self._at_op('('):
            self._next()
            inner = self.parse_expr(0)
            if inner is None or not self._eat_op(')'):
                return None
            return inner
        if self._at_op('{'):
            return self._parse_table()
        self._sink.warn('syntax', f'unexpected {tok.text!r}', tok.span)
        return None

    def _parse_postfix(self, node: ast.Node) -> ast.Node:
        while True:
            if self._at_op('['):
                self._next()
                key = self.parse_expr(0)
                if key is None or not self._eat_op(']'):
                    return node
                node = ast.Index(node, key, span=(node.span[0], key.span[1]))
            elif self._at_op('.'):
                self._next()
                name = self._next()
                if name.kind is not Tok.NAME:
                    return node
                node = ast.Field(node, name.text,
                                 span=(node.span[0], name.span[1]))
            elif self._at_op('('):
                args, end = self._parse_args()
                node = ast.Call(node, args, span=(node.span[0], end))
            elif self._at_op(':'):
                self._next()
                name = self._next()
                if name.kind is not Tok.NAME or not self._at_op('('):
                    return node
                args, end = self._parse_args()
                node = ast.Method(node, name.text, args,
                                  span=(node.span[0], end))
            else:
                return node

    def _parse_args(self) -> tuple[tuple[ast.Node, ...], int]:
        self._eat_op('(')
        args: list[ast.Node] = []
        if not self._at_op(')'):
            while True:
                arg = self.parse_expr(0)
                if arg is None:
                    break
                args.append(arg)
                if not self._eat_op(','):
                    break
        end = self._peek().span[1]
        self._eat_op(')')
        return tuple(args), end

    def _parse_table(self) -> ast.Node | None:
        start = self._next().span[0]        # consume '{'
        array: list[ast.Node] = []
        fields: list[tuple[str, ast.Node]] = []
        while not self._at_op('}') and self._peek().kind is not Tok.EOF:
            if (self._peek().kind is Tok.NAME
                    and self._toks[self._i + 1].text == '='):
                key = self._next().text
                self._next()                # '='
                val = self.parse_expr(0)
                if val is None:
                    return None
                fields.append((key, val))
            else:
                val = self.parse_expr(0)
                if val is None:
                    return None
                array.append(val)
            if not (self._eat_op(',') or self._eat_op(';')):
                break
        end = self._peek().span[1]
        self._eat_op('}')
        return ast.Table(tuple(array), tuple(fields), span=(start, end))

    # -- statements ----------------------------------------------------------

    def parse_body(self, stop: tuple[str, ...] = ()) -> tuple[ast.Node, ...]:
        stmts: list[ast.Node] = []
        while True:
            tok = self._peek()
            if tok.kind is Tok.EOF or self._at_kw(*stop):
                break
            before = self._i
            stmt = self._parse_stmt()
            if stmt is not None:
                stmts.append(stmt)
            if self._i == before:
                stmts.append(self._recover(stop))
        return tuple(stmts)

    def _parse_stmt(self) -> ast.Node | None:
        self._eat_op(';')
        if self._at_kw('local'):
            return self._parse_local()
        if self._at_kw('if'):
            return self._parse_if()
        if self._at_kw('for'):
            return self._parse_for()
        if self._at_kw('while'):
            return self._parse_while()
        if self._at_kw('function'):
            return self._parse_funcdef(is_local=False)
        if self._at_kw('return'):
            return self._parse_return()
        if self._at_kw('do'):
            self._next()
            body = self.parse_body(('end',))
            self._eat_kw('end')
            return ast.If(ast.Bool(True), body)   # a bare `do` block
        return self._parse_expr_or_assign()

    def _parse_expr_or_assign(self) -> ast.Node | None:
        start = self._peek().span[0]
        first = self.parse_expr(0)
        if first is None:
            return None
        targets = [first]
        while self._eat_op(','):
            nxt = self.parse_expr(0)
            if nxt is None:
                break
            targets.append(nxt)
        if self._eat_op('='):
            values = self._parse_expr_list()
            end = values[-1].span[1] if values else self._peek().span[1]
            return ast.Assign(tuple(targets), tuple(values),
                              span=(start, end))
        return ast.ExprStmt(first, span=first.span)

    def _parse_expr_list(self) -> list[ast.Node]:
        values: list[ast.Node] = []
        while True:
            val = self.parse_expr(0)
            if val is None:
                break
            values.append(val)
            if not self._eat_op(','):
                break
        return values

    def _parse_local(self) -> ast.Node | None:
        start = self._next().span[0]        # 'local'
        if self._at_kw('function'):
            return self._parse_funcdef(is_local=True, start=start)
        names: list[str] = []
        while self._peek().kind is Tok.NAME:
            names.append(self._next().text)
            if not self._eat_op(','):
                break
        values: list[ast.Node] = []
        if self._eat_op('='):
            values = self._parse_expr_list()
        end = values[-1].span[1] if values else self._peek().span[1]
        return ast.Local(tuple(names), tuple(values), span=(start, end))

    def _parse_if(self) -> ast.Node | None:
        start = self._next().span[0]        # 'if'
        cond = self.parse_expr(0)
        if cond is None or not self._eat_kw('then'):
            return self._recover(('end',), start)
        body = self.parse_body(('elseif', 'else', 'end'))
        elifs: list[tuple[ast.Node, tuple[ast.Node, ...]]] = []
        while self._at_kw('elseif'):
            self._next()
            ec = self.parse_expr(0)
            if ec is None or not self._eat_kw('then'):
                break
            eb = self.parse_body(('elseif', 'else', 'end'))
            elifs.append((ec, eb))
        orelse: tuple[ast.Node, ...] = ()
        if self._eat_kw('else'):
            orelse = self.parse_body(('end',))
        end = self._peek().span[1]
        self._eat_kw('end')
        return ast.If(cond, body, tuple(elifs), orelse, span=(start, end))

    def _parse_for(self) -> ast.Node | None:
        start = self._next().span[0]        # 'for'
        if self._peek().kind is not Tok.NAME:
            return self._recover(('end',), start)
        var = self._next().text
        # Only numeric `for v = a, b[, c]` is modeled; generic-for
        # (`for k in pairs`) falls back to Unparsed.
        if not self._eat_op('='):
            return self._recover(('end',), start)
        bounds = self._parse_expr_list()
        if not self._eat_kw('do') or len(bounds) < 2:
            return self._recover(('end',), start)
        body = self.parse_body(('end',))
        end = self._peek().span[1]
        self._eat_kw('end')
        step = bounds[2] if len(bounds) > 2 else None
        return ast.NumericFor(var, bounds[0], bounds[1], step, body,
                              span=(start, end))

    def _parse_while(self) -> ast.Node | None:
        start = self._next().span[0]        # 'while'
        cond = self.parse_expr(0)
        if cond is None or not self._eat_kw('do'):
            return self._recover(('end',), start)
        body = self.parse_body(('end',))
        end = self._peek().span[1]
        self._eat_kw('end')
        return ast.While(cond, body, span=(start, end))

    def _parse_funcdef(self, is_local: bool,
                       start: int | None = None) -> ast.Node | None:
        kw = self._next()                   # 'function'
        start = kw.span[0] if start is None else start
        name_parts: list[str] = []
        while self._peek().kind is Tok.NAME:
            name_parts.append(self._next().text)
            if not (self._eat_op('.') or self._eat_op(':')):
                break
        name = '.'.join(name_parts)
        params: list[str] = []
        if self._eat_op('('):
            while self._peek().kind is Tok.NAME:
                params.append(self._next().text)
                if not self._eat_op(','):
                    break
            self._eat_op('...')
            self._eat_op(')')
        body = self.parse_body(('end',))
        end = self._peek().span[1]
        self._eat_kw('end')
        return ast.FuncDef(name, tuple(params), body, is_local,
                           span=(start, end))

    def _parse_return(self) -> ast.Node | None:
        start = self._next().span[0]        # 'return'
        values: list[ast.Node] = []
        if not (self._at_kw('end', 'else', 'elseif')
                or self._peek().kind is Tok.EOF):
            values = self._parse_expr_list()
        end = values[-1].span[1] if values else start
        return ast.Return(tuple(values), span=(start, end))

    # -- error recovery ------------------------------------------------------

    def _recover(self, stop: tuple[str, ...], start: int | None = None):
        """Skip to the next statement boundary (a stop keyword, `;`, or a
        newline-ish gap) and capture the skipped text as `Unparsed`, so a
        construct outside the subset never aborts the whole body."""
        begin = self._peek().span[0] if start is None else start
        while True:
            tok = self._peek()
            if tok.kind is Tok.EOF or self._at_kw(*stop):
                break
            if tok.kind is Tok.OP and tok.text == ';':
                self._next()
                break
            if self._at_kw('end'):
                self._next()
                break
            self._next()
        end = self._peek().span[0]
        raw = self._src[begin:end]
        self._sink.warn('unparsed', f'unmodeled Lua: {raw.strip()[:60]!r}',
                        (begin, end))
        return ast.Unparsed(raw, span=(begin, end))


def _to_number(text: str) -> float:
    if text[:2] in ('0x', '0X'):
        return float(int(text, 16))
    return float(text)


def parse_body(source: str,
               sink: DiagnosticSink | None = None) -> tuple[tuple[ast.Node, ...],
                                                            DiagnosticSink]:
    """Parse a full Lua body into a statement tuple. Returns (stmts, sink);
    the sink holds any warnings/errors collected."""
    sink = sink or DiagnosticSink()
    parser = _Parser(tokenize(source), source, sink)
    return parser.parse_body(), sink


def parse_guard(source: str,
                sink: DiagnosticSink | None = None) -> ast.Node | None:
    """Parse a single guard EXPRESSION (the text between `if` and `then`, or
    a `perframe(...)` call). Returns the AST, or None when unparseable."""
    sink = sink or DiagnosticSink()
    parser = _Parser(tokenize(source), source, sink)
    node = parser.parse_expr(0)
    if node is None:
        sink.warn('syntax', f'unparseable guard: {source.strip()[:60]!r}',
                  (0, len(source)))
    return node
