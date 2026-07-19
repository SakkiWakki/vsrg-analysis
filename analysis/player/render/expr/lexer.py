"""Tokenizer for the chart-Lua subset.

Turns a Lua body string into a flat `Token` list the parser consumes.
Comments are stripped first (line `--...` and block `--[[ ... ]]`) so a
commented-out driver never tokenizes. Each token carries its source span
so parser diagnostics point at exact text.

Only the subset real charts use is tokenized: number and string literals,
identifiers/keywords, the operators charts actually write, and the
delimiters `()[]{},;.:`. A character that starts no known token is emitted
as an `ERROR` token; the parser turns the surrounding region into an
`Unparsed` node rather than aborting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from analysis.player.render.expr.ast import Span


class Tok(Enum):
    NUMBER = 'number'
    STRING = 'string'
    NAME = 'name'
    KEYWORD = 'keyword'
    OP = 'op'
    ERROR = 'error'
    EOF = 'eof'


@dataclass(frozen=True)
class Token:
    kind: Tok
    text: str
    span: Span


# Lua keywords the grammar cares about; other reserved words still tokenize
# as KEYWORD so an identifier check never mistakes them for a Sym.
_KEYWORDS = frozenset({
    'and', 'or', 'not', 'nil', 'true', 'false',
    'if', 'then', 'elseif', 'else', 'end',
    'for', 'while', 'do', 'repeat', 'until',
    'function', 'return', 'local', 'in', 'break'})

# Multi-char operators first so `>=` beats `>`, `==` beats `=`, `..` beats
# `.`. Order matters: the scanner tries these in sequence.
_OPERATORS = ('...', '..', '==', '~=', '<=', '>=',
              '+', '-', '*', '/', '%', '^', '#',
              '<', '>', '=', '(', ')', '[', ']', '{', '}',
              ',', ';', ':', '.')

_NAME_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
# Lua numbers: decimal (incl. `.5`, `5.`, `1e3`, `0x1A`). No sign here -
# a leading `-` is the unary operator, parsed separately.
_NUMBER_RE = re.compile(
    r'0[xX][0-9A-Fa-f]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')


def strip_comments(source: str) -> str:
    """Blank out Lua comments IN PLACE (replaced by spaces of equal length)
    so character spans still line up with the original body. Block comments
    `--[[ ... ]]` first (they may span lines), then line comments `--...`."""
    def blank(match: re.Match) -> str:
        return re.sub(r'[^\n]', ' ', match.group(0))

    without_blocks = re.sub(r'--\[\[.*?\]\]', blank, source, flags=re.DOTALL)
    return re.sub(r'--[^\n]*', blank, without_blocks)


def tokenize(source: str) -> list[Token]:
    """Lex `source` (a Lua body, comments included) into tokens ending with
    an EOF token. Unknown characters become ERROR tokens."""
    text = strip_comments(source)
    tokens: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in ' \t\r\n':
            i += 1
            continue
        if c in '\'"':
            tok, i = _scan_string(text, i)
            tokens.append(tok)
            continue
        number = _NUMBER_RE.match(text, i)
        if number and (c.isdigit() or (c == '.' and i + 1 < n
                                       and text[i + 1].isdigit())):
            tokens.append(Token(Tok.NUMBER, number.group(0),
                                (i, number.end())))
            i = number.end()
            continue
        name = _NAME_RE.match(text, i)
        if name:
            word = name.group(0)
            kind = Tok.KEYWORD if word in _KEYWORDS else Tok.NAME
            tokens.append(Token(kind, word, (i, name.end())))
            i = name.end()
            continue
        op = _match_operator(text, i)
        if op is not None:
            tokens.append(Token(Tok.OP, op, (i, i + len(op))))
            i += len(op)
            continue
        tokens.append(Token(Tok.ERROR, c, (i, i + 1)))
        i += 1
    tokens.append(Token(Tok.EOF, '', (n, n)))
    return tokens


def _match_operator(text: str, i: int) -> str | None:
    for op in _OPERATORS:
        if text.startswith(op, i):
            return op
    return None


def _scan_string(text: str, i: int) -> tuple[Token, int]:
    """A single/double-quoted Lua string from `i`; unterminated strings
    become an ERROR token spanning to end of line."""
    quote = text[i]
    j = i + 1
    n = len(text)
    while j < n and text[j] != quote:
        if text[j] == '\\' and j + 1 < n:
            j += 2
            continue
        if text[j] == '\n':
            return Token(Tok.ERROR, text[i:j], (i, j)), j
        j += 1
    if j >= n:
        return Token(Tok.ERROR, text[i:j], (i, j)), j
    return Token(Tok.STRING, _decode_escapes(text[i + 1:j]), (i, j + 1)), j + 1


# Lua single-char string escapes -> their character. `\ddd` decimal escapes are
# handled separately (variable length); an unknown `\x` keeps `x` (Lua drops the
# backslash). The token value is the DECODED string, so a `'a\nb'` literal
# concatenates as a real newline (the lexer previously stored the raw two-char
# `\n`, which then round-tripped into recorded text as a literal backslash-n).
_STRING_ESCAPES = {
    'n': '\n', 't': '\t', 'r': '\r', 'a': '\a', 'b': '\b', 'f': '\f',
    'v': '\v', '\\': '\\', '"': '"', "'": "'", '\n': '\n',
}


def _decode_escapes(raw: str) -> str:
    if '\\' not in raw:
        return raw
    out = []
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if c != '\\' or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        nxt = raw[i + 1]
        if nxt.isdigit():
            j = i + 1
            while j < n and raw[j].isdigit() and j - (i + 1) < 3:
                j += 1
            out.append(chr(int(raw[i + 1:j]) & 0xFF))
            i = j
            continue
        out.append(_STRING_ESCAPES.get(nxt, nxt))
        i += 2
    return ''.join(out)
