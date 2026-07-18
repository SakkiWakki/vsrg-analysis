"""Lenient parser for StepMania 3.95 / NotITG actor XML.

These files are XML-ISH, not XML: attribute values routinely hold
`%function(self) ... end` Lua bodies with unescaped `<`, `>`, `&`,
quotes-in-comments and raw newlines, so a real XML parser rejects them
(xml.etree raises ParseError on the first `<` inside a Lua chunk). We
scan the tag/attribute structure by hand and treat every attribute
value as an opaque string.

What the scanner recovers:

- the actor tree: nested elements delimited by `<Tag ...>` /
  `<children>` / `</...>` / self-closing `<Tag ... />`. Tag names in
  real charts are decorative (gat renames Layer to LAER/ZZLAER/...),
  so the actor CLASS comes from the `Type=` attribute, not the tag.
- per-actor attributes (last value wins on duplicates, matching the
  engine, which overwrites the actor field as it reads left to right).
- the Lua chunks: any attribute value starting with `%` is a Lua
  body (the leading `%` and, for `%function(self) ... end` wrappers,
  the wrapper, stripped to leave the runnable body).
- classic command strings: attribute values NOT starting with `%`
  whose name ends in `Command` (OnCommand/InitCommand/...), parsed
  into `(verb, args)` command lists.

The scanner never raises on malformed input: an unterminated tag or a
stray delimiter ends the current element and parsing continues, so a
community file with a typo still yields a partial tree.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_TWEEN_VERBS = {
    'linear': 0, 'accelerate': 3, 'decelerate': 4, 'smooth': 5,
    'sleep': None,
}
# Tween verbs we recognize as easings but do not have a curve for; they
# fall back to linear at compile time. Kept explicit so unknown verbs
# (which are property setters) are not mistaken for tweens.
_TWEEN_FALLBACK_VERBS = frozenset({
    'spring', 'bounce', 'bouncebegin', 'bounceend', 'bezier',
    'tween', 'ease',
})


@dataclass
class Actor:
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)

    @property
    def kind(self) -> str:
        """The actor class: the Type= attribute, else the tag name."""
        return str(self.attrs.get('Type', self.tag))

    def command_attrs(self) -> dict:
        return {name: value for name, value in self.attrs.items()
                if name.endswith('Command')}

    def message_commands(self) -> dict:
        """`{message_name: body}` for every `<Name>MessageCommand` attr.
        These run when `MESSAGEMAN:Broadcast('Name')` fires (on every
        actor that defines the command); the body is the raw attribute
        value (a `%`-Lua chunk or a classic command string)."""
        return {name[:-len('MessageCommand')]: value
                for name, value in self.attrs.items()
                if name.endswith('MessageCommand')
                and name != 'MessageCommand'}

    def named_commands(self) -> dict:
        """`{command_name: body}` for the plain `<Name>Command` attrs an
        actor exposes to `play/queuecommand('Name')` - every `*Command`
        that is neither the load-time Init/On nor a broadcast
        `*MessageCommand`. `Name` keeps its case (SM command names are
        case-sensitive: `Spawn`, `HideQ`, `Cast`)."""
        out = {}
        for name, value in self.attrs.items():
            if not name.endswith('Command') or name.endswith('MessageCommand'):
                continue
            if name in ('InitCommand', 'OnCommand'):
                continue
            out[name[:-len('Command')]] = value
        return out


@dataclass
class LuaChunk:
    actor: Actor
    attr: str          # which command attribute held the body
    body: str          # runnable Lua (leading % and wrapper stripped)


@dataclass
class ClassicCommand:
    actor: Actor
    attr: str
    commands: list     # list of (verb, [args]) tuples


@dataclass
class ParsedXml:
    root: Actor
    lua_chunks: list
    classic_commands: list


_NAME = r'[A-Za-z_][A-Za-z0-9_]*'
# SM's XmlFile accepts single- OR double-quoted attribute values;
# charts single-quote attrs whose Lua bodies use double-quoted strings
# (Government Knows' CatEvent rig).
_ATTR_RE = re.compile(r'(' + _NAME + r')\s*=\s*(["\'])', re.DOTALL)

# Tag names are looser than attribute names: SM's XmlFile accepts any
# run of name-ish characters, and charts use digit-leading tags
# (gat's `<0Layer>`). A rejected start tag would leave its `</0Layer>`
# close unmatched, popping the parse stack early and truncating the
# document.
_TAG_RE = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.:-]*')


def _find_attr_value_end(text: str, start: int, quote: str = '"') -> int:
    """Index of the closing quote for an attribute value opened at
    `start`. NotITG never escapes the delimiting quote inside attribute
    values (a body quoted one way uses the other quote for its Lua
    strings), so the next raw occurrence closes it."""
    close = text.find(quote, start)
    return close if close != -1 else len(text)


def _parse_tag_attrs(tag_body: str) -> dict:
    """Attributes from the inside of a start tag. Duplicate names keep
    the last value (engine field-overwrite order). Lua-bearing values
    (`%` bodies and Condition expressions) are rewritten to lex the
    same under LuaJIT as under NotITG's Lua 5.0."""
    attrs = {}
    pos = 0
    while True:
        m = _ATTR_RE.search(tag_body, pos)
        if not m:
            break
        value_start = m.end()
        value_end = _find_attr_value_end(tag_body, value_start, m.group(2))
        value = tag_body[value_start:value_end]
        if value.startswith('%') or m.group(1) == 'Condition':
            value = _lua50_compat(value)
        attrs[m.group(1)] = value
        pos = value_end + 1
    return attrs


# String escapes Lua 5.0 recognizes; for any OTHER `\c` its lexer
# passes `c` through, where 5.1/LuaJIT raise "invalid escape sequence"
# (real charts write `'\+$'` find-patterns).
_LUA50_ESCAPES = frozenset("abfnrtv\\\"'\n0123456789")

# A numeric `for` with a literal-zero step: Lua 5.0 RAISES "'for' step
# is zero" at loop entry, but LuaJIT (5.1) spins forever. A buggy chart
# (`for i = 596, 600, 0 do ... end`) relies on the engine aborting that
# chunk. Replace the literal step with a call that raises, so the body
# never runs and the chunk faults exactly as it does in-engine.
_ZERO_STEP_FOR_RE = re.compile(
    r'(\bfor\b[^\n;]*?=[^\n;]*?,[^\n;]*?,\s*)0(?:\.0*)?(\s+do\b)')


def _lua50_compat(source: str) -> str:
    """Rewrite Lua source so LuaJIT (5.1 lexer) reads it exactly as
    NotITG's Lua 5.0 lexer does. Three divergences real charts hit:

    - a number may run straight into a keyword (`beat < 485then`): 5.0
      ends the number token at the letter; insert the missing space.
    - unknown string escapes pass the char through in 5.0; drop the
      backslash.
    - `[[`/`]]` NEST inside long comments in 5.0, so `--[[ .. [[ .. ]]
      .. ]]` is one comment; blank the interior (newlines kept, so line
      numbers survive) up to the DEPTH-MATCHED close.
    """
    out = []
    # A char-for-char mask of `out` where every string/comment character
    # is blanked to a space, so the zero-step-for rewrite matches only
    # code and never a `for ... , 0 do` sitting inside a string literal.
    mask = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if source.startswith('--[[', i):
            piece = ['--[[']
            i += 4
            depth = 1
            while i < n and depth:
                if source.startswith('[[', i):
                    depth += 1
                    piece.append('  ')
                    i += 2
                elif source.startswith(']]', i):
                    depth -= 1
                    piece.append(']]' if depth == 0 else '  ')
                    i += 2
                else:
                    piece.append(source[i] if source[i] == '\n' else ' ')
                    i += 1
            text = ''.join(piece)
            out.append(text)
            mask.append(' ' * len(text))
        elif source.startswith('--', i):
            stop = source.find('\n', i)
            stop = n if stop == -1 else stop
            out.append(source[i:stop])
            mask.append(' ' * (stop - i))
            i = stop
        elif ch in '\'"':
            piece = [ch]
            i += 1
            while i < n:
                c = source[i]
                if c == '\\':
                    escaped = source[i + 1] if i + 1 < n else ''
                    piece.append(source[i:i + 2] if escaped in _LUA50_ESCAPES
                                 else escaped)
                    i += 2
                    continue
                piece.append(c)
                i += 1
                if c == ch:
                    break
            text = ''.join(piece)
            out.append(text)
            mask.append(' ' * len(text))
        elif source.startswith('[[', i):
            stop = source.find(']]', i + 2)
            stop = n - 2 if stop == -1 else stop
            text = source[i:stop + 2]
            out.append(text)
            mask.append(' ' * len(text))
            i = stop + 2
        elif ch.isdigit():
            start = i
            while i < n and (source[i].isdigit() or source[i] == '.'):
                i += 1
            number = source[start:i]
            begins_token = start == 0 or not (
                source[start - 1].isalnum() or source[start - 1] in '_.')
            if (begins_token and i < n
                    and (source[i].isalpha() or source[i] == '_')
                    and source[i] not in 'eExX'):
                number += ' '
            out.append(number)
            mask.append(number)
        else:
            out.append(ch)
            mask.append(ch)
            i += 1
    rewritten = ''.join(out)
    masked = ''.join(mask)
    for match in reversed(list(_ZERO_STEP_FOR_RE.finditer(masked))):
        s, e = match.span()
        rewritten = (rewritten[:s] + match.group(1)
                     + '(error("\'for\' step is zero"))'
                     + match.group(2) + rewritten[e:])
    return rewritten


def _scan_start_tag(text: str, lt: int):
    """Parse the start tag beginning at `<` (index `lt`). Returns
    (tag_name, attrs, self_closing, end_index) or None if `lt` does not
    begin a usable start/close tag."""
    name_match = _TAG_RE.match(text, lt + 1)
    if not name_match:
        return None
    tag_name = name_match.group(0)

    # Walk to the tag's `>`, skipping any `>` that sits inside a quoted
    # attribute value (Lua bodies contain them constantly).
    pos = lt + 1 + len(tag_name)
    while pos < len(text):
        ch = text[pos]
        match ch:
            case '"' | "'":
                pos = _find_attr_value_end(text, pos + 1, ch) + 1
            case '>':
                inner = text[lt + 1 + len(tag_name):pos]
                self_closing = inner.rstrip().endswith('/')
                if self_closing:
                    inner = inner.rstrip()[:-1]
                attrs = _parse_tag_attrs(inner)
                return tag_name, attrs, self_closing, pos + 1
            case _:
                pos += 1
    return tag_name, _parse_tag_attrs(text[lt + 1 + len(tag_name):]), True, \
        len(text)


def is_lua_function_literal(value: str) -> bool:
    """True for a `%function(...) ... end` command body - the common
    literal shape, as opposed to a `%expr` expression command."""
    body = value[1:].strip()
    return bool(re.match(r'function\s*\(', body)) and body.endswith('end')


def _strip_lua_wrapper(value: str) -> str:
    """A `%`-prefixed attribute is a Lua EXPRESSION whose value is the
    command: the engine compiles `return <expr>` and calls the result
    with the actor. `%function(self) BODY end` unwraps to BODY so it
    runs as a statement chunk (`self` resolves through the sandbox env);
    any other expression becomes an equivalent evaluate-and-call
    statement - it may resolve to a function only at fire time
    (the XGML template's `%prefix.update` reads a global its
    InitCommand binds)."""
    body = value[1:].strip()
    header = re.match(r'function\s*\(([^)]*)\)', body)
    if header and body.endswith('end'):
        return body[header.end():-len('end')]
    return (f'local __cmd = ({body})\n'
            "if type(__cmd) == 'function' then __cmd(self) end")


def _split_top_level(text: str, sep: str) -> list:
    """Split on `sep`, ignoring separators inside (){}[] or quotes.
    Classic command strings never nest, but embedded Lua-ish math like
    `y,-22+(18*0)` puts commas inside parens that must not split."""
    parts = []
    depth = 0
    quote = ''
    start = 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ''
            continue
        match ch:
            case "'" | '"':
                quote = ch
            case '(' | '[' | '{':
                depth += 1
            case ')' | ']' | '}':
                depth = max(0, depth - 1)
            case _ if ch == sep and depth == 0:
                parts.append(text[start:i])
                start = i + 1
    parts.append(text[start:])
    return parts


def _parse_classic_command(value: str) -> list:
    """`x,100;zoom,2;linear,1;y,300` -> [(verb, [args]), ...]. Args stay
    as raw strings; the compiler resolves numeric vs symbolic ones."""
    commands = []
    for segment in _split_top_level(value, ';'):
        segment = segment.strip()
        if not segment:
            continue
        fields = [f.strip() for f in _split_top_level(segment, ',')]
        verb = fields[0]
        if verb:
            commands.append((verb, fields[1:]))
    return commands


def parse_command_string(value: str) -> list:
    """`x,100;zoom,2;linear,1;y,300` -> [(verb, [args]), ...]. Public
    entry for compilers that hold an actor's raw command attribute."""
    return _parse_classic_command(value)


def is_tween_verb(verb: str) -> bool:
    return verb in _TWEEN_VERBS or verb in _TWEEN_FALLBACK_VERBS


def tween_easing(verb: str) -> int | None:
    """Easing id for a tween verb; None for `sleep` (pure time gap).
    Unmodeled easings (spring/bounce/bezier) fall back to linear."""
    if verb in _TWEEN_VERBS:
        return _TWEEN_VERBS[verb]
    return 0


def parse_actor_xml(text: str) -> ParsedXml:
    """Parse one actor XML document into a tree plus flat harvests of
    its Lua chunks and classic command strings."""
    root = Actor(tag='<root>')
    lua_chunks: list = []
    classic_commands: list = []
    _parse_children(text, 0, root, lua_chunks, classic_commands)
    single = root.children[0] if len(root.children) == 1 else root
    return ParsedXml(single, lua_chunks, classic_commands)


def _harvest_attrs(actor: Actor, lua_chunks: list,
                   classic_commands: list) -> None:
    for name, value in actor.attrs.items():
        if value.startswith('%'):
            lua_chunks.append(LuaChunk(actor, name, _strip_lua_wrapper(value)))
        elif name.endswith('Command'):
            commands = _parse_classic_command(value)
            if commands:
                classic_commands.append(
                    ClassicCommand(actor, name, commands))


def _parse_children(text: str, pos: int, parent: Actor,
                    lua_chunks: list, classic_commands: list) -> int:
    """Fill `parent.children` from the elements starting at `pos`.
    Returns the index just past the parent's closing tag."""
    while pos < len(text):
        lt = text.find('<', pos)
        if lt == -1:
            return len(text)

        if text.startswith('</', lt):
            gt = text.find('>', lt)
            return len(text) if gt == -1 else gt + 1

        scanned = _scan_start_tag(text, lt)
        if scanned is None:
            pos = lt + 1
            continue

        tag_name, attrs, self_closing, after = scanned
        if tag_name == 'children':
            # A grouping wrapper: its own children belong to `parent`.
            pos = _parse_children(text, after, parent, lua_chunks,
                                  classic_commands)
            continue

        actor = Actor(tag=tag_name, attrs=attrs)
        _harvest_attrs(actor, lua_chunks, classic_commands)
        parent.children.append(actor)
        if self_closing:
            pos = after
        else:
            pos = _parse_children(text, after, actor, lua_chunks,
                                  classic_commands)
    return len(text)
