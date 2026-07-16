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
_ATTR_RE = re.compile(r'(' + _NAME + r')\s*=\s*"', re.DOTALL)


def _find_attr_value_end(text: str, start: int) -> int:
    """Index of the closing `"` for an attribute value opened at
    `start`. NotITG never escapes `"` inside attribute values (Lua uses
    single quotes for its strings), so the next raw `"` closes it."""
    close = text.find('"', start)
    return close if close != -1 else len(text)


def _parse_tag_attrs(tag_body: str) -> dict:
    """Attributes from the inside of a start tag. Duplicate names keep
    the last value (engine field-overwrite order)."""
    attrs = {}
    pos = 0
    while True:
        m = _ATTR_RE.search(tag_body, pos)
        if not m:
            break
        value_start = m.end()
        value_end = _find_attr_value_end(tag_body, value_start)
        attrs[m.group(1)] = tag_body[value_start:value_end]
        pos = value_end + 1
    return attrs


def _scan_start_tag(text: str, lt: int):
    """Parse the start tag beginning at `<` (index `lt`). Returns
    (tag_name, attrs, self_closing, end_index) or None if `lt` does not
    begin a usable start/close tag."""
    name_match = re.match(_NAME, text[lt + 1:])
    if not name_match:
        return None
    tag_name = name_match.group(0)

    # Walk to the tag's `>`, skipping any `>` that sits inside a quoted
    # attribute value (Lua bodies contain them constantly).
    pos = lt + 1 + len(tag_name)
    while pos < len(text):
        ch = text[pos]
        match ch:
            case '"':
                pos = _find_attr_value_end(text, pos + 1) + 1
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


def _strip_lua_wrapper(value: str) -> str:
    """A `%`-prefixed attribute is a Lua body. `%function(self) BODY end`
    is unwrapped to BODY so it runs as a statement chunk; a bare
    `%expr` keeps its expression."""
    body = value[1:].strip()
    header = re.match(r'function\s*\(([^)]*)\)', body)
    if header and body.endswith('end'):
        return body[header.end():-len('end')]
    return body


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
