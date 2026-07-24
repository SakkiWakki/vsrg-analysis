"""Engine-surface coverage: diff OUR verb classification against the
NotITG binary's own Lua bindings.

The decompile's demangled symbol list (refs/notitg/decompile/
notitg_demangled.txt) carries every `Luna<Class>::method` the FORK
registers - the authoritative verb surface, fork additions included
(openitg-src is only the readable cross-reference; NotITG adds beyond
it). This tool groups those bindings by class and reports which methods
our sim has never classified: not a routed setter, not IGNORED, not
DEFERRED. Unclassified verbs are exactly the ones that would today fall
through to the dropped-verb reporter the first time a chart pokes them -
this surfaces them BEFORE a chart does.

Runtime coverage (what a specific chart exercises) is separate and
always-on: the sweep's compile-done report prints every DEFERRED/
DROPPED verb the chart poked plus structural gaps (see
producers._print_unimplemented).

Usage:
    python -m analysis.games.notitg.coverage [<demangled.txt>]
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

_DEFAULT_SYMBOLS = Path(__file__).resolve().parents[3] / (
    'refs/notitg/decompile/notitg_demangled.txt')

_LUNA_RE = re.compile(r'\bLuna\w*<class (\w+)[^>]*>::(\w+)\b')

# Binding-machinery methods, not chart verbs.
_INFRA = frozenset({
    'Register', 'equal', 'push', 'check', 'create', 'CreateFromStack',
    'PushSelf', 'GetLuaObjectType', 'CreateMethodsVector', 'thunk',
    'tostring_T', '__call', '__len'})

# The actor-family classes whose verbs SimActor is expected to cover.
# Screen/manager singletons (SCREENMAN, GAMESTATE...) route through the
# env's host tables, not the actor poke surface - listed separately.
_ACTOR_CLASSES = frozenset({
    'Actor', 'ActorFrame', 'Sprite', 'BitmapText', 'Model', 'Quad',
    'ActorScroller', 'ActorFrameTexture', 'ActorProxy', 'Polygon',
    'RageTexture', 'RageShaderProgram'})


def engine_surface(symbols_path=None) -> dict:
    """{class name: set of bound method names} from the fork binary's
    demangled Luna symbols."""
    path = Path(symbols_path) if symbols_path else _DEFAULT_SYMBOLS
    surface: dict = defaultdict(set)
    for line in path.read_text(encoding='utf-8',
                               errors='replace').splitlines():
        m = _LUNA_RE.search(line)
        if m and m.group(2) not in _INFRA:
            surface[m.group(1)].add(m.group(2))
    return dict(surface)


def our_surface() -> set:
    """Every verb name the sim classifies somewhere. Drift-proof by
    construction: collects the string keys/members of EVERY module-level
    dict/set/frozenset in the verb-surface modules (routing tables,
    IGNORED, DEFERRED, easings, effect kinds - a new table is picked up
    automatically), plus every dispatch label in the actor/env sources
    (`case 'verb'`, `'a' | 'b'`, `verb == 'x'`, `verb in (...)`)."""
    from analysis.games.notitg import lua_api
    from analysis.games.notitg.sim import actor as sim_actor
    from analysis.games.notitg.sim import env as sim_env
    from analysis.games.notitg.sim import verb_surface

    known: set = set()
    for module in (verb_surface, lua_api, sim_actor):
        for name, value in vars(module).items():
            if name.startswith('__'):
                continue
            if isinstance(value, (dict, set, frozenset)):
                known.update(k for k in value if isinstance(k, str))
    for module in (sim_actor, sim_env, lua_api):
        src = Path(module.__file__).read_text(encoding='utf-8')
        for pattern in (r"case '([A-Za-z_]\w*)'",
                        r"'([A-Za-z_]\w*)'\s*\|",
                        r"\|\s*'([A-Za-z_]\w*)'",
                        r"verb == '([A-Za-z_]\w*)'",
                        r"name == '([A-Za-z_]\w*)'"):
            known.update(m.group(1) for m in re.finditer(pattern, src))
        for m in re.finditer(r"(?:verb|name) in \(([^)]*)\)", src):
            known.update(re.findall(r"'([A-Za-z_]\w*)'", m.group(1)))
    return known


def report(symbols_path=None) -> list:
    """Unclassified actor-family engine verbs, printed and returned."""
    engine = engine_surface(symbols_path)
    ours = {name.casefold() for name in our_surface()}
    lines = []
    for cls in sorted(_ACTOR_CLASSES & set(engine)):
        missing = sorted(name for name in engine[cls]
                         if name.casefold() not in ours)
        if missing:
            lines.append(f'{cls}: {len(missing)} unclassified: '
                         + ', '.join(missing))
    other = sorted(set(engine) - _ACTOR_CLASSES)
    lines.append(f'(non-actor classes not audited here: {len(other)}: '
                 + ', '.join(other[:12]) + ' ...)')
    for line in lines:
        print(line)
    return lines


if __name__ == '__main__':
    report(sys.argv[1] if len(sys.argv) > 1 else None)
