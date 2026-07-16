"""Translate a NotITG chart fragment shader into the library's
fullscreen uniform contract so it can run as one post-process pass.

Chart frags are authored against NotITG's actor shader environment,
not our contract. That environment supplies (via a paired `.vert`,
see the corpus `nop.vert`):

    varying vec2 textureCoord;   raw atlas UV, 0..1 over the texture
    varying vec2 imageCoord;     UV over the actual image region
                                 = textureCoord * textureSize / imageSize
    varying vec4 color;          per-vertex colour (gl_Color), white here
    varying vec3 position;       object-space vertex position
    varying vec3 normal;         transformed normal
    uniform sampler2D sampler0;  the actor's texture
    uniform vec2 textureSize;    physical texture size (power-of-two atlas)
    uniform vec2 imageSize;      the used sub-region size
    uniform vec2 resolution;     screen size in pixels
    uniform float time;          seconds
    uniform float beat;          song beat
    plus chart-specific scalar/vector uniforms poked from Lua.

For a fullscreen pass our capture FBO IS the image and fills the
texture, so textureSize == imageSize == resolution and the atlas
distinction collapses: imageCoord == textureCoord == the fullscreen UV,
and the corpus `img2tex(v) = v / textureSize * imageSize` helper is the
identity. We therefore feed all three the same fullscreen UV derived
from gl_FragCoord / u_resolution.

The translation is source-to-source and version-agnostic: the result
carries our `#version 150` header (so gl_pipeline._adapt_dialect can
still swap in the ES dialect), declares our contract uniforms, and
shims the legacy GLSL the chart body uses (texture2D, gl_FragColor,
varyings). Chart scalar/vector uniforms are re-declared as real
uniforms (default 0) so the custom-uniform path in gl_pipeline can
drive them by name from compiled channels -- their Lua `:uniform1f`
pokes. `beat` maps onto u_strength.z so a stack event can still animate
a beat-reactive frag with no bridge.

Examples:
    the Mod Rush `fisheye.frag` (single `amount` uniform, imageCoord,
    img2tex, resolution, color) and Tetaes `crt.frag` both translate to
    a runnable contract pass with no hand editing.
"""
from __future__ import annotations

import re

CONTRACT_HEADER = '#version 150'

_VARYING_UVS = ('textureCoord', 'imageCoord')
_VARYING_COLOUR = 'color'
_VARYING_VEC3 = ('position', 'normal')

# Contract names we synthesise; a chart re-declaring one must not emit a
# duplicate uniform in the translated source.
_RESERVED = frozenset({
    'textureSize', 'imageSize', 'resolution', 'time', 'beat',
    'u_tex', 'u_resolution', 'u_time', 'u_strength'})

_VERSION_RE = re.compile(r'^[ \t]*#version\b.*$', re.MULTILINE)
_VARYING_RE = re.compile(
    r'^[ \t]*varying[ \t]+(?P<type>vec[234]|float)[ \t]+'
    r'(?P<name>[A-Za-z_]\w*)[ \t]*;[ \t]*$',
    re.MULTILINE)
_SAMPLER0_RE = re.compile(r'\buniform[ \t]+sampler2D[ \t]+sampler0[ \t]*;')
_UNIFORM_DECL_RE = re.compile(
    r'^[ \t]*uniform[ \t]+(?P<type>float|int|bool|vec[234])[ \t]+'
    r'(?P<name>[A-Za-z_]\w*)[ \t]*(?:=[ \t]*[^;]+)?;[ \t]*$',
    re.MULTILINE)
_MAIN_RE = re.compile(r'\bvoid\s+main\s*\(\s*(?:void)?\s*\)\s*\{')

# A file-scope declaration with an initializer: `float x = <expr>;`. Only
# matched at brace-depth 0 (see _hoist_nonconst_globals) and only hoisted
# when the initializer is non-constant.
_GLOBAL_INIT_RE = re.compile(
    r'(?P<type>float|vec[234]|int|bool)[ \t]+(?P<name>[A-Za-z_]\w*)'
    r'[ \t]*=[ \t]*(?P<expr>[^;{}]+);')

# GLSL forbids a non-constant global initializer, which NotITG's lenient
# GL120 driver tolerated but core/ES profiles reject. An initializer is
# constant only if it is a numeric literal expression, optionally wrapped
# in vector constructors; any other identifier (a uniform, our injected
# time/beat aliases, or a function like sin) makes it non-constant.
_CONST_CTOR = frozenset({'vec2', 'vec3', 'vec4', 'float', 'int', 'bool',
                         'mat2', 'mat3', 'mat4', 'true', 'false'})
_IDENT_RE = re.compile(r'[A-Za-z_]\w*')


def _collect_varyings(src: str) -> dict:
    return {m.group('name'): m.group('type')
            for m in _VARYING_RE.finditer(src)}


def _collect_chart_uniforms(src: str) -> dict:
    """Scalar/vector uniforms the chart declares, minus the contract
    names we supply, in declaration order."""
    found = {}
    for m in _UNIFORM_DECL_RE.finditer(src):
        name = m.group('name')
        if name not in _RESERVED:
            found.setdefault(name, m.group('type'))
    return found


def uniform_names(glsl: str) -> tuple:
    """Chart scalar/vector uniform names the custom-uniform path can
    drive, in declaration order. The bridge maps compiled channels onto
    these (their Lua `:uniform1f` pokes)."""
    return tuple(_collect_chart_uniforms(glsl))


def _is_constant_expr(expr: str) -> bool:
    """True if `expr` reads no identifier other than numeric-vector
    constructors -- the only initializer form a core/ES profile accepts
    at file scope."""
    return all(ident in _CONST_CTOR for ident in _IDENT_RE.findall(expr))


def _hoist_nonconst_globals(body: str) -> tuple:
    """Split each file-scope `<type> name = <non-const expr>;` into a bare
    declaration (left in place) plus an assignment (returned, to run at
    main() entry). Only brace-depth-0 declarations are touched; anything
    inside a function body is already a valid local initializer.

    Returns (rewritten_body, [assignment, ...]) preserving source order."""
    out = []
    assignments = []
    depth = 0
    i = 0
    for m in re.finditer(r'[{}]|' + _GLOBAL_INIT_RE.pattern, body):
        token = m.group(0)
        if token in '{}':
            out.append(body[i:m.end()])
            i = m.end()
            depth += 1 if token == '{' else -1
            continue
        if depth != 0 or _is_constant_expr(m.group('expr')):
            continue
        out.append(body[i:m.start()])
        out.append(f"{m.group('type')} {m.group('name')};")
        assignments.append(f"{m.group('name')} ={m.group('expr')};")
        i = m.end()
    out.append(body[i:])
    return ''.join(out), assignments


# Contract values aliased onto the chart's conventional names. Declared
# as file-scope globals but ASSIGNED inside main() -- GLSL forbids a
# non-constant global initializer (a uniform is non-constant), which is
# the whole reason these are split from their declarations.
_ALIAS_GLOBALS = (
    ('vec2', 'textureSize', 'u_resolution'),
    ('vec2', 'imageSize', 'u_resolution'),
    ('vec2', 'resolution', 'u_resolution'),
    ('float', 'time', 'u_time'),
    ('float', 'beat', 'u_strength.z'),
)


def _preamble(varyings: dict, chart_uniforms: dict) -> list:
    lines = [CONTRACT_HEADER,
             'uniform sampler2D u_tex;',
             'uniform vec2 u_resolution;',
             'uniform float u_time;',
             'uniform vec3 u_strength;',
             '#define sampler0 u_tex',
             '#define texture2D texture',
             '#define gl_FragColor _fs_fragcolor',
             'out vec4 _fs_fragcolor;']
    lines += [f'{gtype} {name};' for gtype, name, _ in _ALIAS_GLOBALS]
    if _VARYING_COLOUR in varyings:
        lines.append('vec4 color;')
    lines += [f'vec3 {name};'
              for name in _VARYING_VEC3 if name in varyings]
    lines += [f'uniform {gtype} {name};'
              for name, gtype in chart_uniforms.items()]
    return lines


def _entry_shim(uv_names: tuple, varyings: dict,
                hoisted_assignments: list) -> str:
    """Assignments injected at the top of main(): the fullscreen UV feeds
    the chart's UV varyings, the contract aliases / per-vertex colour take
    their values, and any hoisted global initializer runs (all
    non-constant, so they cannot initialise a global at file scope)."""
    lines = [f'{name} = {expr};' for _, name, expr in _ALIAS_GLOBALS]
    if _VARYING_COLOUR in varyings:
        lines.append('color = vec4(1.0);')
    if uv_names:
        lines.append('vec2 _fs_uv = gl_FragCoord.xy / u_resolution;')
        lines += [f'{name} = _fs_uv;' for name in uv_names]
    # Hoisted global initializers run last: they may read the aliases and
    # UV globals set above.
    lines += hoisted_assignments
    return '\n'.join(lines)


def translate(glsl: str) -> str:
    """Return a contract-compliant fullscreen fragment shader for the raw
    NotITG chart frag `glsl`. Raises ValueError if it declares no
    sampler0 (nothing to sample -- not a fullscreen texture pass)."""
    if not _SAMPLER0_RE.search(glsl):
        raise ValueError('NotITG frag has no sampler0 to translate')

    varyings = _collect_varyings(glsl)
    chart_uniforms = _collect_chart_uniforms(glsl)
    uv_names = tuple(name for name in _VARYING_UVS if name in varyings)

    body = _VERSION_RE.sub('', glsl)
    body = _VARYING_RE.sub('', body)
    body = _SAMPLER0_RE.sub('', body)
    body = _UNIFORM_DECL_RE.sub('', body)
    body, hoisted = _hoist_nonconst_globals(body)

    # UV varyings become mutable globals so the entry shim can assign them
    # the fullscreen UV (a `#define` cannot be an lvalue for that).
    uv_globals = '\n'.join(f'vec2 {name};' for name in uv_names)
    shim = _entry_shim(uv_names, varyings, hoisted)
    injected = _MAIN_RE.sub(lambda m: m.group(0) + '\n' + shim + '\n', body,
                            count=1)
    preamble = '\n'.join(_preamble(varyings, chart_uniforms))
    return preamble + '\n' + uv_globals + '\n' + injected
