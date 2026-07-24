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


def _preamble(varyings: dict, chart_uniforms: dict,
              quad_input: bool = False) -> list:
    lines = [CONTRACT_HEADER,
             'uniform sampler2D u_tex;',
             'uniform vec2 u_resolution;',
             'uniform float u_time;',
             'uniform vec3 u_strength;',
             '#define sampler0 u_tex',
             '#define texture2D texture',
             '#define gl_FragColor _fs_fragcolor',
             'out vec4 _fs_fragcolor;']
    if quad_input:
        # The textured-quad vertex stage's interpolated source UV
        # (gl_capture's _VERTEX_SRC `out vec2 v_uv`), plus the blit
        # opacity: the chart main is renamed and wrapped so its output
        # picks up the instance opacity like every plain blit.
        lines += ['in vec2 v_uv;',
                  'uniform float u_opacity;',
                  'vec4 _fs_shaded;',
                  '#undef gl_FragColor',
                  '#define gl_FragColor _fs_shaded']
    lines += [f'{gtype} {name};' for gtype, name, _ in _ALIAS_GLOBALS]
    if _VARYING_COLOUR in varyings:
        lines.append('vec4 color;')
    lines += [f'vec3 {name};'
              for name in _VARYING_VEC3 if name in varyings]
    lines += [f'uniform {gtype} {name};'
              for name, gtype in chart_uniforms.items()]
    return lines


def _entry_shim(uv_names: tuple, varyings: dict,
                hoisted_assignments: list, uv_source: str) -> str:
    """Assignments injected at the top of main(): the source UV feeds
    the chart's UV varyings, the contract aliases / per-vertex colour take
    their values, and any hoisted global initializer runs (all
    non-constant, so they cannot initialise a global at file scope)."""
    lines = [f'{name} = {expr};' for _, name, expr in _ALIAS_GLOBALS]
    if _VARYING_COLOUR in varyings:
        lines.append('color = vec4(1.0);')
    if uv_names:
        uv = ('v_uv' if uv_source == 'varying'
              else 'gl_FragCoord.xy / u_resolution')
        lines.append(f'vec2 _fs_uv = {uv};')
        lines += [f'{name} = _fs_uv;' for name in uv_names]
    # Hoisted global initializers run last: they may read the aliases and
    # UV globals set above.
    lines += hoisted_assignments
    return '\n'.join(lines)


# ── vertex-stage translation (the Polygon mesh tier) ─────────────────
#
# A NotITG Polygon's `Vert=` shader (crumple.vert) displaces mesh
# vertices in the actor's LOCAL space (gl_Vertex, center origin,
# +y down) and projects through the fixed-function matrices. The mesh
# tier supplies `a_pos` (local xy) / `a_uv` (source UV, already
# converted to our capture orientation) and one `u_mvp` that maps local
# coords straight to clip space (the instance homography embedded in a
# mat4, depth 0 - the engine projection is orthographic so z never
# perspective-divides). Fixed-function inputs the chart reads are
# shimmed: identity texture matrix, identity normal machinery, white
# color. The chart main is renamed and wrapped so the paired fragment
# stage (the plain textured-quad frag) always receives `v_uv`.

_ATTRIBUTE_RE = re.compile(r'\battribute\b')
_VARYING_KEYWORD_RE = re.compile(r'\bvarying\b')
_TEXTURE_MATRIX_RE = re.compile(r'\bgl_TextureMatrix\s*\[[^\]]*\]')
_VERT_MAIN_RE = re.compile(r'\bvoid\s+main\s*\(\s*(?:void)?\s*\)')

_VERT_PREAMBLE = """#version 150
uniform mat4 u_mvp;
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;
vec4 _vs_frontcolor;
vec4 _vs_texcoord[2];
#define texture2D texture
#define gl_Vertex vec4(a_pos, 0.0, 1.0)
#define gl_MultiTexCoord0 vec4(a_uv, 0.0, 1.0)
#define gl_ModelViewProjectionMatrix u_mvp
#define gl_NormalMatrix mat3(1.0)
#define gl_Normal vec3(0.0, 0.0, 1.0)
#define gl_Color vec4(1.0)
#define gl_FrontColor _vs_frontcolor
#define gl_TexCoord _vs_texcoord
"""

_VERT_WRAPPER = """
void main(void) { _vs_chart_main(); v_uv = a_uv; }
"""


def translate_vert(glsl: str) -> str:
    """Return a contract vertex shader for the raw NotITG chart vert
    `glsl` (see the section comment). Raises ValueError when the source
    has no main to wrap."""
    if not _VERT_MAIN_RE.search(glsl):
        raise ValueError('NotITG vert has no main to translate')
    body = _VERSION_RE.sub('', glsl)
    body = _ATTRIBUTE_RE.sub('in', body)
    body = _VARYING_KEYWORD_RE.sub('out', body)
    body = _TEXTURE_MATRIX_RE.sub('mat4(1.0)', body)
    body = _VERT_MAIN_RE.sub('void _vs_chart_main()', body, count=1)
    return _VERT_PREAMBLE + body + _VERT_WRAPPER


# NotITG compiles chart shaders as desktop GLSL 1.20, which has
# implicit int->float conversion; GLSL ES (the dialect Qt picks under
# Wayland/EGL and ANGLE) has NONE, so corpus-legal arithmetic like
# monitor.frag's `10*(-0.1+0.2*rand(beat))` fails to compile. The
# promotion pass rewrites integer literals to floats EXCEPT where an
# int is required or meaningful: preprocessor lines (#define loop
# bounds), array indices/sizes (inside [ ]), for-loop headers (int
# counters compared against their bounds), and statements declaring
# int/uint/bool scalars or vectors. It runs only as a RETRY after the
# faithful translation fails to build, so a shader that compiles
# strictly is never rewritten.
_INT_TOKEN_RE = re.compile(r'(?<![\w.])(\d+)(?![\w.])')
_INT_DECL_STMT_RE = re.compile(
    r'\b(?:int|uint|bool|[iub]vec[234])\b[^;(){}]*;')
_FOR_HEADER_RE = re.compile(r'\bfor\s*\(')
_PREPROC_LINE_RE = re.compile(r'^[ \t]*#.*$', re.MULTILINE)


def promote_int_literals(glsl: str) -> str:
    """`glsl` with bare integer literals promoted to float literals,
    int-typed contexts left alone (see the section comment above)."""
    protected = bytearray(len(glsl))

    def shield(start, end):
        protected[start:end] = b'\1' * (end - start)

    for pattern in (_PREPROC_LINE_RE, _INT_DECL_STMT_RE):
        for m in pattern.finditer(glsl):
            shield(*m.span())
    for m in _FOR_HEADER_RE.finditer(glsl):
        shield(m.start(), _closing_paren(glsl, m.end() - 1))
    depth = 0
    for i, ch in enumerate(glsl):
        if ch == '[':
            depth += 1
        if depth:
            protected[i] = 1
        if ch == ']':
            depth = max(0, depth - 1)
    return _INT_TOKEN_RE.sub(
        lambda m: m.group(0) if protected[m.start()] else m.group(0) + '.0',
        glsl)


def _closing_paren(src: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == '(':
            depth += 1
        elif src[i] == ')':
            depth -= 1
            if depth == 0:
                return i + 1
    return len(src)


def translate(glsl: str, uv_source: str = 'fragcoord') -> str:
    """Return a contract-compliant fragment shader for the raw NotITG
    chart frag `glsl`. Raises ValueError if it declares no sampler0
    (nothing to sample -- not a texture pass).

    `uv_source` picks where the chart's UV varyings read from:
    'fragcoord' (the fullscreen post pass - UV = destination position)
    or 'varying' (a textured-quad blit - UV = the quad's interpolated
    `v_uv` source coordinate, so the frag samples the SOURCE texel the
    vertex mapping chose regardless of where the quad landed)."""
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
    # the source UV (a `#define` cannot be an lvalue for that).
    uv_globals = '\n'.join(f'vec2 {name};' for name in uv_names)
    shim = _entry_shim(uv_names, varyings, hoisted, uv_source)
    quad = uv_source == 'varying'
    entry = 'void _fs_chart_main() {' if quad else None
    injected = _MAIN_RE.sub(
        lambda m: (entry or m.group(0)) + '\n' + shim + '\n', body, count=1)
    if quad:
        injected += ('\nvoid main(void) { _fs_chart_main(); '
                     '_fs_fragcolor = _fs_shaded * u_opacity; }\n')
    preamble = '\n'.join(_preamble(varyings, chart_uniforms,
                                   quad_input=quad))
    return preamble + '\n' + uv_globals + '\n' + injected
