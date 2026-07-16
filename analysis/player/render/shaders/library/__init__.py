"""Builtin fullscreen post-process shaders, one GLSL 1.50 fragment
file per effect, plus a session registry for chart-supplied (tier-2)
shaders.

Every shader served here follows the same uniform contract, so the
GL pipeline can run any of them without per-shader Python code:

    uniform sampler2D u_tex;         source frame (previous pass)
    uniform vec2      u_resolution;  target size in physical pixels
    uniform float     u_time;        song time, seconds
    uniform vec3      u_strength;    effect params (strength1..3)

All shaders read pixel position from `gl_FragCoord`, so passes are
pure pixel-space operations and no vertex UVs or y-flips are needed
anywhere in the chain. Each shader must be an identity map when
`u_strength` is all zeros.

The fluXis set is transpiled from fluXis.Resources/Shaders (Vulkan-
flavored osu-framework GLSL -> desktop GL); ids are the lowercased
fluXis `ShaderType` names. Game-specific param massaging (fluXis
Glitch's /10 scaling, SplitScreen's trunc-to-int splits) lives inside
the shader so `u_strength` always carries the raw event values.

Tier 2 (chart-supplied) shaders are registered at runtime via
`register_source` / `register_file`. They live in a session dict keyed
by namespaced ids (e.g. `chart:<relpath>`); the namespace separator
(`:`) is disallowed in builtin filenames, so a chart shader can never
shadow a builtin. `source()` consults builtins first, then the
registry. NotITG chart frags are not written to our contract; register
them with `compat=True` (or via `register_notitg_frag`) to translate
them through library/notitg_compat.py first.
"""
from __future__ import annotations

from pathlib import Path

from analysis.player.render.shaders.library import notitg_compat

_LIBRARY_DIR = Path(__file__).parent

_NAMESPACE_SEP = ':'

# Session registry of chart-supplied shaders: id -> contract GLSL.
_REGISTRY: dict[str, str] = {}


def _is_builtin_name(name: str) -> bool:
    """A bare, path-safe builtin id: alphanumerics and underscores only,
    so it maps to exactly one `<name>.frag` with no traversal and never
    contains the registry namespace separator."""
    return bool(name) and name.replace('_', '').isalnum()


def source(name: str) -> str | None:
    """GLSL source for the shader `name`, or None if unknown. Builtins
    (bare names -> `<name>.frag`) win over registered chart shaders, so a
    registration can never shadow a builtin."""
    if _is_builtin_name(name):
        path = _LIBRARY_DIR / f'{name}.frag'
        if path.is_file():
            return path.read_text(encoding='utf-8')
    return _REGISTRY.get(name)


def available() -> tuple:
    """All servable ids: builtin filenames plus registered chart ids."""
    builtins = (p.stem for p in _LIBRARY_DIR.glob('*.frag'))
    return tuple(sorted({*builtins, *_REGISTRY}))


def _reject_builtin_shadow(name: str) -> None:
    if _is_builtin_name(name) and (_LIBRARY_DIR / f'{name}.frag').is_file():
        raise ValueError(
            f'{name!r} would shadow a builtin shader; namespace chart '
            f'ids with a {_NAMESPACE_SEP!r} prefix (e.g. chart:{name})')


def register_source(name: str, glsl: str, *, compat: bool = False) -> str:
    """Register chart-supplied GLSL under `name` for the session and
    return the id. `name` must be namespaced (contain a `:`) so it cannot
    collide with a builtin. With `compat=True` the source is a raw NotITG
    chart frag and is translated onto our contract first."""
    if _NAMESPACE_SEP not in name:
        raise ValueError(
            f'chart shader id {name!r} must be namespaced with '
            f'{_NAMESPACE_SEP!r} (e.g. chart:{name})')
    _reject_builtin_shadow(name)
    _REGISTRY[name] = notitg_compat.translate(glsl) if compat else glsl
    return name


def register_file(name: str, path, *, compat: bool = False) -> str:
    """Register the GLSL in `path` under `name` (see `register_source`)."""
    return register_source(name, Path(path).read_text(encoding='utf-8'),
                           compat=compat)


def register_notitg_frag(name: str, glsl: str) -> str:
    """Register a raw NotITG chart frag, translated onto our contract.
    Shorthand for `register_source(name, glsl, compat=True)`."""
    return register_source(name, glsl, compat=True)


def registered_uniform_names(name: str) -> tuple:
    """For a registered NotITG-compat shader, the chart uniform names the
    custom-uniform path can drive (empty for builtins/plain registrations
    with no chart uniforms)."""
    glsl = _REGISTRY.get(name)
    return notitg_compat.uniform_names(glsl) if glsl else ()


def clear_registry() -> None:
    """Drop all session registrations (test isolation, chart unload)."""
    _REGISTRY.clear()
