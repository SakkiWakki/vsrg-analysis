"""Builtin fullscreen post-process shaders, one GLSL 1.50 fragment
file per effect.

Every shader in the library follows the same uniform contract, so the
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
"""
from __future__ import annotations

from pathlib import Path

_LIBRARY_DIR = Path(__file__).parent


def source(name: str) -> str | None:
    """GLSL source for the shader `name`, or None if not in the
    library (unreleased fluXis shaders, typoed map events)."""
    if not name.replace('_', '').isalnum():
        return None
    path = _LIBRARY_DIR / f'{name}.frag'
    if not path.is_file():
        return None
    return path.read_text(encoding='utf-8')


def available() -> tuple:
    return tuple(sorted(p.stem for p in _LIBRARY_DIR.glob('*.frag')))
