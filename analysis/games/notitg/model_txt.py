"""Milkshape 3D ASCII model loader (SM RageModelGeometry's `.txt`).

A NotITG `<Layer File="models/x.txt">` is a Model actor: triangle
geometry with per-vertex UVs and a diffuse texture named by its
material. Government Knows ships obj2ms3dascii conversions (macplus,
think, dick, ftl, cheapsphere), all static single-frame models - so
this reads frame-1 geometry only, flattened to one triangle list per
mesh, and resolves each mesh's diffuse texture against the model's own
directory.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Vertex line: flags, x, y, z, u, v, bone. Triangle line: flags, three
# vertex indices, three normal indices, smoothing group.
_VERTEX_FIELDS = 7
_TRIANGLE_FIELDS = 8

# A material block: name, ambient, diffuse, specular, emissive,
# shininess, transparency, diffuse texture, alpha texture.
_MATERIAL_LINES = 9


def load_model(path) -> list[dict] | None:
    """`[{'vertices': (n, 8) float32 [x, y, z, u, v, nx, ny, nz],
    'texture': absolute path or None, 'spheremap': bool}]` per mesh, or
    None when `path` is not a readable Milkshape ASCII model.

    `spheremap` mirrors SM's texture-name convention: a material whose
    diffuse texture ends in `sphere.png` is environment-mapped - UVs
    come from the view-space NORMALS at draw time, not the (typically
    zero) authored UVs. Government Knows' models shade entirely this
    way."""
    try:
        text = Path(path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not line.strip().startswith('//')]
    try:
        return _parse(lines, Path(path).parent)
    except (ValueError, IndexError) as exc:
        logger.warning('model %s failed to parse: %s', path, exc)
        return None


def _parse(lines, base_dir) -> list[dict] | None:
    cursor = _skip_to(lines, 'Meshes:')
    if cursor is None:
        return None
    mesh_count = int(lines[cursor].split(':')[1])
    cursor += 1

    meshes = []
    for _ in range(mesh_count):
        # "name" flags material_index
        material = int(lines[cursor].rsplit(None, 1)[1])
        cursor += 1
        nverts = int(lines[cursor])
        cursor += 1
        verts = np.array([lines[cursor + i].split()[:_VERTEX_FIELDS]
                          for i in range(nverts)], dtype=np.float64)
        cursor += nverts
        nnormals = int(lines[cursor])
        cursor += 1
        normals = np.array([lines[cursor + i].split()[:3]
                            for i in range(nnormals)], dtype=np.float64)
        cursor += nnormals
        ntris = int(lines[cursor])
        cursor += 1
        tris = np.array([lines[cursor + i].split()[:_TRIANGLE_FIELDS]
                         for i in range(ntris)], dtype=np.int64)
        cursor += ntris
        meshes.append((material, verts, normals, tris))

    cursor = _skip_to(lines, 'Materials:', cursor)
    textures: list = []
    if cursor is not None:
        material_count = int(lines[cursor].split(':')[1])
        cursor += 1
        for _ in range(material_count):
            block = lines[cursor:cursor + _MATERIAL_LINES]
            cursor += _MATERIAL_LINES
            textures.append(_texture_of(block, base_dir))

    out = []
    for material, verts, normals, tris in meshes:
        # verts columns: flag, x, y, z, u, v, bone; triangle lanes 1..4
        # index vertices, 4..7 the normals list.
        corner = verts[tris[:, 1:4].reshape(-1), 1:6]
        corner_normals = (normals[tris[:, 4:7].reshape(-1)]
                          if len(normals) else np.zeros((len(corner), 3)))
        texture = (textures[material]
                   if 0 <= material < len(textures) else None)
        out.append({
            'vertices': np.ascontiguousarray(
                np.hstack([corner, corner_normals]), dtype=np.float32),
            'texture': texture,
            'spheremap': bool(texture)
            and texture.lower().endswith('sphere.png'),
        })
    return out or None


def _texture_of(block, base_dir):
    """The material block's diffuse-texture path, resolved under the
    model's dir (Milkshape writes Windows separators), or None."""
    name = block[_MATERIAL_LINES - 2].strip('"')
    if not name:
        return None
    candidate = base_dir / name.replace('\\', '/')
    return str(candidate) if candidate.is_file() else None


def _skip_to(lines, prefix, start=0):
    for i in range(start, len(lines)):
        if lines[i].startswith(prefix):
            return i
    return None


def is_model_reference(file_attr) -> bool:
    """Whether a `File=` attribute names a Milkshape model (SM loads
    `.txt` files through RageModelGeometry)."""
    return bool(file_attr) and file_attr.lower().endswith('.txt')
