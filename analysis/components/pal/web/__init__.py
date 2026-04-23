"""Web-texture PAL: live web pages as texture handles.

Public API:

    from analysis.components.pal.web import WebTexturePAL, SURFACE_LOCAL_CPU
    pal = WebTexturePAL.default()
    tex = pal.create(surface=SURFACE_LOCAL_CPU, width=800, height=600)
    tex.load_url(...)
    ...
    frame = tex.latest_frame()
    # dispatch on frame.kind

Registering a backend: see :mod:`analysis.components.pal.web.qpixmap_backend`
for the canonical template. Backends implement :class:`WebTextureBackend`
and get registered via ``WebTexturePAL.register(backend)`` at import time.
"""
from analysis.components.pal.web.base import (
    KIND_DMABUF_FD,
    KIND_GL_TEXTURE,
    KIND_QPIXMAP,
    KIND_QSG_TEXTURE,
    KIND_VK_IMAGE,
    KIND_WIN32_SHARED,
    SURFACE_CROSSPROC_GL,
    SURFACE_CROSSPROC_VK,
    SURFACE_LOCAL_CPU,
    SURFACE_LOCAL_GL,
    WebTexture,
    WebTextureBackend,
    WebTextureBackendCaps,
    WebTextureFrame,
)
from analysis.components.pal.web.dispatcher import WebTexturePAL

__all__ = [
    'KIND_DMABUF_FD',
    'KIND_GL_TEXTURE',
    'KIND_QPIXMAP',
    'KIND_QSG_TEXTURE',
    'KIND_VK_IMAGE',
    'KIND_WIN32_SHARED',
    'SURFACE_CROSSPROC_GL',
    'SURFACE_CROSSPROC_VK',
    'SURFACE_LOCAL_CPU',
    'SURFACE_LOCAL_GL',
    'WebTexture',
    'WebTextureBackend',
    'WebTextureBackendCaps',
    'WebTextureFrame',
    'WebTexturePAL',
]
