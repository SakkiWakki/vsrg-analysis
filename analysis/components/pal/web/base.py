"""Core web-texture PAL types.

A WebTexture is a live off-screen web page (html+css+js) that produces
frames at a chosen cadence. Frames travel through the PAL as opaque
handles tagged by ``kind``: a consumer that recognises the tag uses the
handle directly; a consumer that doesn't auto-downgrades to the
CPU-readback (``qpixmap``) path.

Backend selection is host-driven via :class:`WebTexturePAL.select`: the
PAL probes its registered backends and returns the highest-capability
one that works in the caller's environment (GUI sidebar / overlay /
cross-process handoff). The component on top of the returned texture
never sees the backend choice.

Frame-kind taxonomy:

- ``qpixmap``       ; ``QPixmap`` backed by a CPU buffer. Always works.
- ``qsg_texture``   ; ``QSGTexture`` inside Qt's scene graph. In-process.
- ``gl_texture_id`` ; ``int`` GL texture name + context share group id.
- ``dmabuf_fd``     ; Linux dmabuf fd + modifier + strides. Cross-process.
- ``win32_shared``  ; Win32 shared NT handle. Cross-process, Windows.
- ``vk_image``      ; ``VkImage`` + ``VkSemaphore``. Cross-process Vulkan.

Only ``qpixmap`` is implemented today; the tag space is fixed up front so
later backends plug in without touching consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# Frame-kind tags. Keep as module constants so typos are caught at
# import time, not at diff time.
KIND_QPIXMAP       = 'qpixmap'
KIND_QSG_TEXTURE   = 'qsg_texture'
KIND_GL_TEXTURE    = 'gl_texture_id'
KIND_DMABUF_FD     = 'dmabuf_fd'
KIND_WIN32_SHARED  = 'win32_shared'
KIND_VK_IMAGE      = 'vk_image'


# ── Surface selection ───────────────────────────────────────────────

# The "where is this frame headed" signal drives backend picking.
# Intentionally distinct from the component API's SURFACE_* constants ;
# one component may mount on several surfaces with different PAL needs
# (sidebar wants qpixmap; gamescope overlay wants dmabuf).
SURFACE_LOCAL_CPU   = 'local_cpu'      # QPainter compositor, any host widget
SURFACE_LOCAL_GL    = 'local_gl'       # QOpenGLWidget or QML scene graph
SURFACE_CROSSPROC_GL = 'crossproc_gl'  # injected gl_layer in another process
SURFACE_CROSSPROC_VK = 'crossproc_vk'  # injected vulkan_layer


# ── Frame ───────────────────────────────────────────────────────────

@dataclass
class WebTextureFrame:
    """One snapshot of a WebTexture.

    ``kind`` identifies the handle's type; consumers branch on it. All
    backends set ``width``/``height`` in pixels matching the source
    page's viewport, and ``generation`` monotonically so consumers can
    diff against the frame they last uploaded and skip redundant work.

    ``wait_token`` is an opaque handle the producer may stash for
    GPU-side sync (e.g. a Vulkan semaphore, EGL sync object). Consumers
    that got the frame via a backend that set this MUST wait on it
    before sampling the texture; backends that don't need sync leave it
    ``None``.
    """

    width: int
    height: int
    kind: str
    handle: Any          # QPixmap / int / dmabuf dict / ...
    generation: int
    wait_token: Any = None
    # Extra per-kind metadata (stride, modifier, format). Kept as a
    # plain dict so adding fields for a new backend doesn't ripple.
    meta: dict = field(default_factory=dict)


# ── Capabilities ────────────────────────────────────────────────────

@dataclass(frozen=True)
class WebTextureBackendCaps:
    """What a backend can and can't do.

    The PAL dispatcher reads these to match backends against host
    requirements. A backend that supports zero-copy on the host's
    compositor wins over a backend that falls back to readback, even
    if both work.
    """
    # Frame kinds this backend can produce.
    produces: tuple[str, ...] = ()
    # True iff the producing path avoids a CPU readback cycle.
    zero_copy: bool = False
    # True iff the handle can be sent to another process. Required for
    # the game-overlay surfaces.
    cross_process: bool = False
    # True iff the backend needs a live QApplication on the calling
    # thread. QWebEngineView-based backends all do; a headless Chromium
    # backend wouldn't.
    needs_qapplication: bool = True
    # Which :data:`SURFACE_*` this backend is usable for. Surface
    # selection is the primary filter; ties are broken by ``zero_copy``.
    surfaces: frozenset[str] = frozenset()


# ── Backend + texture contracts ─────────────────────────────────────

@runtime_checkable
class WebTexture(Protocol):
    """Handle to a live off-screen page. Produced by a backend's
    :meth:`WebTextureBackend.create`."""

    def resize(self, width: int, height: int) -> None: ...

    def load_url(self, url: str) -> None: ...
    """Load a new document. The frame generation resets so consumers
    re-upload on the next ``latest_frame`` call."""

    def push_js_state(self, json_str: str) -> None: ...
    """Forward a JSON state push to the page. The backend routes this
    through whatever shim/channel the caller wired up after
    ``create``."""

    def active_filters(self) -> frozenset[str]:
        """Filter set the page requested via the bridge. Used by the
        caller to prune state pushes."""
        ...

    def latest_frame(self) -> 'WebTextureFrame | None': ...
    """Latest produced frame, or None before the first render. The
    returned handle lives as long as the WebTexture lives; consumers
    MUST NOT hold references past the next ``latest_frame`` call unless
    the backend documents otherwise."""

    def close(self) -> None: ...


@runtime_checkable
class WebTextureBackend(Protocol):
    """Factory for WebTextures. Backends register with
    :class:`WebTexturePAL` and are instantiated lazily."""

    name: str

    def capabilities(self) -> WebTextureBackendCaps: ...

    def is_available(self) -> bool:
        """Fast probe ; can this backend actually run here and now?
        May inspect QApplication state, check for required extensions,
        etc. Must be safe to call repeatedly."""
        ...

    def create(self, *, width: int, height: int) -> WebTexture: ...
