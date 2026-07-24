"""Seam-B GL executor: the QOpenGLFramebufferObject twin of
``executor.RasterExecutor``. A GL consumer of the ``DrawSchedule`` record
stream produced by ``storyboard_native`` - same records, same painter's
-algorithm contract, backed by FBOs and textured quads instead of
``QImage``/``QPainter``.

Each DrawableId owns a ``QOpenGLFramebufferObject`` sized to that
drawable's logical size (1 logical unit = 1 device px - the reference
mapping, as in the raster backend). BEGIN clears the target FBO by mode
(Retain leaves it); a BLIT draws one textured/fill quad; COPY blits the
open target FBO into a named drawable's FBO (at-position Snapshot); END
pops the target stack. ``execute`` returns the screen drawable's FBO read
back as a ``QImage`` (``toImage``) for tests.

Record layout (frozen; mirrors native/src/evaluate.rs, U_STRIDE=10 /
F_STRIDE=20) - identical to RasterExecutor:

  U lanes: [kind, a, b, c, blend, shader+1, clip+1, screen_space,
            uf_offset, uf_count]
  F lanes: mat3 [0..9], opacity [9], tint rgb [10..13],
           crop l,t,r,b [13..17], reserved [17..20]

The mat3 is the RECORD's column-vector convention (p' = M @ p, translation
in lanes 2/5), source logical -> target logical; homographies welcome (the
quad forwards w through gl_Position, the GPU form of a projective blit).

Conventions follow gl_capture.py (the solved traps - keep in lockstep):
- FBO targets via ``QOpenGLFramebufferObject``; the enclosing target is
  read from ``GL_FRAMEBUFFER_BINDING`` inside a native-painting bracket,
  never assumed.
- premultiplied source-over ``(ONE, ONE_MINUS_SRC_ALPHA)``, additive
  ``(ONE, ONE)``.
- Qt paints y-down into FBOs, so FBO-to-FBO reads need no flip; the quad
  maps logical top-left to NDC (-1, +1) and samples ``v = 1 - y/h``.
- ES-dialect adaptation via ``_adapt_dialect``.

HARD RULE (the glGenTextures lesson): every GL op is individually
guarded. A bad op logs ONCE and is skipped; it never kills the frame or
the executor. Lines / mesh / clip / shader sources are logged-once TODOs
(the raster backend is the reference for those until they land here).

Clear modes match ClearMode in doc.rs: TransparentBlack=0, OpaqueBlack=1,
Retain=2.
"""
from __future__ import annotations

import logging
import struct

import numpy as np
from shiboken6 import VoidPtr

from PySide6.QtGui import QImage, QMatrix3x3, QOpenGLContext
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject,
                              QOpenGLShader, QOpenGLShaderProgram,
                              QOpenGLVertexArrayObject)

from analysis.player.render.shaders.gl_pipeline import (
    _adapt_dialect,
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_DRAW_FRAMEBUFFER, GL_FLOAT, GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_BINDING, GL_LINEAR, GL_NEAREST,
    GL_READ_FRAMEBUFFER, GL_RGBA, GL_SCISSOR_TEST, GL_STENCIL_BUFFER_BIT,
    GL_STENCIL_TEST, GL_TEXTURE0, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_TRIANGLE_STRIP, GL_UNSIGNED_BYTE)
from PySide6.QtOpenGL import QOpenGLPaintDevice

logger = logging.getLogger(__name__)

GL_ONE = 1
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_CLAMP_TO_EDGE = 0x812F

_SCREEN_ID = 0

# u32 lane offsets.
_U_KIND = 0
_U_A = 1
_U_B = 2
_U_C = 3
_U_BLEND = 4
_U_SHADER = 5
_U_CLIP = 6
_U_SCREEN_SPACE = 7
_U_UF_OFFSET = 8
_U_UF_COUNT = 9

# f32 lane offsets.
_F_OPACITY = 9
_F_TINT = 10  # ..13
_F_CROP = 13  # ..17 (l, t, r, b fractions of the SOURCE logical size)

# Op / source / clear codes (kept in sync with storyboard_native; the
# executor runs without importing it at module load so tests importorskip
# cleanly).
_OP_BEGIN, _OP_BLIT, _OP_COPY, _OP_END = 0, 1, 2, 3
_SRC_IMAGE, _SRC_DRAWABLE, _SRC_MESH, _SRC_FILL, _SRC_LINES = 0, 1, 2, 3, 4
_CLEAR_TRANSPARENT, _CLEAR_OPAQUE, _CLEAR_RETAIN = 0, 1, 2

_BLEND_ADDITIVE = 1

_FLOATS_PER_VERTEX = 4
_QUAD_BYTES = 4 * _FLOATS_PER_VERTEX * 4

# A projective textured quad: u_mat is source-logical -> NDC (w forwarded
# through gl_Position, so record homographies interpolate perspective-
# correctly - the GPU form of QPainter's projective drawImage).
_VERTEX_SRC = """#version 150
in vec2 a_pos;
in vec2 a_uv;
uniform mat3 u_mat;
out vec2 v_uv;
void main(void) {
    vec3 p = u_mat * vec3(a_pos, 1.0);
    v_uv = a_uv;
    gl_Position = vec4(p.xy, 0.0, p.z);
}
"""

# FBO content is premultiplied (Qt's GL paint engine and this executor's
# own blits both write premultiplied), so opacity multiplies every channel
# and source-over is (ONE, ONE_MINUS_SRC_ALPHA).
_TEX_FRAG_SRC = """#version 150
uniform sampler2D u_tex;
uniform float u_opacity;
uniform vec3 u_tint;
in vec2 v_uv;
out vec4 fragColor;
void main(void) {
    vec4 c = texture(u_tex, v_uv);
    fragColor = vec4(c.rgb * u_tint, c.a) * u_opacity;
}
"""

_FILL_FRAG_SRC = """#version 150
uniform vec4 u_color;
out vec4 fragColor;
void main(void) { fragColor = u_color; }
"""


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _mat_source_to_ndc(mat3: np.ndarray, w: int, h: int) -> QMatrix3x3:
    """Compose the record's column-vector mat3 (source logical -> target
    logical) with target-logical -> NDC (y-down, top-left at (-1, +1); 1
    logical unit = 1 device px). Returns the 3x3 for u_mat."""
    to_ndc = np.array([[2.0 / w, 0.0, -1.0],
                       [0.0, -2.0 / h, 1.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    m = to_ndc @ mat3.astype(np.float64).reshape(3, 3)
    return QMatrix3x3([float(v) for v in m.flatten()])


def _device_to_ndc(pw: int, ph: int) -> QMatrix3x3:
    """Map device-px coords (top-left origin, y-down) to NDC with Qt's
    top-left at (-1, +1). Used to present the composed screen quad onto the
    host GL target."""
    m = np.array([[2.0 / pw, 0.0, -1.0],
                  [0.0, -2.0 / ph, 1.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return QMatrix3x3([float(v) for v in m.flatten()])


def _target_device_size(painter):
    """(device-px width, height, dpr) of the painter's render target.
    QOpenGLPaintDevice reports device pixels; widget devices report logical
    pixels (scaled by the ratio) - mirrors gl_capture._target_device_size."""
    dev = painter.device()
    dpr = float(dev.devicePixelRatioF())
    if isinstance(dev, QOpenGLPaintDevice):
        size = dev.size()
        return size.width(), size.height(), dpr
    return int(dev.width() * dpr), int(dev.height() * dpr), dpr


def _set_sample_params(f, texture) -> None:
    """Linear + clamp on an FBO texture (sources sample under arbitrary
    homographies; the reference backend filters linearly)."""
    f.glBindTexture(GL_TEXTURE_2D, texture)
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    f.glBindTexture(GL_TEXTURE_2D, 0)


def _build_program(frag_src, uniforms):
    program = QOpenGLShaderProgram()
    built = (program.addShaderFromSourceCode(
                 QOpenGLShader.ShaderTypeBit.Vertex,
                 _adapt_dialect(_VERTEX_SRC))
             and program.addShaderFromSourceCode(
                 QOpenGLShader.ShaderTypeBit.Fragment,
                 _adapt_dialect(frag_src)))
    program.bindAttributeLocation('a_pos', 0)
    program.bindAttributeLocation('a_uv', 1)
    if not (built and program.link()):
        logger.warning('GLExecutor: quad program failed to build: %s',
                       program.log())
        return None
    return program, {u: program.uniformLocation(u) for u in uniforms}


def _upload_image(f, image: QImage):
    """Upload a QImage as a premultiplied RGBA texture: (texture id, w, h),
    or None on failure. Sources arrive as ImageId -> QImage (raster
    parity); GL wants them as textures. glGenTextures binds in PySide's C
    out-param form (a count plus a writable numpy buffer)."""
    conv = image.convertToFormat(QImage.Format.Format_RGBA8888_Premultiplied)
    w, h = conv.width(), conv.height()
    if w <= 0 or h <= 0:
        return None
    ids = np.zeros(1, dtype=np.uint32)
    f.glGenTextures(1, ids)
    texture = int(ids[0])
    if texture == 0:
        return None
    f.glBindTexture(GL_TEXTURE_2D, texture)
    f.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, bytes(conv.constBits()))
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    f.glBindTexture(GL_TEXTURE_2D, 0)
    return texture, w, h


class GLExecutor:
    """Runs DrawSchedule records onto retained per-drawable FBOs.

    Mirrors ``RasterExecutor``'s API: ``images`` maps ImageId -> QImage
    source textures (uploaded to GL lazily); ``drawable_sizes`` is indexed
    by DrawableId and gives each drawable's logical size (allocated as a
    device-px FBO of that integer size, logical == device px here).
    ``clips`` / ``lines`` are accepted for API symmetry - clip and line
    sources are logged-once TODOs on the GL path.

    Requires a current GL 3+ context at construction and execute() time;
    the caller is responsible for making one current (the offscreen-
    surface fixture in the tests, or the widget canvas in production).
    """

    def __init__(
        self,
        images: dict[int, QImage],
        drawable_sizes: list[tuple[float, float]],
        clips: list[tuple] | None = None,
        lines: dict[int, np.ndarray] | None = None,
    ) -> None:
        self._images = images
        self._sizes = drawable_sizes
        self._clips = list(clips or [])
        self._lines: dict[int, np.ndarray] = dict(lines or {})
        self.shader_uniforms: dict[int, list[float]] = {}

        # Lazily built GL objects (need a current context, which the ctor
        # may not have; built on first execute()).
        self._programs = None            # (tex_entry, fill_entry)
        self._targets: dict[int, QOpenGLFramebufferObject] = {}
        self._image_textures: dict[int, tuple] = {}  # image id -> (tex, w, h)
        # EXTERNAL bound textures: DrawableId -> (texture id, pixel w, h),
        # not owned by this executor (never deleted). set_drawable_texture
        # binds a renderer capture FBO's texture as a drawable's content, so
        # a SRC_DRAWABLE blit of that id samples the live capture directly -
        # no QImage readback, no upload. This is the GL app path (D1: bound
        # texture ingest). Sampled like an owned FBO texture, normalized to
        # the drawable's logical box per drawable-ir.md rule 5.
        self._bound_textures: dict[int, tuple] = {}
        # Per-drawable clear-mode overrides (DrawableId -> ClearMode code),
        # API parity with the raster backend. The screen root is minted
        # OpaqueBlack; the GL pipeline overrides it TransparentBlack so the
        # composed screen presents OVER the renderer's painted backdrop
        # instead of covering it opaque black (the black-chart-region fix).
        self._clear_override: dict[int, int] = {}
        # FBOs allocate at logical * scale pixels so the composite is not
        # authored-resolution (640x480) stretched onto the chart rect; the
        # pipeline sets this from the chart rect's device size before the
        # first frame. Geometry stays in logical units throughout.
        self._res_scale = 1.0
        self._vao = None
        self._vbo = None
        self._skipped: set[str] = set()
        self.broken = False

    def set_lines(self, lines_id: int, verts: np.ndarray) -> None:
        """Update a polyline source's vertices (API parity with the raster
        backend). Lines are a logged-once TODO on the GL path."""
        self._lines[int(lines_id)] = np.asarray(verts, dtype=np.float32).reshape(-1, 2)

    def set_clear(self, drawable_id: int, mode: int) -> None:
        """Override the clear mode a drawable's BEGIN applies, keyed by
        DrawableId (parity with RasterExecutor.set_clear). ``mode`` is a
        ClearMode code (TransparentBlack=0, OpaqueBlack=1, Retain=2); set once,
        it holds across execute() calls and wins over the record's own clear.
        The pipeline sets the screen root TransparentBlack so the composite
        presents over the painted backdrop instead of covering it black."""
        self._clear_override[int(drawable_id)] = int(mode)

    def set_resolution_scale(self, scale: float) -> None:
        """Set the FBO allocation scale (device px per logical unit).
        Changing it drops every allocated target so they re-allocate at
        the new size on next use - set it once, before the first frame
        (retained content does not survive a change)."""
        scale = max(1.0, float(scale))
        if abs(scale - self._res_scale) < 1e-6:
            return
        self._res_scale = scale
        for fbo in self._targets.values():
            fbo.release()
        self._targets.clear()

    def set_drawable_texture(self, drawable_id: int, texture_id: int,
                             w_px: int, h_px: int, uv_rect=None) -> None:
        """Bind an EXTERNAL GL texture as a drawable's content, keyed by
        DrawableId. The texture is NOT owned (never generated, bound, or
        deleted here); a SRC_DRAWABLE blit of ``drawable_id`` samples it
        like the drawable's own FBO texture, normalized to the drawable's
        logical box (drawable-ir.md rule 5). ``w_px``/``h_px`` are the
        texture's pixel dimensions (only the aspect matters after
        normalization). Passing texture_id 0 / None un-binds it.

        This is how the GL pipeline hands the renderer's live field / slot /
        backdrop capture (a ``gl_capture._GLHandle``'s ``fbo.texture()``)
        into the doc's command-less field drawables, with no CPU readback -
        the GL-only app path the directive calls for."""
        if not texture_id:
            self._bound_textures.pop(int(drawable_id), None)
            return
        # ``uv_rect`` = (x0, y0, x1, y1) fractions of the texture in y-down
        # content space naming the sub-region that corresponds to the
        # drawable's logical box (an overscan-padded capture's margins crop
        # away here); None = the whole texture.
        self._bound_textures[int(drawable_id)] = (
            int(texture_id), max(1, int(w_px)), max(1, int(h_px)),
            tuple(uv_rect) if uv_rect is not None else None)

    def execute(
        self,
        u: np.ndarray,
        f: np.ndarray,
        uf: np.ndarray | None = None,
    ) -> QImage:
        """Run the records; return the screen drawable's FBO as a QImage.

        Retained (persistent) targets survive across calls, so re-running
        the same doc reuses last frame's content for any Retain drawable
        (feedback)."""
        gf = self._compose(u, f, uf)
        if gf is None:
            return self._empty_screen()
        gf.glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return self._screen_image()

    def render_and_present(self, u: np.ndarray, f: np.ndarray, painter,
                           chart_rect, uf: np.ndarray | None = None) -> bool:
        """Compose the schedule, then present the screen FBO over ``painter``'s
        GL target into ``chart_rect`` - NO QImage readback (the GL-only app
        path). ``chart_rect`` is (x, y, w, h) in the painter's logical units;
        the screen composite is drawn source-over so a transparent screen
        clear lets the painted backdrop show through. Returns True when it
        presented, False (never raising) when it could not - the caller then
        falls through to the normal path.

        The whole compose+present runs inside ONE native-painting bracket:
        the caller's QPainter is mid-frame on its GL target, and raw GL ops
        (the compose FBO walk, our own quad program/VBO) must not run while
        Qt's paint engine holds the context - hence the bracket around both.
        The host framebuffer is captured at bracket entry and restored before
        exit so Qt resumes painting onto its own target."""
        gl = QOpenGLContext.currentContext()
        if gl is None:
            self._log_once('no_context', 'GLExecutor: no current GL context, '
                           'nothing presented')
            return False
        gf = gl.extraFunctions()
        painter.beginNativePainting()
        try:
            host_fbo = int(gf.glGetIntegerv(GL_FRAMEBUFFER_BINDING))
            if self._compose(u, f, uf) is None:
                return False
            screen = self._targets.get(_SCREEN_ID)
            if screen is None:
                return False
            self._present_screen(gf, painter, screen, chart_rect, host_fbo)
        except Exception as exc:  # noqa: BLE001 - one bad present never crashes the frame
            self._log_once('present', f'GLExecutor: present failed ({exc}), skipped')
            return False
        finally:
            gf.glBindFramebuffer(GL_FRAMEBUFFER, host_fbo)
            painter.endNativePainting()
        return True

    def _compose(self, u: np.ndarray, f: np.ndarray, uf):
        """Run the schedule onto the per-drawable FBOs, leaving the screen
        FBO populated. Returns the GL functions object, or None when there is
        no usable context / the GL objects failed to build (the caller
        degrades). Does NOT rebind the default framebuffer - callers finish
        by reading back (execute) or presenting (render_and_present)."""
        u = np.ascontiguousarray(u, dtype=np.uint32)
        f = np.ascontiguousarray(f, dtype=np.float32)
        uf = None if uf is None else np.ascontiguousarray(uf, dtype=np.float32)

        gl = QOpenGLContext.currentContext()
        if gl is None:
            self._log_once('no_context', 'GLExecutor: no current GL context, '
                           'nothing composed')
            return None
        gf = gl.extraFunctions()
        if not self._ensure_gl(gf):
            return None

        self._set_pipeline_state(gf)
        target_stack: list[int] = []

        for i in range(u.shape[0]):
            kind = int(u[i, _U_KIND])
            match kind:
                case n if n == _OP_BEGIN:
                    self._begin(gf, u[i], target_stack)
                case n if n == _OP_BLIT:
                    self._blit(gf, u[i], f[i], target_stack, uf)
                case n if n == _OP_COPY:
                    self._copy(gf, u[i], target_stack)
                case n if n == _OP_END:
                    self._end(gf, target_stack)
        return gf

    def _present_screen(self, gf, painter, screen, chart_rect, host_fbo) -> None:
        """Draw the screen FBO's texture onto ``host_fbo`` (the caller's GL
        target, captured at bracket entry - never assumed 0, Wayland/EGL
        reports 0 wrongly) into ``chart_rect`` (source-over). Maps the chart
        rect (logical -> device px -> NDC) and draws one textured quad. Runs
        inside the caller's native-painting bracket."""
        entry = self._programs[0]
        if entry is None:
            return
        x, y, w, h = chart_rect
        pw, ph, dpr = _target_device_size(painter)
        gf.glBindFramebuffer(GL_FRAMEBUFFER, host_fbo)
        gf.glViewport(0, 0, pw, ph)
        # The compose walk left depth/stencil/cull disabled and blend on, but
        # be explicit - the present quad is a flat source-over blit.
        gf.glDisable(GL_DEPTH_TEST)
        gf.glDisable(GL_STENCIL_TEST)
        gf.glDisable(GL_CULL_FACE)
        gf.glDisable(GL_SCISSOR_TEST)
        gf.glEnable(GL_BLEND)
        gf.glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        # Quad positions are the chart rect's DEVICE-px corners; u_mat maps
        # device px -> NDC. The whole screen texture samples across the rect
        # (uv spans 0..1). The screen FBO texture is y-up relative to the
        # content (gl_capture's v = 1 - y/h convention), so the rect's TOP
        # corner samples v=1 and the bottom v=0 - the same flip
        # gl_capture.present uses, keeping the content upright on the y-down
        # device target.
        dx0, dy0 = x * dpr, y * dpr
        dx1, dy1 = (x + w) * dpr, (y + h) * dpr
        program, locs = entry
        program.bind()
        gf.glActiveTexture(GL_TEXTURE0)
        gf.glBindTexture(GL_TEXTURE_2D, screen.texture())
        program.setUniformValue(locs['u_tex'], 0)
        gf.glUniform1f(locs['u_opacity'], 1.0)
        gf.glUniform3f(locs['u_tint'], 1.0, 1.0, 1.0)
        program.setUniformValue(locs['u_mat'], _device_to_ndc(pw, ph))
        # uv = (u0, v0, u1, v1): top corner v0=1, bottom v1=0 (the flip).
        self._draw_quad(gf, dx0, dy0, dx1, dy1, uv=(0.0, 1.0, 1.0, 0.0))
        gf.glBindTexture(GL_TEXTURE_2D, 0)
        program.release()

    # -- GL object lifecycle ----------------------------------------------

    def _ensure_gl(self, gf) -> bool:
        """Build the quad programs and dynamic quad VAO once. Returns
        False (marking broken) if a build fails - execute() then returns
        an empty screen rather than crashing."""
        if self.broken:
            return False
        if self._programs is None:
            tex = _build_program(_TEX_FRAG_SRC, ('u_mat', 'u_tex', 'u_opacity', 'u_tint'))
            fill = _build_program(_FILL_FRAG_SRC, ('u_mat', 'u_color'))
            if tex is None or fill is None:
                self._mark_broken('quad program build failed')
                return False
            self._programs = (tex, fill)
        if self._vao is None:
            self._build_quad(gf)
        return True

    def _mark_broken(self, why: str) -> None:
        self.broken = True
        logger.warning('GLExecutor disabled: %s', why)

    def _build_quad(self, gf) -> None:
        vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StreamDraw)
        vbo.allocate(_QUAD_BYTES)
        vao = QOpenGLVertexArrayObject()
        vao.create()
        vao.bind()
        stride = _FLOATS_PER_VERTEX * 4
        gf.glEnableVertexAttribArray(0)
        gf.glVertexAttribPointer(0, 2, GL_FLOAT, 0, stride, VoidPtr(0))
        gf.glEnableVertexAttribArray(1)
        gf.glVertexAttribPointer(1, 2, GL_FLOAT, 0, stride, VoidPtr(8))
        vao.release()
        vbo.release()
        self._vbo = vbo
        self._vao = vao

    def _set_pipeline_state(self, gf) -> None:
        gf.glDisable(GL_DEPTH_TEST)
        gf.glDisable(GL_STENCIL_TEST)
        gf.glDisable(GL_SCISSOR_TEST)
        gf.glDisable(GL_CULL_FACE)
        gf.glEnable(GL_BLEND)
        gf.glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

    def _target(self, drawable_id: int) -> QOpenGLFramebufferObject | None:
        """The drawable's FBO, allocated at its logical (== device) size on
        first use. None (logged once) if allocation fails - the op that
        needs it degrades, never crashes."""
        fbo = self._targets.get(drawable_id)
        if fbo is not None:
            return fbo
        w, h = self._sizes[drawable_id]
        pw = max(1, int(round(w * self._res_scale)))
        ph = max(1, int(round(h * self._res_scale)))
        fbo = QOpenGLFramebufferObject(
            pw, ph, QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        if not fbo.isValid():
            self._log_once(f'fbo_{drawable_id}',
                           f'GLExecutor: FBO for drawable {drawable_id} invalid, skipped')
            return None
        gf = QOpenGLContext.currentContext().extraFunctions()
        _set_sample_params(gf, fbo.texture())
        # A fresh FBO's contents are undefined; a drawable can be SAMPLED
        # before any BEGIN targets it (a segment with no feed items this
        # frame), so it must read as empty, not garbage.
        prev = int(gf.glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        fbo.bind()
        gf.glClearColor(0.0, 0.0, 0.0, 0.0)
        gf.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT
                   | GL_STENCIL_BUFFER_BIT)
        gf.glBindFramebuffer(GL_FRAMEBUFFER, prev)
        self._targets[drawable_id] = fbo
        return fbo

    # -- ops ---------------------------------------------------------------

    def _begin(self, gf, rec: np.ndarray, target_stack: list[int]) -> None:
        drawable_id = int(rec[_U_A])
        clear = self._clear_override.get(drawable_id, int(rec[_U_B]))
        fbo = self._target(drawable_id)
        if fbo is None:
            return
        if not fbo.bind():
            self._log_once(f'bind_{drawable_id}',
                           f'GLExecutor: FBO bind for drawable {drawable_id} failed, skipped')
            return
        w, h = fbo.width(), fbo.height()
        gf.glViewport(0, 0, w, h)
        if clear == _CLEAR_TRANSPARENT:
            gf.glClearColor(0.0, 0.0, 0.0, 0.0)
            gf.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        elif clear == _CLEAR_OPAQUE:
            gf.glClearColor(0.0, 0.0, 0.0, 1.0)
            gf.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        # Retain: leave existing content untouched (feedback / snapshots).
        target_stack.append(drawable_id)

    def _end(self, gf, target_stack: list[int]) -> None:
        if target_stack:
            target_stack.pop()
        # Rebind the enclosing target (rare - flat schedules end empty).
        if target_stack:
            fbo = self._targets.get(target_stack[-1])
            if fbo is not None:
                fbo.bind()
                gf.glViewport(0, 0, fbo.width(), fbo.height())

    def _copy(self, gf, rec: np.ndarray, target_stack: list[int]) -> None:
        into = int(rec[_U_A])
        if not target_stack:
            return
        source_id = target_stack[-1]
        source = self._targets.get(source_id)
        dest = self._target(into)
        if source is None or dest is None:
            return
        # COPY duplicates the CURRENT in-progress target (at-position
        # Snapshot): the slot freezes here, unaffected by later drawing.
        sw, sh = source.width(), source.height()
        dw, dh = dest.width(), dest.height()
        gf.glBindFramebuffer(GL_READ_FRAMEBUFFER, source.handle())
        gf.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dest.handle())
        gf.glBlitFramebuffer(0, 0, sw, sh, 0, 0, dw, dh,
                             GL_COLOR_BUFFER_BIT, GL_NEAREST)
        # Restore the open target as the bound framebuffer.
        source.bind()
        gf.glViewport(0, 0, sw, sh)

    def _blit(self, gf, urec: np.ndarray, frec: np.ndarray,
              target_stack: list[int], uf: np.ndarray | None) -> None:
        if not target_stack:
            return
        self._stash_uniforms(urec, uf)
        self._log_todo_lanes(urec)

        src_kind = int(urec[_U_A])
        opacity = _clamp01(float(frec[_F_OPACITY]))
        tint = (float(frec[_F_TINT]), float(frec[_F_TINT + 1]), float(frec[_F_TINT + 2]))
        additive = int(urec[_U_BLEND]) == _BLEND_ADDITIVE

        target_id = target_stack[-1]
        if self._targets.get(target_id) is None:
            return
        # NDC mapping runs in the target's LOGICAL units; the FBO may be
        # allocated at logical * resolution_scale pixels (the viewport
        # handles that - geometry must not).
        tw, th = self._sizes[target_id]
        mat3 = np.array(frec[:9], dtype=np.float64)

        if additive:
            gf.glBlendFunc(GL_ONE, GL_ONE)
        match src_kind:
            case n if n == _SRC_FILL:
                self._draw_fill(gf, mat3, tw, th, tint, opacity)
            case n if n == _SRC_IMAGE:
                # An image's logical box is its natural (pixel) size; the
                # item transform scales it (source-logical == pixels).
                uploaded = self._image_texture(gf, int(urec[_U_B]))
                if uploaded is not None and len(uploaded) == 3:
                    uploaded = (*uploaded, None)
                logical = None if uploaded is None else (uploaded[1], uploaded[2])
                self._draw_texture(gf, mat3, tw, th, frec, tint, opacity,
                                   uploaded, logical)
            case n if n == _SRC_DRAWABLE:
                # SOURCE NORMALIZATION (drawable-ir.md rule 5): the source
                # drawable's content covers its LOGICAL box regardless of its
                # backing FBO pixel size (a chart-rect-sized bound capture is
                # 640x480 logical). The quad spans the logical box; uv samples
                # the whole texture. Same zoom fix as the raster backend.
                src_id = int(urec[_U_B])
                uploaded = self._drawable_texture(src_id)
                lw, lh = self._sizes[src_id]
                self._draw_texture(gf, mat3, tw, th, frec, tint, opacity,
                                   uploaded, (float(lw), float(lh)))
            case n if n == _SRC_LINES:
                self._log_once('lines', 'GLExecutor: Lines source not implemented (TODO), skipped')
            case n if n == _SRC_MESH:
                self._log_once('mesh', 'GLExecutor: Mesh source not implemented (TODO), skipped')
        if additive:
            gf.glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

    # -- source draws ------------------------------------------------------

    def _image_texture(self, gf, image_id: int):
        """(texture, w, h) for an ImageId, uploaded once; None (logged
        once) if the image is missing or upload fails."""
        cached = self._image_textures.get(image_id)
        if cached is not None:
            return cached
        source = self._images.get(image_id)
        if source is None:
            self._log_once(f'img_{image_id}',
                           f'GLExecutor: missing image id {image_id}, drew nothing')
            return None
        try:
            uploaded = _upload_image(gf, source)
        except Exception as exc:  # noqa: BLE001 - one bad upload never kills the frame
            self._log_once(f'imgup_{image_id}',
                           f'GLExecutor: image {image_id} upload failed ({exc}), drew nothing')
            return None
        if uploaded is None:
            self._log_once(f'imgup_{image_id}',
                           f'GLExecutor: image {image_id} upload returned nothing, drew nothing')
            return None
        self._image_textures[image_id] = uploaded
        return uploaded

    def _drawable_texture(self, drawable_id: int):
        """(texture, w, h) for a source Drawable; None when there is nothing
        to sample. An externally bound texture (a live renderer capture)
        wins over the owned FBO, so a command-less field drawable reads the
        handed capture directly. Absent that, the owned FBO; None when the
        drawable has not been composed this run and holds no retained
        content (a feedback read of a never-drawn drawable is transparent)."""
        bound = self._bound_textures.get(drawable_id)
        if bound is not None:
            return bound
        fbo = self._targets.get(drawable_id)
        if fbo is None:
            return None
        return fbo.texture(), fbo.width(), fbo.height(), None

    def _draw_fill(self, gf, mat3, tw, th, tint, opacity) -> None:
        entry = self._programs[1]
        if entry is None:
            return
        program, locs = entry
        program.bind()
        a = opacity
        r, g, b = _clamp01(tint[0]), _clamp01(tint[1]), _clamp01(tint[2])
        # Premultiplied: color carries tint * opacity, alpha carries opacity.
        gf.glUniform4f(locs['u_color'], r * a, g * a, b * a, a)
        program.setUniformValue(locs['u_mat'], _mat_source_to_ndc(mat3, tw, th))
        # A unit quad in source space (0,0)-(1,1), matching the raster
        # backend's fillRect(0,0,1,1).
        self._draw_quad(gf, 0.0, 0.0, 1.0, 1.0, uv=None)
        program.release()

    def _draw_texture(self, gf, mat3, tw, th, frec, tint, opacity, uploaded,
                      logical) -> None:
        if uploaded is None or logical is None:
            return
        texture, sw, sh = uploaded[0], uploaded[1], uploaded[2]
        sub = uploaded[3] if len(uploaded) > 3 else None
        lw, lh = logical
        if lw <= 0.0 or lh <= 0.0 or sw <= 0 or sh <= 0:
            return
        entry = self._programs[0]
        if entry is None:
            return
        program, locs = entry
        # Crops are fractions of the source's LOGICAL box; inset the quad
        # geometry in logical units and the sampled uv in fractions of the
        # texture (the two decouple - the backing texture may be a chart-
        # rect-sized capture whose logical box is 640x480). y-down FBO
        # convention: v = 1 - fraction (gl_capture's mapping).
        crop_l = _clamp01(float(frec[_F_CROP]))
        crop_t = _clamp01(float(frec[_F_CROP + 1]))
        crop_r = _clamp01(float(frec[_F_CROP + 2]))
        crop_b = _clamp01(float(frec[_F_CROP + 3]))
        vis_fw = max(0.0, 1.0 - crop_l - crop_r)
        vis_fh = max(0.0, 1.0 - crop_t - crop_b)
        if vis_fw <= 0.0 or vis_fh <= 0.0:
            return
        x0, y0 = crop_l * lw, crop_t * lh
        x1, y1 = x0 + vis_fw * lw, y0 + vis_fh * lh
        # Sample fractions compose into the bound sub-rect (an overscanned
        # capture's design-box region); the flip to the y-up FBO texture
        # convention happens last.
        bx0, by0, bx1, by1 = sub if sub is not None else (0.0, 0.0, 1.0, 1.0)
        span_u, span_v = bx1 - bx0, by1 - by0
        u0 = bx0 + crop_l * span_u
        u1 = bx0 + (crop_l + vis_fw) * span_u
        v0 = 1.0 - (by0 + crop_t * span_v)
        v1 = 1.0 - (by0 + (crop_t + vis_fh) * span_v)

        program.bind()
        gf.glActiveTexture(GL_TEXTURE0)
        gf.glBindTexture(GL_TEXTURE_2D, texture)
        program.setUniformValue(locs['u_tex'], 0)
        gf.glUniform1f(locs['u_opacity'], opacity)
        gf.glUniform3f(locs['u_tint'], _clamp01(tint[0]), _clamp01(tint[1]), _clamp01(tint[2]))
        program.setUniformValue(locs['u_mat'], _mat_source_to_ndc(mat3, tw, th))
        self._draw_quad(gf, x0, y0, x1, y1, uv=(u0, v0, u1, v1))
        gf.glBindTexture(GL_TEXTURE_2D, 0)
        program.release()

    def _draw_quad(self, gf, x0, y0, x1, y1, uv) -> None:
        """Upload and draw one quad: positions in source logical coords,
        optional uv (fill quads carry none)."""
        if uv is None:
            u0 = v0 = u1 = v1 = 0.0
        else:
            u0, v0, u1, v1 = uv
        data = struct.pack(
            '16f',
            x0, y0, u0, v0,
            x1, y0, u1, v0,
            x0, y1, u0, v1,
            x1, y1, u1, v1)
        self._vao.bind()
        self._vbo.bind()
        self._vbo.write(0, data, len(data))
        gf.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        self._vao.release()
        self._vbo.release()

    # -- shader uniforms (parity: bound + introspectable, drawn unshaded) --

    def _stash_uniforms(self, urec: np.ndarray, uf: np.ndarray | None) -> None:
        shader_plus_one = int(urec[_U_SHADER])
        count = int(urec[_U_UF_COUNT])
        if shader_plus_one == 0 or count == 0 or uf is None:
            return
        offset = int(urec[_U_UF_OFFSET])
        if offset + count > uf.shape[0]:
            self._log_once('uf_range', 'GLExecutor: uniform window out of range, ignored')
            return
        self.shader_uniforms[shader_plus_one - 1] = [
            float(v) for v in uf[offset:offset + count]
        ]

    def _log_todo_lanes(self, urec: np.ndarray) -> None:
        if int(urec[_U_SHADER]) != 0:
            self._log_once('shader', 'GLExecutor: shader lane not implemented (TODO), drawn unshaded')
        if int(urec[_U_CLIP]) != 0:
            self._log_once('clip', 'GLExecutor: clip lane not implemented (TODO), drawn unclipped')

    # -- readback ----------------------------------------------------------

    def _screen_image(self) -> QImage:
        fbo = self._targets.get(_SCREEN_ID)
        if fbo is None:
            return self._empty_screen()
        return fbo.toImage()

    def _empty_screen(self) -> QImage:
        w, h = self._sizes[_SCREEN_ID]
        img = QImage(max(1, int(round(w))), max(1, int(round(h))),
                     QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        return img

    def _log_once(self, key: str, message: str) -> None:
        if key not in self._skipped:
            self._skipped.add(key)
            logger.warning(message)
