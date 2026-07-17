"""FBO-backed capture backend for the field-instance composite.

GL twin of capture.RasterCaptureBackend, used when the host painter
renders on a GL 3+ context (the QOpenGLWidget canvas). Capture slots
become framebuffer objects painted through QPainter-on-
QOpenGLPaintDevice (chart content rendering is unchanged); instance
blits become textured quads; node-point snapshots become
glBlitFramebuffer into retained textures. This removes the CPU raster
blits and per-frame texture uploads that dominated dense NotITG frames
(proxy walls blit the window-sized field capture a dozen times per
frame), and it matches the engine's own structure: ActorFrameTexture
IS a render-to-texture framebuffer.

Conventions shared with shaders/gl_pipeline.py (the traps are solved
there - keep them in lockstep):
- The enclosing render target is read back from GL_FRAMEBUFFER_BINDING
  inside a `beginNativePainting` bracket, never assumed from
  `defaultFramebufferObject()` (Wayland/EGL reports 0 there).
- Shader sources are desktop GLSL 1.50 adapted to the ES dialect when
  the context is OpenGL ES.
- Qt paints y-down into FBOs everywhere (the widget's backing FBO
  included; the real window flip happens in Qt's compositing), so
  FBO-to-FBO blits need no orientation flips. The quad path maps
  logical top-left to NDC (-1, +1) and samples the source at
  v = 1 - y/h, both following that one convention.

Instance transforms are full projective QTransforms (NotITG 3D mods
sample to homographies), so the quad program carries the source->NDC
map as a mat3 and forwards the homogeneous w through gl_Position for
perspective-correct interpolation - the GPU form of QPainter's
projective drawPixmap.

Any GL failure marks the backend broken; the renderer selects the
raster backend from the next frame (`usable` returns False).
"""
from __future__ import annotations

import struct

from shiboken6 import VoidPtr

import numpy as np

from PySide6.QtCore import QRectF
from PySide6.QtGui import (QMatrix3x3, QOpenGLContext, QPainter,
                           QPaintEngine)
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject,
                              QOpenGLPaintDevice, QOpenGLShader,
                              QOpenGLShaderProgram,
                              QOpenGLVertexArrayObject)

from analysis.player.render.shaders.gl_pipeline import (
    _adapt_dialect, GL_BLEND, GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_FLOAT,
    GL_FRAMEBUFFER, GL_FRAMEBUFFER_BINDING, GL_DRAW_FRAMEBUFFER,
    GL_LINEAR, GL_NEAREST, GL_READ_FRAMEBUFFER, GL_SCISSOR_TEST,
    GL_STENCIL_BUFFER_BIT, GL_STENCIL_TEST, GL_TEXTURE0, GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_TRIANGLE_STRIP)

GL_ONE = 1
GL_ONE_MINUS_SRC_ALPHA = 0x0303

# Capture content is premultiplied (Qt's GL paint engine renders
# premultiplied ARGB), so instance opacity multiplies every channel and
# source-over is (ONE, ONE_MINUS_SRC_ALPHA).
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

_TEX_FRAG_SRC = """#version 150
uniform sampler2D u_tex;
uniform float u_opacity;
in vec2 v_uv;
out vec4 fragColor;
void main(void) { fragColor = texture(u_tex, v_uv) * u_opacity; }
"""

_FILL_FRAG_SRC = """#version 150
uniform vec4 u_color;
out vec4 fragColor;
void main(void) { fragColor = u_color; }
"""

_FLOATS_PER_VERTEX = 4
_QUAD_BYTES = 4 * _FLOATS_PER_VERTEX * 4


def usable(painter) -> bool:
    """Whether `painter` can host the GL backend this frame: an active
    GL 3+ context with the painter on a GL paint engine (the
    QOpenGLWidget canvas; raster hosts and headless tests fall out
    here)."""
    glctx = QOpenGLContext.currentContext()
    if glctx is None or glctx.format().majorVersion() < 3:
        return False
    engine = painter.paintEngine() if isinstance(painter, QPainter) else None
    return (engine is not None
            and engine.type() == QPaintEngine.Type.OpenGL2)


class _GLHandle:
    """A drawable capture: an FBO's texture plus the geometry needed to
    blit it. Slot handles (refs None) alias their slot's FBO and are
    valid until the slot reopens; snapshot handles are refcounted and
    recycled through the backend's freelist. `gen` invalidates handles
    that outlive their GL context."""

    __slots__ = ('fbo', 'w', 'h', 'dpr', 'gen', 'refs')

    def __init__(self, fbo, w, h, dpr, gen, refs=None):
        self.fbo = fbo
        self.w = w
        self.h = h
        self.dpr = dpr
        self.gen = gen
        self.refs = refs


class _Slot:
    """One capture slot's persistent FBO plus its per-open painting
    state (the paint device must outlive its painter - QPainter holds a
    bare pointer)."""

    __slots__ = ('fbo', 'pw', 'ph', 'w', 'h', 'dpr', 'device', 'painter',
                 'host_painter', 'prev_fbo')

    def __init__(self):
        self.fbo = None
        self.pw = 0
        self.ph = 0
        self.w = 0
        self.h = 0
        self.dpr = 1.0
        self.device = None
        self.painter = None
        self.host_painter = None
        self.prev_fbo = 0


def _target_device_size(painter):
    """(device-px width, height, dpr) of the painter's render target.
    QOpenGLPaintDevice reports device pixels from its metrics; widget
    devices report logical pixels."""
    dev = painter.device()
    dpr = float(dev.devicePixelRatioF())
    if isinstance(dev, QOpenGLPaintDevice):
        size = dev.size()
        return size.width(), size.height(), dpr
    return int(dev.width() * dpr), int(dev.height() * dpr), dpr


class GLCaptureBackend:
    """FBO capture slots + textured-quad blits (the GL path)."""

    def __init__(self):
        self._slots: dict[str, _Slot] = {}
        self._freelist: list[_GLHandle] = []
        self._programs = None       # (tex_entry, fill_entry) or None
        self._vao = None
        self._vbo = None
        self._context = None
        self._gen = 0
        self._batch = None
        self.broken = False

    # -- context lifecycle -------------------------------------------------

    def _sync_context(self, f) -> None:
        """Reset every cached GL object when the context changed (first
        frame, widget reparented): they belong to the old context, and
        handles the renderer retained from it must stop drawing (their
        generation goes stale; blits skip them until the retention
        machinery re-primes)."""
        glctx = QOpenGLContext.currentContext()
        if glctx is self._context:
            return
        self._slots = {}
        self._freelist = []
        self._programs = None
        self._vao = None
        self._vbo = None
        self._context = glctx
        self._gen += 1

    def _mark_broken(self, why: str) -> None:
        self.broken = True
        print(f'GL capture backend disabled: {why}')

    # -- slot painting -----------------------------------------------------

    def open(self, slot: str, host_painter, w: int, h: int) -> QPainter:
        """An active painter into the named slot's FBO, cleared
        transparent. Brackets the enclosing painter with
        `beginNativePainting` (nesting is fine - the screen slot's own
        painter encloses the field slots), so `close` must run before
        the enclosing painter paints again."""
        f = QOpenGLContext.currentContext().extraFunctions()
        self._sync_context(f)
        state = self._slots.setdefault(slot, _Slot())
        dpr = float(host_painter.device().devicePixelRatioF())
        pw = max(1, int(w * dpr))
        ph = max(1, int(h * dpr))

        host_painter.beginNativePainting()
        state.prev_fbo = int(f.glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        fbo = self._ensure_slot_fbo(f, state, pw, ph)
        if fbo is None or not fbo.bind():
            self._mark_broken(f'slot {slot!r} framebuffer failed')
            f.glBindFramebuffer(GL_FRAMEBUFFER, state.prev_fbo)
            host_painter.endNativePainting()
            return self._raster_fallback().open(slot, host_painter, w, h)

        f.glDisable(GL_SCISSOR_TEST)
        f.glViewport(0, 0, pw, ph)
        f.glClearColor(0.0, 0.0, 0.0, 0.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT
                  | GL_STENCIL_BUFFER_BIT)

        device = QOpenGLPaintDevice(pw, ph)
        device.setDevicePixelRatio(dpr)
        painter = QPainter(device)
        if not painter.isActive():
            self._mark_broken(f'slot {slot!r} painter failed')
            f.glBindFramebuffer(GL_FRAMEBUFFER, state.prev_fbo)
            host_painter.endNativePainting()
            return self._raster_fallback().open(slot, host_painter, w, h)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        state.w, state.h, state.dpr = w, h, dpr
        state.device = device
        state.painter = painter
        state.host_painter = host_painter
        return painter

    def close(self, slot: str):
        """End the slot's painter and return to the enclosing painter's
        target; returns the slot's drawable handle, valid until the
        slot is next opened."""
        state = self._slots.get(slot)
        if state is None or state.painter is None:
            return self._raster_fallback().close(slot)
        state.painter.end()
        state.painter = None
        state.device = None
        f = QOpenGLContext.currentContext().extraFunctions()
        f.glBindFramebuffer(GL_FRAMEBUFFER, state.prev_fbo)
        state.host_painter.endNativePainting()
        state.host_painter = None
        return _GLHandle(state.fbo, state.w, state.h, state.dpr, self._gen)

    def _ensure_slot_fbo(self, f, state, pw, ph):
        """The slot's FBO at the current size. Combined depth+stencil so
        QPainter clipping works while the slot is painted."""
        if state.fbo is not None and (state.pw, state.ph) == (pw, ph):
            return state.fbo
        state.fbo = None
        fbo = QOpenGLFramebufferObject(
            pw, ph,
            QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        if not fbo.isValid():
            return None
        _set_sample_params(f, fbo.texture())
        state.fbo = fbo
        state.pw, state.ph = pw, ph
        return fbo

    def _raster_fallback(self):
        """A raster backend standing in after a GL failure, so the
        frame that hit the failure still completes; `usable`/`broken`
        route the next frame to raster entirely."""
        from analysis.player.render.capture import RasterCaptureBackend
        if not isinstance(getattr(self, '_fallback', None),
                          RasterCaptureBackend):
            self._fallback = RasterCaptureBackend()
        return self._fallback

    # -- snapshots ---------------------------------------------------------

    def snapshot(self, slot: str):
        """The slot's current pixels blitted into a retained texture
        (recycled through the freelist). Must run inside an open blits
        batch targeting the slot - the batch's native bracket has
        flushed the slot painter, so the FBO holds every draw so far
        (the AFT node's mid-composite capture point)."""
        state = self._slots.get(slot)
        if state is None or state.fbo is None:
            return self._raster_fallback().snapshot(slot)
        f = QOpenGLContext.currentContext().extraFunctions()
        handle = self._fresh_snapshot(f, state)
        if handle is None:
            return None
        f.glBindFramebuffer(GL_READ_FRAMEBUFFER, state.fbo.handle())
        f.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle.fbo.handle())
        f.glBlitFramebuffer(0, 0, state.pw, state.ph,
                            0, 0, state.pw, state.ph,
                            GL_COLOR_BUFFER_BIT, GL_NEAREST)
        if self._batch is not None:
            f.glBindFramebuffer(GL_FRAMEBUFFER, self._batch.target_fbo)
        return handle

    def _fresh_snapshot(self, f, state):
        """A refcounted snapshot handle sized like `state`, reusing a
        released texture when one matches."""
        for i, cand in enumerate(self._freelist):
            if (cand.fbo.width(), cand.fbo.height()) == (state.pw, state.ph):
                handle = self._freelist.pop(i)
                handle.w, handle.h, handle.dpr = state.w, state.h, state.dpr
                handle.refs = 1
                return handle
        fbo = QOpenGLFramebufferObject(state.pw, state.ph)
        if not fbo.isValid():
            return None
        _set_sample_params(f, fbo.texture())
        return _GLHandle(fbo, state.w, state.h, state.dpr, self._gen, refs=1)

    def retain(self, handle):
        if isinstance(handle, _GLHandle) and handle.refs is not None:
            handle.refs += 1
        return handle

    def release(self, handle) -> None:
        if not isinstance(handle, _GLHandle) or handle.refs is None:
            return
        handle.refs -= 1
        if handle.refs == 0 and handle.gen == self._gen:
            self._freelist.append(handle)

    # -- blits -------------------------------------------------------------

    def blit(self, painter, handle, clip, transform=None, src_box=None,
             opacity=1.0) -> None:
        with self.blits(painter, clip) as batch:
            batch.blit(handle, transform=transform, src_box=src_box,
                       opacity=opacity)

    def blits(self, painter, clip):
        return _GLBlits(self, painter, clip)

    def present(self, painter, slot: str) -> None:
        """Composite the (closed) slot over `painter`'s target: a
        full-surface source-over quad, matching the raster
        drawPixmap-at-origin hand-off."""
        state = self._slots.get(slot)
        if state is None or state.fbo is None:
            self._raster_fallback().present(painter, slot)
            return
        handle = _GLHandle(state.fbo, state.w, state.h, state.dpr, self._gen)
        self.blit(painter, handle, QRectF(0, 0, state.w, state.h))

    # -- GL objects shared by the quad path --------------------------------

    def _quad_programs(self):
        """(textured, fill) program entries, built once per context;
        None when a build failed (backend marked broken)."""
        if self._programs is not None:
            return self._programs
        tex = _build_program(_TEX_FRAG_SRC, ('u_mat', 'u_tex', 'u_opacity'))
        fill = _build_program(_FILL_FRAG_SRC, ('u_mat', 'u_color'))
        if tex is None or fill is None:
            self._mark_broken('quad program build failed')
            return None
        self._programs = (tex, fill)
        return self._programs

    def _bind_quad(self, f) -> None:
        """Bind the shared dynamic quad VAO (interleaved pos+uv, one
        4-vertex strip re-uploaded per blit - 64 bytes)."""
        if self._vao is not None:
            self._vao.bind()
            self._vbo.bind()
            return
        vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StreamDraw)
        vbo.allocate(_QUAD_BYTES)
        vao = QOpenGLVertexArrayObject()
        vao.create()
        vao.bind()
        stride = _FLOATS_PER_VERTEX * 4
        f.glEnableVertexAttribArray(0)
        f.glVertexAttribPointer(0, 2, GL_FLOAT, 0, stride, VoidPtr(0))
        f.glEnableVertexAttribArray(1)
        f.glVertexAttribPointer(1, 2, GL_FLOAT, 0, stride, VoidPtr(8))
        self._vbo = vbo
        self._vao = vao


def _set_sample_params(f, texture) -> None:
    """Linear + clamp on a capture texture: instances sample under
    arbitrary homographies, and the engine's render-to-texture sprites
    filter linearly."""
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
        print(f'capture quad program failed to build: {program.log()}')
        return None
    return program, {u: program.uniformLocation(u) for u in uniforms}


def _source_to_ndc(transform, dpr, pw, ph) -> QMatrix3x3:
    """The mat3 mapping source logical coords through the instance
    transform to target NDC (w in the third component): T then
    logical->device (dpr) then device->NDC with Qt's top-left at
    (-1, +1). Column-vector convention, QTransform's row-vector grid
    transposed in."""
    if transform is None:
        t_cv = np.eye(3)
    else:
        t_cv = np.array([
            [transform.m11(), transform.m21(), transform.m31()],
            [transform.m12(), transform.m22(), transform.m32()],
            [transform.m13(), transform.m23(), transform.m33()],
        ])
    ndc = np.array([[2.0 * dpr / pw, 0.0, -1.0],
                    [0.0, -2.0 * dpr / ph, 1.0],
                    [0.0, 0.0, 1.0]])
    m = ndc @ t_cv
    return QMatrix3x3([float(v) for v in m.flatten()])


class _GLBlits:
    """One batch of instance blits inside a single native-painting
    bracket on the target painter: state set once, one quad draw per
    blit/fill, snapshots legal mid-batch."""

    def __init__(self, backend, painter, clip):
        self._backend = backend
        self._painter = painter
        self._clip = clip
        self.target_fbo = 0

    def __enter__(self):
        backend = self._backend
        painter = self._painter
        painter.beginNativePainting()
        f = QOpenGLContext.currentContext().extraFunctions()
        backend._sync_context(f)
        self._f = f
        self.target_fbo = int(f.glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        self._pw, self._ph, self._dpr = _target_device_size(painter)

        f.glDisable(GL_DEPTH_TEST)
        f.glDisable(GL_STENCIL_TEST)
        f.glDisable(GL_CULL_FACE)
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        f.glViewport(0, 0, self._pw, self._ph)
        self._apply_scissor(f)
        backend._bind_quad(f)
        backend._batch = self
        return self

    def __exit__(self, *exc):
        backend = self._backend
        f = self._f
        backend._batch = None
        if backend._vao is not None:
            backend._vao.release()
        f.glDisable(GL_SCISSOR_TEST)
        f.glBindTexture(GL_TEXTURE_2D, 0)
        f.glBindFramebuffer(GL_FRAMEBUFFER, self.target_fbo)
        self._painter.endNativePainting()
        return False

    def _apply_scissor(self, f) -> None:
        """Clip in target space: the clip rect is axis-aligned (the
        chart region), so a scissor covers it exactly."""
        clip = self._clip
        x0 = int(round(clip.x() * self._dpr))
        y1 = int(round((clip.y() + clip.height()) * self._dpr))
        w = int(round(clip.width() * self._dpr))
        h = int(round(clip.height() * self._dpr))
        f.glEnable(GL_SCISSOR_TEST)
        f.glScissor(x0, self._ph - y1, w, h)

    def blit(self, handle, transform=None, src_box=None,
             opacity=1.0) -> None:
        backend = self._backend
        if not isinstance(handle, _GLHandle) or handle.gen != backend._gen:
            # A handle from a lost context or a raster-fallback frame:
            # skip the draw; the retention machinery re-primes within a
            # frame once the backends settle.
            return
        programs = backend._quad_programs()
        if programs is None:
            return
        program, locs = programs[0]
        f = self._f
        program.bind()
        f.glActiveTexture(GL_TEXTURE0)
        f.glBindTexture(GL_TEXTURE_2D, handle.fbo.texture())
        program.setUniformValue(locs['u_tex'], 0)
        f.glUniform1f(locs['u_opacity'], min(1.0, float(opacity)))
        self._draw_quad(f, program, locs, transform,
                        src_box or QRectF(0, 0, handle.w, handle.h), handle)
        program.release()

    def fill(self, rgb, opacity) -> None:
        """The curtain quad: a flat premultiplied fill covering the
        clip rect at its position among the blits."""
        backend = self._backend
        programs = backend._quad_programs()
        if programs is None:
            return
        program, locs = programs[1]
        f = self._f
        program.bind()
        a = min(1.0, float(opacity))
        r, g, b = rgb
        f.glUniform4f(locs['u_color'], r * a, g * a, b * a, a)
        self._draw_quad(f, program, locs, None, self._clip, None)
        program.release()

    def _draw_quad(self, f, program, locs, transform, box, handle) -> None:
        """Upload the quad for `box` (source logical coords; uv from
        the handle's texture when texturing) and draw it under the
        source->NDC matrix."""
        program.setUniformValue(
            locs['u_mat'],
            _source_to_ndc(transform, self._dpr, self._pw, self._ph))
        x0, y0 = box.x(), box.y()
        x1, y1 = x0 + box.width(), y0 + box.height()
        if handle is not None:
            u0, u1 = x0 / handle.w, x1 / handle.w
            v0, v1 = 1.0 - y0 / handle.h, 1.0 - y1 / handle.h
        else:
            u0 = u1 = v0 = v1 = 0.0
        data = struct.pack(
            '16f',
            x0, y0, u0, v0,
            x1, y0, u1, v0,
            x0, y1, u0, v1,
            x1, y1, u1, v1)
        backend = self._backend
        backend._vbo.write(0, data, len(data))
        f.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
