"""GL execution of the fullscreen shader stack.

Runs inside the QOpenGLWidget paint pass. `begin_capture` redirects
the frame's chart painting into an offscreen framebuffer (QPainter on
a `QOpenGLPaintDevice` renders into whatever FBO is bound);
`end_capture` then runs the composited shader passes over the
captured texture, intermediates ping-ponging between two FBOs and the
last pass rendering straight into the widget's backing framebuffer,
all inside a `beginNativePainting` bracket on the host painter. The
frame never leaves the GPU. The HUD is drawn by the host painter
afterwards, so it is never post-processed.

Every pass draws one fullscreen quad with the library's uniform
contract (see shaders/library/__init__.py); passes are pixel-space
maps, so no orientation flips are needed anywhere in the chain.

This is the only Qt/GL-specific piece of the shader system; sampling
lives in stack.py as pure Python. Any GL failure (no context, FBO or
shader build failure) disables the pipeline and the caller falls back
to direct, unshaded painting.
"""
from __future__ import annotations

import struct

from shiboken6 import VoidPtr

from PySide6.QtGui import QOpenGLContext, QPainter, QVector2D, QVector3D
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject,
                              QOpenGLPaintDevice, QOpenGLShader,
                              QOpenGLShaderProgram,
                              QOpenGLVertexArrayObject)

from analysis.player.render.shaders import library

# QtOpenGL exposes GL entry points but not the GL enum constants.
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE0 = 0x84C0
GL_TRIANGLE_STRIP = 0x0005
GL_FLOAT = 0x1406
GL_BLEND = 0x0BE2
GL_DEPTH_TEST = 0x0B71
GL_STENCIL_TEST = 0x0B90
GL_SCISSOR_TEST = 0x0C11
GL_CULL_FACE = 0x0B44
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_LINEAR = 0x2601
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_CLAMP_TO_EDGE = 0x812F
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_STENCIL_BUFFER_BIT = 0x00000400
GL_FRAMEBUFFER = 0x8D40
GL_READ_FRAMEBUFFER = 0x8CA8
GL_DRAW_FRAMEBUFFER = 0x8CA9
GL_FRAMEBUFFER_BINDING = 0x8CA6
GL_NEAREST = 0x2600

_VERTEX_SRC = """#version 150
in vec2 a_pos;
void main(void) { gl_Position = vec4(a_pos, 0.0, 1.0); }
"""

_UNIFORMS = ('u_tex', 'u_resolution', 'u_time', 'u_strength')

_QUAD = struct.pack('8f', -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0)


def _uniform_floats(raw):
    """The pass-dict value as a tuple of 1..4 floats, or None for shapes
    we don't drive here (extra sampler textures are deferred, see module
    notes)."""
    if isinstance(raw, (int, float)):
        return (float(raw),)
    seq = tuple(raw) if isinstance(raw, (tuple, list)) else ()
    if 2 <= len(seq) <= 4:
        return tuple(float(v) for v in seq)
    return None


def _set_custom_uniforms(f, program, uniforms) -> None:
    """Set arbitrary float/vec2/vec3/vec4 uniforms by name from the pass
    dict, so the bridge can drive registered chart shaders' own uniforms
    (their Lua `:uniform1f` pokes) from compiled channels.

    Set through glUniformNf on the GL functions, NOT
    QOpenGLShaderProgram.setUniformValue: under PySide6 a plain Python
    float argument to setUniformValue(int, ...) binds the QColor/int
    overload, not the float one, so the value silently never reaches a
    scalar uniform. Contract names are set by the caller; unknown names
    resolve to location -1 and glUniform ignores them."""
    setters = (None, f.glUniform1f, f.glUniform2f, f.glUniform3f,
               f.glUniform4f)
    for name, raw in uniforms.items():
        if name in _UNIFORMS:
            continue
        values = _uniform_floats(raw)
        if values is not None:
            setters[len(values)](program.uniformLocation(name), *values)


def _adapt_dialect(src: str) -> str:
    """Library sources are desktop GLSL 1.50; ES contexts (Qt picks
    OpenGL ES under Wayland/EGL and ANGLE) reject that header, so swap
    it for the equivalent ES dialect. The body syntax is compatible."""
    if not QOpenGLContext.currentContext().isOpenGLES():
        return src
    return src.replace('#version 150',
                       '#version 300 es\nprecision highp float;', 1)


class ShaderGLPipeline:
    def __init__(self):
        self._programs = {}          # name -> (program, locs) | None
        self._fbos = [None, None]
        self._size = (0, 0)
        self._vao = None
        self._vbo = None
        self._capture_painter = None
        # The paint device must outlive its painter (QPainter holds a
        # bare pointer), so the pipeline keeps the reference.
        self._capture_device = None
        self._host_painter = None
        self._host_fbo = 0
        self._context = None
        self._broken = False

    def begin_capture(self, host_painter, w, h) -> QPainter | None:
        """Redirect chart painting into the capture FBO. Returns the
        capture painter, or None when GL isn't usable (raster host,
        headless test, pre-GL3 context, earlier failure) so the caller
        paints direct."""
        glctx = QOpenGLContext.currentContext()
        if (self._broken or glctx is None
                or glctx.format().majorVersion() < 3):
            return None
        if glctx is not self._context:
            # New/recreated context (first frame, widget reparented):
            # cached GL objects belong to the old one, start over.
            self._programs = {}
            self._fbos = [None, None]
            self._size = (0, 0)
            self._vao = None
            self._vbo = None
            self._context = glctx
        dpr = float(host_painter.device().devicePixelRatioF())
        pw = max(1, int(w * dpr))
        ph = max(1, int(h * dpr))

        host_painter.beginNativePainting()
        # The host's render target, read back rather than assumed:
        # inside a QOpenGLWidget paint this is the widget's backing
        # FBO, which `defaultFramebufferObject()` does NOT report on
        # every platform (Wayland/EGL returns the window surface's 0).
        f = QOpenGLContext.currentContext().extraFunctions()
        self._host_fbo = int(f.glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        painter = self._begin_fbo_painter(pw, ph, dpr)
        if painter is None:
            self._mark_broken('capture framebuffer/painter setup failed')
            f.glBindFramebuffer(GL_FRAMEBUFFER, self._host_fbo)
            host_painter.endNativePainting()
            return None

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self._capture_painter = painter
        self._host_painter = host_painter
        return painter

    def end_capture(self, passes, t_now: float) -> None:
        """End the capture painter and run `passes` over the captured
        frame; the last pass renders straight into the host
        framebuffer, so the whole chain stays on the GPU with no extra
        copies. Pairs with a successful `begin_capture`."""
        self._capture_painter.end()
        self._capture_painter = None
        self._capture_device = None
        f = QOpenGLContext.currentContext().extraFunctions()
        try:
            if not self._run_passes(f, passes, t_now):
                # No runnable pass: the capture already holds the
                # final image; one GPU-side blit moves it across.
                pw, ph = self._size
                f.glBindFramebuffer(GL_READ_FRAMEBUFFER,
                                    self._fbos[0].handle())
                f.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._host_fbo)
                f.glBlitFramebuffer(0, 0, pw, ph, 0, 0, pw, ph,
                                    GL_COLOR_BUFFER_BIT, GL_NEAREST)
        finally:
            f.glBindTexture(GL_TEXTURE_2D, 0)
            f.glBindFramebuffer(GL_FRAMEBUFFER, self._host_fbo)
            self._host_painter.endNativePainting()
            self._host_painter = None

    def _mark_broken(self, why: str) -> None:
        self._broken = True
        print(f'shader pipeline disabled: {why}')

    def _ensure_fbo(self, f, index):
        """FBO `index` at the current size, created on first use so
        the ping-pong buffer only exists once a map actually chains
        two shaders."""
        if self._fbos[index] is not None:
            return self._fbos[index]
        pw, ph = self._size
        fbo = QOpenGLFramebufferObject(
            pw, ph,
            QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        if not fbo.isValid():
            return None
        # Shaders sample at arbitrary UVs (fisheye, glitch); linear +
        # clamp avoids blockiness and edge wraparound.
        f.glBindTexture(GL_TEXTURE_2D, fbo.texture())
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        f.glBindTexture(GL_TEXTURE_2D, 0)
        self._fbos[index] = fbo
        return fbo

    def _begin_fbo_painter(self, pw, ph, dpr) -> QPainter | None:
        f = QOpenGLContext.currentContext().extraFunctions()
        if self._size != (pw, ph):
            # Dropping the old FBOs while the context is current lets
            # their GL resources delete cleanly.
            self._fbos = [None, None]
            self._size = (pw, ph)

        capture = self._ensure_fbo(f, 0)
        if capture is None or not capture.bind():
            return None
        f.glViewport(0, 0, pw, ph)
        f.glClearColor(0.0, 0.0, 0.0, 0.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT
                  | GL_STENCIL_BUFFER_BIT)

        device = QOpenGLPaintDevice(pw, ph)
        device.setDevicePixelRatio(dpr)
        self._capture_device = device
        painter = QPainter(device)
        return painter if painter.isActive() else None

    def _run_passes(self, f, passes, t_now) -> bool:
        """Run the runnable passes over the capture: intermediates
        ping-pong between the two FBOs, the last pass renders into the
        host framebuffer. Returns False when nothing was runnable (the
        capture FBO still holds the frame)."""
        runnable = []
        for name, uniforms in passes:
            entry = self._program(name)
            if entry is not None:
                runnable.append((entry, uniforms))
        if not runnable:
            return False

        # Chaining needs the ping-pong buffer; if it can't be created,
        # degrade to the last pass alone rather than dropping the frame.
        if len(runnable) > 1 and self._ensure_fbo(f, 1) is None:
            runnable = runnable[-1:]

        pw, ph = self._size
        f.glDisable(GL_BLEND)
        f.glDisable(GL_DEPTH_TEST)
        f.glDisable(GL_STENCIL_TEST)
        f.glDisable(GL_SCISSOR_TEST)
        f.glDisable(GL_CULL_FACE)
        f.glActiveTexture(GL_TEXTURE0)
        self._bind_quad(f)

        last = len(runnable) - 1
        src = 0
        for i, ((program, locs), uniforms) in enumerate(runnable):
            if i == last:
                f.glBindFramebuffer(GL_FRAMEBUFFER, self._host_fbo)
            else:
                self._fbos[1 - src].bind()
            f.glViewport(0, 0, pw, ph)
            f.glBindTexture(GL_TEXTURE_2D, self._fbos[src].texture())
            program.bind()
            program.setUniformValue(locs['u_tex'], 0)
            program.setUniformValue(locs['u_resolution'],
                                    QVector2D(pw, ph))
            program.setUniformValue(
                locs['u_strength'],
                QVector3D(*uniforms.get('u_strength', (0.0, 0.0, 0.0))))
            # u_time is a scalar float: set it through glUniform1f, not
            # setUniformValue -- a Python float there binds PySide6's
            # QColor/int overload and never reaches the uniform (time-
            # animated shaders like noise/glitch would otherwise freeze).
            f.glUniform1f(locs['u_time'], float(t_now))
            _set_custom_uniforms(f, program, uniforms)
            f.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            program.release()
            src = 1 - src

        self._vao.release()
        return True

    def _bind_quad(self, f):
        if self._vao is not None:
            self._vao.bind()
            return
        vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.allocate(_QUAD, len(_QUAD))
        vao = QOpenGLVertexArrayObject()
        vao.create()
        vao.bind()
        # Attribute 0 is `a_pos` in every program (bound pre-link), so
        # one VAO serves the whole library.
        f.glEnableVertexAttribArray(0)
        f.glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, VoidPtr(0))
        self._vbo = vbo
        self._vao = vao

    def _program(self, name):
        """Build (once) and return `(program, uniform_locations)` for
        `name`, or None when the shader is unknown or failed to build.
        Failures are cached so each warns once."""
        if name in self._programs:
            return self._programs[name]

        entry = None
        src = library.source(name)
        if src is None:
            print(f'unknown shader {name!r}; pass skipped')
        else:
            program = QOpenGLShaderProgram()
            built = (program.addShaderFromSourceCode(
                         QOpenGLShader.ShaderTypeBit.Vertex,
                         _adapt_dialect(_VERTEX_SRC))
                     and program.addShaderFromSourceCode(
                         QOpenGLShader.ShaderTypeBit.Fragment,
                         _adapt_dialect(src)))
            program.bindAttributeLocation('a_pos', 0)
            if built and program.link():
                entry = (program,
                         {u: program.uniformLocation(u) for u in _UNIFORMS})
            else:
                print(f'shader {name!r} failed to build: {program.log()}')

        self._programs[name] = entry
        return entry
