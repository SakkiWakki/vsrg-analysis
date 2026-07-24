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

from PySide6.QtGui import (QImage, QOpenGLContext, QPainter, QVector2D,
                           QVector3D)
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject,
                              QOpenGLPaintDevice, QOpenGLShader,
                              QOpenGLShaderProgram, QOpenGLTexture,
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

# Second sampler for two-input passes (bloom's compose reads the original
# pre-chain frame alongside the blurred glow). Tracked apart from the
# required contract so single-input shaders need not declare it; bound to
# texture unit 1 only when a program actually resolves the location.
_SECOND_SAMPLER = 'u_tex2'
_CAPTURE_UNIT = 1

# File-backed sampler binds (a chart's uniformTexture pokes, registered
# with the shader in the library) occupy units from here up: unit 0 is
# the pass source, unit 1 the u_tex2 capture.
_FILE_SAMPLER_UNIT0 = 2

# Multi-pass shaders whose single stack id fans out to several library
# frags. Bloom mirrors fluXis: separable gaussian blur (horizontal then
# vertical) into a compose that adds the glow onto the original frame.
_EXPANSIONS = {
    'bloom': ('bloom_blur_h', 'bloom_blur_v', 'bloom_compose'),
}


def _expand(passes):
    """Fan multi-pass shader ids out to their sub-pass frags, carrying the
    same uniforms to each; single-pass ids pass through unchanged."""
    for name, uniforms in passes:
        for sub in _EXPANSIONS.get(name, (name,)):
            yield sub, uniforms


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
        self._programs = {}   # name -> (program, locs, file_samplers) | None
        # Slot 0 is the capture (read-only during passes so two-input
        # passes can re-read the original frame); slots 1 and 2 ping-pong
        # the intermediates without ever clobbering the capture.
        self._fbos = [None, None, None]
        # An externally-captured frame standing in for slot 0 while
        # `run_over` executes (the GL capture backend's post slot).
        self._external = None
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
        # File-backed sampler textures (a chart's uniformTexture binds):
        # path -> QOpenGLTexture | None (None = unreadable, warned once).
        self._file_textures = {}

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
            self._fbos = [None, None, None]
            self._size = (0, 0)
            self._vao = None
            self._vbo = None
            self._file_textures = {}
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

    def abort_capture(self) -> None:
        """Unwind an interrupted begin_capture (mid-frame exception
        between begin and end): end the capture painter, restore the
        host framebuffer, and close the native bracket, so the next
        frame starts from clean paint state."""
        if self._capture_painter is None:
            return
        if self._capture_painter.isActive():
            self._capture_painter.end()
        self._capture_painter = None
        self._capture_device = None
        glctx = QOpenGLContext.currentContext()
        if glctx is not None:
            glctx.extraFunctions().glBindFramebuffer(
                GL_FRAMEBUFFER, self._host_fbo)
        if self._host_painter is not None:
            self._host_painter.endNativePainting()
            self._host_painter = None

    def run_over(self, host_painter, capture_fbo, passes, t_now: float,
                 pw: int, ph: int) -> None:
        """Run `passes` over an externally-captured frame (the GL
        capture backend's post slot), the last pass rendering into
        `host_painter`'s target - the unified chain's shader stage,
        replacing the begin/end capture pair when captures already live
        on FBOs. When nothing is runnable the capture is blitted across
        unshaded."""
        host_painter.beginNativePainting()
        f = QOpenGLContext.currentContext().extraFunctions()
        self._host_fbo = int(f.glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        if self._size != (pw, ph):
            self._fbos = [None, None, None]
            self._size = (pw, ph)
        self._external = capture_fbo
        try:
            if not self._run_passes(f, passes, t_now):
                f.glBindFramebuffer(GL_READ_FRAMEBUFFER,
                                    capture_fbo.handle())
                f.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._host_fbo)
                f.glBlitFramebuffer(0, 0, pw, ph, 0, 0, pw, ph,
                                    GL_COLOR_BUFFER_BIT, GL_NEAREST)
        finally:
            self._external = None
            f.glBindTexture(GL_TEXTURE_2D, 0)
            f.glBindFramebuffer(GL_FRAMEBUFFER, self._host_fbo)
            host_painter.endNativePainting()

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
            self._fbos = [None, None, None]
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
        """Run the runnable passes over the capture: the capture stays in
        slot 0 (read-only, so two-input passes can re-read the original
        frame), intermediates ping-pong between slots 1 and 2, and the
        last pass renders into the host framebuffer. Returns False when
        nothing was runnable (the capture FBO still holds the frame)."""
        runnable = self._prepare_runnable(f, passes)
        if not runnable:
            return False

        pw, ph = self._size
        f.glDisable(GL_BLEND)
        f.glDisable(GL_DEPTH_TEST)
        f.glDisable(GL_STENCIL_TEST)
        f.glDisable(GL_SCISSOR_TEST)
        f.glDisable(GL_CULL_FACE)
        self._bind_quad(f)

        last = len(runnable) - 1
        # Slot 0 is the capture (first pass' input); intermediates land in
        # slots 1 and 2 alternately, leaving slot 0 intact for u_tex2.
        src = 0
        dst = 1
        for i, (entry, uniforms) in enumerate(runnable):
            if i == last:
                f.glBindFramebuffer(GL_FRAMEBUFFER, self._host_fbo)
            else:
                self._fbos[dst].bind()
            self._draw_pass(f, entry, uniforms, src, t_now)
            # Ping-pong between slots 1 and 2 (which sum to 3), never
            # slot 0: the source just written becomes next input, its
            # partner becomes the next target.
            src, dst = dst, 3 - dst

        self._vao.release()
        # Unbind the quad VBO from GL_ARRAY_BUFFER: Qt's paint engine
        # may draw with client-side vertex arrays (compatibility
        # contexts), and a foreign buffer left bound corrupts its
        # vertex pointers when the host painter resumes.
        self._vbo.release()
        return True

    def _prepare_runnable(self, f, passes):
        """The `(entry, uniforms)` list to draw: multi-pass ids expanded,
        unknown/failed shaders dropped, and -- when the ping-pong slots
        can't be created for a chain -- degraded to the last pass alone
        rather than dropping the frame."""
        runnable = [(entry, uniforms)
                    for name, uniforms in _expand(passes)
                    if (entry := self._program(name)) is not None]
        degrade = len(runnable) > 1 and not self._pingpong_ready(f)
        return runnable[-1:] if degrade else runnable

    def _pingpong_ready(self, f) -> bool:
        """Both intermediate ping-pong slots exist (slot 0 is the capture)."""
        return (self._ensure_fbo(f, 1) is not None
                and self._ensure_fbo(f, 2) is not None)

    def _draw_pass(self, f, entry, uniforms, src, t_now) -> None:
        """Bind `entry`'s program and its contract uniforms, then draw the
        fullscreen quad. The target FBO and viewport are set by the caller."""
        program, locs, file_samplers = entry
        pw, ph = self._size
        f.glViewport(0, 0, pw, ph)
        program.bind()
        self._bind_file_samplers(f, file_samplers)
        self._bind_pass_textures(f, locs, src)
        program.setUniformValue(locs['u_tex'], 0)
        program.setUniformValue(locs['u_resolution'], QVector2D(pw, ph))
        program.setUniformValue(
            locs['u_strength'],
            QVector3D(*uniforms.get('u_strength', (0.0, 0.0, 0.0))))
        # u_time is a scalar float: set it through glUniform1f, not
        # setUniformValue -- a Python float there binds PySide6's QColor/int
        # overload and never reaches the uniform (time-animated shaders like
        # noise/glitch would otherwise freeze).
        f.glUniform1f(locs['u_time'], float(t_now))
        _set_custom_uniforms(f, program, uniforms)
        f.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        program.release()

    def _bind_file_samplers(self, f, file_samplers) -> None:
        """Bind each of the pass's file-backed sampler textures (chart
        uniformTexture binds registered with the shader) to its own unit
        above the contract units. Unit 0 is re-activated by the pass
        source bind that follows."""
        for unit, (loc, path) in enumerate(file_samplers,
                                           start=_FILE_SAMPLER_UNIT0):
            texture = self._file_texture(path)
            if texture is None:
                continue
            f.glActiveTexture(GL_TEXTURE0 + unit)
            texture.bind()
            f.glUniform1i(loc, unit)

    def _file_texture(self, path):
        """The uploaded texture for `path`, cached; None (warned once)
        when the image is unreadable. Repeat wrap: SM charts scroll and
        tile bound atlases (texturewrapping is the engine's default idiom
        for these), and in-range UVs are unaffected."""
        if path not in self._file_textures:
            image = QImage(path)
            if image.isNull():
                print(f'shader sampler texture unreadable: {path}')
                self._file_textures[path] = None
            else:
                texture = QOpenGLTexture(image)
                texture.setMinMagFilters(QOpenGLTexture.Filter.Linear,
                                         QOpenGLTexture.Filter.Linear)
                texture.setWrapMode(QOpenGLTexture.WrapMode.Repeat)
                self._file_textures[path] = texture
        return self._file_textures[path]

    def _bind_pass_textures(self, f, locs, src) -> None:
        """Bind the pass source (slot `src`) to unit 0, and, when the
        program declares `u_tex2`, the untouched capture (slot 0) to unit
        1 so a two-input pass (bloom compose) reads the original frame
        alongside the chain source. Leaves unit 0 active so the next pass'
        single-texture default lands correctly."""
        if locs[_SECOND_SAMPLER] != -1:
            f.glActiveTexture(GL_TEXTURE0 + _CAPTURE_UNIT)
            f.glBindTexture(GL_TEXTURE_2D, self._slot_texture(0))
            f.glUniform1i(locs[_SECOND_SAMPLER], _CAPTURE_UNIT)
        f.glActiveTexture(GL_TEXTURE0)
        f.glBindTexture(GL_TEXTURE_2D, self._slot_texture(src))

    def _slot_texture(self, index) -> int:
        """Slot `index`'s texture; slot 0 is the external capture when
        `run_over` supplied one (unified chain), else the pipeline's
        own capture FBO."""
        if index == 0 and self._external is not None:
            return self._external.texture()
        return self._fbos[index].texture()

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
        """Build (once) and return `(program, uniform_locations,
        file_samplers)` for `name`, or None when the shader is unknown or
        failed to build. `file_samplers` are the shader's registered
        uniformTexture binds as (location, path) pairs, resolved once.
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
                locs = {u: program.uniformLocation(u) for u in _UNIFORMS}
                locs[_SECOND_SAMPLER] = program.uniformLocation(_SECOND_SAMPLER)
                file_samplers = tuple(
                    (loc, path)
                    for sampler, path in library.sampler_files(name).items()
                    if (loc := program.uniformLocation(sampler)) != -1)
                entry = (program, locs, file_samplers)
            else:
                print(f'shader {name!r} failed to build: {program.log()}')

        self._programs[name] = entry
        return entry
