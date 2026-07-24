"""NotITG Polygon mesh tier: the sim's vertex-vector recording
(SetDrawMode/SetNumVertices/SetVertexPosition/SetVertexTexCoord), the
producers' mesh payload (UV unscale/flip, quads expansion), and the
Vert= shader translation (crumple.vert -> the mesh vertex contract)."""
from pathlib import Path

import numpy as np
import pytest

from analysis.games.notitg.sim.actor import SimActor
from analysis.games.notitg.sim.producers import (_AFT_UV_SCALE_X,
                                                 _AFT_UV_SCALE_Y,
                                                 _mesh_payload)
from analysis.player.render.shaders.library import notitg_compat

_CRUMPLE = Path('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/UKSRT9/'
                '5. getfucked2/lua/shaders/crumple.vert')


# -- sim recording ------------------------------------------------------------

def _grid_actor():
    a = SimActor()
    a.poke('SetDrawMode', ['triangles'])
    a.poke('SetNumVertices', [6])
    for i in range(6):
        a.poke('SetVertexPosition', [i, float(i) * 10.0 - 320.0, 240.0, 0.0])
        a.poke('SetVertexTexCoord', [i, i / 6.0 * _AFT_UV_SCALE_X, 0.0])
    return a


def test_mesh_pokes_record_static_vertex_state():
    a = _grid_actor()
    assert a.mesh_mode == 'triangles'
    assert len(a.mesh_positions) == 6
    assert a.mesh_positions[1] == [-310.0, 240.0, 0.0]
    assert a.mesh_uvs[3] == pytest.approx([0.5 * _AFT_UV_SCALE_X, 0.0])
    # Mesh writes are actor state, not keyframe channels.
    assert not any(p.startswith('mesh') for p in a.keyframes())


def test_mesh_writes_before_allocation_or_out_of_range_drop():
    a = SimActor()
    dropped = []
    a.dropped_notify = dropped.append
    a.poke('SetVertexPosition', [0, 1.0, 2.0, 0.0])  # no SetNumVertices yet
    a.poke('SetNumVertices', [2])
    a.poke('SetVertexPosition', [5, 1.0, 2.0, 0.0])  # index out of range
    assert a.mesh_positions == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert dropped == []


# -- producers mesh payload ---------------------------------------------------

class _FakeXmlActor:
    kind = 'Polygon'

    def __init__(self, attrs=None):
        self.attrs = attrs or {}


def test_mesh_payload_unscales_uvs():
    a = _grid_actor()
    a.poke('SetVertexTexCoord', [0, _AFT_UV_SCALE_X, _AFT_UV_SCALE_Y])
    payload = _mesh_payload(a, _FakeXmlActor())
    assert payload['mode'] == 'triangles'
    assert payload['vert'] is None
    vertices = payload['vertices']
    assert vertices.shape == (6, 4)
    assert vertices.dtype == np.float32
    # Chart UV (scaleX, scaleY) = the content's far corner: ours is
    # (1, 1) - unscaled from the pow2 padding, NO v flip (our capture
    # FBOs are bottom-up exactly like the engine's AFT textures).
    assert vertices[0, 2:] == pytest.approx([1.0, 1.0])
    # z drops (orthographic projection), local xy pass through.
    assert vertices[1, :2] == pytest.approx([-310.0, 240.0])


def test_mesh_payload_expands_quads_to_triangles():
    a = SimActor()
    a.poke('SetDrawMode', ['quads'])
    a.poke('SetNumVertices', [4])
    for i, (x, y) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        a.poke('SetVertexPosition', [i, float(x), float(y), 0.0])
    payload = _mesh_payload(a, _FakeXmlActor())
    assert payload['mode'] == 'triangles'
    corners = [tuple(v) for v in payload['vertices'][:, :2]]
    assert corners == [(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)]


def test_mesh_payload_none_without_mesh_or_for_unknown_mode():
    assert _mesh_payload(SimActor(), _FakeXmlActor()) is None
    a = _grid_actor()
    a.poke('SetDrawMode', ['lines'])
    assert _mesh_payload(a, _FakeXmlActor()) is None


# -- Vert= translation --------------------------------------------------------

_MINI_VERT = """#version 120
attribute vec4 TextureMatrixScale;
varying vec2 textureCoord;
uniform sampler2D samplerRandom;
uniform float amp;
void main() {
  vec3 rnd = texture2D(samplerRandom, gl_Vertex.xy * 0.001).xyz;
  gl_Position = gl_ModelViewProjectionMatrix
    * vec4(gl_Vertex.xyz + rnd * amp, 1.0);
  gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
  textureCoord = gl_TexCoord[0].xy;
  gl_FrontColor = gl_Color;
}
"""


def test_translate_vert_shims_fixed_function_inputs():
    out = notitg_compat.translate_vert(_MINI_VERT)
    assert 'attribute' not in out
    assert 'varying' not in out
    assert 'gl_TextureMatrix' not in out.replace('#define gl_TexCoord', '')
    assert 'void _vs_chart_main()' in out
    assert 'void main(void) { _vs_chart_main(); v_uv = a_uv; }' in out
    assert 'uniform mat4 u_mvp;' in out
    # The chart's own uniforms stay declared (the pipeline pokes amp,
    # binds samplerRandom).
    assert 'uniform sampler2D samplerRandom;' in out
    assert 'uniform float amp;' in out


def test_translate_vert_requires_a_main():
    with pytest.raises(ValueError):
        notitg_compat.translate_vert('uniform float amp;')


@pytest.mark.skipif(not _CRUMPLE.is_file(),
                    reason='getfucked2 corpus not installed')
def test_translate_vert_handles_crumple():
    out = notitg_compat.translate_vert(
        _CRUMPLE.read_text(encoding='utf-8'))
    assert 'void _vs_chart_main()' in out
    assert 'gl_ModelViewProjectionMatrix' in out  # via the #define
    assert '#define gl_ModelViewProjectionMatrix u_mvp' in out


# -- GL: the translated vert builds into a runnable mesh program --------------

@pytest.fixture(scope='module')
def gl(_qapp):
    from PySide6.QtGui import (QOffscreenSurface, QOpenGLContext,
                               QSurfaceFormat)
    fmt = QSurfaceFormat()
    fmt.setMajorVersion(3)
    fmt.setMinorVersion(2)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    context = QOpenGLContext()
    context.setFormat(fmt)
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    if not (context.create() and surface.isValid()
            and context.makeCurrent(surface)):
        pytest.skip('no OpenGL context on this platform')
    yield context
    context.doneCurrent()


def test_gl_default_mesh_program_builds(gl):
    from analysis.player.render.gl_capture import GLCaptureBackend
    backend = GLCaptureBackend()
    entry = backend._mesh_program(None)
    assert entry is not None
    program, locs = entry
    assert locs['u_mvp'] != -1
    assert locs['u_tex'] != -1


def test_gl_mini_vert_program_builds(gl, tmp_path):
    from analysis.player.render.gl_capture import GLCaptureBackend
    vert = tmp_path / 'mini.vert'
    vert.write_text(_MINI_VERT)
    backend = GLCaptureBackend()
    entry = backend._mesh_program(str(vert))
    assert entry is not None
    program, _locs = entry
    # The chart's own uniforms are live in the linked program.
    assert program.uniformLocation('amp') != -1


@pytest.mark.skipif(not _CRUMPLE.is_file(),
                    reason='getfucked2 corpus not installed')
def test_gl_crumple_vert_program_builds(gl):
    from analysis.player.render.gl_capture import GLCaptureBackend
    backend = GLCaptureBackend()
    entry = backend._mesh_program(str(_CRUMPLE))
    assert entry is not None
    program, _locs = entry
    assert program.uniformLocation('amp') != -1
    assert program.uniformLocation('samplerRandom') != -1


def test_draw_mode_aliases_normalize_at_record():
    # Library census: Quads/fan/strip/linestrip spellings dominate.
    for token, expected in (('Quads', 'quads'), ('fan', 'trianglefan'),
                            ('Strip', 'trianglestrip'),
                            ('LineStrip', 'linestrip'),
                            ('QuadStrip', 'quadstrip')):
        a = SimActor()
        a.poke('SetDrawMode', [token])
        assert a.mesh_mode == expected, token


def test_quadstrip_payload_maps_to_triangle_strip():
    a = SimActor()
    a.poke('SetDrawMode', ['QuadStrip'])
    a.poke('SetNumVertices', [4])
    for i in range(4):
        a.poke('SetVertexPosition', [i, float(i), 0.0, 0.0])
    payload = _mesh_payload(a, _FakeXmlActor())
    assert payload['mode'] == 'trianglestrip'
    assert len(payload['vertices']) == 4
