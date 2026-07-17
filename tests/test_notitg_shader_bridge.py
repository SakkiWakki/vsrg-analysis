"""NotITG tier-2 map-supplied fragment shaders: the shader_bridge path
that registers a chart's own `.frag` files and drives their per-frame
uniforms, verified against real Government Knows corpus frags.

Two layers are covered:
- pure sampling (`chart_shader_effect` / `ChartShaderEffect`): which
  frags are fullscreen-expressible, uniform stream sampling, identity
  at rest, active windows;
- GL (skipped without a context, like test_shaders.py): the registered
  chart frags compile and run through the pipeline, and a broken frag
  degrades gracefully instead of crashing.

Real corpus frags live in the local NotITG library; the module is
skipped when that install is absent so CI without the songs stays green.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.games.notitg import shader_bridge as sb
from analysis.player.render.shaders import library

_GK_SHADERS = Path(
    '/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/FMS_Cat/'
    'The Government Knows [FMS_Cat]/fg/shaders')

pytestmark = pytest.mark.skipif(
    not _GK_SHADERS.is_dir(),
    reason='Government Knows corpus not installed')


def _frag(name: str) -> str:
    return (_GK_SHADERS / name).read_text(encoding='utf-8')


def _ctx(t: float):
    return SimpleNamespace(t_now=t)


@pytest.fixture(autouse=True)
def _clean_registry():
    library.clear_registry()
    yield
    library.clear_registry()


# ── which frags are fullscreen-expressible ---------------------------

def test_fullscreen_frags_register_stage_b_frags_skipped():
    # invert/unclesam2 are plain fullscreen post; vhs/cloud need engine
    # noise textures (samplerRandom) the fullscreen pipeline can't feed.
    entries = [
        {'name': 'gk_invert', 'frag': _frag('invert.frag')},
        {'name': 'gk_unclesam2', 'frag': _frag('unclesam2.frag')},
        {'name': 'gk_vhs', 'frag': _frag('vhs.frag')},
        {'name': 'gk_cloud', 'frag': _frag('cloud.frag')},
    ]
    assert sb.chart_shader_effect(entries) is not None
    registered = {i for i in library.available() if i.startswith('chart:')}
    assert 'chart:notitg:gk_invert' in registered
    assert 'chart:notitg:gk_unclesam2' in registered
    assert 'chart:notitg:gk_vhs' not in registered      # samplerRandom
    assert 'chart:notitg:gk_cloud' not in registered    # samplerRandom


def test_no_expressible_frags_gives_no_effect():
    assert sb.chart_shader_effect(
        [{'name': 'gk_vhs', 'frag': _frag('vhs.frag')}]) is None
    assert sb.chart_shader_effect([]) is None
    assert sb.chart_shader_effect(None) is None


def test_frag_path_source_is_read():
    eff = sb.chart_shader_effect(
        [{'name': 'gk_invert', 'frag_path': str(_GK_SHADERS / 'invert.frag')}])
    assert eff is not None
    assert 'chart:notitg:gk_invert' in library.available()


# ── uniform stream sampling ------------------------------------------

def test_declared_uniform_is_sampled_from_its_stream():
    eff = sb.chart_shader_effect([
        {'name': 'gk_unclesam2', 'frag': _frag('unclesam2.frag'),
         'uniforms': {'phase': [{'time': 1000, 'duration': 1000,
                                 'strength': 1.0}]}}])
    (sid, u0), = eff.at(_ctx(0.0)).shaders
    (_, u_mid), = eff.at(_ctx(1.5)).shaders
    (_, u_end), = eff.at(_ctx(2.5)).shaders
    assert sid == 'chart:notitg:gk_unclesam2'
    assert u0['phase'] == pytest.approx(0.0)     # rest before the event
    assert u_mid['phase'] == pytest.approx(0.5)  # easing 0 -> 1
    assert u_end['phase'] == pytest.approx(1.0)  # held after


def test_undeclared_uniform_stream_is_dropped():
    # A poke stream for a name the frag never declares must not appear in
    # the emitted pass (it would set nothing and only add noise).
    eff = sb.chart_shader_effect([
        {'name': 'gk_invert', 'frag': _frag('invert.frag'),
         'uniforms': {'nonexistent': [{'time': 0, 'strength': 1.0}]}}])
    (_, uniforms), = eff.at(_ctx(1.0)).shaders
    assert uniforms == {}


def test_no_uniform_streams_sets_no_custom_uniforms():
    # With no poke streams the pass carries no custom uniforms; each
    # declared uniform keeps its GL default (0), which is the frag's
    # rest state. The pass sets only the contract uniforms.
    eff = sb.chart_shader_effect(
        [{'name': 'gk_unclesam2', 'frag': _frag('unclesam2.frag')}])
    (_, uniforms), = eff.at(_ctx(3.3)).shaders
    assert uniforms == {}


# ── active windows ----------------------------------------------------

def test_window_gates_the_pass():
    eff = sb.chart_shader_effect([
        {'name': 'gk_invert', 'frag': _frag('invert.frag'),
         'windows': [{'time': 2000, 'duration': 0, 'strength': 1.0},
                     {'time': 3000, 'duration': 0, 'strength': 0.0}]}])
    assert eff.at(_ctx(1.0)) is None    # before the window
    assert eff.at(_ctx(2.5)) is not None  # inside
    assert eff.at(_ctx(3.5)) is None    # after it closes


# ── GL: real GK frags compile + run + degrade gracefully -------------

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


@pytest.mark.parametrize('frag', ['invert.frag', 'unclesam2.frag',
                                  'lumikey.frag', 'kiwotukeyou.frag',
                                  'weirdshit.frag'])
def test_gl_gk_frag_compiles(gl, frag):
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    sid = library.register_notitg_frag(f'chart:notitg:{frag}', _frag(frag))
    pipeline = ShaderGLPipeline()
    entry = pipeline._program(sid)
    assert entry is not None, f'{frag} failed to build'
    _, locs = entry
    # u_tex is used by every fullscreen post (they all sample the scene);
    # u_strength/u_time are stripped by the linker when a frag doesn't
    # read them, so only the always-used sampler is guaranteed present.
    assert locs['u_tex'] != -1


def test_gl_gk_invert_roundtrips(gl):
    # invert.frag: gl_FragColor = 1 - texture(sampler0). Run it twice so
    # the chain product lands in the readable ping-pong FBO.
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtOpenGL import QOpenGLPaintDevice
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    sid = library.register_notitg_frag('chart:notitg:inv', _frag('invert.frag'))
    pipeline = ShaderGLPipeline()
    host_device = QOpenGLPaintDevice(64, 64)
    host = QPainter(host_device)
    try:
        painter = pipeline.begin_capture(host, 64, 64)
        assert painter is not None
        painter.fillRect(0, 0, 64, 64, QColor(200, 40, 10))
        pipeline.end_capture(((sid, {}),) * 2, t_now=0.0)
    finally:
        host.end()
    assert not pipeline._broken
    inverted = pipeline._fbos[1].toImage().pixelColor(32, 32)
    assert (inverted.red(), inverted.green(), inverted.blue()) == (55, 215, 245)


def test_gl_custom_uniform_reaches_the_shader(gl):
    # unclesam2's `phase` scales output by 0.2*exp(-0.2*phase): a larger
    # phase darkens the frame. Driving phase from the pass dict must
    # actually change the pixel (the PySide setUniformValue(int, float)
    # trap would leave it frozen), so phase 0 and phase 3 differ.
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtOpenGL import QOpenGLPaintDevice
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    sid = library.register_notitg_frag('chart:notitg:us',
                                       _frag('unclesam2.frag'))

    def run(phase):
        pipeline = ShaderGLPipeline()
        host_device = QOpenGLPaintDevice(64, 64)
        host = QPainter(host_device)
        try:
            painter = pipeline.begin_capture(host, 64, 64)
            painter.fillRect(0, 0, 64, 64, QColor(150, 150, 150))
            pipeline.end_capture(((sid, {'phase': phase}),) * 2, t_now=0.0)
        finally:
            host.end()
        assert not pipeline._broken
        c = pipeline._fbos[1].toImage().pixelColor(32, 32)
        return (c.red(), c.green(), c.blue())

    bright, dark = run(0.0), run(3.0)
    assert sum(dark) < sum(bright), (bright, dark)


def test_gl_broken_frag_degrades_without_crashing(gl):
    # A frag that references an engine texture we don't feed is skipped by
    # the bridge, but if a malformed shader ever reaches the pipeline it
    # must fall back to the unshaded frame, never crash. Register broken
    # GLSL directly under a namespaced id and confirm the pass is dropped.
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtOpenGL import QOpenGLPaintDevice
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    library.register_source('chart:broken', 'this is not glsl {{{')
    pipeline = ShaderGLPipeline()
    host_device = QOpenGLPaintDevice(64, 64)
    host = QPainter(host_device)
    try:
        painter = pipeline.begin_capture(host, 64, 64)
        assert painter is not None
        painter.fillRect(0, 0, 64, 64, QColor(30, 90, 210))
        pipeline.end_capture((('chart:broken', {}),), t_now=0.0)
    finally:
        host.end()
    assert not pipeline._broken            # a bad shader disables one pass,
    assert pipeline._program('chart:broken') is None   # not the pipeline
    # The capture (unshaded frame) is intact and blitted through.
    frame = pipeline._fbos[0].toImage().pixelColor(32, 32)
    assert (frame.red(), frame.green(), frame.blue()) == (30, 90, 210)
