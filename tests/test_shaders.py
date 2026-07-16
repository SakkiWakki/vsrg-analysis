"""Shader system: stack sampling, timeline start override, library
contract, and (context permitting) real GL compile + capture runs."""
from types import SimpleNamespace

import pytest

from analysis.player.render.effects import composite
from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import EventTimeline, Keyframe
from analysis.player.render.shaders import ShaderStackEffect, library


def _ctx(t=0.0):
    return SimpleNamespace(t_now=t, player=SimpleNamespace(keycount=4))


def _event(shader, time_ms, duration_ms=0, ease=0, end=None, start=None):
    e = {'time': time_ms, 'shader': shader, 'duration': duration_ms,
         'ease': ease, 'end-params': end or {}}
    if start is not None:
        e['use-start'] = True
        e['start-params'] = start
    return e


# ── timeline start override ------------------------------------------

def test_timeline_start_override_eases_from_keyframe_start():
    tl = EventTimeline(
        [Keyframe(0.0, (10.0,), 0.0, 0),
         Keyframe(5.0, (0.0,), 2.0, 0, start=(4.0,))],
        rest=(0.0,))
    assert tl.sample(5.0) == pytest.approx((4.0,))   # snaps to start
    assert tl.sample(6.0) == pytest.approx((2.0,))   # 4 -> 0 halfway
    assert tl.sample(7.0) == pytest.approx((0.0,))


def test_timeline_without_override_eases_from_previous_target():
    tl = EventTimeline(
        [Keyframe(0.0, (10.0,), 0.0, 0),
         Keyframe(5.0, (0.0,), 2.0, 0)],
        rest=(0.0,))
    assert tl.sample(6.0) == pytest.approx((5.0,))


# ── stack sampling ----------------------------------------------------

def test_stack_inactive_when_all_strengths_zero():
    eff = ShaderStackEffect([_event('Vignette', 0, end={'strength': 0.0})])
    assert eff.at(_ctx(1.0)) is None


def test_stack_eases_strength_and_emits_pass():
    eff = ShaderStackEffect(
        [_event('Vignette', 1000, duration_ms=2000, end={'strength': 1.0})])
    frame = eff.at(_ctx(2.0))
    (name, uniforms), = frame.shaders
    assert name == 'vignette'
    assert uniforms['u_strength'] == pytest.approx((0.5, 0.0, 0.0))


def test_stack_use_start_snaps_then_eases():
    eff = ShaderStackEffect(
        [_event('Mosaic', 0, duration_ms=1000, end={'strength': 0.0},
                start={'strength': 1.0})])
    frame = eff.at(_ctx(0.5))
    (_, uniforms), = frame.shaders
    assert uniforms['u_strength'] == pytest.approx((0.5, 0.0, 0.0))


def test_stack_keeps_first_appearance_order():
    events = [_event('Noise', 500, end={'strength': 0.3}),
              _event('Vignette', 0, end={'strength': 0.8}),
              _event('Noise', 0, end={'strength': 0.1})]
    frame = ShaderStackEffect(events).at(_ctx(1.0))
    assert [name for name, _ in frame.shaders] == ['noise', 'vignette']


def test_stack_reads_legacy_params_key():
    eff = ShaderStackEffect(
        [{'time': 0, 'shader': 'Invert', 'params': {'strength': 1.0}}])
    frame = eff.at(_ctx(1.0))
    (name, uniforms), = frame.shaders
    assert name == 'invert'
    assert uniforms['u_strength'] == pytest.approx((1.0, 0.0, 0.0))


def test_stack_tolerates_junk_events():
    eff = ShaderStackEffect([None, 'x', {}, {'shader': ''},
                             _event('Greyscale', 0, end={'strength': 1.0})])
    assert len(eff.at(_ctx(1.0)).shaders) == 1


def test_stack_empty_is_falsy():
    assert not ShaderStackEffect([])
    assert not ShaderStackEffect(None)


# ── compositor channel ------------------------------------------------

def test_composite_concatenates_shader_passes_in_effect_order():
    class _E:
        def __init__(self, frame):
            self._frame = frame

        def at(self, ctx):
            return self._frame

    a = EffectFrame(shaders=(('vignette', {'u_strength': (1, 0, 0)}),))
    b = EffectFrame(shaders=(('invert', {'u_strength': (1, 0, 0)}),))
    frame = composite([_E(a), _E(b)], _ctx())
    assert [name for name, _ in frame.shaders] == ['vignette', 'invert']
    assert not frame.is_identity


# ── adapter wiring -----------------------------------------------------

def test_fluxis_adapter_effects_include_shader_stack():
    from analysis.games.fluxis.adapter import FluxisAdapter
    replay = {'_fluxis_effect_streams': {
        'shader': [_event('Bloom', 0, end={'strength': 1.0})]}}
    effects = FluxisAdapter().effects(replay)
    assert any(isinstance(e, ShaderStackEffect) for e in effects)


# ── library contract ---------------------------------------------------

FLUXIS_PORTED = ('chromatic', 'fisheye', 'glitch', 'greyscale', 'hueshift',
                 'invert', 'mosaic', 'noise', 'reflections', 'retro',
                 'splitscreen', 'vignette')


def test_library_lists_ported_fluxis_set():
    assert set(FLUXIS_PORTED) <= set(library.available())


def test_library_sources_follow_uniform_contract():
    for name in library.available():
        src = library.source(name)
        assert 'uniform sampler2D u_tex;' in src, name
        assert 'uniform vec2 u_resolution;' in src, name
        assert 'uniform vec3 u_strength;' in src, name
        assert 'gl_FragCoord' in src, name


def test_library_rejects_path_like_names():
    assert library.source('../secrets') is None
    assert library.source('nope') is None


# ── GL execution (skipped when the platform has no GL) ----------------

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


def test_gl_all_library_shaders_compile(gl):
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    pipeline = ShaderGLPipeline()
    for name in library.available():
        entry = pipeline._program(name)
        assert entry is not None, f'{name} failed to build'
        _, locs = entry
        for uniform in ('u_tex', 'u_resolution', 'u_strength'):
            assert locs[uniform] != -1, f'{name} lost {uniform}'
    for name in ('noise', 'glitch'):
        assert pipeline._programs[name][1]['u_time'] != -1


def test_gl_capture_chain_runs_and_inverts(gl):
    from PySide6.QtGui import QPainter, QColor
    from PySide6.QtOpenGL import QOpenGLPaintDevice
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline

    host_device = QOpenGLPaintDevice(64, 64)
    host = QPainter(host_device)
    pipeline = ShaderGLPipeline()
    try:
        painter = pipeline.begin_capture(host, 64, 64)
        assert painter is not None
        painter.fillRect(0, 0, 64, 64, QColor(255, 0, 0))
        # Two passes force the ping-pong path; the intermediate FBO
        # holds pass one's output, the only chain product an offscreen
        # test can read back (the final pass lands in the default FBO).
        pipeline.end_capture(
            (('invert', {'u_strength': (1.0, 0.0, 0.0)}),) * 2, t_now=0.0)
    finally:
        host.end()
    assert not pipeline._broken

    capture = pipeline._fbos[0].toImage().pixelColor(32, 32)
    assert (capture.red(), capture.green(), capture.blue()) == (255, 0, 0)
    inverted = pipeline._fbos[1].toImage().pixelColor(32, 32)
    assert (inverted.red(), inverted.green(), inverted.blue()) == (0, 255, 255)
