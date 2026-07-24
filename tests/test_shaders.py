"""Shader system: stack sampling, timeline start override, library
contract, and (context permitting) real GL compile + capture runs."""
from types import SimpleNamespace

import pytest

from pathlib import Path

from analysis.player.render.effects import composite
from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import EventTimeline, Keyframe
from analysis.player.render.shaders import ShaderStackEffect, library
from analysis.player.render.shaders.library import notitg_compat


# A real NotITG chart frag from the corpus (Mod Rush,
# "Catapult_Marshmallow/mods/fisheye.frag"): single `amount` uniform,
# imageCoord varying, img2tex helper, resolution, per-vertex color.
# Embedded so the tests run without the Songs library present; the
# guarded tests below re-validate against the on-disk file when it
# exists.
NOTITG_FISHEYE_FRAG = """\
uniform float amount;

varying vec4 color;
varying vec2 imageCoord;
uniform vec2 resolution;
uniform vec2 textureSize;
uniform vec2 imageSize;
uniform sampler2D sampler0;

vec2 img2tex( vec2 v ) { return v / textureSize * imageSize; }

void main()
{
\tvec2 uv = imageCoord;
\tuv -= 0.5;
\tuv *= 1.0 - amount / 2.0;

\tfloat r = sqrt(dot(uv,uv));
\tuv *= 1.0 + r * amount;
\tuv += 0.5;

\tvec2 res = resolution;
\tuv = clamp( uv, 1.0 / res, (res - 1.0) / res );

\tvec3 col = texture2D( sampler0, img2tex(uv) ).rgb;

\tgl_FragColor = vec4( col, 1.0 ) * color;
}
"""

_REAL_FISHEYE_PATH = Path(
    '/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/Mod Rush/'
    'Catapult_Marshmallow/mods/fisheye.frag')


@pytest.fixture(autouse=True)
def _clean_registry():
    library.clear_registry()
    yield
    library.clear_registry()


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
    replay = {'keycount': 4, '_fluxis_effect_streams': {
        'shader': [_event('Bloom', 0, end={'strength': 1.0})]}}
    effects = FluxisAdapter().effects(replay)
    assert any(isinstance(e, ShaderStackEffect) for e in effects)


# ── library contract ---------------------------------------------------

FLUXIS_PORTED = ('chromatic', 'fisheye', 'glitch', 'glitch2', 'greyscale',
                 'hueshift', 'invert', 'mosaic', 'noise', 'reflections',
                 'retro', 'splitscreen', 'vignette')

# Bloom has no single frag: its stack id fans out to these sub-passes.
BLOOM_SUBPASSES = ('bloom_blur_h', 'bloom_blur_v', 'bloom_compose')


def test_library_lists_ported_fluxis_set():
    assert set(FLUXIS_PORTED) <= set(library.available())


def test_library_serves_bloom_subpasses():
    assert set(BLOOM_SUBPASSES) <= set(library.available())


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
        _, locs, _samplers = entry
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


# ── tier-2 registry: precedence + namespacing ------------------------

def test_register_source_serves_namespaced_id():
    library.register_source('chart:demo', 'FOO')
    assert library.source('chart:demo') == 'FOO'
    assert 'chart:demo' in library.available()


def test_register_file_reads_from_disk(tmp_path):
    frag = tmp_path / 'x.frag'
    frag.write_text('BAR', encoding='utf-8')
    library.register_file('chart:x', frag)
    assert library.source('chart:x') == 'BAR'


def test_registry_requires_namespace_separator():
    with pytest.raises(ValueError):
        library.register_source('demo', 'FOO')


def test_registration_cannot_shadow_builtin():
    # Even namespaced, an id resolving to a builtin filename is refused;
    # and a bare builtin name never reaches the registry at all.
    with pytest.raises(ValueError):
        library.register_source('invert', 'HACK')
    library.register_source('chart:invert', 'HACK')   # namespaced is fine
    assert 'uniform sampler2D u_tex;' in library.source('invert')  # builtin
    assert library.source('chart:invert') == 'HACK'


def test_builtin_wins_over_same_bare_name():
    # A registry entry can only exist under a namespaced key, so a bare
    # builtin lookup can never be intercepted.
    library.register_source('chart:invert', 'HACK')
    assert library.source('invert') != 'HACK'


def test_unknown_id_is_none():
    assert library.source('chart:nope') is None
    assert library.source('nope') is None


# ── NotITG compat translation ----------------------------------------

def test_translate_produces_contract_shader():
    out = notitg_compat.translate(NOTITG_FISHEYE_FRAG)
    assert out.startswith('#version 150')
    assert 'uniform sampler2D u_tex;' in out
    assert 'uniform vec2 u_resolution;' in out
    assert 'uniform vec3 u_strength;' in out
    assert '#define sampler0 u_tex' in out
    assert '#define texture2D texture' in out
    assert 'out vec4 _fs_fragcolor;' in out
    # varyings gone as `varying`, present as fed globals / aliases
    assert 'varying' not in out
    assert 'uniform float amount;' in out   # chart uniform survives, drivable


def test_translate_reports_chart_uniforms():
    assert notitg_compat.uniform_names(NOTITG_FISHEYE_FRAG) == ('amount',)
    assert library.register_notitg_frag('chart:fish', NOTITG_FISHEYE_FRAG)
    assert library.registered_uniform_names('chart:fish') == ('amount',)


def test_translate_rejects_frag_without_sampler0():
    with pytest.raises(ValueError):
        notitg_compat.translate('void main() { gl_FragColor = vec4(1.0); }')


def test_translate_hoists_nonconstant_global_initializers():
    # ~a third of the corpus has file-scope `float x = f(time);` which a
    # core/ES profile rejects; the initializer must move into main().
    frag = ('uniform sampler2D sampler0;\n'
            'uniform float amp;\n'
            'float k = 0.5;\n'                   # constant -- stays put
            'float wobble = sin(amp) * 2.0;\n'   # non-const -- hoisted
            'void main() { gl_FragColor = texture2D(sampler0, vec2(wobble, k)); }')
    out = notitg_compat.translate(frag)
    assert 'float wobble;' in out                # declaration only at scope
    assert 'wobble =sin(amp) * 2.0;' in out      # initializer inside main()
    assert 'float k = 0.5;' in out               # constant left untouched


def test_translate_leaves_local_initializers_alone():
    frag = ('uniform sampler2D sampler0;\n'
            'void main() {\n'
            '  float local = float(gl_FragCoord.x);\n'
            '  gl_FragColor = texture2D(sampler0, vec2(local));\n}')
    out = notitg_compat.translate(frag)
    assert 'float local = float(gl_FragCoord.x);' in out


def test_register_notitg_frag_translates():
    library.register_notitg_frag('chart:fish', NOTITG_FISHEYE_FRAG)
    assert '#define sampler0 u_tex' in library.source('chart:fish')


# ── custom uniform coercion -------------------------------------------

def test_uniform_floats_coercions():
    from analysis.player.render.shaders.gl_pipeline import _uniform_floats
    assert _uniform_floats(0.5) == (0.5,)
    assert _uniform_floats(3) == (3.0,)
    assert _uniform_floats((1.0, 2.0)) == (1.0, 2.0)
    assert _uniform_floats([1.0, 2.0, 3.0]) == (1.0, 2.0, 3.0)
    assert _uniform_floats((1.0, 2.0, 3.0, 4.0)) == (1.0, 2.0, 3.0, 4.0)
    assert _uniform_floats('a-texture-name') is None   # samplers deferred
    assert _uniform_floats((1.0, 2, 3, 4, 5)) is None  # unsupported width


def test_expand_fans_bloom_out_and_passes_others_through():
    from analysis.player.render.shaders.gl_pipeline import _expand
    u = {'u_strength': (0.6, 0.0, 0.0)}
    bloom = list(_expand([('bloom', u)]))
    assert [name for name, _ in bloom] == ['bloom_blur_h', 'bloom_blur_v',
                                           'bloom_compose']
    assert all(sub_u is u for _, sub_u in bloom)   # same uniforms to each
    passthrough = list(_expand([('invert', u), ('glitch2', u)]))
    assert [name for name, _ in passthrough] == ['invert', 'glitch2']


# ── GL: register a real corpus frag and run it as a pass --------------

def _run_single_pass(pipeline, shader_id, uniforms, w=64, h=64, split=False):
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtOpenGL import QOpenGLPaintDevice
    host_device = QOpenGLPaintDevice(w, h)
    host = QPainter(host_device)
    try:
        painter = pipeline.begin_capture(host, w, h)
        assert painter is not None
        painter.fillRect(0, 0, w, h, QColor(200, 120, 40))
        if split:
            # A small off-centre patch so a distortion sampling a
            # different texel produces a visibly different pixel (a
            # centred split would sit on the fisheye's fixed point).
            painter.fillRect(4, 4, 16, 16, QColor(10, 200, 220))
        # Duplicate the pass so the chain product lands in the readable
        # ping-pong FBO (the final pass renders into the default FBO).
        pipeline.end_capture(((shader_id, uniforms),) * 2, t_now=0.5)
    finally:
        host.end()
    return pipeline


def test_gl_registered_notitg_frag_compiles_and_runs(gl):
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    shader_id = library.register_notitg_frag('chart:fish', NOTITG_FISHEYE_FRAG)
    pipeline = ShaderGLPipeline()
    _run_single_pass(pipeline, shader_id, {'amount': 0.4, 'u_strength': (0, 0, 0)})
    assert not pipeline._broken
    entry = pipeline._programs.get(shader_id)
    assert entry is not None, 'registered NotITG frag failed to build'
    # amount=0 is identity (uv unchanged); centre pixel keeps its colour.
    pipeline2 = ShaderGLPipeline()
    _run_single_pass(pipeline2, shader_id, {'amount': 0.0})
    out = pipeline2._fbos[1].toImage().pixelColor(32, 32)
    assert (out.red(), out.green(), out.blue()) == (200, 120, 40)


def test_gl_custom_uniform_changes_output(gl):
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    shader_id = library.register_notitg_frag('chart:fish', NOTITG_FISHEYE_FRAG)
    # A strong fisheye pulls a different texel into an off-centre pixel,
    # so the driven `amount` uniform must reach the shader.
    pipeline = ShaderGLPipeline()
    _run_single_pass(pipeline, shader_id, {'amount': 0.0}, split=True)
    calm = pipeline._fbos[1].toImage()
    pipeline2 = ShaderGLPipeline()
    _run_single_pass(pipeline2, shader_id, {'amount': 0.9}, split=True)
    warped = pipeline2._fbos[1].toImage()

    def _rgb(img, x, y):
        c = img.pixelColor(x, y)
        return (c.red(), c.green(), c.blue())

    changed = sum(_rgb(calm, x, y) != _rgb(warped, x, y)
                  for y in range(0, 64, 4) for x in range(0, 64, 4))
    assert changed > 0, 'driven amount uniform did not reach the shader'


def test_gl_u_time_reaches_shader(gl):
    # Regression: u_time must be set via glUniform1f, not setUniformValue
    # (whose Python-float path is a no-op under PySide6), or time-animated
    # builtins freeze.
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtOpenGL import QOpenGLPaintDevice
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline

    def sample(t):
        pipeline = ShaderGLPipeline()
        device = QOpenGLPaintDevice(64, 64)
        host = QPainter(device)
        try:
            painter = pipeline.begin_capture(host, 64, 64)
            painter.fillRect(0, 0, 64, 64, QColor(128, 128, 128))
            pipeline.end_capture(
                (('noise', {'u_strength': (1.0, 0.0, 0.0)}),) * 2, t_now=t)
        finally:
            host.end()
        c = pipeline._fbos[1].toImage().pixelColor(20, 40)
        return (c.red(), c.green(), c.blue())

    assert sample(0.0) != sample(9.0)


def _run_bloom(strength, w=64, h=64, fill=(120, 120, 120)):
    """Run bloom (blur h/v -> compose) at `strength`, then a strength-0
    invert (identity) so the compose output lands in a readable ping-pong
    slot instead of the write-only host FBO. Returns that slot's image."""
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtOpenGL import QOpenGLPaintDevice
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    pipeline = ShaderGLPipeline()
    host_device = QOpenGLPaintDevice(w, h)   # keep referenced for the painter
    host = QPainter(host_device)
    try:
        painter = pipeline.begin_capture(host, w, h)
        assert painter is not None
        painter.fillRect(0, 0, w, h, QColor(*fill))
        pipeline.end_capture(
            (('bloom', {'u_strength': (strength, 0.0, 0.0)}),
             ('invert', {'u_strength': (0.0, 0.0, 0.0)})),
            t_now=0.0)
    finally:
        host.end()
    assert not pipeline._broken
    return pipeline._fbos[1].toImage()


def test_gl_bloom_expands_and_composes(gl):
    # Blur of a solid fill is that fill, so compose = scene + glow*strength:
    # strength 0 is identity (glow*0), a positive strength brightens.
    def grey(img):
        c = img.pixelColor(32, 32)
        return (c.red(), c.green(), c.blue())

    assert grey(_run_bloom(0.0)) == (120, 120, 120)
    lit = grey(_run_bloom(0.6))
    assert all(v > 120 for v in lit), lit


@pytest.mark.skipif(not _REAL_FISHEYE_PATH.is_file(),
                    reason='NotITG Songs library not present')
def test_gl_real_corpus_file_compiles_and_runs(gl):
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    shader_id = library.register_file(
        'chart:fisheye-real', _REAL_FISHEYE_PATH, compat=True)
    pipeline = ShaderGLPipeline()
    _run_single_pass(pipeline, shader_id, {'amount': 0.3})
    assert not pipeline._broken
    assert pipeline._programs.get(shader_id) is not None
