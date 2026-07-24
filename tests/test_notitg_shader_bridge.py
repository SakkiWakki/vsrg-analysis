"""NotITG shader_bridge: the shader-flag pulse bridge, plus GL checks
that real Government Knows corpus frags translate and run through the
fullscreen pipeline (chart `Frag=` actors themselves compile as shaded
field-instance blits - see test_notitg_chart_shaders.py - but the
translator and pipeline these GL tests exercise are shared).

Real corpus frags live in the local NotITG library; those tests are
skipped when that install is absent so CI without the songs stays green.
The flag-bridge tests are pure lifetime logic and always run.
"""
from pathlib import Path

import pytest

from analysis.games.notitg import shader_bridge as sb
from analysis.player.render.shaders import library

_GK_SHADERS = Path(
    '/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/FMS_Cat/'
    'The Government Knows [FMS_Cat]/fg/shaders')

# The GL corpus tests read the real install; the flag-bridge tests
# below do not (pure lifetime logic) and always run.
_needs_corpus = pytest.mark.skipif(
    not _GK_SHADERS.is_dir(),
    reason='Government Knows corpus not installed')


def _frag(name: str) -> str:
    return (_GK_SHADERS / name).read_text(encoding='utf-8')


@pytest.fixture(autouse=True)
def _clean_registry():
    library.clear_registry()
    yield
    library.clear_registry()


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


@_needs_corpus
@pytest.mark.parametrize('frag', ['invert.frag', 'unclesam2.frag',
                                  'lumikey.frag', 'kiwotukeyou.frag',
                                  'weirdshit.frag'])
def test_gl_gk_frag_compiles(gl, frag):
    from analysis.player.render.shaders.gl_pipeline import ShaderGLPipeline
    sid = library.register_notitg_frag(f'chart:notitg:{frag}', _frag(frag))
    pipeline = ShaderGLPipeline()
    entry = pipeline._program(sid)
    assert entry is not None, f'{frag} failed to build'
    _, locs, _samplers = entry
    # u_tex is used by every fullscreen post (they all sample the scene);
    # u_strength/u_time are stripped by the linker when a frag doesn't
    # read them, so only the always-used sampler is guaranteed present.
    assert locs['u_tex'] != -1


@_needs_corpus
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


@_needs_corpus
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


@_needs_corpus
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


# ── shader-flag bridge: pulse lifetimes ──────────────────────────────
#
# These need no corpus (pure logic) - they guard the class of bug the
# user reported (a transparent mirror overlay that never turns off). The
# stream is the compiled `shader_flags`: dicts with `key`, `which` (the
# SetShaderFlagNum slot; absent = plain SetShaderFlag = slot 0), and `t`
# in seconds. On/off windows drive a mirror/tile fullscreen pass, so a
# missed OFF = a mirror pass that persists forever.

def _windows(*flags):
    return sb._pair_pulses(sb._clean(list(flags)))[0]


def _flag(key, t, which=None):
    row = {'key': key, 't': t}
    if which is not None:
        row['which'] = which
    return row


def test_empty_flag_stream_makes_no_pass():
    # gat's mod_shader calls are all commented out -> the compiled stream
    # is empty -> the flag bridge must contribute nothing (the mirror the
    # user saw is NOT from here).
    assert sb.build_shader_events([]) == ([], [])
    assert sb.build_shader_events(None) == ([], [])
    assert sb.notitg_shader_effects([]) == []


def test_paired_pulse_opens_and_closes():
    # mod_shader(beat, 55): set at t, clear (key 0) 0.5 beats later.
    assert _windows(_flag(55, 10.0), _flag(0, 10.5)) == [(55, 10.0, 10.5)]


def test_same_slot_rewrite_closes_the_previous_flag():
    # THE REGRESSION: NotITG's registry holds one value per slot, so
    # writing a new key into the slot turns the old flag OFF. The mirror
    # (55) must close when the tile (48) replaces it - not persist through
    # the tile's window and overlay it (the reported transparent mirror).
    windows = _windows(_flag(55, 10.0), _flag(48, 11.0), _flag(0, 12.0))
    assert windows == [(55, 10.0, 11.0), (48, 11.0, 12.0)]


def test_independent_slots_stay_simultaneously_live():
    # Distinct SetShaderFlagNum slots do NOT clobber each other; each
    # closes only when ITS slot is rewritten.
    windows = _windows(
        _flag(55, 10.0, which=0), _flag(48, 10.2, which=1),
        _flag(0, 12.0, which=0), _flag(0, 13.0, which=1))
    assert sorted(windows) == [(48, 10.2, 13.0), (55, 10.0, 12.0)]


def test_setshaderflagnum_clears_its_own_slot():
    # mod_shader's SetShaderFlagNum(0, which) clear path targets the slot
    # it opened.
    assert _windows(_flag(48, 50.0, which=1),
                    _flag(0, 50.5, which=1)) == [(48, 50.0, 50.5)]


def test_redundant_reset_of_same_key_does_not_reopen():
    # Writing the identical key into the same slot is an engine no-op: no
    # spurious close/reopen (which would flicker the pass to 0 and back).
    assert _windows(_flag(55, 10.0), _flag(55, 11.0),
                    _flag(0, 12.0)) == [(55, 10.0, 12.0)]


def test_unclosed_flag_persists_without_a_fabricated_off():
    # A set with no matching clear stays on to the chart's end (the
    # chart's own oversight). No fabricated off event - build_shader_events
    # emits only the ON event.
    assert _windows(_flag(55, 10.0)) == [(55, 10.0, None)]
    events, _skipped = sb.build_shader_events([_flag(55, 10.0)])
    strengths = [e['end-params']['strength'] for e in events]
    assert strengths == [1.0]     # one ON ramp, no OFF


def test_unmapped_and_skipped_keys_are_reported_not_guessed():
    # gat's would-be keys 49/124/217 have no feasible fullscreen shader;
    # they are skipped (reported), never mapped to a wrong effect.
    events, skipped = sb.build_shader_events(
        [_flag(49, 1.0), _flag(217, 2.0), _flag(999, 3.0)])
    assert events == []
    assert skipped == [49, 217, 999]


def test_events_ramp_up_at_on_and_down_at_off():
    # The emitted .ffx events: strength 0->1 at t_on, 1->0 at t_off, so
    # the pass is live only inside the window.
    events, _skipped = sb.build_shader_events([_flag(55, 10.0),
                                               _flag(0, 10.5)])
    on, off = events
    assert on['time'] == 10000.0 and on['end-params']['strength'] == 1.0
    assert off['time'] == 10500.0 and off['end-params']['strength'] == 0.0
