"""Per-note shader stamping (feed v3): the Government Knows tier.

The pipeline hands `feed_from_context` a `note_shader` dict per player
per frame; engine-drawn items (receptors, taps, mines, hold bodies and
tails, their glows) stamp the active category's program and carry a
pre-sampled uniform block - the chart's values followed by the six
NOTE_SHADER_BUILTIN_NAMES - while replay overlays never stamp.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.player.render.shaders.library import notitg_compat
from analysis.player.render.storyboard import note_feed as nf

from tests.test_sbnative_note_feed import (_ctx, _note_view, _stream_view,
                                           _IMAGE_MAP)

_ARROW = (5, [0.25, 0.75])       # shader_plus_one, chart uniform values
_RECEPTOR = (9, [])


def _shader(t=12.0, beat=24.0, to_beat=None):
    return {'arrow': _ARROW, 'hold': _ARROW, 'receptor': _RECEPTOR,
            'time': t, 'beat': beat, 'to_beat': to_beat}


def _rows(u, count):
    return u.reshape(count, nf.FEED_U_STRIDE)


def test_engine_items_stamp_their_category_and_overlays_do_not():
    views = [_note_view(col=1, lx=64.0, y=100.0, state='tap', miss=True,
                        judgment='great', press_t=None)]
    ctx = _ctx(views, stream_views=[_stream_view(col=2, lx=128.0, y=60.0)])
    u, _f, x, count, report = nf.feed_from_context(ctx, _IMAGE_MAP,
                                                   note_shader=_shader())
    assert report['receptors'] == 4 and report['taps'] == 1
    rows = _rows(u, count)
    image = rows[:, 1]
    shader = rows[:, 4]
    receptor_id = _IMAGE_MAP['receptor'][0]
    assert set(shader[image == receptor_id]) == {_RECEPTOR[0]}
    assert set(shader[image == _IMAGE_MAP['tap'][0]]) == {_ARROW[0]}
    assert set(shader[image == _IMAGE_MAP['mine'][0]]) == {_ARROW[0]}
    # The miss X is a replay overlay, never shaded.
    assert set(shader[image == _IMAGE_MAP['miss_x_great'][0]]) == {0}


def test_uniform_block_is_chart_values_then_builtins():
    views = [_note_view(col=3, lx=192.0, y=100.0, state='tap',
                        press_t=10.5, off=0.5)]
    to_beat = lambda t: t * 2.0     # note time 10.0 -> beat 20.0
    ctx = _ctx(views)
    u, _f, x, count, _report = nf.feed_from_context(
        ctx, _IMAGE_MAP, note_shader=_shader(to_beat=to_beat))
    rows = _rows(u, count)
    tap = np.nonzero(rows[:, 1] == _IMAGE_MAP['tap'][0])[0][0]
    off, n = int(rows[tap, 5]), int(rows[tap, 6])
    block = [float(v) for v in x[off:off + n]]
    chart = list(_ARROW[1])
    # iCol, isHold, isReceptor, fNoteBeat, beat, time.
    assert block == pytest.approx(chart + [3.0, 0.0, 0.0, 20.0, 24.0, 12.0])


def test_receptor_block_flags_is_receptor():
    ctx = _ctx([])
    u, _f, x, count, _report = nf.feed_from_context(ctx, _IMAGE_MAP,
                                                    note_shader=_shader())
    rows = _rows(u, count)
    off, n = int(rows[0, 5]), int(rows[0, 6])
    block = [float(v) for v in x[off:off + n]]
    assert block == pytest.approx([0.0, 0.0, 1.0, 0.0, 24.0, 12.0])


def test_no_shader_ctx_leaves_lanes_zero_and_x_empty():
    ctx = _ctx([_note_view(col=0, lx=0.0, y=100.0, state='tap')])
    u, _f, x, count, _report = nf.feed_from_context(ctx, _IMAGE_MAP)
    rows = _rows(u, count)
    assert not rows[:, 4:].any()
    assert len(x) == 0


def test_translate_vert_note_quad_scales_unit_geometry():
    src = 'void main() { gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; }'
    translated = notitg_compat.translate_vert(src, note_quad=True)
    assert 'u_note_size' in translated
    assert 'vec4(a_pos * u_note_size, 0.0, 1.0)' in translated
    plain = notitg_compat.translate_vert(src)
    assert 'u_note_size' not in plain


def test_int_uniform_names_read_the_declarations():
    from analysis.player.render.storyboard.gl_executor import (
        _int_uniform_names)
    src = ('uniform bool isHold;\nuniform int iCol;\n'
           'uniform float wave;\nuniform sampler2D sampler0;\n')
    assert _int_uniform_names(src) == {'isHold', 'iCol'}
    assert _int_uniform_names(None) == set()
