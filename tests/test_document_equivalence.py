"""Exact-equivalence gate for the document-driven storyboard render.

Phase 3 routes StoryboardEffect THROUGH the CompiledDocument node tree.
The acceptance bar is ZERO visual change, so these tests render BOTH the
Element-walk (`StoryboardEffect`) and the node-walk
(`DocumentStoryboardRenderer`) into offscreen images at a battery of
timestamps over a battery of storyboards -- flat draws, nested ActorFrame
groups, sheet sprites, bitmap text, the SM 'white' texture, hidden gates,
group alpha, multi-z banding, and the clipped design box -- and assert the
two produce byte-identical frames AND identical z-slot structure. When
this holds, the document path can be the default with no oracle drift.

The document leaves rendering to StoryboardEffect's paint helpers, so
equivalence is expected to be exact (not merely close); the test asserts
raw-buffer equality, which would catch a single-pixel divergence.
"""
import json

import numpy as np
import pytest
from types import SimpleNamespace

from PySide6.QtGui import QImage, QPainter

from analysis.games.fluxis.fsb_storyboard import parse_fsb
from analysis.player.render.document.builder import storyboard_document
from analysis.player.render.document.design_space import DesignSpace, FIT_MIN
from analysis.player.render.document.render import DocumentStoryboardRenderer
from analysis.player.render.storyboard import Storyboard, StoryboardEffect
from analysis.player.render.storyboard.model import Element, build_timelines

_SIZE = 220
_RECT = (0, 0, _SIZE, _SIZE)
_TIMES = (0.0, 0.4, 0.5, 1.2, 2.0, 2.999, 3.5, 5.0, 7.0)


def _ctx(t, rect=_RECT):
    return SimpleNamespace(t_now=t, chart_rect=rect)


def _leaf(kind, z=0, z_index=0, t_start=0.0, t_end=float('inf'), **overrides):
    fields = dict(
        kind=kind, z=z, z_index=z_index, t_start=t_start, t_end=t_end,
        anchor=(0.0, 0.0), origin=(0.0, 0.0), timelines=build_timelines())
    fields.update(overrides)
    return Element(**fields)


def _group(children, **overrides):
    return _leaf('group', children=tuple(children), **overrides)


def _design_for(sb):
    return DesignSpace(width=sb.design_w, height=sb.design_h,
                       fit=sb.fit, clip=sb.clip_design_box)


def _render(effect, t):
    """Render one storyboard effect's frame into an offscreen ARGB image;
    None when the effect draws nothing at t."""
    frame = effect.at(_ctx(t))
    if frame is None:
        return None, ()
    image = QImage(_SIZE, _SIZE, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    for _z, draw in frame.draws:
        draw(_ctx(t), painter)
    painter.end()
    zs = tuple(z for z, _draw in frame.draws)
    return image, zs


def _buffer(image):
    ptr = image.constBits()
    return bytes(ptr[: image.sizeInBytes()])


def _element_effect(storyboard):
    """A StoryboardEffect pinned to the direct Element walk, regardless of
    the module default -- the equivalence reference."""
    import analysis.player.render.storyboard.render as render_mod

    saved = render_mod.USE_DOCUMENT_PATH
    render_mod.USE_DOCUMENT_PATH = False
    try:
        return StoryboardEffect(storyboard)
    finally:
        render_mod.USE_DOCUMENT_PATH = saved


def _assert_equivalent(storyboard):
    element_effect = _element_effect(storyboard)
    document, index = storyboard_document(storyboard, _design_for(storyboard))
    document_effect = DocumentStoryboardRenderer(
        document, index, storyboard, element_effect)
    assert bool(element_effect) == bool(document_effect)
    for t in _TIMES:
        el_image, el_zs = _render(element_effect, t)
        doc_image, doc_zs = _render(document_effect, t)
        assert el_zs == doc_zs, f'z-slot banding differs at t={t}: {el_zs} vs {doc_zs}'
        both_none = el_image is None and doc_image is None
        if both_none:
            continue
        assert el_image is not None and doc_image is not None, (
            f'one path drew and the other did not at t={t}')
        assert _buffer(el_image) == _buffer(doc_image), (
            f'pixels differ at t={t}')


# ── fixtures ------------------------------------------------------------


def _fsb(tmp_path, payload):
    path = tmp_path / 'sb.fsb'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def _fsb_element(**overrides):
    element = {
        'type': 0, 'layer': 2, 'z-index': 0,
        'start': 1000.0, 'end': 3000.0,
        'anchor': 18, 'origin': 18, 'x': 10.0, 'y': -20.0,
        'width': 100.0, 'height': 50.0,
        'color': 0xFF0000FF, 'parameters': {}, 'animations': [],
    }
    element.update(overrides)
    return element


def test_equiv_fluxis_flat_rects(tmp_path):
    fade = {'start': 1000.0, 'duration': 1000.0, 'easing': 0,
            'type': 7, 'start-value': '1', 'end-value': '0'}
    sb = parse_fsb(_fsb(tmp_path, {'elements': [
        _fsb_element(animations=[fade]),
        _fsb_element(layer=0, x=-30.0, y=40.0, color=0x00FF00FF),
    ]}))
    _assert_equivalent(sb)


def test_equiv_multiple_z_slots():
    below = _leaf('rect', z=-5, timelines=build_timelines(
        {'x': 20.0, 'y': 20.0, 'w': 30.0, 'h': 30.0}))
    mid = _leaf('rect', z=0, timelines=build_timelines(
        {'x': 90.0, 'y': 90.0, 'w': 30.0, 'h': 30.0, 'color': (0, 1, 0)}))
    above = _leaf('rect', z=3, timelines=build_timelines(
        {'x': 150.0, 'y': 150.0, 'w': 30.0, 'h': 30.0, 'color': (0, 0, 1)}))
    _assert_equivalent(Storyboard(_SIZE, _SIZE, 'height', (below, mid, above)))


def test_equiv_nested_group_transforms():
    child = _leaf('rect', timelines=build_timelines(
        {'x': 60.0, 'w': 12.0, 'h': 12.0}))
    inner = _group([child], anchor=(0.0, 0.0),
                   timelines=build_timelines({'rotation': 45.0}))
    outer = _group([inner], anchor=(0.5, 0.5),
                   timelines=build_timelines({'rotation': 30.0, 'x': 10.0}))
    _assert_equivalent(Storyboard(_SIZE, _SIZE, 'height', (outer,)))


def test_equiv_group_alpha_and_hidden():
    faded = _leaf('rect', timelines=build_timelines(
        {'x': 40.0, 'w': 20.0, 'h': 20.0, 'alpha': 0.5}))
    hidden_child = _leaf('rect', timelines=build_timelines(
        {'x': 120.0, 'w': 20.0, 'h': 20.0, 'hidden': 1.0}))
    grp = _group([faded, hidden_child], anchor=(0.5, 0.5),
                 timelines=build_timelines({'alpha': 0.4}))
    _assert_equivalent(Storyboard(_SIZE, _SIZE, 'height', (grp,)))


def test_equiv_child_window_culls_while_siblings_draw():
    early = _leaf('rect', t_start=0.0, t_end=2.0,
                  timelines=build_timelines({'x': 10.0, 'w': 10.0, 'h': 10.0}))
    late = _leaf('rect', t_start=5.0, t_end=9.0,
                 timelines=build_timelines({'x': 120.0, 'w': 10.0, 'h': 10.0}))
    grp = _group([early, late], timelines=build_timelines())
    _assert_equivalent(Storyboard(_SIZE, _SIZE, 'height', (grp,)))


def test_equiv_group_window_culls_whole_subtree():
    child = _leaf('rect', timelines=build_timelines({'w': 20.0, 'h': 20.0}))
    grp = _group([child], t_start=5.0, t_end=10.0,
                 timelines=build_timelines())
    _assert_equivalent(Storyboard(_SIZE, _SIZE, 'height', (grp,)))


def test_equiv_white_sprite():
    sprite = _leaf('sprite', asset='white',
                   timelines=build_timelines(
                       {'x': 40.0, 'y': 40.0, 'w': 30.0, 'h': 30.0}))
    _assert_equivalent(Storyboard(_SIZE, _SIZE, 'height', (sprite,)))


def test_equiv_clipped_design_box():
    # NotITG's hard 640x480 crop: an actor running past the design edge
    # must clip identically through both walks.
    off = _leaf('rect', timelines=build_timelines(
        {'x': 600.0, 'y': 400.0, 'w': 200.0, 'h': 200.0}))
    inside = _leaf('rect', timelines=build_timelines(
        {'x': 100.0, 'y': 100.0, 'w': 40.0, 'h': 40.0, 'color': (0, 1, 0)}))
    sb = Storyboard(640, 480, 'min', (off, inside), clip_design_box=True)
    _assert_equivalent(sb)


def _sheet_png(tmp_path, cols, rows, cell=16):
    from PySide6.QtGui import QColor, QImage, QPainter
    img = QImage(cols * cell, rows * cell, QImage.Format.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    for row in range(rows):
        for col in range(cols):
            frame = col + row * cols
            p.fillRect(col * cell, row * cell, cell, cell,
                       QColor(frame * 20 % 256, 40, 200))
    p.end()
    path = tmp_path / f'sheet {cols}x{rows}.png'
    img.save(str(path))
    return str(path)


def test_equiv_sheet_sprite_animation(tmp_path):
    from analysis.games.notitg import sprite_sheet as sm_sheet
    asset = _sheet_png(tmp_path, 3, 2)
    el = _leaf('sprite', asset=asset, sheet_cols=3, sheet_rows=2,
               sheet_states=sm_sheet.default_states(6),
               timelines=build_timelines(
                   {'x': 40.0, 'y': 40.0, 'scale_x': 3.0, 'scale_y': 3.0}))
    _assert_equivalent(Storyboard(_SIZE, _SIZE, 'height', (el,)))


def test_equiv_plain_sprite_from_file(tmp_path):
    asset = _sheet_png(tmp_path, 1, 1)
    el = _leaf('sprite', asset=asset, additive=True,
               timelines=build_timelines(
                   {'x': 50.0, 'y': 50.0, 'scale_x': 2.0, 'scale_y': 2.0,
                    'rotation': 15.0}))
    _assert_equivalent(Storyboard(_SIZE, _SIZE, 'height', (el,)))


def test_equiv_hidden_root_in_its_own_z_slot():
    # A hidden root occupying a z-slot alone: the Element walk gates it by
    # window (slot survives, paints nothing) while the node walk gates by
    # visibility. The z-slot banding of the emitted frame must still agree.
    hidden = _leaf('rect', z=-3, timelines=build_timelines(
        {'x': 30.0, 'w': 20.0, 'h': 20.0, 'hidden': 1.0}))
    shown = _leaf('rect', z=1, timelines=build_timelines(
        {'x': 100.0, 'y': 100.0, 'w': 20.0, 'h': 20.0, 'color': (0, 1, 0)}))
    _assert_equivalent(Storyboard(_SIZE, _SIZE, 'height', (hidden, shown)))


def test_equiv_deep_group_multi_z():
    # A group in the below band and independent leaves above it: exercises
    # banding + descent together.
    leaf_child = _leaf('rect', timelines=build_timelines(
        {'x': 30.0, 'w': 10.0, 'h': 10.0}))
    grp = _group([leaf_child], z=-2, anchor=(0.3, 0.3),
                 timelines=build_timelines({'rotation': 20.0}))
    hud = _leaf('ellipse', z=4, timelines=build_timelines(
        {'x': 150.0, 'y': 150.0, 'w': 24.0, 'h': 24.0, 'color': (1, 1, 0)}))
    _assert_equivalent(Storyboard(_SIZE, _SIZE, 'height', (grp, hud)))
