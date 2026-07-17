"""NotITG AFT/proxy field-instance producer, RecordingActor getters +
AFT classification, and SM bitmap-font parsing/layout."""
from pathlib import Path

import pytest

pytest.importorskip('lupa')
pytest.importorskip('PySide6')

from PySide6.QtCore import QPointF

from analysis.games.notitg.field_instances import (NotitgFieldInstances,
                                                   _copy_transform, _design_map)
from analysis.games.notitg.modfile import compile_modfile
from analysis.games.notitg.recording_actor import RecordingActor
from analysis.player.render.storyboard import bitmap_font

_GAT_SM = Path('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
               'UKSRT8/5. gat/gat.sm')
_FONTS = Path('/mnt/Yucky/Rhythm Games/Players/NotITG/Themes/default/Fonts')


# -- RecordingActor getters + AFT classification --------------------------

def test_getter_reads_last_set_scalar():
    actor = RecordingActor()
    actor.poke('x', [120])
    actor.poke('zoom', [0.5])
    assert actor.get('x') == 120
    assert actor.read('GetX') == 120
    assert actor.read('GetZoom') == 0.5


def test_getter_of_unset_property_returns_rest():
    actor = RecordingActor()
    assert actor.read('GetY') == 0.0
    assert actor.read('GetZoom') == 1.0


def test_unmodelled_getter_returns_none():
    assert RecordingActor().read('GetChild') is None


def test_getrotation_returns_three_components():
    actor = RecordingActor()
    actor.poke('rotationz', [30])
    actor.poke('rotationx', [10])
    assert actor.getrotation() == (10.0, 0.0, 30.0)


def test_settexturename_marks_actor_as_aft():
    aft = RecordingActor()
    aft.poke('SetTextureName', ['gat_aft'])
    assert aft.is_aft
    assert aft.read('GetTexture') == 'aft:gat_aft'


def test_settexture_from_aft_marks_copy_source():
    copy = RecordingActor()
    copy.poke('SetTexture', ['aft:gat_aft'])
    assert copy.aft_source == 'gat_aft'
    assert not copy.is_aft


def test_plain_texture_is_not_an_aft_copy():
    sprite = RecordingActor()
    sprite.poke('SetTexture', ['white'])
    assert sprite.aft_source is None


def test_basezoom_records_separately_from_zoom():
    actor = RecordingActor()
    actor.poke('basezoomy', [-1])
    actor.poke('zoom', [0.8])
    frames = actor.keyframes()
    assert frames['base_scale_y'][-1].values == (-1.0,)
    assert frames['scale_y'][-1].values == (0.8,)


# -- field-instance transform ---------------------------------------------

def test_identity_copy_is_screen_identity():
    k, ox, oy = _design_map((100.0, 0.0, 400.0, 600.0))
    transform = _copy_transform(320, 240, 0, 1, 1, k, ox, oy)
    assert transform.isIdentity()


def test_copy_translation_scales_by_design_ratio():
    chart_rect = (100.0, 0.0, 400.0, 600.0)
    k, ox, oy = _design_map(chart_rect)
    centre = QPointF(ox + 320 * k, oy + 240 * k)
    shifted = _copy_transform(420, 240, 0, 1, 1, k, ox, oy).map(centre)
    assert shifted.x() == pytest.approx(centre.x() + 100 * k)
    assert shifted.y() == pytest.approx(centre.y())


def test_vertical_mirror_flips_about_screen_centre():
    chart_rect = (100.0, 0.0, 400.0, 600.0)
    k, ox, oy = _design_map(chart_rect)
    point = QPointF(ox + 320 * k, oy + 100 * k)  # design (320, 100)
    mirrored = _copy_transform(320, 240, 0, 1, -1, k, ox, oy).map(point)
    assert mirrored.y() == pytest.approx(oy + (480 - 100) * k)


class _Ctx:
    chart_rect = (100.0, 0.0, 400.0, 600.0)

    def __init__(self, t):
        self.t_now = t


def _copy(name, keyframes):
    from analysis.player.render.storyboard.model import build_timelines
    rests = {'x': 0.0, 'y': 0.0, 'rotation': 0.0, 'scale_x': 1.0,
             'scale_y': 1.0, 'base_scale_x': 1.0, 'base_scale_y': 1.0,
             'alpha': 1.0}
    return {'name': name, 'source': name,
            'timelines': build_timelines(rests=rests, keyframes=keyframes)}


def test_identity_original_always_present_with_copies():
    from analysis.player.render.effects.timeline import Keyframe
    copy = _copy('P1p', {'x': [Keyframe(0.0, (160.0,), 0.0, 0)],
                         'alpha': [Keyframe(0.0, (1.0,), 0.0, 0)]})
    frame = NotitgFieldInstances([copy]).at(_Ctx(1.0))
    assert frame.fields[0] == (None, 1.0)
    assert len(frame.fields) == 2


def test_invisible_copy_is_dropped():
    from analysis.player.render.effects.timeline import Keyframe
    copy = _copy('P1p', {'alpha': [Keyframe(0.0, (0.0,), 0.0, 0)]})
    assert NotitgFieldInstances([copy]).at(_Ctx(1.0)) is None


# -- SM bitmap font -------------------------------------------------------

@pytest.mark.skipif(not _FONTS.exists(), reason='NotITG theme fonts absent')
def test_font_parses_grid_and_advances():
    font = bitmap_font.load_font('_eurostile normal', [str(_FONTS)])
    assert font is not None
    assert (font.cols, font.rows) == (16, 16)
    assert font.advance(ord('A')) > 0


@pytest.mark.skipif(not _FONTS.exists(), reason='NotITG theme fonts absent')
def test_font_cell_maps_codepoint_to_grid():
    font = bitmap_font.load_font('_eurostile normal', [str(_FONTS)])
    # 'A' = 65: col 65 % 16 = 1, row 65 // 16 = 4, in a 512x512 / 16 grid.
    cell = font.cell(ord('A'), 512, 512)
    assert cell == (32.0, 128.0, 32.0, 32.0)


def test_unresolvable_font_returns_none():
    assert bitmap_font.load_font('no such font', ['/tmp']) is None


# -- integration on the gat pilot -----------------------------------------

@pytest.mark.skipif(not _GAT_SM.exists(), reason='NotITG gat pilot not present')
def test_gat_field_copies_include_proxies_and_aft():
    result = compile_modfile(str(_GAT_SM))
    names = {c['name'] for c in result['field_copies']}
    assert {'P1p', 'P2p'} <= names
    assert any(c['source'].startswith('gat_aft') for c in result['field_copies'])


@pytest.mark.skipif(not _GAT_SM.exists(), reason='NotITG gat pilot not present')
def test_gat_proxies_split_screen_when_active():
    """The P1p/P2p proxies fan out to opposite half-screens once the
    tiling section starts, producing an off-centre field copy."""
    result = compile_modfile(str(_GAT_SM))
    copies = {c['name']: c for c in result['field_copies']}
    p1x = copies['P1p']['timelines']['x'].sample(370.0)[0]
    p2x = copies['P2p']['timelines']['x'].sample(370.0)[0]
    assert p1x != p2x
    assert min(p1x, p2x) < 320 < max(p1x, p2x)


@pytest.mark.skipif(not _GAT_SM.exists(), reason='NotITG gat pilot not present')
def test_gat_bitmaptext_actors_resolve_font():
    result = compile_modfile(str(_GAT_SM))
    bitmaptexts = [e for e in _flatten(result['tree'])
                   if e.kind == 'bitmaptext']
    assert bitmaptexts
    assert all(e.font is not None for e in bitmaptexts)


def _flatten(elements):
    for element in elements:
        yield element
        yield from _flatten(element.children)
