"""NotITG AFT/proxy field-instance producer, RecordingActor getters +
AFT classification, and SM bitmap-font parsing/layout."""
from pathlib import Path

import pytest

pytest.importorskip('lupa')
pytest.importorskip('PySide6')

from PySide6.QtCore import QPointF

from analysis.games.notitg.field_instances import (NotitgFieldInstances,
                                                   _copy_transform, _design_map,
                                                   design_box)
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


# -- design box (min-fit crop) + screen-constant resolution ---------------

def test_design_box_is_centered_min_fit():
    """The 640x480 design box letterboxes ('min' fit) centered in a wide
    chart region - its center coincides with the region center so the
    notefield and actors agree on where 320 is."""
    rect = (0, 0, 900, 600)
    box = design_box(rect)
    assert box.center().x() == pytest.approx(450)
    assert box.center().y() == pytest.approx(300)
    # min fit: scaled by the tighter axis (height here), 4:3 box.
    assert box.height() == pytest.approx(600)
    assert box.width() == pytest.approx(800)


def test_design_map_matches_design_box():
    rect = (0, 0, 900, 600)
    k, ox, oy = _design_map(rect)
    box = design_box(rect)
    assert (ox, oy) == pytest.approx((box.x(), box.y()))
    assert k == pytest.approx(box.height() / 480.0)


def test_screen_constants_resolve_in_classic_commands():
    """`x,SCREEN_CENTER_X` records the numeric center, not a dropped
    None - so an AFT copy's InitCommand base sits at screen center."""
    actor = RecordingActor()
    actor.poke('x', ['SCREEN_CENTER_X'])
    actor.poke('y', ['SCREEN_CENTER_Y'])
    assert actor.get('x') == pytest.approx(320.0)
    assert actor.get('y') == pytest.approx(240.0)


# -- gat: AFT driver, load ordering, background layering ------------------

@pytest.mark.skipif(not _GAT_SM.exists(), reason='NotITG gat pilot not present')
def test_gat_aft_target_driven_by_data_holder_quads():
    """gat_aft_target sits at the identity base (320,240) before its
    driver window and translates/zooms/rotates inside it (beat 1140-1146
    ~ t=369), sampled from the gat_aftx/afty/aftzoom/aftrz quad curves."""
    result = compile_modfile(str(_GAT_SM))
    copies = {c['name']: c for c in result['field_copies']}
    tl = copies['gat_aft_target']['timelines']
    # Base: screen center, unit zoom, no rotation (identity placement).
    assert tl['x'].sample(42.0)[0] == pytest.approx(320.0)
    assert tl['y'].sample(42.0)[0] == pytest.approx(240.0)
    # Driven inside the window: shifted off center and rotated/zoomed.
    assert tl['x'].sample(369.0)[0] != pytest.approx(320.0, abs=1.0)
    assert tl['rotation'].sample(369.0)[0] != pytest.approx(0.0, abs=1.0)


@pytest.mark.skipif(not _GAT_SM.exists(), reason='NotITG gat pilot not present')
def test_gat_intro_chara_spawns_at_load():
    """char_shame's beat-0 Spawn mod_action must win over its later-timed
    InitCommand zoom(0): the intro chara is at unit scale, centered, at
    t=5 (before the load-order fix it sampled scale 0 = invisible)."""
    from analysis.games.notitg.mod_stubs import StubEnvironment
    from analysis.games.notitg.modfile import (_beat_to_seconds,
                                               _load_document, _resolve_lua_dir,
                                               _timing, parse_fgchanges)
    from analysis.games.etterna import sm_chart
    from analysis.player.render.storyboard.model import build_timelines

    entries = parse_fgchanges(str(_GAT_SM))
    lua_dir = _resolve_lua_dir(str(_GAT_SM), entries)
    data = sm_chart.parse_sm(str(_GAT_SM))
    _b, _o, chart = _timing(data)
    to_s = _beat_to_seconds(data, chart)
    root, _lc, _cc = _load_document(lua_dir)
    start_beat = min((b for b, _n, k in entries if k == 'FGCHANGES'),
                     default=0.0)
    env = StubEnvironment(start_beat, to_seconds=to_s)
    env.load_actors(root)
    env.replay_mod_actions()
    tl = build_timelines(keyframes=env.named_actor_keyframes()['char_shame'])
    assert tl['scale_x'].sample(5.0)[0] == pytest.approx(1.0)
    assert tl['scale_y'].sample(5.0)[0] == pytest.approx(1.0)
    assert tl['x'].sample(5.0)[0] == pytest.approx(320.0)


@pytest.mark.skipif(not _GAT_SM.exists(), reason='NotITG gat pilot not present')
def test_gat_background_hoisted_below_notes():
    """The BGCHANGES bg tree is a top-level element at a below-the-notes
    z (behind the field); the foreground tree stays at z=0."""
    result = compile_modfile(str(_GAT_SM))
    tree = result['tree']
    zs = sorted(e.z for e in tree)
    assert zs[0] < 0 < len(tree)  # at least one below-band element
    assert 0 in zs                # foreground stays at z=0
    assert result['has_background'] is True


@pytest.mark.skipif(not _GAT_SM.exists(), reason='NotITG gat pilot not present')
def test_gat_adapter_drops_builtin_background():
    """With its own background actors compiled, the notitg adapter returns
    no #BACKGROUND path (the built-in MapBackground would duplicate it)."""
    from analysis.games.notitg.adapter import NotitgAdapter
    replay = {'filepath': f'{_GAT_SM}::0'}
    assert NotitgAdapter().background_path(replay) is None
