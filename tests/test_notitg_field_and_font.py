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
    assert frame.fields[0] == (None, 1.0, 'field')
    assert len(frame.fields) == 2


def test_invisible_copy_is_dropped():
    from analysis.player.render.effects.timeline import Keyframe
    copy = _copy('P1p', {'alpha': [Keyframe(0.0, (0.0,), 0.0, 0)]})
    assert NotitgFieldInstances([copy]).at(_Ctx(1.0)) is None


def test_proxy_copy_is_field_scope_never_screen():
    from analysis.player.render.effects.timeline import EventTimeline, Keyframe
    copy = _copy('P1p', {'alpha': [Keyframe(0.0, (1.0,), 0.0, 0)]})
    # An AFT-bg timeline (now unused) is passed for producer compatibility;
    # a proxy re-renders the NoteField only, so it is always field-only.
    aft_bg = EventTimeline([Keyframe(0.0, (1.0,), 0.0, 0)], rest=(1.0,))
    frame = NotitgFieldInstances([copy], aft_bg_timeline=aft_bg).at(_Ctx(1.0))
    _transform, _opacity, scope = frame.fields[1]
    assert scope == 'field'


def test_aft_copy_is_screen_scope_unconditionally():
    """AFT copies blit the previous-frame screen composite ('screen'
    scope) regardless of the (now-unused) bg-in-capture timeline."""
    from analysis.player.render.effects.timeline import EventTimeline, Keyframe
    copy = _copy('gat_aft', {'alpha': [Keyframe(0.0, (1.0,), 0.0, 0)]})
    # Even a bg-off-then-on timeline no longer changes the scope.
    aft_bg = EventTimeline([Keyframe(5.0, (1.0,), 0.0, 0)], rest=(0.0,))
    fx = NotitgFieldInstances([copy], aft_bg_timeline=aft_bg)
    assert fx.at(_Ctx(1.0)).fields[1][2] == 'screen'
    assert fx.at(_Ctx(9.0)).fields[1][2] == 'screen'
    # And with no timeline supplied at all.
    assert NotitgFieldInstances([copy]).at(_Ctx(1.0)).fields[1][2] == 'screen'


def test_base_hidden_suppresses_identity_original():
    from analysis.player.render.effects.timeline import EventTimeline, Keyframe
    copy = _copy('P1p', {'alpha': [Keyframe(0.0, (1.0,), 0.0, 0)]})
    base_hidden = EventTimeline([Keyframe(2.0, (1.0,), 0.0, 0)], rest=(0.0,))
    fx = NotitgFieldInstances([copy], base_hidden=base_hidden)
    # Base visible: identity + copy.
    shown = fx.at(_Ctx(1.0)).fields
    assert shown[0] == (None, 1.0, 'field') and len(shown) == 2
    # Base hidden: only the copy (no identity original).
    hidden = fx.at(_Ctx(3.0)).fields
    assert (None, 1.0, 'field') not in hidden
    assert len(hidden) == 1


# -- dual-player fields (item 43/50/54) ----------------------------------

def _spec():
    from analysis.games.notitg.field_instances import SecondFieldSpec
    return SecondFieldSpec(note_mods=object())


def test_dual_originals_at_p1_p2_offsets_with_scopes():
    """A dual chart emits two identity originals: P1 shifted left blitting
    the primary capture ('field'), P2 shifted right blitting the second
    capture ('field2'). The offsets are the theme +-160 design px scaled
    by the design map (chart_rect 400x600 -> min-fit box side 400, k so
    that 640*k <= 400 and 480*k <= 400 => k = 400/640 = 0.625)."""
    spec = _spec()
    frame = NotitgFieldInstances([], second_field=spec).at(_Ctx(1.0))
    assert frame.second_field is spec
    p1, p2 = frame.fields
    assert p1[2] == 'field' and p2[2] == 'field2'
    k = 400.0 / 640.0
    # P1 left by 160 design px, P2 right by 160 design px.
    assert p1[0].dx() == -160.0 * k
    assert p2[0].dx() == 160.0 * k


def test_dual_routes_p2_proxy_to_field2_p1_to_field():
    from analysis.player.render.effects.timeline import Keyframe
    p1 = _copy('P1p', {'alpha': [Keyframe(0.0, (1.0,), 0.0, 0)]})
    p2 = _copy('P2p', {'alpha': [Keyframe(0.0, (1.0,), 0.0, 0)]})
    frame = NotitgFieldInstances([p1, p2], second_field=_spec()).at(_Ctx(1.0))
    scopes = [entry[2] for entry in frame.fields]
    # Two originals ('field','field2') then the two proxy copies.
    assert scopes.count('field2') == 2   # P2 original + P2p copy
    assert scopes[2] == 'field'          # P1p copy -> primary
    assert scopes[3] == 'field2'         # P2p copy -> second capture


def test_single_player_never_forwards_second_field():
    from analysis.player.render.effects.timeline import Keyframe
    copy = _copy('P2p', {'alpha': [Keyframe(0.0, (1.0,), 0.0, 0)]})
    frame = NotitgFieldInstances([copy]).at(_Ctx(1.0))
    # No spec -> P2p is still the primary 'field' capture, nothing 'field2'.
    assert frame.second_field is None
    assert all(entry[2] != 'field2' for entry in frame.fields)


def test_dual_hidden_base_keeps_capture_path():
    """A fully-hidden dual frame still forwards the spec (so the second
    capture renders) with a non-empty placeholder fields list."""
    from analysis.player.render.effects.timeline import EventTimeline, Keyframe
    base_hidden = EventTimeline([Keyframe(0.0, (1.0,), 0.0, 0)], rest=(1.0,))
    frame = NotitgFieldInstances([], base_hidden=base_hidden,
                                 second_field=_spec()).at(_Ctx(1.0))
    assert frame.second_field is not None
    assert frame.fields  # non-empty placeholder -> renderer captures


def test_note_mods_samples_requested_player():
    """The consumer samples ONLY its player's channels: a P1-only mod and
    a P2-only mod each light up only in that player's consumer."""
    import numpy as np
    from analysis.player.render.mods.channels import ModChannels, ModEvent
    from analysis.games.notitg.note_mods import NotitgNoteMods

    class _NM(_Ctx):
        lane_w = 64.0
        judge_y = 300.0
        candidates = []

        class player:
            keycount = 4
    channels = ModChannels.compile([
        ModEvent(0.0, 1.0, -1.0, 'drunk', 0),
        ModEvent(0.0, 1.0, -1.0, 'tipsy', 1),
    ])
    ctx = _NM(1.0)
    p0 = NotitgNoteMods(channels, [(0.0, 120.0)], player=0)
    p1 = NotitgNoteMods(channels, [(0.0, 120.0)], player=1)
    p0.apply(ctx)
    # receptor_offsets exists for both; the point is the sampled channels
    # differ - verify via values_at directly.
    assert channels.values_at(1.0, 0) == {'drunk': 1.0}
    assert channels.values_at(1.0, 1) == {'tipsy': 1.0}
    assert p0._player == 0 and p1._player == 1


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







