"""Field-instance transform composition: engine transform order, chain
folding, oscillator overlay, AFT flip translation, visibility gating."""
import numpy as np
import pytest

from analysis.games.notitg import field_compose
from analysis.player.render.effects.timeline import EventTimeline, Keyframe


def _kf(t, value):
    return Keyframe(t, (value,), 0.0, 0)


def _channel(keyframes, **kwargs):
    return field_compose.TransformChannel(
        [field_compose.link_timelines(keyframes)], **kwargs)


def _map_capture(H, x, y):
    px, py, w = np.array([x, y, 1.0]) @ H
    return px / w, py / w


def test_center_unit_instance_is_identity():
    H, alpha = _channel({'x': [_kf(0.0, 320.0)],
                         'y': [_kf(0.0, 240.0)]}).at(1.0)
    assert alpha == 1.0
    assert np.allclose(H, np.eye(3))


def test_rotation_applies_before_scale():
    """Engine order: content rotates, THEN scales - a non-uniform scale
    stretches the rotated result. Capture (330, 240) is +10 x from
    centre; a 90 deg spin turns it to +10 y, then scale_y 2 stretches
    it to +20 y."""
    H, _alpha = _channel({'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)],
                          'rotation': [_kf(0.0, 90.0)],
                          'scale_y': [_kf(0.0, 2.0)]}).at(1.0)
    assert _map_capture(H, 330.0, 240.0) == pytest.approx((320.0, 260.0))


def test_skew_applies_before_rotation():
    """Engine order: skew shears the content first, the rotation then
    turns the sheared result. Capture (320, 250) is +10 y from centre;
    skew_x 1 shears it to (+10, +10), a 90 deg spin turns that to
    (-10, +10)."""
    H, _alpha = _channel({'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)],
                          'rotation': [_kf(0.0, 90.0)],
                          'skew_x': [_kf(0.0, 1.0)]}).at(1.0)
    assert _map_capture(H, 320.0, 250.0) == pytest.approx((310.0, 250.0))


def test_rotation_y_projects_perspective():
    """An out-of-plane turn is true perspective, not a flat squash: the
    homography goes projective and the field narrows."""
    H, _alpha = _channel({'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)],
                          'rotation_y': [_kf(0.0, 60.0)]}).at(1.0)
    from analysis.player.render import transform3d
    assert not transform3d.is_affine(H)
    left, _ = _map_capture(H, 160.0, 240.0)
    right, _ = _map_capture(H, 480.0, 240.0)
    assert abs(right - left) < 320.0


def test_chain_composes_child_onto_parent():
    """A parent frame's spin carries its child copy around the parent's
    position (rotation-around-offset, which per-property folding cannot
    express): child at +100 x inside a parent at centre rotated 90 deg
    lands at +100 y."""
    parent = field_compose.link_timelines(
        {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)],
         'rotation': [_kf(0.0, 90.0)]})
    child = field_compose.link_timelines({'x': [_kf(0.0, 100.0)]})
    H, _alpha = field_compose.TransformChannel([parent, child]).at(1.0)
    assert _map_capture(H, 320.0, 240.0) == pytest.approx((320.0, 340.0))


def test_hidden_anywhere_in_chain_hides():
    parent = field_compose.link_timelines({'hidden': [_kf(0.0, 1.0)]})
    child = field_compose.link_timelines({'x': [_kf(0.0, 320.0)]})
    assert field_compose.TransformChannel([parent, child]).at(1.0) is None


def test_alpha_multiplies_along_chain():
    parent = field_compose.link_timelines({'alpha': [_kf(0.0, 0.5)]})
    child = field_compose.link_timelines({'alpha': [_kf(0.0, 0.5)]})
    _H, alpha = field_compose.TransformChannel([parent, child]).at(1.0)
    assert alpha == pytest.approx(0.25)


def test_zero_scale_is_invisible():
    assert _channel({'scale_x': [_kf(0.0, 0.0)]}).at(1.0) is None


def test_t0_clamp_holds_load_state():
    """Samples before the compile start read the load state, not the
    pre-load rests: a proxy hidden from its InitCommand must not flash
    visible at t=0."""
    keyframes = {'hidden': [_kf(0.5, 1.0)]}
    assert _channel(keyframes, t0=0.5).at(0.0) is None
    assert _channel(keyframes).at(0.0) is not None


def test_aft_flip_translates_basezoom_sign():
    """basezoomy(-1) on an AFT sampler compensates bottom-up GL captures;
    our capture is top-down, so the sign is negated - the poked -1 lands
    upright, and an unpoked +1 yields the raw flipped texture."""
    link = field_compose.link_timelines(
        {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)],
         'base_scale_y': [_kf(0.0, -1.0)]})
    upright = field_compose.TransformChannel([link], flip_base_y=True)
    H, _alpha = upright.at(1.0)
    assert np.allclose(H, np.eye(3))

    unpoked = field_compose.link_timelines(
        {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)]})
    flipped = field_compose.TransformChannel([unpoked], flip_base_y=True)
    H, _alpha = flipped.at(1.0)
    assert _map_capture(H, 320.0, 100.0) == pytest.approx((320.0, 380.0))


def test_aft_flip_mirrors_the_valign_anchor():
    """The afthell band rig: basezoomy(-1) + valign(0.75) + y(100). The
    engine's flipped quad spans the OPPOSITE side of the position
    (valign 0.75 flipped places like valign 0.25 upright: quad y
    [-120, 360] about the position), so canceling the sign must negate
    the anchor offset too - without the mirror the band lands ~half a
    screen high and shows as a sliver."""
    link = field_compose.link_timelines(
        {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 100.0)],
         'base_scale_y': [_kf(0.0, -1.0)], 'valign': [_kf(0.0, 0.75)]})
    H, _ = field_compose.TransformChannel([link], flip_base_y=True).at(1.0)
    # Source center (240) sits at position + mirrored anchor (+120).
    assert _map_capture(H, 320.0, 240.0) == pytest.approx((320.0, 220.0))
    assert _map_capture(H, 320.0, 480.0) == pytest.approx((320.0, 460.0))


def test_aft_flip_swaps_vertical_crop_edges():
    """cropbottom on a flipped sampler hides the engine quad's local
    bottom, which the source mirror puts at OUR source's top: crop_at
    reports it as a top inset (the surviving band is the source's
    bottom half, exactly what the engine shows on screen)."""
    link = field_compose.link_timelines(
        {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)],
         'base_scale_y': [_kf(0.0, -1.0)],
         'crop_bottom': [_kf(0.0, 0.5)], 'crop_left': [_kf(0.0, 0.1)]})
    flipped = field_compose.TransformChannel([link], flip_base_y=True)
    assert flipped.crop_at(1.0) == pytest.approx((0.1, 0.5, 0.0, 0.0))
    plain = field_compose.TransformChannel([link])
    assert plain.crop_at(1.0) == pytest.approx((0.1, 0.0, 0.0, 0.5))


def test_player_instance_rests_at_versus_seats():
    for number, dx in ((1, -160.0), (2, 160.0)):
        inst = field_compose.player_instance(number, None)
        assert inst['kind'] == 'player' and inst['player'] == number
        H, alpha = inst['transform'].at(1.0)
        assert alpha == 1.0
        assert H[2, 0] == pytest.approx(dx)
        assert H[2, 1] == pytest.approx(0.0)


def test_oscillator_deltas_ride_recorded_stream():
    deltas = {'x': EventTimeline([_kf(0.0, 25.0)], rest=(0.0,))}
    inst = field_compose.player_instance(
        1, {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)]}, deltas)
    H, _alpha = inst['transform'].at(1.0)
    assert H[2, 0] == pytest.approx(25.0)


def test_harvest_copies_and_players_share_the_contract():
    copies = [{'name': 'P2p', 'source': 'P2p',
               'timelines': {'x': EventTimeline([_kf(0.0, 200.0)],
                                                rest=(0.0,))}},
              {'name': 'gat_aft', 'source': 'gat_aft', 'timelines': {}}]
    instances = field_compose.harvest_instances(copies, dual=True)
    kinds = [(inst['kind'], inst['player']) for inst in instances]
    assert kinds == [('proxy', 2), ('aft', 0), ('player', 1), ('player', 2)]
    H, _alpha = instances[0]['transform'].at(1.0)
    assert H[2, 0] == pytest.approx(200.0 - 320.0)


# -- fork transform-order / spherical rotation / skew-order (parity + effect) -

def _instant_kf(t, value_tuple):
    """A whole-tuple instant keyframe (rotation_order string / quat 4-tuple)."""
    return Keyframe(t, value_tuple, 0.0, 0)


def test_rest_link_ignores_new_order_channels_byte_identical():
    """An untouched link (default order, identity quat, skew-after) composes
    to exactly the pre-order matrix - the parity anchor for every flat
    chart and gat."""
    from analysis.player.render import transform3d as t3
    keyframes = {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)],
                 'rotation': [_kf(0.0, 30.0)], 'rotation_x': [_kf(0.0, 20.0)],
                 'scale_x': [_kf(0.0, 1.5)], 'skew_x': [_kf(0.0, 0.2)]}
    with_channels, _a = _channel(keyframes).at(1.0)
    m = t3.rotate_xyz(20.0, 0.0, 30.0)
    m = m @ t3.scale(1.5, 1.0, 1.0)
    m = m @ t3.translate(320.0, 240.0, 0.0)
    m = t3.skew_x(0.2) @ m
    expected = t3.homography(field_compose._TO_CONTENT @ m,
                             field_compose.field_projection.design_projection())
    assert np.allclose(with_channels, t3.normalize_h(expected))


def test_rotation_order_channel_changes_the_map():
    base = {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)],
            'rotation_x': [_kf(0.0, 40.0)], 'rotation_y': [_kf(0.0, 60.0)]}
    H_xyz, _ = _channel(base).at(1.0)
    reordered = {**base, 'rotation_order': [_instant_kf(0.0, ('zyx',))]}
    H_zyx, _ = _channel(reordered).at(1.0)
    assert not np.allclose(H_xyz, H_zyx)


def test_identity_quat_channel_is_a_no_op():
    base = {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)],
            'rotation': [_kf(0.0, 45.0)]}
    H_no_quat, _ = _channel(base).at(1.0)
    with_ident = {**base, 'quat': [_instant_kf(0.0, (0.0, 0.0, 0.0, 1.0))]}
    H_ident, _ = _channel(with_ident).at(1.0)
    assert np.allclose(H_no_quat, H_ident)


def test_spherical_quat_channel_tilts_the_field():
    from analysis.player.render import transform3d as t3
    base = {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)]}
    H_flat, _ = _channel(base).at(1.0)
    q = t3.quat_from_axis('y', 45.0)
    tilted = {**base, 'quat': [_instant_kf(0.0, q)]}
    H_tilt, _ = _channel(tilted).at(1.0)
    assert not np.allclose(H_flat, H_tilt)


def test_skew_before_flag_flips_the_compose_side():
    base = {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)],
            'rotation': [_kf(0.0, 30.0)], 'skew_x': [_kf(0.0, 0.4)]}
    H_after, _ = _channel(base).at(1.0)
    before = {**base, 'skew_x_before': [_kf(0.0, 1.0)]}
    H_before, _ = _channel(before).at(1.0)
    assert not np.allclose(H_after, H_before)


# -- crop + align (the AFT sampler band rig) ---------------------------------

_CENTERED = {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)]}


def test_crop_at_rests_none():
    assert _channel(_CENTERED).crop_at(1.0) is None


def test_crop_at_reads_the_leaf_crop_channels():
    ch = _channel({**_CENTERED, 'crop_left': [_kf(0.0, 0.25)],
                   'crop_top': [_kf(0.0, 0.5)]})
    assert ch.crop_at(1.0) == pytest.approx((0.25, 0.5, 0.0, 0.0))


def test_crop_does_not_touch_the_transform():
    plain, _ = _channel(_CENTERED).at(1.0)
    cropped, _ = _channel({**_CENTERED,
                           'crop_bottom': [_kf(0.0, 0.5)]}).at(1.0)
    assert np.allclose(plain, cropped)


def test_valign_anchors_content_on_the_position():
    """valign 0.75 puts the texture's 75% line on the actor's y: the
    half-screen band trick (cropbottom 0.5 + valign 0.75 shows the top
    half in place). Texture y=360 lands on y=240; the center rides up."""
    H, _ = _channel({**_CENTERED, 'valign': [_kf(0.0, 0.75)]}).at(1.0)
    assert _map_capture(H, 320.0, 360.0) == pytest.approx((320.0, 240.0))
    assert _map_capture(H, 320.0, 240.0) == pytest.approx((320.0, 120.0))


def test_halign_scales_with_the_leaf_zoom():
    """The anchor shift is content-side: a zoomed sprite shifts by the
    ZOOMED anchor distance (the engine offsets the quad's vertices, and
    the zoom then scales them)."""
    H, _ = _channel({**_CENTERED, 'halign': [_kf(0.0, 0.0)],
                     'scale_x': [_kf(0.0, 0.5)]}).at(1.0)
    # halign 0: left edge on the position. Texture x=0 lands at x=320.
    assert _map_capture(H, 0.0, 240.0) == pytest.approx((320.0, 240.0))


def test_centered_align_is_identity():
    H, _ = _channel({**_CENTERED, 'halign': [_kf(0.0, 0.5)],
                     'valign': [_kf(0.0, 0.5)]}).at(1.0)
    assert np.allclose(H, np.eye(3))


def test_instances_carry_crop_in_their_entries():
    from types import SimpleNamespace

    from analysis.games.notitg.field_instances import NotitgFieldInstances

    cropped = field_compose.instance(
        'A', 'aft', 0,
        [field_compose.link_timelines({'crop_left': [_kf(0.0, 0.5)]})],
        aft_order='post')
    plain = field_compose.instance(
        'B', 'aft', 0, [field_compose.link_timelines(None)],
        aft_order='post')
    frame = NotitgFieldInstances([cropped, plain]).at(
        SimpleNamespace(t_now=1.0, chart_rect=(0, 0, 640, 480)))
    base, first, second = frame.fields
    assert first[4] == pytest.approx((0.5, 0.0, 0.0, 0.0))
    assert second[4] is None


# -- playfield transform mods -----------------------------------------------
#
# `x`/`y`/`z`/`rotation*`/`zoom*`/`skewx` move the whole notefield, so they
# compose as one more link under the field's chain rather than displacing
# each arrow (field_compose.playfield_mod_link).

def _mods(modstring, player=1, t_start=1.0, t_end=3.0):
    from analysis.games.notitg.mod_channels import compile_mod_channels
    return compile_mod_channels([{'t_start': t_start, 't_end': t_end,
                                  'modstring': modstring, 'player': player}])


def test_playfield_mod_link_is_none_without_the_mods():
    assert field_compose.playfield_mod_link(_mods('*-1 50 drunk'), 1) is None
    assert field_compose.playfield_mod_link(None, 1) is None


def test_playfield_mods_reach_their_link_units():
    """Pixels and degrees are the raw percent; the scale family is the
    fraction (`*-1 72 zoomx` is 0.72x, not 72x)."""
    link = field_compose.playfield_mod_link(
        _mods('*-1 -215 x, *-1 22 rotationz, *-1 72 zoomx'), 1)
    assert link['x'].sample(2.0) == pytest.approx((-215.0,))
    assert link['rotation'].sample(2.0) == pytest.approx((22.0,))
    assert link['scale_x'].sample(2.0) == pytest.approx((0.72,))


def test_zoom_mods_rest_at_full_size():
    """`clearall` leaves the scale family at 100%, not 0 - the field keeps
    its size before the chart's first zoom window and after its last, where
    a 0 rest would collapse it to a point."""
    link = field_compose.playfield_mod_link(_mods('*-1 40 zoomy'), 1)
    assert link['scale_y'].sample(0.0) == pytest.approx((1.0,))
    assert link['scale_y'].sample(2.0) == pytest.approx((0.4,))
    # Reverting at clearall speed (1.0/sec) from 0.4 back to 1.0.
    assert link['scale_y'].sample(9.0) == pytest.approx((1.0,))


def test_playfield_mods_compose_inside_the_field_chain():
    """The mod link is INNERMOST: its offset rides the actor chain above it
    rather than replacing it, so a seated field moves BY the mod."""
    channel = field_compose.TransformChannel(field_compose.with_playfield_mods(
        [field_compose.link_timelines({'x': [_kf(0.0, 320.0)],
                                       'y': [_kf(0.0, 240.0)]})],
        _mods('*-1 -160 x'), 1))
    H, _alpha = channel.at(2.0)
    assert _map_capture(H, 320.0, 240.0) == pytest.approx((160.0, 240.0))


# -- face culling (SetCullMode) ----------------------------------------------
#
# The two-sided-card idiom: front/back sprite pairs, the back at
# rotationx+180, both `cullmode('front')`, so the projected winding picks
# exactly ONE face per frame. Winding is judged in ENGINE terms - the chart's
# basezoomy(-1) reverses it, and flip_base_y translates that sign away, so a
# flipped leaf's engine winding is the negated determinant of ours.

def _culled_channel(pokes, flip=False):
    return field_compose.TransformChannel(
        [field_compose.link_timelines(
            {'x': [_kf(0.0, 320.0)], 'y': [_kf(0.0, 240.0)], **pokes})],
        flip_base_y=flip)


def test_cull_front_drops_the_front_face_and_keeps_the_flipped_one():
    front = _culled_channel({'cull': [_kf(0.0, 2.0)]})
    assert front.at(1.0) is None, 'front-facing + cull front: dropped'
    turned = _culled_channel({'cull': [_kf(0.0, 2.0)],
                              'rotation_x': [_kf(0.0, 180.0)]})
    assert turned.at(1.0) is not None, 'the 180-degree face survives'


def test_cull_back_is_the_complement():
    front = _culled_channel({'cull': [_kf(0.0, 1.0)]})
    assert front.at(1.0) is not None
    turned = _culled_channel({'cull': [_kf(0.0, 1.0)],
                              'rotation_x': [_kf(0.0, 180.0)]})
    assert turned.at(1.0) is None


def test_cull_judges_engine_winding_for_flipped_leaves():
    # A chart sampler pokes basezoomy(-1); flip_base_y negates that to +1,
    # so OUR det is positive while the ENGINE winding is reversed. Culling
    # must judge the ENGINE side: cull 'front' KEEPS the sampler (that is
    # how every sampler in the chart survives its own cullmode) and drops
    # its rotationx+180 backface twin - gat 2's chicken pairs exactly.
    sampler = _culled_channel({'cull': [_kf(0.0, 2.0)],
                               'base_scale_y': [_kf(0.0, -1.0)]}, flip=True)
    assert sampler.at(1.0) is not None
    backface = _culled_channel({'cull': [_kf(0.0, 2.0)],
                                'base_scale_y': [_kf(0.0, -1.0)],
                                'rotation_x': [_kf(0.0, 180.0)]}, flip=True)
    assert backface.at(1.0) is None


def test_uncalled_actors_never_pay_for_culling():
    plain = _culled_channel({})
    assert plain.at(1.0) is not None
    assert plain.cull_mode_at(1.0) == 0.0
