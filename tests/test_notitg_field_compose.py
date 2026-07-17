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
