"""Hand-derived tests for the SM/NotITG 3D transform math (transform3d.py).

Each test is a closed-form geometric identity, not a golden dump: rotationx(90)
collapses a quad to a line, rotationy foreshortens by cos in the ortho limit,
the vanish point moves the horizon linearly, ActorFrame nesting equals composed
matrices, an affine stack has an exactly affine homography, qtransform_from_h
round-trips the corners, and batched == scalar. One test cross-checks the 2D
arrow_effects confusionx |cos| zoom against the real ortho rotationx homography.
"""

import numpy as np
import pytest

from analysis.player.render import transform3d as t3

QUAD = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def _corners3(corners, z=0.0):
    c = np.asarray(corners, float)
    return np.column_stack([c, np.full(len(c), z)])


# --- affine core -----------------------------------------------------------

def test_row_vector_translate():
    out = t3.apply([1.0, 2.0, 0.0], t3.translate(10, 20, 0))
    assert np.allclose(out[:2], [11.0, 22.0])


def test_local_matrix_sprite_semantics():
    # scale x2, rotate 90 about z, translate to (5,5): (1,0) -> (5,7).
    L = t3.local_matrix(pos=(5, 5, 0), rot=(0, 0, 90), scl=(2, 2, 1))
    assert np.allclose(t3.apply([1.0, 0.0, 0.0], L)[:2], [5.0, 7.0])


def test_fused_equals_sequence():
    # RageMatrixRotationXYZ fused == Rz @ Ry @ Rx (SM "Rx*Ry*Rz" under B@A).
    rx, ry, rz = 23.0, -47.0, 61.0
    fused = t3.rotate_xyz(rx, ry, rz)
    seq = t3.rotate_z(rz) @ t3.rotate_y(ry) @ t3.rotate_x(rx)
    assert np.allclose(fused, seq)


def test_baserotation_adds_into_rot():
    # local_matrix takes rot already including the baserotation add (per axis).
    tween = (10.0, 0.0, 5.0)
    base = (0.0, 30.0, 0.0)
    combined = tuple(a + b for a, b in zip(tween, base))
    L = t3.local_matrix(rot=combined)
    assert np.allclose(L, t3.rotate_xyz(*combined))


# --- rotationx(90) collapses a centered quad to a line ---------------------

def test_rotationx_90_collapses_to_line():
    # Ortho projection so there is no perspective; a 90deg X tilt folds the
    # quad edge-on -> all screen-y coincide (a horizontal line).
    P = t3.projection(0, 640, 480)
    M = t3.translate(320, 240, 0)  # center it in the viewport
    M = t3.rotate_x(90) @ M
    world = _corners3(QUAD * 50.0)
    H = t3.homography(M, P)
    px = t3.project_corners(QUAD * 50.0, H)
    assert np.ptp(px[:, 1]) < 1e-6  # collapsed vertically -> a line
    assert np.ptp(px[:, 0]) > 1.0   # still has horizontal extent
    _ = world


# --- rotationy foreshortening -> cos(theta) at fov->0 (ortho limit) ---------

@pytest.mark.parametrize("theta", [0.0, 15.0, 35.0, 60.0, 80.0])
def test_rotationy_foreshortens_by_cos_ortho(theta):
    # Under orthographic projection, a Y-axis tilt scales the horizontal
    # (x) extent by cos(theta); vertical extent unchanged.
    P = t3.projection(0, 640, 480)
    M = t3.rotate_y(theta) @ t3.translate(320, 240, 0)
    H = t3.homography(M, P)
    px = t3.project_corners(QUAD * 100.0, H)
    x_extent = np.ptp(px[:, 0])
    y_extent = np.ptp(px[:, 1])
    assert np.isclose(x_extent, 200.0 * abs(np.cos(np.radians(theta))), atol=1e-6)
    assert np.isclose(y_extent, 200.0, atol=1e-6)


# --- vanish point moves the horizon linearly -------------------------------

def test_vanish_shift_moves_horizon_linearly():
    W, H = 640, 480
    far = np.array([W / 2, H / 2, -1e6, 1.0])

    def horizon_y(vany):
        P = t3.projection(60, W, H, vanish=(W / 2, vany))
        return t3.apply(far, P)[1]

    ys = [horizon_y(v) for v in (240.0, 200.0, 160.0, 120.0)]
    deltas = np.diff(ys)
    assert np.allclose(deltas, deltas[0])  # equal steps -> linear in vanishY


# --- ActorFrame nesting == composed matrices -------------------------------

def test_actorframe_nesting():
    parent = t3.local_matrix(pos=(100, 50, 0), rot=(0, 0, 30), scl=(2, 2, 1))
    child = t3.local_matrix(pos=(10, 0, 0), rot=(0, 0, 15), scl=(0.5, 0.5, 1))
    world = t3.compose(parent, child)
    p_local = np.array([4.0, 3.0, 0.0])
    # world = child @ parent: point through child, then parent.
    expected = t3.apply(t3.apply(p_local, child), parent)
    assert np.allclose(t3.apply(p_local, world), expected)


# --- affine-only stack -> exactly affine homography ------------------------

def test_affine_stack_homography_is_affine():
    # No perspective anywhere -> m13 = m23 = 0.
    P = t3.projection(0, 640, 480)  # ortho
    M = t3.local_matrix(pos=(200, 100, 0), rot=(0, 0, 40), scl=(1.5, 0.8, 1),
                        skewx=0.3)
    H = t3.normalize_h(t3.homography(M, P))
    assert t3.is_affine(H)
    assert abs(H[0, 2]) < 1e-9 and abs(H[1, 2]) < 1e-9  # Qt m13, m23


def test_perspective_stack_is_not_affine():
    # A z-tilted plane under real fov is genuinely projective.
    P = t3.projection(60, 640, 480)
    M = t3.rotate_x(40) @ t3.translate(320, 240, 0)
    H = t3.homography(M, P)
    assert not t3.is_affine(H)


# --- round-trip: qtransform maps the 4 corners to the projected corners -----

def test_qtransform_round_trip():
    from PySide6.QtCore import QPointF

    P = t3.projection(55, 640, 480, vanish=(280, 210))
    M = t3.rotate_xyz(25, -18, 12) @ t3.translate(300, 260, 0)
    corners = QUAD * 80.0
    H = t3.homography(M, P)
    px_np = t3.project_corners(corners, H)
    qt = t3.qtransform_from_h(H)
    for (cx, cy), (ex, ey) in zip(corners, px_np):
        p = qt.map(QPointF(float(cx), float(cy)))
        assert np.isclose(p.x(), ex, atol=1e-6)
        assert np.isclose(p.y(), ey, atol=1e-6)


# --- batched == scalar loop ------------------------------------------------

def test_batched_equals_scalar_loop():
    rxs = np.array([10.0, -20.0, 33.0])
    rys = np.array([5.0, 40.0, -15.0])
    rzs = np.array([0.0, 90.0, 45.0])
    batched = t3.rotate_xyz(rxs, rys, rzs)
    for i in range(3):
        assert np.allclose(batched[i], t3.rotate_xyz(rxs[i], rys[i], rzs[i]))

    P = t3.projection(50, 640, 480)
    models = t3.translate(np.array([100.0, 200.0, 300.0]),
                          np.array([50.0, 60.0, 70.0]), 0.0)
    Hb = t3.homography(models, P)
    for i in range(3):
        assert np.allclose(Hb[i], t3.homography(models[i], P))


# --- degeneracy verdicts ----------------------------------------------------

def test_verdict_ok_gone_clipped():
    W, H = 640, 480
    P = t3.projection(60, W, H)
    corners = QUAD * 60.0

    # ok: a plain z=0 quad in front of the eye.
    M = t3.translate(320, 240, 0)
    verdict, Hm, poly = t3.project_with_verdict(M, P, corners)
    assert verdict == "ok" and poly is None
    assert Hm.shape == (3, 3)

    # gone: push the whole plane far behind the eye (past camera distance d).
    d = (W / 2) / np.tan(np.radians(60) / 2)
    M_far = t3.translate(320, 240, d + 500)  # camera at z=d looking to z=0
    verdict, _, poly = t3.project_with_verdict(M_far, P, corners)
    assert verdict == "gone"
    assert len(poly) == 0

    # clipped: a big X-tilted plane straddling the eye plane.
    M_tilt = t3.rotate_x(89.0) @ t3.translate(320, 240, 0)
    big = QUAD * 4000.0
    verdict, _, poly = t3.project_with_verdict(M_tilt, P, big)
    assert verdict == "clipped"
    assert len(poly) >= 3  # a real content-space polygon in front of the eye


# --- cross-check: confusionx |cos| zoom == fov->0 rotationx homography ------

def test_confusionx_is_ortho_rotationx_limit():
    from analysis.player.render.mods import arrow_effects as ae

    P = t3.projection(0, 640, 480)  # ortho == fov->0 limit
    for percent, beat, offset in [(1.0, 0.7, 0.0), (0.5, 3.1, 0.25),
                                  (2.0, 1.9, -0.1)]:
        angle_deg = ae._confusion_axis_degrees(percent, beat, offset)
        zoom = ae.confusionx_zoom(percent, beat, offset)  # |cos(angle)|

        M = t3.rotate_x(angle_deg) @ t3.translate(320, 240, 0)
        H = t3.homography(M, P)
        px = t3.project_corners(QUAD * 100.0, H)
        y_scale = np.ptp(px[:, 1]) / 200.0  # = |cos(angle)|
        assert np.isclose(y_scale, zoom, atol=1e-9)


# --- cross-check: perspective_z_scale == real projection of a z push --------

def test_perspective_z_scale_matches_projection():
    """The per-note z->zoom contract (arrow_effects.perspective_z_scale,
    d/(d-z)) is EXACTLY the scale the real LoadMenuPerspective gives a
    z-translated plane at the design center - the sanctioned center-plane
    degradation of per-note 3D."""
    from analysis.player.render.mods import arrow_effects as ae

    d = t3.eye_distance(45.0, 640.0)
    assert np.isclose(d, ae.EYE_DISTANCE)
    P = t3.projection(45.0, 640.0, 480.0)
    for z in (0.0, 50.0, -50.0, 200.0, -600.0):
        M = t3.translate(0.0, 0.0, z)
        H = t3.homography(M, P)
        px = t3.project_corners(QUAD * 50.0 + (320.0, 240.0), H)
        scale = np.ptp(px[:, 0]) / 100.0
        assert np.isclose(scale, ae.perspective_z_scale(z), atol=1e-9)
    # at z=0 the scale is exactly 1 (design plane maps 1:1).
    assert ae.perspective_z_scale(0.0) == 1.0
    # a push at/behind the eye saturates instead of exploding/flipping.
    assert ae.perspective_z_scale(d * 2.0) == ae._MAX_Z_SCALE
