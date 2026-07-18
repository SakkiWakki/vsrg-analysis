"""The 3D LN body sweep: a flat head yields a flat body seam (no
perpendicular pop-off), and a body bent out of plane lays its width in
the bent plane (spans xz when the body tilts into z)."""
import numpy as np

from analysis.player.render.mods import body_sweep as bs
from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves as mc


def _ctx(n):
    return cv.Ctx(t=0.0, beat=0.0, cols=np.zeros(n, dtype=np.int64),
                  arrow_size=64.0)


def test_straight_body_seam_is_flat():
    """A plain vertical body (no x/z curves): every tangent is the head
    forward (+y) and every width vector is the head right (+x). The seam
    does not pop off perpendicular -- the whole ribbon lies flat in the
    receptor plane."""
    n = 16
    pts = bs.sample_points(0.0, 400.0, axis_curves={}, ctx=_ctx(n), n=n)
    tangs = bs.tangents(pts)
    basis = bs.width_basis(tangs)

    np.testing.assert_allclose(tangs, np.tile([0.0, 1.0, 0.0], (n, 1)),
                               atol=1e-12)
    np.testing.assert_allclose(basis, np.tile([1.0, 0.0, 0.0], (n, 1)),
                               atol=1e-12)
    # z stays 0 across the whole ribbon.
    left, right = bs.ribbon(0.0, 400.0, {}, _ctx(n), width=64.0, n=n)
    assert np.allclose(left[:, 2], 0.0) and np.allclose(right[:, 2], 0.0)


def test_head_seam_matches_head_frame():
    """Boundary condition: the head-end (k=0) tangent is the head forward
    and the head-end width is the head right, so the body starts aligned
    with the flat head regardless of how it bends later."""
    n = 16
    # A body that bends hard into z further down (bumpy), but the seam
    # must still start flat.
    curves = {'z': mc.bumpy_z(percent=1.0)}
    pts = bs.sample_points(0.0, 600.0, curves, _ctx(n), n=n)
    tangs = bs.tangents(pts)
    basis = bs.width_basis(tangs)

    # Seam tangent has no x component and points down-screen (+y); its
    # width is in the x-z plane (perpendicular to a y-and-z tangent).
    assert abs(tangs[0][0]) < 1e-12
    assert tangs[0][1] > 0.0
    assert abs(float(np.dot(basis[0], tangs[0]))) < 1e-9


def test_bumpy_body_tilts_out_of_receptor_plane():
    """A body pushed into z (bumpy) leaves the flat receptor plane: its
    tangent gains a large z component. A pure z-bend keeps the width along
    x (x is still perpendicular to a y-z tangent), but the ribbon SURFACE
    now faces partly sideways -- its normal (tangent x width) tilts out of
    the screen. This is the 3D structure the 2D screen-stroke could never
    carry."""
    n = 64
    curves = {'z': mc.bumpy_z(percent=1.0)}
    pts = bs.sample_points(0.0, 600.0, curves, _ctx(n), n=n)
    tangs = bs.tangents(pts)
    basis = bs.width_basis(tangs)

    assert np.max(np.abs(tangs[:, 2])) > 0.5
    normals = np.cross(tangs, basis)
    # Where the body dives into z, the ribbon normal leaves +z (the
    # screen-facing direction) -- the ribbon is no longer flat-on.
    assert np.max(np.abs(normals[:, 1])) > 0.3
    dots = np.abs(np.einsum('ij,ij->i', basis, tangs))
    assert np.max(dots) < 1e-9


def test_xz_bent_body_width_spans_xz():
    """When the body bends in BOTH x and z (drunk + bumpy), the tangent
    leaves the x-y plane, so the width basis -- perpendicular to it --
    genuinely gains a z component: the ribbon cross-section spans x-z, the
    literal property in the design ('head in xz -> width spans xz')."""
    n = 128
    curves = {'x': mc.drunk_x(percent=1.0), 'z': mc.bumpy_z(percent=1.0)}
    pts = bs.sample_points(0.0, 600.0, curves, _ctx(n), n=n)
    tangs = bs.tangents(pts)
    basis = bs.width_basis(tangs)

    assert np.max(np.abs(basis[:, 2])) > 1e-2
    dots = np.abs(np.einsum('ij,ij->i', basis, tangs))
    assert np.max(dots) < 1e-9


def test_screen_ribbon_foreshortens_and_stays_perpendicular():
    """project_screen_ribbon: an in-plane sample (depth scale 1) keeps the
    full half-width; a sample pushed toward the camera (scale 2) doubles
    it (foreshortening); and on a tilted spine the width lies perpendicular
    to the screen tangent (so a strong bend never bowties)."""
    centers = np.stack([np.full(10, 100.0), np.linspace(0, 400, 10)], axis=1)
    w = 64.0

    left, right = bs.project_screen_ribbon(centers, np.ones(10), w)
    assert np.allclose(right[:, 0] - centers[:, 0], 32.0)
    assert np.allclose(centers[:, 0] - left[:, 0], 32.0)

    _l2, r2 = bs.project_screen_ribbon(centers, np.full(10, 2.0), w)
    assert np.allclose(r2[:, 0] - centers[:, 0], 64.0)

    diag = np.stack([np.linspace(0, 300, 10), np.linspace(0, 400, 10)], axis=1)
    _l3, r3 = bs.project_screen_ribbon(diag, np.ones(10), w)
    tang = np.array([3.0, 4.0]) / 5.0
    assert np.max(np.abs((r3 - diag)[1:-1] @ tang)) < 1e-6


def test_width_basis_is_unit_and_continuous():
    """Parallel transport keeps the width basis unit-length, and the
    frame CONVERGES with sampling density: as N rises the largest
    adjacent-basis step shrinks toward 0. A fixed threshold at low N would
    just measure the bend rate, not a flaw -- so we assert convergence,
    which is the real continuity guarantee (and tells the renderer to
    scale body sample count with curvature)."""
    curves = {'x': mc.drunk_x(percent=1.0), 'z': mc.bumpy_z(percent=1.0)}

    def max_step(n):
        ctx = _ctx(n)
        basis = bs.width_basis(bs.tangents(
            bs.sample_points(0.0, 600.0, curves, ctx, n=n)))
        assert np.allclose(np.linalg.norm(basis, axis=1), 1.0, atol=1e-12)
        return float(np.max(np.linalg.norm(np.diff(basis, axis=0), axis=1)))

    coarse = max_step(64)
    fine = max_step(256)
    assert fine < coarse
    assert fine < 0.2
