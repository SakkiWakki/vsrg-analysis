"""3D swept ribbon for LN bodies/tails (prototype, curve-native).

A note in the curve model is an infinitesimal point sampled at one
y_offset. An LN body is EXTENDED: it spans the head y_offset to the tail
y_offset and has width. So it is a curve SEGMENT, not a point -- we
sample the axis-curves along the body, lift each sample to a 3D point in
the field/group frame, and orient a constant-width ribbon along the
curve's local tangent (its derivative). A body tilted into xz then has
its width span xz; a drunk body wiggles with the sine.

Seam boundary condition: heads render FLAT (sprite in the receptor
plane). The body must begin aligned with the head's frame and only bend
as the curve bends -- it must not pop off perpendicular at the join. So
the head-end tangent is pinned to the head's forward direction, and the
width basis is parallel-transported from the head's flat right-vector
along the tangents (no twist flip).

Output is geometry in the group frame; the caller projects it through the
shared field camera (one projection per body group) and strokes the
projected ribbon in screen space. The existing 2D QPainterPathStroker
stays as that screen-space fill; what changes is the ribbon is built in
3D first.
"""
from __future__ import annotations

import numpy as np

# The scroll axis points DOWN-screen in design space (y grows toward the
# receptor); a note's travel direction along its path is +y before mods
# bend it. The head sits flat in the z=0 receptor plane, so its forward
# is +y and its right (the width axis) is +x.
_HEAD_FORWARD = np.array([0.0, 1.0, 0.0])
_HEAD_RIGHT = np.array([1.0, 0.0, 0.0])


def sample_points(y_head, y_tail, axis_curves, ctx, n=16):
    """Sample the body at `n` y_offsets from head to tail, returning an
    (n, 3) array of 3D points in the group frame.

    `axis_curves` is a dict of the position axis curves that write the
    body: {'x': Curve, 'z': Curve} (each `f(y_offset, ctx) -> array`).
    The y coordinate is the scroll offset itself; x/z are the curve
    displacements (absent axis = 0). Orientation curves (rot_*) are the
    caller's concern -- they twist the head sprite, not the body spine."""
    ys = np.linspace(float(y_head), float(y_tail), int(n))
    x = axis_curves['x'](ys, ctx) if 'x' in axis_curves else np.zeros_like(ys)
    z = axis_curves['z'](ys, ctx) if 'z' in axis_curves else np.zeros_like(ys)
    return np.stack([x, ys, z], axis=1)


def tangents(points):
    """Unit tangent at each sample -- the discrete curve derivative.

    Interior samples use a central difference; the ENDS are one-sided so
    the head-end tangent is the direction the body actually leaves the
    head (pinned to the head forward when the first segment is straight),
    honoring the flat-seam boundary condition rather than averaging in a
    phantom sample past the head."""
    p = np.asarray(points, dtype=np.float64)
    d = np.empty_like(p)
    d[1:-1] = p[2:] - p[:-2]
    d[0] = p[1] - p[0]
    d[-1] = p[-1] - p[-2]
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return d / norms


def width_basis(tangs, head_right=_HEAD_RIGHT):
    """Parallel-transport the head's flat right-vector along the tangents
    so the ribbon's width axis stays continuous from the flat seam (no
    sudden twist). At each step the previous right-vector is reprojected
    perpendicular to the new tangent and renormalized (a discrete rotation
    minimizing frame; the standard bishop/parallel-transport frame).

    Returns an (n, 3) array of unit width vectors. Segment k's ribbon
    edges are point_k +/- (W/2) * basis_k."""
    t = np.asarray(tangs, dtype=np.float64)
    n = t.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    right = np.asarray(head_right, dtype=np.float64)
    # Seat the seam: make the head right perpendicular to the first
    # tangent (it already is when the body leaves straight down), so the
    # very first ribbon segment lies flat in the head's plane.
    right = _orthonormalize(right, t[0])
    out[0] = right
    for k in range(1, n):
        right = _orthonormalize(right, t[k])
        out[k] = right
    return out


def _orthonormalize(vec, tangent):
    """Project `vec` perpendicular to `tangent` and renormalize. If they
    are parallel (degenerate), fall back to any axis perpendicular to the
    tangent so the frame never collapses."""
    v = vec - tangent * float(np.dot(vec, tangent))
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        v = np.cross(tangent, _HEAD_FORWARD)
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            v = np.cross(tangent, _HEAD_RIGHT)
            norm = float(np.linalg.norm(v))
    return v / norm


def ribbon(y_head, y_tail, axis_curves, ctx, width, n=16):
    """The full 3D body ribbon: (left_edge, right_edge) each (n, 3).

    Convenience assembling sample_points -> tangents -> width_basis and
    offsetting by +/- width/2 along the transported basis. The caller
    projects both edge polylines through the field camera and fills
    between them."""
    pts = sample_points(y_head, y_tail, axis_curves, ctx, n=n)
    basis = width_basis(tangents(pts))
    half = 0.5 * float(width) * basis
    return pts - half, pts + half


def project_screen_ribbon(centers_xy, depth_scale, width):
    """Screen-space left/right edges of the body ribbon, with the width
    foreshortened by depth.

    `centers_xy` is the (n, 2) per-sample body spine in OUR pixel space
    (lane_x + bent dx, screen y) -- already the correct screen position of
    each body point, so the spine is NOT re-projected. `depth_scale` is
    the (n,) per-sample d/(d-z) factor (`arrow_effects.perspective_z_scale`
    of the sample's +z push), the same scale the note head uses.

    The width axis is the perpendicular of the spine's SCREEN tangent (the
    2D image of the 3D tangent already carries the in-plane bend). Each
    cross-section's half-width is scaled by its depth: a sample pushed
    toward the camera (z > 0, scale > 1) widens, one pushed away narrows,
    so the ribbon foreshortens on tilt. The head-end tangent is the
    spine's leaving direction, so the first cross-section stays flat
    against the flat head (the seam condition)."""
    c = np.asarray(centers_xy, dtype=np.float64)
    s = np.broadcast_to(np.asarray(depth_scale, dtype=np.float64),
                        (c.shape[0],)).reshape(-1, 1)
    tang = _screen_tangents(c)
    # Perpendicular in screen space (y-down): rotate the tangent -90 deg so
    # a downward spine (0, 1) yields a rightward normal (+1, 0) -- `right`
    # is then +x of center, matching the left/right naming.
    normal = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    half = 0.5 * float(width) * s * normal
    return c - half, c + half


def _screen_tangents(centers_xy):
    """Unit tangents of a 2D screen polyline, one-sided at the ends so the
    head-end direction is the spine's actual leaving direction (the flat
    seam condition, in screen space)."""
    c = np.asarray(centers_xy, dtype=np.float64)
    d = np.empty_like(c)
    d[1:-1] = c[2:] - c[:-2]
    d[0] = c[1] - c[0]
    d[-1] = c[-1] - c[-2]
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return d / norms
