"""SM/NotITG 3D transform mathematics for the compiled document.

Pure-numpy port of StepMania's ACTOR transform stack and RageDisplay
perspective projection, plus the planar-homography extraction that lets a
QPainter executor render 3D-transformed planar content exactly. A future GL
executor consumes the 4x4 model + projection directly. See TRANSFORM3D.md
(next to this file) for the full derivations and SM source line references.

Convention (RageMatrix and Qt QTransform agree): row-vector, translation in
the bottom row. A point is v = [x y z 1] and v' = v @ M. `mat_mul(A, B) ==
A @ B`; a row point right of `A @ B` applies A first, then B.

Qt imports live only in `qtransform_from_h` (the executor edge); the core is
pure math -- a clean rust/wgpu port boundary.
"""

from __future__ import annotations

import numpy as np

DEG = np.pi / 180.0

# glFrustum near plane and the +1000 far-plane slack SM uses (RageDisplay
# LoadMenuPerspective L721-722).
_NEAR = 1.0
_FAR_SLACK = 1000.0

# Below this the perspective w is treated as on/behind the eye plane.
_EYE_EPS = 1e-6
# Guard for a degenerate projective denominator when normalizing H.
_TINY_M33 = 1e-12


def _as_batch(m):
    """View a (4,4) or (N,4,4) array as (N,4,4); returns (batched, was_scalar)."""
    a = np.asarray(m, dtype=np.float64)
    if a.ndim == 2:
        return a[None], True
    return a, False


# ---------------------------------------------------------------------------
# Affine core: 4x4 model matrices (row-vector), batched where useful.
# ---------------------------------------------------------------------------

def identity(n=None):
    """Identity model matrix, (4,4) or (n,4,4)."""
    if n is None:
        return np.eye(4)
    return np.broadcast_to(np.eye(4), (n, 4, 4)).copy()


def translate(x, y, z=0.0):
    """T: translation in the bottom row (RageMatrixTranslate, RageMath L334).

    Scalars -> (4,4); array-likes -> (N,4,4) broadcast over x/y/z."""
    x, y, z = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (x, y, z)))
    m = np.broadcast_to(np.eye(4), x.shape + (4, 4)).copy()
    m[..., 3, 0] = x
    m[..., 3, 1] = y
    m[..., 3, 2] = z
    return m


def scale(x, y, z=1.0):
    """S = diag(x, y, z, 1) (RageMatrixScale, RageMath L366)."""
    x, y, z = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (x, y, z)))
    m = np.broadcast_to(np.eye(4), x.shape + (4, 4)).copy()
    m[..., 0, 0] = x
    m[..., 1, 1] = y
    m[..., 2, 2] = z
    return m


def _rot(axis, deg):
    """One-axis rotation, verbatim RageMatrixRotationX/Y/Z (RageMath L396-435)."""
    t = np.asarray(deg, np.float64) * DEG
    c, s = np.cos(t), np.sin(t)
    m = np.broadcast_to(np.eye(4), t.shape + (4, 4)).copy()
    match axis:
        case "x":
            m[..., 1, 1] = c
            m[..., 2, 2] = c
            m[..., 2, 1] = s
            m[..., 1, 2] = -s
        case "y":
            m[..., 0, 0] = c
            m[..., 2, 2] = c
            m[..., 0, 2] = s
            m[..., 2, 0] = -s
        case "z":
            m[..., 0, 0] = c
            m[..., 1, 1] = c
            m[..., 0, 1] = s
            m[..., 1, 0] = -s
    return m


def rotate_x(deg):
    """Rotation about x, in degrees (RageMatrixRotationX)."""
    return _rot("x", deg)


def rotate_y(deg):
    """Rotation about y (RageMatrixRotationY)."""
    return _rot("y", deg)


def rotate_z(deg):
    """Rotation about z, the in-plane spin (RageMatrixRotationZ)."""
    return _rot("z", deg)


def rotate_xyz(rx, ry, rz):
    """Fused X*Y*Z rotation, closed form from RageMatrixRotationXYZ (L442-490).

    Equals rotate_z(rz) @ rotate_y(ry) @ rotate_x(rx) in numpy (SM's
    "Rx * Ry * Rz" under its B@A multiply); a row point v @ result rotates
    about X first, then Y, then Z. Batched over broadcast rx/ry/rz."""
    rx, ry, rz = np.broadcast_arrays(
        *(np.asarray(v, np.float64) * DEG for v in (rx, ry, rz))
    )
    cX, sX = np.cos(rx), np.sin(rx)
    cY, sY = np.cos(ry), np.sin(ry)
    cZ, sZ = np.cos(rz), np.sin(rz)
    m = np.broadcast_to(np.eye(4), rx.shape + (4, 4)).copy()
    m[..., 0, 0] = cZ * cY
    m[..., 0, 1] = cZ * sY * sX + sZ * cX
    m[..., 0, 2] = cZ * sY * cX - sZ * sX
    m[..., 1, 0] = -sZ * cY
    m[..., 1, 1] = -sZ * sY * sX + cZ * cX
    m[..., 1, 2] = -sZ * sY * cX - cZ * sX
    m[..., 2, 0] = -sY
    m[..., 2, 1] = cY * sX
    m[..., 2, 2] = cY * cX
    return m


# NotITG's fork adds SetRotationOrder: the actor picks which axis-rotation
# order composes its Euler rotation, instead of the stock fixed X*Y*Z.
# The engine builds it with RageMatrixMultiply(out, x, y, z, order) - the
# same per-axis rotations, multiplied in the token's order (Actor.clean.c
# BeginDraw @ 004a4320; the SetRotationOrder token->enum swizzle @
# 004abd70). 'xyz' is the stock order and MUST equal rotate_xyz exactly.
# The fork accepts exactly these swizzle tokens (SetRotationOrder @ 004abd70);
# an unknown string logs 'Invalid Rotation mode' and leaves the order be.
_ROTATION_ORDERS = ('xyz', 'xzy', 'yzx', 'zxy', 'zyx')
_AXIS_ROT = {'x': rotate_x, 'y': rotate_y, 'z': rotate_z}


def rotate_ordered(rx, ry, rz, order='xyz'):
    """Euler rotation composed in `order` (a permutation of 'xyz'), the
    fork's SetRotationOrder (Actor::BeginDraw @ 004a4320). The order names
    the axis applied FIRST to a row content point: 'xyz' rotates about X,
    then Y, then Z, identical to rotate_xyz (the stock RageMatrixRotationXYZ
    default). A row point v @ result applies order[0] first, so in numpy
    matmul the matrices multiply in reverse: R(order[2]) @ R(order[1]) @
    R(order[0]). Batched over broadcast rx/ry/rz."""
    if order == 'xyz':
        return rotate_xyz(rx, ry, rz)
    angles = {'x': rx, 'y': ry, 'z': rz}
    first, second, third = order
    return (_AXIS_ROT[third](angles[third])
            @ _AXIS_ROT[second](angles[second])
            @ _AXIS_ROT[first](angles[first]))


def matrix_from_quat(quat):
    """RageMatrixFromQuat (RageMath.cpp:395): a unit quaternion (x, y, z, w)
    to a row-vector 4x4 rotation. The fork's spherical adds (AddRotationH/P/R
    = heading/pitch/roll) accumulate onto this quat channel and the engine
    MultMatrixes it after the Euler rotation (Actor.cpp BeginDraw:424-429).
    Row-major param order = the transpose of the OpenGL layout, matching our
    row-vector convention. Identity quat (0,0,0,1) -> identity matrix."""
    x, y, z, w = (float(c) for c in quat)
    xx, xy, xz = x * (x + x), x * (y + y), x * (z + z)
    wx, wy, wz = w * (x + x), w * (y + y), w * (z + z)
    yy, yz = y * (y + y), y * (z + z)
    zz = z * (z + z)
    return np.array([
        [1 - (yy + zz), xy + wz, xz - wy, 0.0],
        [xy - wz, 1 - (xx + zz), yz + wx, 0.0],
        [xz + wy, yz - wx, 1 - (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def quat_from_axis(axis, deg):
    """Unit quaternion for the fork's single-axis spherical adds:
    RageQuatFromH (y-axis / heading), RageQuatFromP (x-axis / pitch),
    RageQuatFromR (z-axis / roll) (RageMath.cpp:311-341). Each halves and
    NEGATES the angle before the sin/cos, so the quat spins the content the
    same visual direction the Euler rotations do."""
    theta = -float(deg) * DEG / 2.0
    c, s = np.cos(theta), np.sin(theta)
    match axis:
        case 'x':   # pitch (P)
            return (s, 0.0, 0.0, c)
        case 'y':   # heading (H)
            return (0.0, s, 0.0, c)
        case 'z':   # roll (R)
            return (0.0, 0.0, s, c)


def quat_multiply(a, b):
    """Hamilton product a*b of two (x,y,z,w) quats, normalized
    (RageQuatMultiply, RageMath.cpp:287). The fork accumulates spherical
    adds as DestQuat = DestQuat * QuatFromAxis(angle)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by + ay * bw + az * bx - ax * bz
    z = aw * bz + az * bw + ax * by - ay * bx
    w = aw * bw - ax * bx - ay * by - az * bz
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    inv = 1.0 / norm if norm > 0.0 else 1.0
    return (x * inv, y * inv, z * inv, w * inv)


def skew_x(amount):
    """SkewX: x' = x + amount*y  (RageMatrixSkewX, m10 = amount, RageMath L312)."""
    a = np.asarray(amount, np.float64)
    m = np.broadcast_to(np.eye(4), a.shape + (4, 4)).copy()
    m[..., 1, 0] = a
    return m


def skew_y(amount):
    """SkewY: y' = y + amount*x  (RageMatrixSkewY, m01 = amount, RageMath L319)."""
    a = np.asarray(amount, np.float64)
    m = np.broadcast_to(np.eye(4), a.shape + (4, 4)).copy()
    m[..., 0, 1] = a
    return m


def mat_mul(a, b):
    """Row-vector composition: a @ b (v @ (a @ b) applies a first). Batched."""
    return np.asarray(a, np.float64) @ np.asarray(b, np.float64)


def apply(points, m):
    """Map row points by v @ m, returning euclidean coords (divide by w).

    points: (..., 3) or (..., 4). Returns (..., 3) with the perspective divide
    applied (w folded out). For pure-affine m the w is 1 and this is exact."""
    p = np.asarray(points, np.float64)
    if p.shape[-1] == 3:
        p = np.concatenate([p, np.ones(p.shape[:-1] + (1,))], axis=-1)
    out = p @ m
    w = out[..., 3:4]
    return out[..., :3] / w


def local_matrix(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), scl=(1.0, 1.0, 1.0),
                 skewx=0.0, skewy=0.0, align=None):
    """A node's local matrix per Actor::BeginDraw (Actor.cpp L737-810).

    The premult run there builds, in row-vector storage,
        L = S(scl) @ Rxyz(rot) @ T(pos) @ [T(align)] @ [SkewX] @ [SkewY]
    so that v @ L scales, then rotates, then translates to position -- the
    SM sprite reading. `rot` should already include the m_baseRotation add
    (angle = tween + baserotation per axis, L754-756). `align` is the
    (dx, dy) alignment offset (default center = no offset)."""
    L = scale(*scl) @ rotate_xyz(*rot) @ translate(*pos)
    if align is not None:
        L = L @ translate(align[0], align[1], 0.0)
    if skewx:
        L = L @ skew_x(skewx)
    if skewy:
        L = L @ skew_y(skewy)
    return L


def compose(parent_world, local):
    """Child world = local @ parent_world (row-vector; ActorFrame nesting).

    A row content point v maps by v @ (local @ parent_world): local first,
    then the parent chain. Batched over a leading N on either operand."""
    return np.asarray(local, np.float64) @ np.asarray(parent_world, np.float64)


# ---------------------------------------------------------------------------
# Projection: LoadMenuPerspective (fov + vanish point) -> viewport pixels.
# ---------------------------------------------------------------------------

def _frustum(l, r, b, t, zn, zf):
    """glFrustum matrix, row-vector storage (RageDisplay GetFrustumMatrix L859)."""
    A = (r + l) / (r - l)
    B = (t + b) / (t - b)
    C = -(zf + zn) / (zf - zn)
    D = -(2.0 * zf * zn) / (zf - zn)
    return np.array([
        [2.0 * zn / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 * zn / (t - b), 0.0, 0.0],
        [A, B, C, -1.0],
        [0.0, 0.0, D, 0.0],
    ])


def _viewport(width, height):
    """NDC [-1,1]^2 -> pixel (x right, y down), as a row-vector 4x4.

    x_px = (ndc_x+1)/2 * W ; y_px = (1-ndc_y)/2 * H. Leaves z, w untouched so
    the perspective divide still happens on w afterward."""
    m = np.eye(4)
    m[0, 0] = width / 2.0
    m[3, 0] = width / 2.0
    m[1, 1] = -height / 2.0
    m[3, 1] = height / 2.0
    return m


def eye_distance(fov, width):
    """Camera distance for LoadMenuPerspective: the eye sits this many px
    from the z=0 plane so that `width` subtends `fov` degrees
    (RageDisplay L707, d = (W/2)/tan(fov/2)). The center-plane
    perspective scale of a +z push is d / (d - z)."""
    return (width / 2.0) / np.tan((fov * DEG) / 2.0)


def projection(fov, width, height, vanish=None):
    """world -> screen-pixel projection for (fov deg, viewport, vanish point).

    Ports RageDisplay::LoadMenuPerspective (L691-734). `vanish` is the screen
    (px, py) vanishing point; None (or exactly the screen center) gives a
    centered frustum. fov == 0 is the orthographic branch (L698-702). Returns
    a row-vector 4x4 P: [x y z 1] @ P = [px*w, py*w, *, w], pixel = (Xw/w, Yw/w).

    Property (verified): the z=0 design plane maps 1:1 to pixels under a
    centered vanish, so unmodded (z=0) content is untouched by perspective."""
    W, H = float(width), float(height)
    if vanish is None:
        vx_screen, vy_screen = W / 2.0, H / 2.0
    else:
        vx_screen, vy_screen = float(vanish[0]), float(vanish[1])

    if fov == 0:
        # GetOrthoMatrix(0, W, H, 0, -1000, 1000); view = identity.
        l, r, b, t, zn, zf = 0.0, W, H, 0.0, -1000.0, 1000.0
        ortho = np.array([
            [2.0 / (r - l), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (t - b), 0.0, 0.0],
            [0.0, 0.0, -2.0 / (zf - zn), 0.0],
            [-(r + l) / (r - l), -(t + b) / (t - b), -(zf + zn) / (zf - zn), 1.0],
        ])
        return ortho @ _viewport(W, H)

    d = eye_distance(fov, W)

    # SCALE(v, 0, W, W, 0) = W - v, then - W/2 (L709-713).
    vx = (W - vx_screen) - W / 2.0
    vy = (H - vy_screen) - H / 2.0

    frust = _frustum((vx - W / 2.0) / d, (vx + W / 2.0) / d,
                     (vy + H / 2.0) / d, (vy - H / 2.0) / d,
                     _NEAR, d + _FAR_SLACK)
    # RageLookAt eye = (-vx+W/2, -vy+H/2, d) -> pure world translation by -eye.
    view = translate(vx - W / 2.0, vy - H / 2.0, -d)
    return view @ frust @ _viewport(W, H)


# ---------------------------------------------------------------------------
# Planar homography extraction (z=0 content -> screen pixels).
# ---------------------------------------------------------------------------

def homography(model, proj):
    """Exact content(x,y,z=0) -> screen-pixel 3x3 homography.

    Q = model @ proj; a content point [x y 0 1] @ Q ignores row 2 of Q (times
    z=0), and pixel (px,py,w) live in columns (0,1,3). So H is the 3x3 minor
    Q[rows (0,1,3), cols (0,1,3)] with [x y 1] @ H = [px*w, py*w, w]. Batched:
    model (N,4,4) or (4,4), proj (4,4) or (N,4,4) -> (N,3,3) or (3,3)."""
    m, m_scalar = _as_batch(model)
    p, p_scalar = _as_batch(proj)
    Q = m @ p
    idx = np.array([0, 1, 3])
    H = Q[:, idx[:, None], idx[None, :]]
    if m_scalar and p_scalar:
        return H[0]
    return H


def normalize_h(H):
    """Divide H by H[2,2] so the projective denominator is anchored at 1.

    Guards a near-zero H[2,2] (leaves H untouched there; degeneracy is a
    verdict from project_with_verdict, not a silent blowup). Batched."""
    a = np.asarray(H, np.float64)
    scal = a.ndim == 2
    a = a[None] if scal else a
    m33 = a[:, 2, 2]
    safe = np.abs(m33) >= _TINY_M33
    out = a.copy()
    out[safe] = a[safe] / m33[safe, None, None]
    return out[0] if scal else out


def is_affine(H, eps=1e-9):
    """True iff the projective COLUMN of H is null -> affine map.

    In row-vector layout w = x*H[0,2] + y*H[1,2] + H[2,2]; the map is affine
    iff the perspective entries H[0,2], H[1,2] (Qt m13, m23) vanish. Tests
    |H[0,2]|, |H[1,2]| < eps * |H[2,2]|. Batched -> bool or (N,) bool."""
    a = np.asarray(H, np.float64)
    scal = a.ndim == 2
    a = a[None] if scal else a
    thresh = eps * np.abs(a[:, 2, 2])
    flat = (np.abs(a[:, 0, 2]) <= thresh) & (np.abs(a[:, 1, 2]) <= thresh)
    return bool(flat[0]) if scal else flat


def project_corners(corners, H):
    """Batch-project content quads to screen pixels for culling/bounds.

    corners: (4,2) or (N,4,2) content-space quad(s). H: (3,3) or (N,3,3).
    Returns (4,2) or (N,4,2) pixels (perspective divide applied)."""
    c = np.asarray(corners, np.float64)
    scal_c = c.ndim == 2
    c = c[None] if scal_c else c
    h = np.asarray(H, np.float64)
    if h.ndim == 2:
        h = np.broadcast_to(h, (c.shape[0], 3, 3))
    hom = np.concatenate([c, np.ones(c.shape[:-1] + (1,))], axis=-1)
    out = hom @ h
    px = out[..., :2] / out[..., 2:3]
    return px[0] if scal_c else px


# ---------------------------------------------------------------------------
# Degeneracy + visibility verdict.
# ---------------------------------------------------------------------------

def _clip_polygon_front(corners, w):
    """Sutherland-Hodgman clip of the quad against the front half-plane w>eps.

    w is the per-corner projective denominator (affine in (x,y) on a plane),
    so the w=eps boundary is a straight line in content space and the clip is
    exact. Returns the content-space polygon (M,2) in front of the eye."""
    verts = np.asarray(corners, np.float64)
    ws = np.asarray(w, np.float64)
    out_pts = []
    n = len(verts)
    for i in range(n):
        cur, nxt = verts[i], verts[(i + 1) % n]
        wc, wn = ws[i], ws[(i + 1) % n]
        cur_in, nxt_in = wc > _EYE_EPS, wn > _EYE_EPS
        if cur_in:
            out_pts.append(cur)
        if cur_in != nxt_in:
            # crossing point where w == _EYE_EPS; w is linear along the edge.
            tcross = (_EYE_EPS - wc) / (wn - wc)
            out_pts.append(cur + tcross * (nxt - cur))
    return np.array(out_pts) if out_pts else np.empty((0, 2))


def project_with_verdict(model, proj, corners):
    """Project a planar quad and return (verdict, H, clip_polygon).

    verdict: 'ok' | 'clipped' | 'gone' -- classified by the perspective w at
    the content corners (w = plane point's homogeneous denominator).
    - 'ok':      all corners w > eps; clip_polygon is None.
    - 'clipped': mixed; clip_polygon (M,2) is the content-space region in
                 front of the eye (apply H after clipping content to it).
    - 'gone':    all corners w <= eps (fully behind/through the eye); H is
                 whatever homography() gives, clip_polygon is empty.
    Scalar model/proj only (a per-node verdict); use project_corners for
    batched bounds once a node is known visible."""
    H = homography(model, proj)
    c = np.asarray(corners, np.float64)
    hom = np.concatenate([c, np.ones((len(c), 1))], axis=-1)
    w = hom @ H[:, 2]
    front = w > _EYE_EPS
    if np.all(front):
        return "ok", normalize_h(H), None
    if not np.any(front):
        return "gone", normalize_h(H), np.empty((0, 2))
    return "clipped", normalize_h(H), _clip_polygon_front(c, w)


# ---------------------------------------------------------------------------
# QPainter executor edge: 3x3 homography -> QTransform (Qt import local).
# ---------------------------------------------------------------------------

def qtransform_from_h(H):
    """Convert a row-vector 3x3 homography to a Qt QTransform (full projective).

    Qt QTransform is row-vector [x y 1] @ M with layout
        (m11 m12 m13 ; m21 m22 m23 ; m31 m32 m33)
    -- identical to our H after normalizing by H[2,2]. QPainter then executes
    the perspective (m13/m23 non-zero) directly. This is the only Qt-touching
    function; keep it at the executor edge."""
    from PySide6.QtGui import QTransform

    h = normalize_h(np.asarray(H, np.float64))
    return QTransform(
        h[0, 0], h[0, 1], h[0, 2],
        h[1, 0], h[1, 1], h[1, 2],
        h[2, 0], h[2, 1], h[2, 2],
    )
