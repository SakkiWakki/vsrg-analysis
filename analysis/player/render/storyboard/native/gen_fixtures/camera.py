"""Fixture generator for src/camera.rs (A3 - camera port).

Samples the REAL Python 3D math (analysis/player/render/transform3d.py)
plus the LoadMenuPerspective fov camera build (the design_projection
model in analysis/games/notitg/field_projection.py) over a case grid and
dumps inputs + expected matrices / projected points as JSON. The Rust
#[cfg(test)] include_str!s the JSON and asserts parity to 1e-3.

Run:  PYTHONPATH=/home/yucky/dev/vsrg-analysis \
      python analysis/player/render/storyboard/native/gen_fixtures/camera.py

Matrices are dumped ROW-MAJOR as 16 floats (Rust Mat4 = [f32; 16]
row-major); the Python core stores row-vector 4x4, whose numpy .ravel()
is already row-major, so no transpose is needed. `design_projection`'s
frozen signature takes an explicit far-plane z; the stock
transform3d.projection hardcodes far = eye_distance + 1000, so the
generator drives the frustum with that explicit far to lock the exact
LoadMenuPerspective matrix (verified equal to the stock projection when
far == d + 1000).
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.player.render import transform3d as t3

FAR_SLACK = 1000.0


def local_matrix(pos, rot, scl, skew, order):
    """local_matrix with a rotation ORDER (the fork's SetRotationOrder).

    transform3d.local_matrix hardcodes rotate_xyz; the frozen Rust
    local_matrix takes a RotOrder, so drive the ordered rotation here and
    reproduce the same premult run (scale @ rot @ translate @ skewx @
    skewy) that Actor::BeginDraw builds."""
    L = t3.scale(*scl) @ t3.rotate_ordered(rot[0], rot[1], rot[2], order) @ t3.translate(*pos)
    if skew[0]:
        L = L @ t3.skew_x(skew[0])
    if skew[1]:
        L = L @ t3.skew_y(skew[1])
    return L


def design_projection(fov_deg, W, H, vanish, far):
    """LoadMenuPerspective world->design-pixel projection, explicit far.

    Replicates transform3d.projection (RageDisplay::LoadMenuPerspective)
    but with the far-plane z as a caller argument (the frozen signature),
    rather than the hardcoded eye_distance + 1000. `vanish` is None for a
    centered frustum or a (px, py) screen point."""
    if vanish is None:
        vx_screen, vy_screen = W / 2.0, H / 2.0
    else:
        vx_screen, vy_screen = float(vanish[0]), float(vanish[1])
    d = t3.eye_distance(fov_deg, W)
    vx = (W - vx_screen) - W / 2.0
    vy = (H - vy_screen) - H / 2.0
    frust = t3._frustum((vx - W / 2.0) / d, (vx + W / 2.0) / d,
                        (vy + H / 2.0) / d, (vy - H / 2.0) / d,
                        t3._NEAR, far)
    view = t3.translate(vx - W / 2.0, vy - H / 2.0, -d)
    return view @ frust @ t3._viewport(W, H)


def flat(m):
    return [float(v) for v in np.asarray(m, np.float64).ravel()]


def project_w(m, p):
    """[x y z 1] @ m -> (px, py, w); pixel = (px/w, py/w). w<=eps = behind eye."""
    hom = np.array([p[0], p[1], p[2], 1.0]) @ np.asarray(m, np.float64)
    return float(hom[0]), float(hom[1]), float(hom[3])


ROT_ORDERS = ("xyz", "xzy", "yzx", "zxy", "zyx")


def local_cases():
    cases = []
    grid = [
        ((0, 0, 0), (0, 0, 0), (1, 1, 1), (0.0, 0.0)),
        ((10, 20, 5), (15, 30, 45), (2, 3, 1), (0.0, 0.0)),
        ((-30, 40, -12), (60, -20, 90), (1.5, 0.5, 2.0), (0.3, 0.0)),
        ((5, -5, 8), (0, 0, 33), (1, 1, 1), (0.0, 0.4)),
        ((100, 50, 0), (25, -70, 10), (0.8, 1.2, 1.0), (0.15, -0.25)),
        ((0, 0, -20), (-45, 45, -45), (3, 3, 3), (0.0, 0.0)),
        ((7, 3, 1), (12, 34, 56), (1, 1, 1), (0.5, 0.5)),
    ]
    for order in ROT_ORDERS:
        for pos, rot, scl, skew in grid:
            cases.append({
                "pos": [float(v) for v in pos],
                "rot": [float(v) for v in rot],
                "scl": [float(v) for v in scl],
                "skew": [float(v) for v in skew],
                "order": order,
                "expected": flat(local_matrix(pos, rot, scl, skew, order)),
            })
    return cases


def compose_cases():
    cases = []
    parents = [
        ((0, 0, 0), (0, 0, 0), (1, 1, 1)),
        ((50, 25, 10), (10, 20, 30), (2, 2, 1)),
        ((-10, 5, 0), (0, 90, 0), (1, 1, 1)),
    ]
    children = [
        ((5, 5, 5), (15, 0, 0), (1, 1, 1)),
        ((0, 20, 0), (0, 0, 45), (0.5, 0.5, 0.5)),
    ]
    for p in parents:
        for c in children:
            parent = local_matrix(p[0], p[1], p[2], (0.0, 0.0), "xyz")
            local = local_matrix(c[0], c[1], c[2], (0.0, 0.0), "xyz")
            # compose(parent, local) = local @ parent (child world).
            world = t3.compose(parent, local)
            cases.append({
                "parent": flat(parent),
                "local": flat(local),
                "expected": flat(world),
            })
    return cases


def projection_cases():
    W, H = 640.0, 480.0
    probe_pts = [
        [320.0, 240.0, 0.0],     # design center, on plane -> 1:1
        [0.0, 0.0, 0.0],         # corner on plane
        [640.0, 480.0, 0.0],     # far corner on plane
        [320.0, 240.0, 100.0],   # push toward eye (still in front)
        [320.0, 240.0, 400.0],   # deeper push
        [100.0, 380.0, 200.0],   # off-center + z
        [320.0, 240.0, 900.0],   # near the eye plane (d ~ 772 at fov45)
        [320.0, 240.0, 800.0],   # just in front of a fov45 eye
    ]
    cams = []
    for fov in (45.0, 60.0, 80.0):
        d = t3.eye_distance(fov, W)
        far = d + FAR_SLACK
        cams.append((fov, None, far))
        cams.append((fov, [300.0, 200.0], far))          # off-center vanish
        cams.append((fov, [420.0, 260.0], far))          # off-center other way
    # far-dist variants at the default fov
    d45 = t3.eye_distance(45.0, W)
    cams.append((45.0, None, d45 + 2000.0))
    cams.append((45.0, None, d45 + 500.0))
    cams.append((45.0, [360.0, 300.0], d45 + 3000.0))

    cases = []
    for fov, vanish, far in cams:
        m = design_projection(fov, W, H, vanish, far)
        pts = []
        for p in probe_pts:
            px, py, w = project_w(m, p)
            behind = w <= t3._EYE_EPS
            entry = {"p": p, "w": w, "behind_eye": bool(behind)}
            if not behind:
                entry["expected"] = [px / w, py / w]
            pts.append(entry)
        cases.append({
            "fov": fov,
            "w": W,
            "h": H,
            "vanish": vanish,
            "far": far,
            "matrix": flat(m),
            "points": pts,
        })
    return cases


def main():
    data = {
        "local": local_cases(),
        "compose": compose_cases(),
        "projection": projection_cases(),
    }
    out = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures", "camera_cases.json")
    out = os.path.abspath(out)
    with open(out, "w") as f:
        json.dump(data, f, indent=1)
    n = len(data["local"]) + len(data["compose"]) + len(data["projection"])
    npts = sum(len(c["points"]) for c in data["projection"])
    print(f"wrote {out}: {n} cases "
          f"({len(data['local'])} local, {len(data['compose'])} compose, "
          f"{len(data['projection'])} projection incl. {npts} projected points)")


if __name__ == "__main__":
    main()
