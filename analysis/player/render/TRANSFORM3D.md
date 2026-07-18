# transform3d: SM/NotITG 3D transform mathematics

Pure-numpy port of StepMania's ACTOR transform + RageDisplay projection
math, plus the planar-homography extraction that lets a 2D QPainter
executor render 3D-transformed planar content exactly (no 4-point DLT).
The GL executor consumes the same 4x4 stack directly.

Sources (local Etterna checkout, the authoritative NotITG-era RageMatrix):
- `src/RageUtil/Misc/RageMath.cpp`
  - `RageMatrixMultiply` L255-283, `RageMatrixTranslate` L334-364,
    `RageMatrixScale` L366-393, `RageMatrixRotationX/Y/Z` L396-435,
    `RageMatrixRotationXYZ` L442-490 (fused X*Y*Z), `RageMatrixSkewX/Y`
    L312-323, `RageLookAt` L720-768.
- `src/Etterna/Actor/Base/Actor.cpp`
  - `Actor::BeginDraw` L737-810 (the per-actor compose order),
    `m_baseRotation` add L754-756.
- `src/RageUtil/Graphics/RageDisplay.cpp`
  - `RageMatrixStack::MultMatrix/MultMatrixLocal` L423-433,
    `PreMultMatrix` L661, `SkewX` L515-520,
    `LoadMenuPerspective` L691-734 (fov + vanish point),
    `GetFrustumMatrix` L859-888, `GetOrthoMatrix` L832-856.

## Convention

RageMatrix and Qt QTransform agree: **row-vector**, translation in the
bottom row. A point is `v = [x y z 1]`, transformed by `v' = v M`.
`RageMatrixMultiply(out, A, B)` computes (L264-280) `out[i][j] =
sum_k B[i][k] A[k][j]`, i.e. `out = B A` in ordinary product terms. In
row-vector algebra `v (B A)` means "apply B, then A", so with args
(A, B) the C call composes m onto the right/left depending on which
stack helper calls it -- see the compose order below. This module stores
M so that `v M` is the correct action, and `mat_mul(A, B) == A @ B`
(ordinary numpy product): a row point right-multiplied by `A B` applies
A first, then B, reading left to right.

Basis matrices (row-vector; translation row = index 3):

    T(x,y,z)     I with (m30,m31,m32) = (x,y,z)
    S(x,y,z)     diag(x,y,z,1)
    Rz(t)        [[ c, s,0,0],[-s, c,0,0],[0,0,1,0],[0,0,0,1]]
    Rx(t)        [[1,0,0,0],[0, c, s,0],[0,-s, c,0],[0,0,0,1]]
    Ry(t)        [[ c,0,-s,0],[0,1,0,0],[ s,0, c,0],[0,0,0,1]]
    SkewX(a)     I with m10 = a       (x' = x + a*y)
    SkewY(a)     I with m01 = a       (y' = y + a*x)

(signs are RageMath.cpp verbatim; note Ry has +s at m20, -s at m02.)

## Fused XYZ rotation

SM never multiplies Rx Ry Rz at runtime; it fuses them
(`RageMatrixRotationXYZ`, docstring L437 "Return Rx(rX) * Ry(rY) *
Rz(rZ)"). In our row-vector storage that fused matrix equals
`Rx @ Ry @ Rz` (numpy product; verified in `test_fused_equals_sequence`).
Applied to a row point the visual order is X first, then Y, then Z. We
port the closed form (L474-489) so a chart's `(rotationx, rotationy,
rotationz)` triple reproduces SM bit-for-bit, including the shared
`m_baseRotation` add (Actor.cpp L754-756: angle = tween + baserotation,
per axis, before the fuse).

## Per-actor compose order (Actor::BeginDraw)

`BeginDraw` builds the local matrix by a run of `PreMultMatrix`, which is
`MultMatrixLocal`: `top <- RageMatrixMultiply(top, top, m)`. With C args
(A=top, B=m) that writes `out[i][j] = m[i][k] top[k][j] = (m top)`
elementwise -- but read in row-vector storage this is exactly `top @ m`
(a row point sees `v top m`: top applied first, then m). Verified in
`test_actorframe_nesting`. Starting from the parent's `top`, the calls in
source order (L742-804) accumulate:

    top <- top @ T(pos)                    L745
    top <- top @ Rxyz(rot + baserot)       L760
    top <- top @ S(scale * basescale)      L773
    top <- top @ T(align offset)           L785   (default align 0.5 -> 0)
    top <- top @ Quat                      L792   (rare; skipped unless set)
    top <- top @ SkewX(skewx)              L799
    top <- top @ SkewY(skewy)              L803

So a node's LOCAL matrix is

    L = T(pos) @ Rxyz @ S(scale) @ [T(align)] @ [SkewX] @ [SkewY]

and a row content point maps `v @ (parent_world @ L)`: translate last in
world means pos is applied in the PARENT's frame after the child's own
rotate/scale -- the standard "scale, then rotate, then translate to
position" reading for a row vector left of the product. The child's
WORLD matrix is `world_child = parent_world @ L`; ActorFrame nesting is a
left-fold of local matrices down the tree, matching the compiled
document group tree: `world = root @ L1 @ L2 @ ... @ Ln`. `local_matrix()`
builds L; `compose(parent, L)` = `parent @ L`.

Content note: a leaf's own quad lives in the z=0 plane centered on its
origin (SM sprites center; compiled NotITG actors use anchor (0,0)
origin (0.5,0.5), scoping item 23). The planar homography assumes z=0
content; the caller supplies content-space corners.

## Projection: LoadMenuPerspective (fov + vanish point)

SM whole-scene 3D uses `LoadMenuPerspective(fov, W, H, vanishX, vanishY)`
(L691-734). world->screen derivation for (fov, vanish, WxH):

Let `theta = fovRad/2`; the camera distance making WxH subtend fov
horizontally:

    d = (W/2) / tan(theta)                                (L707)

Vanish point (screen px) is remapped (L709-713):

    vx = SCALE(vanishX, 0,W, W,0) - W/2 = (W/2 - vanishX)
    vy = (H/2 - vanishY)

vanish at screen center (W/2,H/2) -> vx=vy=0. Projection is an
**off-center frustum** (glFrustum, L716-722):

    l=(vx-W/2)/d, r=(vx+W/2)/d,  b=(vy+H/2)/d, t=(vy-H/2)/d
    near=1, far=d+1000

The VIEW (`RageLookAt`, L724-732): eye `(-vx+W/2, -vy+H/2, d)` looking
down -z to the same XY -> pure translation `T(vx-W/2, vy-H/2, -d)` (eye
XY == target XY, so the lookat basis is world-axis-aligned).

`View @ Frustum` maps world -> clip; NDC = clip/clip_w in [-1,1]^2. The
viewport map (y down, matching Qt + SM's flipped screen):

    x_px = (ndc_x + 1)/2 * W,   y_px = (1 - ndc_y)/2 * H

We fold View, Frustum, and the viewport map into one `projection(fov,
vanish, W, H) -> 4x4 P` (row-vector) with `[x y z 1] P = [px*w, py*w, *,
w]`. `fov == 0` takes the orthographic branch (L698-702): scaled
identity, `w == 1`, no divide.

Sanity limits (tested):
- `fov -> 0+`: `d -> inf`, frustum -> ortho, no foreshortening.
- vanish shift: the horizon (image of z=+inf) moves linearly in
  (vanishX, vanishY) -- vanish enters P only through the affine view
  translation and the frustum (A,B) shear, both linear in (vx,vy).

## Planar homography extraction (the key derivation)

A leaf's content is a **plane** z=0. Compose model M (4x4 world) with
projection P: `Q = M @ P`. A content point `[x y 0 1]` maps by
`[x y 0 1] Q`. Because z=0, **row 2 of Q is multiplied by 0 and never
contributes** -- delete it. The output pixel is `(Xw/w, Yw/w)` and
`(px*w, py*w, w)` live in columns (0,1,3) of Q. Hence the exact
content(x,y) -> screen homography is the 3x3 MINOR

    H = Q[ rows (0,1,3), cols (0,1,3) ],   [x y 1] H = [px*w, py*w, w]

No DLT, no 4-point solve: the homography is literally a 3x3 submatrix of
the composed 4x4. `homography(model4x4, projection)` returns this
(batched over a leading N). In row-vector layout `w = x H[0,2] + y H[1,2]
+ H[2,2]`, so `is_affine(H)` tests the projective COLUMN (Qt m13, m23):
`|H[0,2]|, |H[1,2]| < eps*|H[2,2]|`.

Qt executes full projective transforms (QTransform has m13/m23), so
`qtransform_from_h(H)` maps our 3x3 (row-vector) directly onto Qt's
`(m11 m12 m13; m21 m22 m23; m31 m32 m33)` -- same convention, same
layout -- after normalizing by H[2,2]. QPainter draws the leaf pixmap
under it and the perspective is exact for the plane.

### Cross-check vs the 2D confusion approximation

`arrow_effects.confusionx_zoom` returns `|cos(angle)|` as the 2D proxy
for a rotationx tilt. Claim: that is the `fov -> 0` (orthographic) limit
of the real rotationx homography's vertical scale. Under ortho P the
w-row is constant (H affine); `Rx(angle)` on a z=0 plane leaves x-extent
fixed and scales y-extent by the projection of the tilted plane onto the
image = `cos(angle)`. So H's y-scale = `cos(angle)`, matching
`confusionx_zoom`'s `|cos|` on the y-axis in the ortho limit
(`test_confusionx_is_ortho_rotationx_limit`).

## Degeneracy + bounds

Perspective can send content behind the eye (`w <= 0` after divide),
where H is meaningless. `project_with_verdict` classifies against the eye
plane in content space:
- `'ok'`      -- all content corners `w > eps`; H valid.
- `'clipped'` -- some `w > eps`, some `<= eps`; returns the clip polygon
  in CONTENT space (Sutherland-Hodgman of the quad against the `w = eps`
  line, which is straight in content space because `w` is affine in
  (x,y) on a plane). Caller clips content to it, then applies H.
- `'gone'`    -- all corners `w <= eps`; nothing to draw.

Conditioning: `normalize_h` divides H by H[2,2] (guard `|H[2,2]| < tiny`
-> verdict degenerate); `project_corners` gives batched (N,4,2) pixel
bounds for culling.

## What each executor consumes

- **QPainter (2D)**: `homography()` + `qtransform_from_h()` per planar
  leaf, gated by `project_with_verdict`; `project_corners()` for bounds.
  Exact for planar content; a note under a per-note 3D path degrades to
  its center-plane homography.
- **GL (future rust/wgpu)**: the 4x4 `model` and `projection` directly as
  MVP; no homography (the rasterizer divides per fragment). The
  homography path is the QPainter specialization of the same matrices,
  so both executors are backends of one transform tree (DESIGN axis 3).

## Consumers: the NotITG field projection

`analysis/games/notitg/field_projection.py` is the single construction
point for the NotITG field's use of this module: one
`LoadMenuPerspective` (fov 45, the recorded per-player `SetVanishPoint`
stream when compiled, else centered), one field model matrix summing
BOTH tilt producers (recorded actor rotation pokes + the scalar
confusionx/y mod channels), one homography out. `field_3d.NotitgField3D`
emits it for the base field; `field_compose` imports the same centered
projection for instance channels. The per-note z->zoom contract
(`arrow_effects.perspective_z_scale`, d/(d-z) with d =
`eye_distance(fov, W)`) is the exact center-plane scale of this
projection - the confusionx ortho cross-check above generalizes: the 2D
kernels are the degenerate limits of this one projection, kept only for
per-column variants and the capture-deferral fallback.
