//! Wave-1 port target - the 2D field-instance transform math.
//!
//! Ports `analysis/games/notitg/field_compose.py`: `TransformChannel`'s
//! leaf-link 2D composition (`at` / `crop_at` / `_local`) plus the
//! `link_timelines` rest semantics. Channel SAMPLING stays OUTSIDE this
//! file (the caller samples ChannelRefs into a `TransformState` per
//! link); this module is the pure math that folds those sampled scalars
//! into a homography + alpha + crop.
//!
//! Scope: the leaf-link 2D affine path only (z=0, centered fov default,
//! no rotation_x/y, no quat, no skew-before toggle - those are the
//! camera area's perspective math). Under the centered LoadMenuPerspective
//! the z=0 design plane maps 1:1 to design pixels, so `at()`'s normalized
//! homography is exactly the affine block of the composed
//! `_TO_CONTENT @ world` (verified against the Python via the fixture).
//!
//! Engine order (openitg Actor::BeginDraw, row-vector storage): a leaf's
//! local map is, innermost-first,
//!   skew -> rotate_z -> scale -> translate,
//! with a halign/valign anchor riding innermost of all, and a child
//! composes onto its parent as `local @ parent`. `flip_base_y` mirrors
//! the leaf's vertical source axis (AFT bottom-up capture compensation):
//! it negates base_scale_y AND the anchor y offset (both innermost/
//! content-side) and swaps the vertical crop edges.
#![allow(dead_code)]

/// Design space (SM SCREEN metrics); the anchor offset scales by these.
const DESIGN_W: f32 = 640.0;
const DESIGN_H: f32 = 480.0;

/// Visibility floors, mirroring field_compose's constants.
const MIN_ALPHA: f32 = 1.0 / 255.0;
const MIN_DET: f32 = 1e-9;
const REST_EPS: f32 = 1e-4;

/// Row-major 3x3 homography (row-vector convention: `v @ M`, translation
/// in the bottom row) - matches transform3d.py's returned H.
pub type Mat3 = [f32; 9];

const IDENTITY3: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// One link's sampled 2D transform scalars (the caller fills these from
/// ChannelRefs). Field names track `field_compose._LINK_RESTS`; rests
/// are the SM defaults an untouched link carries.
#[derive(Clone, Copy, Debug)]
pub struct TransformState {
    pub x: f32,
    pub y: f32,
    pub zoom_x: f32,
    pub zoom_y: f32,
    pub rot: f32,
    pub skew_x: f32,
    pub skew_y: f32,
    pub base_scale_x: f32,
    pub base_scale_y: f32,
    pub halign: f32,
    pub valign: f32,
    pub hidden: f32,
    pub awake: f32,
    pub alpha: f32,
    /// left, top, right, bottom edge insets (fractions of the texture).
    pub crop: [f32; 4],
    pub natural_w: f32,
    pub natural_h: f32,
    /// Out-of-plane rotation (degrees) and the depth translate/scale. All
    /// rest at engine identity, so a 2D link composes exactly as before.
    pub rotation_x: f32,
    pub rotation_y: f32,
    pub z: f32,
    pub scale_z: f32,
    pub base_scale_z: f32,
    /// The Euler order `rotation_x/y/rot` multiply in (SetRotationOrder).
    pub rotation_order: crate::camera::RotOrder,
}

impl Default for TransformState {
    fn default() -> Self {
        TransformState {
            x: 0.0,
            y: 0.0,
            zoom_x: 1.0,
            zoom_y: 1.0,
            rot: 0.0,
            skew_x: 0.0,
            skew_y: 0.0,
            base_scale_x: 1.0,
            base_scale_y: 1.0,
            halign: 0.5,
            valign: 0.5,
            hidden: 0.0,
            awake: 1.0,
            alpha: 1.0,
            crop: [0.0; 4],
            natural_w: DESIGN_W,
            natural_h: DESIGN_H,
            rotation_x: 0.0,
            rotation_y: 0.0,
            z: 0.0,
            scale_z: 1.0,
            base_scale_z: 1.0,
            rotation_order: crate::camera::RotOrder::Xyz,
        }
    }
}

// --- 3x3 row-vector primitives (transform3d.py, reduced to the plane) --

pub(crate) fn mul3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[row * 3 + k] * b[k * 3 + col];
            }
            m[row * 3 + col] = s;
        }
    }
    m
}

fn translate3(x: f32, y: f32) -> Mat3 {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, 1.0]
}

fn scale3(x: f32, y: f32) -> Mat3 {
    [x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 1.0]
}

/// rotate_z(deg): the in-plane spin (RageMatrixRotationZ), row-vector.
fn rotate_z3(deg: f32) -> Mat3 {
    let t = deg * std::f32::consts::PI / 180.0;
    let (c, s) = (t.cos(), t.sin());
    [c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0]
}

/// skew_x: x' = x + amount*y (m10 = amount).
fn skew_x3(amount: f32) -> Mat3 {
    [1.0, 0.0, 0.0, amount, 1.0, 0.0, 0.0, 0.0, 1.0]
}

/// skew_y: y' = y + amount*x (m01 = amount).
fn skew_y3(amount: f32) -> Mat3 {
    [1.0, amount, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

fn det3(m: &Mat3) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// True when the link leaves every out-of-plane term at rest, so its map
/// is a plane-preserving 2D affine and the Mat3 path is exact.
fn is_planar(link: &TransformState) -> bool {
    link.rotation_x == 0.0
        && link.rotation_y == 0.0
        && link.z == 0.0
        && link.scale_z == 1.0
        && link.base_scale_z == 1.0
}

/// One link's local matrix in 3D - `field_compose._local`'s general branch.
/// Delegates the engine-ordered rotate/scale/translate/skew build to
/// `camera::local_matrix` (the shared port) rather than repeating it; the
/// anchor offset rides innermost, exactly as in the 2D path.
fn local4(link: &TransformState, flip: bool, leaf: bool) -> crate::camera::Mat4 {
    let (adx, ady) = anchor(link, flip, leaf);
    let mut base_sy = link.base_scale_y;
    if flip {
        base_sy = -base_sy;
    }
    let m = crate::camera::local_matrix(
        [link.x, link.y, link.z],
        [link.rotation_x, link.rotation_y, link.rot],
        [
            link.zoom_x * link.base_scale_x,
            link.zoom_y * base_sy,
            link.scale_z * link.base_scale_z,
        ],
        [link.skew_x, link.skew_y],
        link.rotation_order,
    );
    if adx == 0.0 && ady == 0.0 {
        return m;
    }
    crate::camera::mat_mul(&mat4_translate(adx, ady), &m)
}

fn mat4_translate(x: f32, y: f32) -> crate::camera::Mat4 {
    let mut m = [0.0f32; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m[12] = x;
    m[13] = y;
    m
}

/// The leaf's halign/valign anchor offset (0 for a non-leaf link).
fn anchor(link: &TransformState, flip: bool, leaf: bool) -> (f32, f32) {
    if !leaf {
        return (0.0, 0.0);
    }
    // The anchor offsets the quad by its OWN size, so it scales with the
    // leaf's natural box - not a global design constant. A leaf that leaves
    // natural_w/h at the 640x480 default composes exactly as before.
    let adx = (0.5 - link.halign) * link.natural_w;
    let mut ady = (0.5 - link.valign) * link.natural_h;
    if flip {
        ady = -ady;
    }
    (adx, ady)
}

/// Embed a planar Mat3 (row-vector) in a Mat4, and project a Mat4 back to
/// the z=0 plane's Mat3. The pair is lossless for planar maps, letting a
/// chain mix 2D and 3D links without forcing every link through Mat4.
fn mat3_to_mat4(m: &Mat3) -> crate::camera::Mat4 {
    let mut o = [0.0f32; 16];
    o[10] = 1.0;
    for (row, src) in [0usize, 1, 3].iter().enumerate() {
        for (col, dst) in [0usize, 1, 3].iter().enumerate() {
            o[src * 4 + dst] = m[row * 3 + col];
        }
    }
    o
}

/// A 4x4 translation, for lifting the content-centring offset into the
/// projected path (row-vector convention, matching `camera::compose`).
fn translate4(tx: f32, ty: f32) -> crate::camera::Mat4 {
    let mut m = crate::camera::IDENTITY;
    m[12] = tx;
    m[13] = ty;
    m
}

fn mat4_to_mat3(m: &crate::camera::Mat4) -> Mat3 {
    let mut o = [0.0f32; 9];
    for (row, src) in [0usize, 1, 3].iter().enumerate() {
        for (col, dst) in [0usize, 1, 3].iter().enumerate() {
            o[row * 3 + col] = m[src * 4 + dst];
        }
    }
    o
}

/// One link's local matrix - the 2D reduction of field_compose._local.
/// `flip` and `leaf` gate the mirror + anchor exactly as the Python does.
fn local(link: &TransformState, flip: bool, leaf: bool) -> Mat3 {
    let (mut adx, mut ady) = (0.0f32, 0.0f32);
    if leaf {
        adx = (0.5 - link.halign) * DESIGN_W;
        ady = (0.5 - link.valign) * DESIGN_H;
        if flip {
            ady = -ady;
        }
    }

    let mut base_sy = link.base_scale_y;
    if flip {
        base_sy = -base_sy;
    }
    let sx = link.zoom_x * link.base_scale_x;
    let sy = link.zoom_y * base_sy;
    let (skew, skewy) = (link.skew_x, link.skew_y);

    // The overwhelmingly common plain-positioned link: a single
    // translation (matches field_compose's fast path bit-for-bit).
    if link.rot == 0.0 && skew == 0.0 && skewy == 0.0 && sx == 1.0 && sy == 1.0 {
        return translate3(link.x + adx, link.y + ady);
    }

    let mut m = rotate_z3(link.rot);
    m = mul3(&m, &scale3(sx, sy));
    m = mul3(&m, &translate3(link.x, link.y));
    // skew_*_before rests 0 (stock: skew composes on the content side).
    if skew != 0.0 {
        m = mul3(&skew_x3(skew), &m);
    }
    if skewy != 0.0 {
        m = mul3(&skew_y3(skewy), &m);
    }
    if adx != 0.0 || ady != 0.0 {
        m = mul3(&translate3(adx, ady), &m);
    }
    m
}

/// Fold the link chain (root-first) into `(H, alpha, crop)`, or None
/// when the instance is invisible: hidden anywhere, alpha ~0, or a
/// degenerate composed scale. Ports `TransformChannel.at` + `crop_at`.
///
/// `flip_base_y` mirrors the LEAF's vertical source axis (AFT capture
/// compensation). The returned H is the row-major affine homography
/// mapping capture coords onto the design screen; crop is the leaf's
/// `(left, top, right, bottom)` insets (top/bottom swapped under flip),
/// or None when no edge is cropped.
pub fn compose_links(
    links: &[TransformState],
    flip_base_y: bool,
    projection: Option<&crate::camera::Mat4>,
) -> Option<(Mat3, f32, [f32; 4])> {
    if links.is_empty() {
        return None;
    }
    let leaf = links.len() - 1;
    let mut alpha = 1.0f32;
    // The chain stays on the exact Mat3 path until a link uses an
    // out-of-plane term; from there it composes in Mat4 (a planar link
    // embeds losslessly, so a mixed chain is still engine-faithful).
    let mut world: Option<Mat3> = None;
    let mut world4: Option<crate::camera::Mat4> = None;
    for (i, link) in links.iter().enumerate() {
        if link.awake < 0.5 {
            return None;
        }
        if link.hidden >= 0.5 {
            return None;
        }
        alpha *= link.alpha;
        let flip = flip_base_y && i == leaf;
        if world4.is_none() && is_planar(link) {
            let local = local(link, flip, i == leaf);
            // world = compose(world, local) = local @ world.
            world = Some(match world {
                None => local,
                Some(w) => mul3(&local, &w),
            });
            continue;
        }
        let local = local4(link, flip, i == leaf);
        let parent = world4.or_else(|| world.as_ref().map(mat3_to_mat4));
        world4 = Some(match parent {
            None => local,
            Some(w) => crate::camera::compose(&w, &local),
        });
    }
    if alpha < MIN_ALPHA {
        return None;
    }

    // Content is centred about its OWN box, so the offset is the leaf's
    // natural size - a drawable sized to what it draws still composes
    // correctly. The 640x480 default keeps every design-sized leaf identical.
    let leaf_link = &links[leaf];
    let (tcx, tcy) = (-(leaf_link.natural_w / 2.0), -(leaf_link.natural_h / 2.0));

    // THE PROJECTION MUST HAPPEN BEFORE THE COLLAPSE. `homography(model, proj)`
    // (transform3d.py:366) is `minor{0,1,3}(model @ proj)` - the 4x4s multiply
    // first and the z row/column is dropped only afterwards. Collapsing to a
    // Mat3 up front and folding the camera onto THAT (the old shape) throws
    // away the Z the perspective divide takes its w from, so an out-of-plane
    // chain came out a flat squash: the fold measured as a 0.00px change.
    // Mirrors field_compose.py:216-218 - `project_with_verdict(_TO_CONTENT @
    // world, projection, _CORNERS)` - with _TO_CONTENT lifted into 4D too.
    let h = match (world4, projection) {
        (Some(w4), Some(proj)) => {
            // `mat_mul(a, b) = a @ b`, so this is `_TO_CONTENT @ world` -
            // the content centring applies INNERMOST, matching the Mat3 path's
            // `mul3(&to_content, &world)`. (`camera::compose(parent, local)`
            // is `local @ parent`, i.e. reversed; using it here silently
            // produced `world @ _TO_CONTENT` and 2780px of corner error.)
            let model = crate::camera::mat_mul(&translate4(tcx, tcy), &w4);
            mat4_to_mat3(&crate::camera::mat_mul(&model, proj))
        }
        (Some(w4), None) => mul3(&translate3(tcx, tcy), &mat4_to_mat3(&w4)),
        // A planar chain never leaves z=0, where the centered design
        // projection maps 1:1 - the affine Mat3 IS the homography, and the
        // exact 2D path stays exact.
        (None, _) => mul3(&translate3(tcx, tcy), &world.expect("non-empty links")),
    };
    if det3(&h).abs() < MIN_DET {
        return None;
    }

    let crop = leaf_crop(&links[leaf], flip_base_y);
    Some((h, alpha, crop))
}

/// The leaf link's crop insets `(left, top, right, bottom)`; a flipped
/// leaf swaps top/bottom (the AFT source-mirror puts cropbottom's hidden
/// band at our source's top). Ports `TransformChannel.crop_at`.
fn leaf_crop(leaf: &TransformState, flip_base_y: bool) -> [f32; 4] {
    let [left, mut top, right, mut bottom] = leaf.crop;
    if flip_base_y {
        std::mem::swap(&mut top, &mut bottom);
    }
    [left, top, right, bottom]
}

/// True when no edge is cropped (field_compose returns None here). The
/// caller decides whether to elide a rest crop; compose_links always
/// returns the swapped crop so the executor sees the real leaf edges.
pub fn crop_is_rest(crop: &[f32; 4]) -> bool {
    crop.iter().all(|&edge| edge <= REST_EPS)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = include_str!("../fixtures/transform_cases.json");

    /// Minimal reader for the fixed transform_cases.json schema (avoids a
    /// serde dependency - the crate's Cargo.toml carries only pyo3).
    mod json {
        /// All f64 numbers in `s`, in source order (the fixture holds no
        /// other numerics, so positional extraction is unambiguous).
        pub fn numbers(s: &str) -> Vec<f64> {
            let bytes = s.as_bytes();
            let mut out = Vec::new();
            let mut i = 0;
            while i < bytes.len() {
                let c = bytes[i];
                let starts_num = c == b'-' || c.is_ascii_digit();
                if starts_num {
                    let start = i;
                    i += 1;
                    while i < bytes.len() {
                        let d = bytes[i];
                        if d.is_ascii_digit() || d == b'.' || d == b'-' || d == b'+'
                            || d == b'e' || d == b'E'
                        {
                            i += 1;
                        } else {
                            break;
                        }
                    }
                    out.push(s[start..i].parse::<f64>().unwrap());
                } else {
                    i += 1;
                }
            }
            out
        }
    }

    /// A structural view of one fixture case, extracted by scanning for
    /// the field markers (the schema is flat and fixed per the generator).
    struct Case {
        name: String,
        flip_base_y: bool,
        links: Vec<TransformState>,
        expected: Option<(Mat3, f32, Option<[f32; 4]>)>,
    }

    /// Field order the generator writes per link (see _STATE_FIELDS).
    fn parse_link(nums: &[f64]) -> TransformState {
        let g = |i: usize| nums[i] as f32;
        TransformState {
            x: g(0),
            y: g(1),
            zoom_x: g(2),
            zoom_y: g(3),
            rot: g(4),
            skew_x: g(5),
            skew_y: g(6),
            base_scale_x: g(7),
            base_scale_y: g(8),
            halign: g(9),
            valign: g(10),
            hidden: g(11),
            awake: 1.0,
            alpha: g(12),
            crop: [g(13), g(14), g(15), g(16)],
            natural_w: DESIGN_W,
            natural_h: DESIGN_H,
            // The fixture is the 2D corpus (LINK_FIELDS), so the
            // out-of-plane terms stay at rest and it keeps pinning the
            // exact planar path.
            ..TransformState::default()
        }
    }

    const LINK_FIELDS: usize = 17;

    fn parse_cases() -> Vec<Case> {
        // Split on the top-level "name" markers; each case block runs to
        // the next one. The generator emits one flat object per case.
        let mut cases = Vec::new();
        let markers: Vec<usize> = FIXTURES
            .match_indices("\"name\":")
            .map(|(i, _)| i)
            .collect();
        for (idx, &start) in markers.iter().enumerate() {
            let end = markers.get(idx + 1).copied().unwrap_or(FIXTURES.len());
            let block = &FIXTURES[start..end];

            let name_start = block.find('"').and_then(|_| {
                let after = &block[block.find(':').unwrap() + 1..];
                let q0 = after.find('"').unwrap() + 1;
                let q1 = after[q0..].find('"').unwrap();
                Some(after[q0..q0 + q1].to_string())
            });
            let name = name_start.unwrap();
            let flip_base_y = block[..block.find("\"links\"").unwrap()].contains("true");

            // The links array: numbers between "links": and "expected".
            let links_seg = {
                let a = block.find("\"links\"").unwrap();
                let b = block.find("\"expected\"").unwrap();
                &block[a..b]
            };
            let link_nums = json::numbers(links_seg);
            assert_eq!(link_nums.len() % LINK_FIELDS, 0, "case {name}");
            let links: Vec<TransformState> = link_nums
                .chunks(LINK_FIELDS)
                .map(parse_link)
                .collect();

            let exp_seg = &block[block.find("\"expected\"").unwrap()..];
            let expected = if exp_seg.contains("null")
                && !exp_seg.contains('[')
            {
                None
            } else {
                parse_expected(exp_seg)
            };
            cases.push(Case {
                name,
                flip_base_y,
                links,
                expected,
            });
        }
        cases
    }

    fn parse_expected(seg: &str) -> Option<(Mat3, f32, Option<[f32; 4]>)> {
        // "h": [9], "alpha": f, "crop": [4] | null.
        let h_seg = {
            let a = seg.find("\"h\"").unwrap();
            let b = seg.find("\"alpha\"").unwrap();
            &seg[a..b]
        };
        let h_nums = json::numbers(h_seg);
        assert_eq!(h_nums.len(), 9);
        let mut h = [0.0f32; 9];
        for (i, v) in h_nums.iter().enumerate() {
            h[i] = *v as f32;
        }
        let alpha_seg = {
            let a = seg.find("\"alpha\"").unwrap();
            let b = seg.find("\"crop\"").unwrap();
            &seg[a..b]
        };
        let alpha = json::numbers(alpha_seg)[0] as f32;

        let crop_seg = &seg[seg.find("\"crop\"").unwrap()..];
        let crop = if crop_seg.contains('[') {
            let c = json::numbers(crop_seg);
            assert_eq!(c.len(), 4);
            Some([c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32])
        } else {
            None
        };
        Some((h, alpha, crop))
    }

    #[test]
    fn fixture_parity() {
        let cases = parse_cases();
        assert!(cases.len() >= 40, "expected >= 40 cases, got {}", cases.len());
        let mut max_err = 0.0f32;
        let mut checked = 0;
        for case in &cases {
            let got = compose_links(&case.links, case.flip_base_y, None);
            match (&case.expected, got) {
                (None, None) => {}
                (Some(_), None) => panic!("{}: expected visible, got None", case.name),
                (None, Some(_)) => panic!("{}: expected None, got visible", case.name),
                (Some((eh, ea, ecrop)), Some((gh, ga, gcrop))) => {
                    for k in 0..9 {
                        let e = (eh[k] - gh[k]).abs();
                        max_err = max_err.max(e);
                        assert!(e < 1e-4, "{} H[{k}]: {} vs {}", case.name, eh[k], gh[k]);
                    }
                    let ae = (ea - ga).abs();
                    max_err = max_err.max(ae);
                    assert!(ae < 1e-4, "{} alpha: {ea} vs {ga}", case.name);
                    // Python's crop_at returns None at rest; compose_links
                    // returns the (swapped) leaf crop always, so a None
                    // expectation must equal an all-rest returned crop.
                    match ecrop {
                        Some(ec) => {
                            for k in 0..4 {
                                let ce = (ec[k] - gcrop[k]).abs();
                                max_err = max_err.max(ce);
                                assert!(
                                    ce < 1e-4,
                                    "{} crop[{k}]: {} vs {}",
                                    case.name,
                                    ec[k],
                                    gcrop[k]
                                );
                            }
                        }
                        None => assert!(
                            crop_is_rest(&gcrop),
                            "{}: expected rest crop, got {:?}",
                            case.name,
                            gcrop
                        ),
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked >= 40, "checked only {checked} visible cases");
        eprintln!("transform fixture_parity: {} cases, max err {max_err:e}", cases.len());
    }

    #[test]
    fn identity_link_is_centered_translate() {
        // A single rest link sits at the design centre offset by
        // _TO_CONTENT: content (320,240) -> (0,0). H maps that way.
        let (h, alpha, crop) = compose_links(&[TransformState::default()], false, None).unwrap();
        assert_eq!(alpha, 1.0);
        assert!(crop_is_rest(&crop));
        // v @ H for v=(320,240,1): x' = 320 - 320 = 0.
        let (vx, vy) = (320.0f32, 240.0f32);
        let px = vx * h[0] + vy * h[3] + h[6];
        let py = vx * h[1] + vy * h[4] + h[7];
        assert!(px.abs() < 1e-4 && py.abs() < 1e-4, "{px} {py}");
    }

    #[test]
    fn content_centres_about_its_own_natural_box() {
        // A drawable sized to what it DRAWS (not the design screen) must still
        // compose correctly: its content centre maps to the same place a
        // design-sized leaf's does. Here a 200x100 leaf centres at (100, 50).
        let leaf = TransformState {
            natural_w: 200.0,
            natural_h: 100.0,
            ..Default::default()
        };
        let (h, _, _) = compose_links(&[leaf], false, None).unwrap();
        let (vx, vy) = (100.0f32, 50.0f32);
        let px = vx * h[0] + vy * h[3] + h[6];
        let py = vx * h[1] + vy * h[4] + h[7];
        assert!(px.abs() < 1e-4 && py.abs() < 1e-4, "{px} {py}");
    }

    #[test]
    fn anchor_offsets_by_the_leafs_own_size() {
        // halign rides the leaf's natural width, so a 200-wide leaf anchored
        // left shifts by 100 (half its own box), not half the design screen.
        let leaf = TransformState {
            natural_w: 200.0,
            natural_h: 100.0,
            halign: 0.0,
            ..Default::default()
        };
        let (adx, ady) = anchor(&leaf, false, true);
        assert!((adx - 100.0).abs() < 1e-4, "{adx}");
        assert!(ady.abs() < 1e-4, "{ady}");
    }

    #[test]
    fn out_of_plane_terms_at_rest_take_the_exact_planar_path() {
        // The 3D extension must not perturb a 2D chain: an explicitly
        // rest-filled out-of-plane link folds bit-identically to the
        // planar composition (gat's field instances all live here).
        let planar = TransformState {
            x: 40.0,
            y: -25.0,
            zoom_x: 1.5,
            rot: 30.0,
            skew_x: 0.2,
            ..Default::default()
        };
        let parent = TransformState {
            x: 12.0,
            rot: -10.0,
            ..Default::default()
        };
        assert!(is_planar(&planar) && is_planar(&parent));
        let (h, _, _) = compose_links(&[parent, planar], false, None).unwrap();
        let expect = mul3(
            &translate3(-(DESIGN_W / 2.0), -(DESIGN_H / 2.0)),
            &mul3(&local(&planar, false, true), &local(&parent, false, false)),
        );
        assert_eq!(h, expect);
    }

    #[test]
    fn a_tipped_chain_with_a_camera_makes_a_trapezoid_not_a_squash() {
        // The perspective divide has to happen while the 4x4 still has its Z:
        // `homography(model, proj)` is `minor{0,1,3}(model @ proj)`, so the
        // projection multiplies BEFORE the z row/column is dropped. Collapsing
        // first and folding a camera onto the resulting Mat3 (the shape this
        // replaced) is arithmetic on data with no depth left, and measured as
        // a 0.00px change on a chart whose whole effect is `rotationy`.
        //
        // A y-axis tip must therefore make the near edge GROW and the far edge
        // SHRINK. Without the fold both edges keep the same height and the
        // quad merely narrows - the flat-squash signature.
        let tipped = TransformState {
            rotation_y: 45.0,
            ..Default::default()
        };
        assert!(!is_planar(&tipped));
        let proj = crate::camera::design_projection(
            45.0, DESIGN_W, DESIGN_H, [DESIGN_W / 2.0, DESIGN_H / 2.0], 1772.7);

        let edge_height = |h: &Mat3, x: f32| {
            let top = project_point(h, x, 0.0);
            let bottom = project_point(h, x, DESIGN_H);
            (bottom.1 - top.1).abs()
        };

        let (flat, _, _) = compose_links(&[tipped.clone()], false, None).unwrap();
        let (proj_h, _, _) =
            compose_links(&[tipped], false, Some(&proj)).unwrap();

        // Unprojected: both edges identical (the squash).
        let (fl, fr) = (edge_height(&flat, 0.0), edge_height(&flat, DESIGN_W));
        assert!((fl - fr).abs() < 1e-3, "unprojected edges differ: {fl} vs {fr}");

        // Projected: one edge genuinely taller than the other. NOTE this
        // pins that a camera is APPLIED, not that it is applied correctly -
        // a reversed `_TO_CONTENT @ world` still trapezoids, just in the
        // wrong place. The order is guarded by the Python parity harness
        // (test_out_of_plane_chain_matches_the_legacy_homography), which
        // compares the whole homography against TransformChannel.
        // Projected: one edge genuinely taller than the other.
        let (pl, pr) = (edge_height(&proj_h, 0.0), edge_height(&proj_h, DESIGN_W));
        assert!(
            (pl - pr).abs() > 0.1 * pl.max(pr),
            "a camera must make a trapezoid, got {pl} vs {pr}"
        );
    }

    /// Apply a column-vector Mat3 to a content point, with the divide.
    fn project_point(h: &Mat3, x: f32, y: f32) -> (f32, f32) {
        let px = x * h[0] + y * h[3] + h[6];
        let py = x * h[1] + y * h[4] + h[7];
        let pw = x * h[2] + y * h[5] + h[8];
        if pw.abs() < 1e-9 {
            return (px, py);
        }
        (px / pw, py / pw)
    }

    #[test]
    fn rotation_x_foreshortens_the_vertical_axis() {
        // gat 1's `gat_bg:rotationx(60)`: tipping about x shrinks the
        // projected y extent by cos(60) = 0.5 while x is untouched. The
        // z=0 block is what compose_links hands back (the camera folds in
        // at the item level), so this pins the rotation reaching the mat.
        let tipped = TransformState {
            rotation_x: 60.0,
            ..Default::default()
        };
        assert!(!is_planar(&tipped));
        let (h, _, _) = compose_links(&[tipped], false, None).unwrap();
        // A content point one unit "down" from centre: (320, 241).
        let (vx, vy) = (320.0f32, 241.0f32);
        let px = vx * h[0] + vy * h[3] + h[6];
        let py = vx * h[1] + vy * h[4] + h[7];
        assert!(px.abs() < 1e-4, "x untouched by an x-axis tip: {px}");
        let cos60 = (60.0f32).to_radians().cos();
        assert!((py - cos60).abs() < 1e-4, "y foreshortened: {py} vs {cos60}");
    }

    #[test]
    fn a_planar_parent_composes_onto_a_tipped_child() {
        // Mixed chains are the gat background shape (a positioned frame
        // over a rotationx'd one): the planar prefix must survive the
        // promotion to Mat4 rather than being dropped.
        let parent = TransformState {
            x: 15.0,
            y: 7.0,
            ..Default::default()
        };
        let child = TransformState {
            rotation_x: 60.0,
            ..Default::default()
        };
        let (h, _, _) = compose_links(&[parent, child], false, None).unwrap();
        let (vx, vy) = (320.0f32, 240.0f32);
        let px = vx * h[0] + vy * h[3] + h[6];
        let py = vx * h[1] + vy * h[4] + h[7];
        assert!((px - 15.0).abs() < 1e-4, "parent x carried: {px}");
        assert!((py - 7.0).abs() < 1e-4, "parent y carried: {py}");
    }

    #[test]
    fn hidden_and_zero_alpha_and_degenerate_gate_to_none() {
        let hidden = TransformState {
            hidden: 1.0,
            ..Default::default()
        };
        assert!(compose_links(&[hidden], false, None).is_none());
        let faint = TransformState {
            alpha: 0.0,
            ..Default::default()
        };
        assert!(compose_links(&[faint], false, None).is_none());
        let flat = TransformState {
            zoom_x: 0.0,
            ..Default::default()
        };
        assert!(compose_links(&[flat], false, None).is_none());
    }

    #[test]
    fn identity3_is_unused_sanity() {
        assert_eq!(mul3(&IDENTITY3, &IDENTITY3), IDENTITY3);
    }
}
