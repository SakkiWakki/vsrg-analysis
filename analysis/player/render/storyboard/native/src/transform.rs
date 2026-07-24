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
    pub alpha: f32,
    /// left, top, right, bottom edge insets (fractions of the texture).
    pub crop: [f32; 4],
    pub natural_w: f32,
    pub natural_h: f32,
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
            alpha: 1.0,
            crop: [0.0; 4],
            natural_w: DESIGN_W,
            natural_h: DESIGN_H,
        }
    }
}

// --- 3x3 row-vector primitives (transform3d.py, reduced to the plane) --

fn mul3(a: &Mat3, b: &Mat3) -> Mat3 {
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
pub fn compose_links(links: &[TransformState], flip_base_y: bool) -> Option<(Mat3, f32, [f32; 4])> {
    if links.is_empty() {
        return None;
    }
    let leaf = links.len() - 1;
    let mut alpha = 1.0f32;
    let mut world: Option<Mat3> = None;
    for (i, link) in links.iter().enumerate() {
        if link.hidden >= 0.5 {
            return None;
        }
        alpha *= link.alpha;
        let flip = flip_base_y && i == leaf;
        let local = local(link, flip, i == leaf);
        // world = compose(world, local) = local @ world.
        world = Some(match world {
            None => local,
            Some(w) => mul3(&local, &w),
        });
    }
    if alpha < MIN_ALPHA {
        return None;
    }

    // _TO_CONTENT = translate(-320, -240) applied on the content side.
    // Under the centered design projection the z=0 plane maps 1:1, so
    // the normalized homography IS this affine 3x3.
    let world = world.expect("non-empty links");
    let to_content = translate3(-(DESIGN_W / 2.0), -(DESIGN_H / 2.0));
    let h = mul3(&to_content, &world);
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
            alpha: g(12),
            crop: [g(13), g(14), g(15), g(16)],
            natural_w: DESIGN_W,
            natural_h: DESIGN_H,
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
            let got = compose_links(&case.links, case.flip_base_y);
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
        let (h, alpha, crop) = compose_links(&[TransformState::default()], false).unwrap();
        assert_eq!(alpha, 1.0);
        assert!(crop_is_rest(&crop));
        // v @ H for v=(320,240,1): x' = 320 - 320 = 0.
        let (vx, vy) = (320.0f32, 240.0f32);
        let px = vx * h[0] + vy * h[3] + h[6];
        let py = vx * h[1] + vy * h[4] + h[7];
        assert!(px.abs() < 1e-4 && py.abs() < 1e-4, "{px} {py}");
    }

    #[test]
    fn hidden_and_zero_alpha_and_degenerate_gate_to_none() {
        let hidden = TransformState {
            hidden: 1.0,
            ..Default::default()
        };
        assert!(compose_links(&[hidden], false).is_none());
        let faint = TransformState {
            alpha: 0.0,
            ..Default::default()
        };
        assert!(compose_links(&[faint], false).is_none());
        let flat = TransformState {
            zoom_x: 0.0,
            ..Default::default()
        };
        assert!(compose_links(&[flat], false).is_none());
    }

    #[test]
    fn identity3_is_unused_sanity() {
        assert_eq!(mul3(&IDENTITY3, &IDENTITY3), IDENTITY3);
    }
}
