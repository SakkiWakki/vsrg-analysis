//! 3D transform math - port of analysis/player/render/transform3d.py plus
//! the LoadMenuPerspective fov camera build in
//! analysis/games/notitg/field_projection.py (design_projection).
//!
//! Convention (row-vector, matching RageMatrix and Qt QTransform): a
//! point is v = [x y z 1], mapped by v @ M. `mat_mul(a, b)` is the
//! standard row-major product a * b, and `v @ (a @ b)` applies a first.
//! `Mat4` is [f32; 16] ROW-MAJOR: index r*4 + c is row r, column c.
//!
//! See .claude/plans/drawable-port-wave1.md (A3) and transform3d.py for
//! the SM source-line derivations.

pub type Mat4 = [f32; 16];

/// Euler rotation order (the fork's SetRotationOrder swizzle). Names the
/// axis applied FIRST to a row point; 'xyz' is the stock RageMatrix
/// default (rotate_xyz).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotOrder {
    Xyz,
    Xzy,
    Yzx,
    Zxy,
    Zyx,
}

impl RotOrder {
    /// The three axes in the order they multiply for a row point: order[0]
    /// applied first. `from_str` accepts the fork's swizzle tokens.
    fn axes(self) -> [Axis; 3] {
        match self {
            RotOrder::Xyz => [Axis::X, Axis::Y, Axis::Z],
            RotOrder::Xzy => [Axis::X, Axis::Z, Axis::Y],
            RotOrder::Yzx => [Axis::Y, Axis::Z, Axis::X],
            RotOrder::Zxy => [Axis::Z, Axis::X, Axis::Y],
            RotOrder::Zyx => [Axis::Z, Axis::Y, Axis::X],
        }
    }

    pub fn from_str(s: &str) -> Option<RotOrder> {
        match s {
            "xyz" => Some(RotOrder::Xyz),
            "xzy" => Some(RotOrder::Xzy),
            "yzx" => Some(RotOrder::Yzx),
            "zxy" => Some(RotOrder::Zxy),
            "zyx" => Some(RotOrder::Zyx),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

const DEG: f32 = std::f32::consts::PI / 180.0;

/// glFrustum near plane and the +1000 far-plane slack SM uses
/// (RageDisplay LoadMenuPerspective). Exposed as the design_projection
/// far default; callers pass an explicit far.
pub const NEAR: f32 = 1.0;
pub const FAR_SLACK: f32 = 1000.0;

pub const IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

#[inline]
fn at(m: &Mat4, r: usize, c: usize) -> f32 {
    m[r * 4 + c]
}

/// Standard row-major product a * b. For row points, v @ (mat_mul(a, b))
/// applies a first, then b. The dot products accumulate in f64 (Python
/// does the whole chain in float64); pixel-scale entries otherwise lose
/// the exact cancellations - e.g. a centered projection's x-translation
/// must vanish to 0, not to a 1/64 f32 residue - and the fixture parity
/// tolerance is 1e-3.
pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            let mut sum = 0.0f64;
            for k in 0..4 {
                sum += at(a, r, k) as f64 * at(b, k, c) as f64;
            }
            out[r * 4 + c] = sum as f32;
        }
    }
    out
}

/// T: translation in the bottom row (RageMatrixTranslate).
fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = IDENTITY;
    m[3 * 4] = x;
    m[3 * 4 + 1] = y;
    m[3 * 4 + 2] = z;
    m
}

/// S = diag(x, y, z, 1) (RageMatrixScale).
fn scale(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = IDENTITY;
    m[0] = x;
    m[5] = y;
    m[10] = z;
    m
}

/// One-axis rotation, verbatim RageMatrixRotationX/Y/Z. `deg` in degrees.
fn rotate_axis(axis: Axis, deg: f32) -> Mat4 {
    let t = deg * DEG;
    let (c, s) = (t.cos(), t.sin());
    let mut m = IDENTITY;
    match axis {
        Axis::X => {
            m[5] = c; // (1,1)
            m[10] = c; // (2,2)
            m[9] = s; // (2,1)
            m[6] = -s; // (1,2)
        }
        Axis::Y => {
            m[0] = c; // (0,0)
            m[10] = c; // (2,2)
            m[2] = s; // (0,2)
            m[8] = -s; // (2,0)
        }
        Axis::Z => {
            m[0] = c; // (0,0)
            m[5] = c; // (1,1)
            m[1] = s; // (0,1)
            m[4] = -s; // (1,0)
        }
    }
    m
}

/// Fused X*Y*Z rotation, closed form from RageMatrixRotationXYZ. A row
/// point v @ result rotates about X first, then Y, then Z (the stock
/// order); equals rotate_z(rz) @ rotate_y(ry) @ rotate_x(rx).
fn rotate_xyz(rx: f32, ry: f32, rz: f32) -> Mat4 {
    let (rx, ry, rz) = (rx * DEG, ry * DEG, rz * DEG);
    let (c_x, s_x) = (rx.cos(), rx.sin());
    let (c_y, s_y) = (ry.cos(), ry.sin());
    let (c_z, s_z) = (rz.cos(), rz.sin());
    let mut m = IDENTITY;
    m[0] = c_z * c_y;
    m[1] = c_z * s_y * s_x + s_z * c_x;
    m[2] = c_z * s_y * c_x - s_z * s_x;
    m[4] = -s_z * c_y;
    m[5] = -s_z * s_y * s_x + c_z * c_x;
    m[6] = -s_z * s_y * c_x - c_z * s_x;
    m[8] = -s_y;
    m[9] = c_y * s_x;
    m[10] = c_y * c_x;
    m
}

/// Euler rotation composed in `order` (the fork's SetRotationOrder). The
/// order names the axis applied FIRST to a row point; in matmul the
/// matrices multiply in reverse: R(order[2]) * R(order[1]) * R(order[0]).
/// 'xyz' short-circuits to the fused closed form and MUST equal it.
pub fn rotate_ordered(rx: f32, ry: f32, rz: f32, order: RotOrder) -> Mat4 {
    if order == RotOrder::Xyz {
        return rotate_xyz(rx, ry, rz);
    }
    let angle = |a: Axis| match a {
        Axis::X => rx,
        Axis::Y => ry,
        Axis::Z => rz,
    };
    let [first, second, third] = order.axes();
    let m = mat_mul(
        &rotate_axis(third, angle(third)),
        &rotate_axis(second, angle(second)),
    );
    mat_mul(&m, &rotate_axis(first, angle(first)))
}

/// SkewX: x' = x + amount*y (RageMatrixSkewX, m10 = amount).
pub fn skew_x(amount: f32) -> Mat4 {
    let mut m = IDENTITY;
    m[4] = amount; // (1,0)
    m
}

/// SkewY: y' = y + amount*x (RageMatrixSkewY, m01 = amount).
pub fn skew_y(amount: f32) -> Mat4 {
    let mut m = IDENTITY;
    m[1] = amount; // (0,1)
    m
}

/// A node's local matrix per Actor::BeginDraw:
///   L = S(scl) @ R(rot, order) @ T(pos) @ SkewX @ SkewY
/// so v @ L scales, then rotates, then translates (the SM sprite
/// reading). Skews are applied only when nonzero (matching the Python
/// guard). `skew` is (skewx, skewy).
pub fn local_matrix(
    pos: [f32; 3],
    rot: [f32; 3],
    scl: [f32; 3],
    skew: [f32; 2],
    order: RotOrder,
) -> Mat4 {
    let mut l = mat_mul(
        &mat_mul(
            &scale(scl[0], scl[1], scl[2]),
            &rotate_ordered(rot[0], rot[1], rot[2], order),
        ),
        &translate(pos[0], pos[1], pos[2]),
    );
    if skew[0] != 0.0 {
        l = mat_mul(&l, &skew_x(skew[0]));
    }
    if skew[1] != 0.0 {
        l = mat_mul(&l, &skew_y(skew[1]));
    }
    l
}

/// Child world = local @ parent (row-vector; ActorFrame nesting). A row
/// point v maps by v @ compose(parent, local): local first, then parent.
pub fn compose(parent: &Mat4, local: &Mat4) -> Mat4 {
    mat_mul(local, parent)
}

/// Camera distance for LoadMenuPerspective: d = (W/2) / tan(fov/2). The
/// center-plane perspective scale of a +z push is d / (d - z).
pub fn eye_distance(fov_deg: f32, width: f32) -> f32 {
    (width / 2.0) / ((fov_deg * DEG) / 2.0).tan()
}

/// LoadMenuPerspective world -> design-pixel projection for
/// (fov, viewport, vanish, far). Ports RageDisplay::LoadMenuPerspective.
/// `vanish` is the screen (px, py) vanishing point; None (or exactly the
/// screen center) gives a centered frustum. `far` is the far-plane z
/// (the stock projection uses eye_distance + FAR_SLACK). Returns a
/// row-vector 4x4 P: [x y z 1] @ P = [px*w, py*w, *, w].
///
/// (fov == 0 is the orthographic branch in the Python source; the field
/// only ever builds a perspective camera through this path - the fov->0
/// degradation is handled by the 2D kernels, not here - so this port
/// covers the perspective branch that design_projection actually uses.)
pub fn design_projection(fov_deg: f32, w: f32, h: f32, vanish: [f32; 2], far: f32) -> Mat4 {
    design_projection_opt(fov_deg, w, h, Some(vanish), far)
}

/// design_projection with an explicit centered case (vanish = None).
pub fn design_projection_opt(
    fov_deg: f32,
    w: f32,
    h: f32,
    vanish: Option<[f32; 2]>,
    far: f32,
) -> Mat4 {
    // The whole view @ frust @ viewport chain runs in f64 (as Python's
    // numpy does) and downcasts once at the end. These matrices carry
    // pixel-scale entries (~772, ~320) whose exact cancellations - a
    // centered camera's x/y translation must vanish to 0 - are destroyed
    // by an f32-intermediate chain (the residue lands at 2^-5, far past
    // the 1e-3 tolerance).
    let (w, h, far) = (w as f64, h as f64, far as f64);
    let (vx_screen, vy_screen) = match vanish {
        Some(v) => (v[0] as f64, v[1] as f64),
        None => (w / 2.0, h / 2.0),
    };
    let d = (w / 2.0) / ((fov_deg as f64 * DEG as f64) / 2.0).tan();
    // SCALE(v, 0, W, W, 0) = W - v, then - W/2.
    let vx = (w - vx_screen) - w / 2.0;
    let vy = (h - vy_screen) - h / 2.0;
    let frust = frustum_d(
        (vx - w / 2.0) / d,
        (vx + w / 2.0) / d,
        (vy + h / 2.0) / d,
        (vy - h / 2.0) / d,
        NEAR as f64,
        far,
    );
    // RageLookAt eye = (-vx+W/2, -vy+H/2, d) -> world translation by -eye.
    let view = translate_d(vx - w / 2.0, vy - h / 2.0, -d);
    let m = mul_d(&mul_d(&view, &frust), &viewport_d(w, h));
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = m[i] as f32;
    }
    out
}

type Mat4d = [f64; 16];

fn mul_d(a: &Mat4d, b: &Mat4d) -> Mat4d {
    let mut out = [0.0f64; 16];
    for r in 0..4 {
        for c in 0..4 {
            let mut sum = 0.0f64;
            for k in 0..4 {
                sum += a[r * 4 + k] * b[k * 4 + c];
            }
            out[r * 4 + c] = sum;
        }
    }
    out
}

fn translate_d(x: f64, y: f64, z: f64) -> Mat4d {
    let mut m = [0.0f64; 16];
    for i in 0..4 {
        m[i * 4 + i] = 1.0;
    }
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m
}

fn viewport_d(width: f64, height: f64) -> Mat4d {
    let mut m = [0.0f64; 16];
    for i in 0..4 {
        m[i * 4 + i] = 1.0;
    }
    m[0] = width / 2.0;
    m[12] = width / 2.0;
    m[5] = -height / 2.0;
    m[13] = height / 2.0;
    m
}

fn frustum_d(l: f64, r: f64, b: f64, t: f64, zn: f64, zf: f64) -> Mat4d {
    let a = (r + l) / (r - l);
    let bb = (t + b) / (t - b);
    let cc = -(zf + zn) / (zf - zn);
    let d = -(2.0 * zf * zn) / (zf - zn);
    [
        2.0 * zn / (r - l), 0.0, 0.0, 0.0, //
        0.0, 2.0 * zn / (t - b), 0.0, 0.0, //
        a, bb, cc, -1.0, //
        0.0, 0.0, d, 0.0,
    ]
}

/// Map a row point through m with the perspective (w) divide, returning
/// the 2D pixel. Callers must first classify w<=eps (behind/through the
/// eye) as gone; here w==0 yields infinities rather than a panic.
pub fn project(m: &Mat4, p: [f32; 3]) -> [f32; 2] {
    let x = dot_col(m, p, 0);
    let y = dot_col(m, p, 1);
    let w = dot_col(m, p, 3);
    [(x / w) as f32, (y / w) as f32]
}

/// The full projective w at a row point (for the behind-eye verdict).
pub fn project_w(m: &Mat4, p: [f32; 3]) -> f32 {
    dot_col(m, p, 3) as f32
}

/// [x y z 1] . column c of m, accumulated in f64 (matching Python).
fn dot_col(m: &Mat4, p: [f32; 3], c: usize) -> f64 {
    p[0] as f64 * at(m, 0, c) as f64
        + p[1] as f64 * at(m, 1, c) as f64
        + p[2] as f64 * at(m, 2, c) as f64
        + at(m, 3, c) as f64
}

#[cfg(test)]
mod tests {
    //! Fixture parity. `fixtures/camera_cases.json` is written by the REAL
    //! Python 3D math (gen_fixtures/camera.py); we parse it with a tiny
    //! dependency-free JSON reader (the crate carries no serde, and
    //! Cargo.toml is not ours to touch) and assert matrix/point parity to
    //! 1e-3 (trig accumulation).
    use super::*;

    const EPS: f32 = 1e-3;
    const EYE_EPS: f32 = 1e-6;

    // -- minimal JSON value + parser (objects, arrays, numbers, strings,
    //    true/false/null; enough for our own generated fixtures) ---------
    #[derive(Clone, Debug)]
    enum Json {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Json>),
        Obj(Vec<(String, Json)>),
    }

    impl Json {
        fn get(&self, key: &str) -> Option<&Json> {
            match self {
                Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }
        fn arr(&self) -> &[Json] {
            match self {
                Json::Arr(v) => v,
                _ => panic!("expected array"),
            }
        }
        fn f32(&self) -> f32 {
            match self {
                Json::Num(n) => *n as f32,
                _ => panic!("expected number"),
            }
        }
        fn bool(&self) -> bool {
            match self {
                Json::Bool(b) => *b,
                _ => panic!("expected bool"),
            }
        }
        fn s(&self) -> &str {
            match self {
                Json::Str(s) => s,
                _ => panic!("expected string"),
            }
        }
        fn f3(&self) -> [f32; 3] {
            let a = self.arr();
            [a[0].f32(), a[1].f32(), a[2].f32()]
        }
        fn f2(&self) -> [f32; 2] {
            let a = self.arr();
            [a[0].f32(), a[1].f32()]
        }
        fn vecf(&self) -> Vec<f32> {
            self.arr().iter().map(|v| v.f32()).collect()
        }
    }

    struct Parser<'a> {
        b: &'a [u8],
        i: usize,
    }

    impl<'a> Parser<'a> {
        fn ws(&mut self) {
            while self.i < self.b.len() && (self.b[self.i] as char).is_whitespace() {
                self.i += 1;
            }
        }
        fn value(&mut self) -> Json {
            self.ws();
            match self.b[self.i] {
                b'{' => self.object(),
                b'[' => self.array(),
                b'"' => Json::Str(self.string()),
                b't' => {
                    self.i += 4;
                    Json::Bool(true)
                }
                b'f' => {
                    self.i += 5;
                    Json::Bool(false)
                }
                b'n' => {
                    self.i += 4;
                    Json::Null
                }
                _ => self.number(),
            }
        }
        fn object(&mut self) -> Json {
            let mut kv = Vec::new();
            self.i += 1; // {
            self.ws();
            if self.b[self.i] == b'}' {
                self.i += 1;
                return Json::Obj(kv);
            }
            loop {
                self.ws();
                let key = self.string();
                self.ws();
                self.i += 1; // :
                let val = self.value();
                kv.push((key, val));
                self.ws();
                let c = self.b[self.i];
                self.i += 1; // , or }
                if c == b'}' {
                    break;
                }
            }
            Json::Obj(kv)
        }
        fn array(&mut self) -> Json {
            let mut v = Vec::new();
            self.i += 1; // [
            self.ws();
            if self.b[self.i] == b']' {
                self.i += 1;
                return Json::Arr(v);
            }
            loop {
                v.push(self.value());
                self.ws();
                let c = self.b[self.i];
                self.i += 1; // , or ]
                if c == b']' {
                    break;
                }
            }
            Json::Arr(v)
        }
        fn string(&mut self) -> String {
            self.i += 1; // opening quote
            let mut s = String::new();
            while self.b[self.i] != b'"' {
                if self.b[self.i] == b'\\' {
                    self.i += 1;
                    s.push(self.b[self.i] as char);
                } else {
                    s.push(self.b[self.i] as char);
                }
                self.i += 1;
            }
            self.i += 1; // closing quote
            s
        }
        fn number(&mut self) -> Json {
            let start = self.i;
            while self.i < self.b.len() {
                let c = self.b[self.i];
                if c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' || c.is_ascii_digit() {
                    self.i += 1;
                } else {
                    break;
                }
            }
            let text = std::str::from_utf8(&self.b[start..self.i]).unwrap();
            Json::Num(text.parse().unwrap())
        }
    }

    fn parse(text: &str) -> Json {
        let mut p = Parser { b: text.as_bytes(), i: 0 };
        p.value()
    }

    fn opt_f2(j: &Json) -> Option<[f32; 2]> {
        match j {
            Json::Null => None,
            other => Some(other.f2()),
        }
    }

    fn to_mat(v: &[f32]) -> Mat4 {
        let mut m = [0.0f32; 16];
        m.copy_from_slice(&v[..16]);
        m
    }

    fn assert_mat_close(got: &Mat4, want: &[f32], ctx: &str) -> f32 {
        let mut max = 0.0f32;
        for i in 0..16 {
            let d = (got[i] - want[i]).abs();
            assert!(
                d <= EPS,
                "{ctx}: entry {i} got {} want {} (|d|={d})",
                got[i],
                want[i]
            );
            max = max.max(d);
        }
        max
    }

    const FIXTURES: &str = include_str!("../fixtures/camera_cases.json");

    #[test]
    fn parity_against_python_fixtures() {
        let root = parse(FIXTURES);
        let mut max_err = 0.0f32;
        let mut n_local = 0;
        let mut n_compose = 0;
        let mut n_proj = 0;
        let mut n_pts = 0;

        for c in root.get("local").unwrap().arr() {
            let order = RotOrder::from_str(c.get("order").unwrap().s()).expect("known order");
            let got = local_matrix(
                c.get("pos").unwrap().f3(),
                c.get("rot").unwrap().f3(),
                c.get("scl").unwrap().f3(),
                c.get("skew").unwrap().f2(),
                order,
            );
            max_err = max_err.max(assert_mat_close(&got, &c.get("expected").unwrap().vecf(), "local"));
            n_local += 1;
        }

        for c in root.get("compose").unwrap().arr() {
            let got = compose(
                &to_mat(&c.get("parent").unwrap().vecf()),
                &to_mat(&c.get("local").unwrap().vecf()),
            );
            max_err =
                max_err.max(assert_mat_close(&got, &c.get("expected").unwrap().vecf(), "compose"));
            n_compose += 1;
        }

        for c in root.get("projection").unwrap().arr() {
            let vanish = opt_f2(c.get("vanish").unwrap());
            let got = design_projection_opt(
                c.get("fov").unwrap().f32(),
                c.get("w").unwrap().f32(),
                c.get("h").unwrap().f32(),
                vanish,
                c.get("far").unwrap().f32(),
            );
            max_err = max_err.max(assert_mat_close(&got, &c.get("matrix").unwrap().vecf(), "projection"));
            n_proj += 1;
            for pt in c.get("points").unwrap().arr() {
                let p = pt.get("p").unwrap().f3();
                let w = project_w(&got, p);
                let behind = w <= EYE_EPS;
                assert_eq!(
                    behind,
                    pt.get("behind_eye").unwrap().bool(),
                    "behind-eye verdict mismatch at p={p:?} (rust w={w} vs python w={})",
                    pt.get("w").unwrap().f32()
                );
                if let Some(want) = pt.get("expected") {
                    let want = want.f2();
                    let px = project(&got, p);
                    for k in 0..2 {
                        let d = (px[k] - want[k]).abs();
                        assert!(
                            d <= 1e-2,
                            "projected point {k}: got {} want {} (|d|={d}) p={p:?}",
                            px[k],
                            want[k]
                        );
                        max_err = max_err.max(d.min(EPS));
                    }
                    n_pts += 1;
                }
            }
        }

        eprintln!(
            "camera parity: {n_local} local + {n_compose} compose + {n_proj} projection \
             ({n_pts} front-of-eye points), max matrix |err| = {max_err}"
        );
        assert!(n_local >= 30, "spec wants >= 30 cases; got {n_local} local");
    }

    #[test]
    fn xyz_order_matches_fused_closed_form() {
        // rotate_ordered('xyz') short-circuits; check it equals the
        // per-axis product for a nontrivial angle triple.
        let fused = rotate_ordered(17.0, -33.0, 61.0, RotOrder::Xyz);
        let manual = mat_mul(
            &mat_mul(&rotate_axis(Axis::Z, 61.0), &rotate_axis(Axis::Y, -33.0)),
            &rotate_axis(Axis::X, 17.0),
        );
        for i in 0..16 {
            assert!((fused[i] - manual[i]).abs() <= 1e-5, "entry {i}");
        }
    }

    #[test]
    fn centered_design_plane_maps_one_to_one() {
        // The z=0 design plane maps 1:1 to pixels under a centered vanish.
        let w = 640.0;
        let h = 480.0;
        let far = eye_distance(45.0, w) + FAR_SLACK;
        let m = design_projection_opt(45.0, w, h, None, far);
        // The frozen non-optional entry with the vanish at the exact
        // center must agree with the centered (None) build.
        let m_explicit = design_projection(45.0, w, h, [w / 2.0, h / 2.0], far);
        for i in 0..16 {
            assert!((m[i] - m_explicit[i]).abs() <= 1e-4, "explicit-center entry {i}");
        }
        let center = project(&m, [320.0, 240.0, 0.0]);
        assert!((center[0] - 320.0).abs() <= 1e-2);
        assert!((center[1] - 240.0).abs() <= 1e-2);
        let corner = project(&m, [640.0, 480.0, 0.0]);
        assert!((corner[0] - 640.0).abs() <= 1e-2);
        assert!((corner[1] - 480.0).abs() <= 1e-2);
    }
}
