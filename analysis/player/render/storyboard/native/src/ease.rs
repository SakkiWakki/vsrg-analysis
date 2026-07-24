//! Easing curves, ported verbatim from
//! `analysis/player/render/effects/easing.py`.
//!
//! Two id families share one integer space, matching the Python module so
//! Python-compiled keyframes cross Seam A losslessly:
//!   * 0..=23  osu.Framework `Easing` enum ordering (fluXis serializes
//!     these; 0 = linear/None). Only the polynomial / sine / expo / circ
//!     families are implemented; the exotic tail (elastic/back/bounce/pow10)
//!     falls back to OutQuint, as Python does.
//!   * negative ids  StepMania tween curves (openitg Actor.cpp:522-526),
//!     addressed negatively so osu enum growth can never collide.
//!
//! `ease(kind, u)` clamps `u` to [0, 1] then returns eased progress; a
//! channel ramp toward its next value eases by the START breakpoint's id.

use std::f32::consts::PI;

// osu.Framework Easing enum ids (index-aligned with `_CURVES` below).
pub const LINEAR: i32 = 0;

// StepMania tween curves, negative to avoid colliding with future osu ids.
pub const EASE_SM_BOUNCE_BEGIN: i32 = -1;
pub const EASE_SM_BOUNCE_END: i32 = -2;
pub const EASE_SM_SPRING: i32 = -3;

fn in_pow(u: f32, p: f32) -> f32 {
    u.powf(p)
}

fn out_pow(u: f32, p: f32) -> f32 {
    1.0 - (1.0 - u).powf(p)
}

fn in_out_pow(u: f32, p: f32) -> f32 {
    if u < 0.5 {
        (2.0 * u).powf(p) / 2.0
    } else {
        1.0 - (2.0 * (1.0 - u)).powf(p) / 2.0
    }
}

fn in_sine(u: f32) -> f32 {
    1.0 - (u * PI / 2.0).cos()
}

fn out_sine(u: f32) -> f32 {
    (u * PI / 2.0).sin()
}

fn in_out_sine(u: f32) -> f32 {
    (1.0 - (u * PI).cos()) / 2.0
}

fn in_expo(u: f32) -> f32 {
    if u <= 0.0 {
        0.0
    } else {
        2.0f32.powf(10.0 * (u - 1.0))
    }
}

fn out_expo(u: f32) -> f32 {
    if u >= 1.0 {
        1.0
    } else {
        1.0 - 2.0f32.powf(-10.0 * u)
    }
}

fn in_out_expo(u: f32) -> f32 {
    if u < 0.5 {
        in_expo(2.0 * u) / 2.0
    } else {
        0.5 + out_expo(2.0 * u - 1.0) / 2.0
    }
}

fn in_circ(u: f32) -> f32 {
    1.0 - (1.0 - u * u).sqrt()
}

fn out_circ(u: f32) -> f32 {
    (1.0 - (1.0 - u).powi(2)).sqrt()
}

fn in_out_circ(u: f32) -> f32 {
    if u < 0.5 {
        in_circ(2.0 * u) / 2.0
    } else {
        0.5 + out_circ(2.0 * u - 1.0) / 2.0
    }
}

// osu.Framework Easing enum, index-aligned with easing.py::_CURVES.
fn osu_curve(kind: i32, u: f32) -> f32 {
    match kind {
        0 => u,                       // None
        1 => out_pow(u, 2.0),         // Out (quad)
        2 => in_pow(u, 2.0),          // In (quad)
        3 => in_pow(u, 2.0),          // InQuad
        4 => out_pow(u, 2.0),         // OutQuad
        5 => in_out_pow(u, 2.0),      // InOutQuad
        6 => in_pow(u, 3.0),          // InCubic
        7 => out_pow(u, 3.0),         // OutCubic
        8 => in_out_pow(u, 3.0),      // InOutCubic
        9 => in_pow(u, 4.0),          // InQuart
        10 => out_pow(u, 4.0),        // OutQuart
        11 => in_out_pow(u, 4.0),     // InOutQuart
        12 => in_pow(u, 5.0),         // InQuint
        13 => out_pow(u, 5.0),        // OutQuint
        14 => in_out_pow(u, 5.0),     // InOutQuint
        15 => in_sine(u),             // InSine
        16 => out_sine(u),            // OutSine
        17 => in_out_sine(u),         // InOutSine
        18 => in_expo(u),             // InExpo
        19 => out_expo(u),            // OutExpo
        20 => in_out_expo(u),         // InOutExpo
        21 => in_circ(u),             // InCirc
        22 => out_circ(u),            // OutCirc
        23 => in_out_circ(u),         // InOutCirc
        _ => out_pow(u, 5.0),         // OutQuint fallback (the exotic tail)
    }
}

fn sm_bounce_begin(u: f32) -> f32 {
    1.0 - (1.1 + u * (PI - 1.1)).sin() / 0.89
}

fn sm_bounce_end(u: f32) -> f32 {
    (1.1 + (1.0 - u) * (PI - 1.1)).sin() / 0.89
}

fn sm_spring(u: f32) -> f32 {
    1.0 - (u * PI * 2.5).cos() / (1.0 + u * 3.0)
}

/// Eased progress for raw progress `u` in [0, 1]. `bounce`/`spring` may
/// leave the unit interval by design; only `u` is clamped, never output.
pub fn ease(kind: i32, u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    match kind {
        EASE_SM_BOUNCE_BEGIN => sm_bounce_begin(u),
        EASE_SM_BOUNCE_END => sm_bounce_end(u),
        EASE_SM_SPRING => sm_spring(u),
        k if k < 0 => out_pow(u, 5.0), // unknown negative id: OutQuint
        k => osu_curve(k, u),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_is_identity() {
        for &u in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(ease(LINEAR, u), u);
        }
    }

    #[test]
    fn endpoints_are_pinned() {
        for kind in 0..=23 {
            assert!((ease(kind, 0.0)).abs() < 1e-6, "kind {kind} at 0");
            assert!((ease(kind, 1.0) - 1.0).abs() < 1e-6, "kind {kind} at 1");
        }
    }

    #[test]
    fn u_is_clamped() {
        assert_eq!(ease(LINEAR, -1.0), 0.0);
        assert_eq!(ease(LINEAR, 2.0), 1.0);
    }

    #[test]
    fn unknown_ids_fall_back_to_out_quint() {
        assert_eq!(ease(999, 0.3), out_pow(0.3, 5.0));
        assert_eq!(ease(-99, 0.3), out_pow(0.3, 5.0));
    }

    // -- Fixture parity: build the equivalent one-ramp channel per case and
    //    assert eased sampling matches the Python EventTimeline grid. -------
    use crate::channels::{ChannelRef, ChannelTable};

    /// A pinhole number/array reader for our own generated ease fixtures
    /// (numbers, arrays, and the two string keys we look up). Enough for
    /// ease_cases.json; not a general JSON parser.
    struct Reader<'a> {
        b: &'a [u8],
        i: usize,
    }

    impl<'a> Reader<'a> {
        fn ws(&mut self) {
            while self.i < self.b.len() && (self.b[self.i] as char).is_whitespace() {
                self.i += 1;
            }
        }

        /// Advance to just past the next occurrence of `"key":` and return
        /// the following JSON value's byte position.
        fn seek(&mut self, key: &str) {
            let needle = format!("\"{key}\"");
            let from = self.i;
            let rel = std::str::from_utf8(&self.b[from..])
                .unwrap()
                .find(&needle)
                .expect("key present");
            self.i = from + rel + needle.len();
            self.ws();
            debug_assert_eq!(self.b[self.i], b':');
            self.i += 1;
            self.ws();
        }

        fn number(&mut self) -> f64 {
            self.ws();
            let start = self.i;
            while self.i < self.b.len() {
                let c = self.b[self.i];
                if c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' || c.is_ascii_digit() {
                    self.i += 1;
                } else {
                    break;
                }
            }
            std::str::from_utf8(&self.b[start..self.i]).unwrap().parse().unwrap()
        }

        fn num_array(&mut self) -> Vec<f64> {
            self.ws();
            debug_assert_eq!(self.b[self.i], b'[');
            self.i += 1;
            let mut out = Vec::new();
            loop {
                self.ws();
                if self.b[self.i] == b']' {
                    self.i += 1;
                    break;
                }
                out.push(self.number());
                self.ws();
                if self.b[self.i] == b',' {
                    self.i += 1;
                }
            }
            out
        }

        /// Count top-level objects inside the `"cases"` array and return the
        /// byte position just after `"cases": [`.
        fn enter_cases(&mut self) {
            self.seek("cases");
            debug_assert_eq!(self.b[self.i], b'[');
            self.i += 1;
        }

        fn at_array_end(&mut self) -> bool {
            self.ws();
            self.b[self.i] == b']'
        }
    }

    const EASE_FIXTURES: &str = include_str!("../fixtures/ease_cases.json");

    #[test]
    fn parity_against_python_timeline() {
        let mut r = Reader { b: EASE_FIXTURES.as_bytes(), i: 0 };
        r.enter_cases();

        let mut cases = 0usize;
        let mut max_err = 0.0f64;
        while !r.at_array_end() {
            let easing = r.number_after("easing");
            let t0 = r.number_after("t0") as f32;
            let dur = r.number_after("dur") as f32;
            let v0 = r.number_after("v0") as f32;
            let v1 = r.number_after("v1") as f32;
            r.seek("grid");
            let grid = r.num_array();
            r.seek("expected");
            let expected = r.num_array();
            assert_eq!(grid.len(), expected.len(), "grid/expected mismatch");

            // The channel EventTimeline plays for one eased keyframe: rest
            // v0 before t0, eased ramp v0->v1 over [t0, t0+dur], hold after.
            let mut table = ChannelTable::default();
            let id = table.push_eased(
                &[t0, t0 + dur],
                &[v0, v1],
                &[dur, 0.0],
                &[easing as i32, 0],
                v0,
            );
            let chan = ChannelRef { id, rest: v0 };
            for (t, want) in grid.iter().zip(&expected) {
                let got = table.sample(chan, *t as f32) as f64;
                let d = (got - want).abs();
                max_err = max_err.max(d);
                assert!(d < 1e-4, "ease {easing} at t={t}: got {got}, want {want}, d={d}");
            }

            cases += 1;
            // Advance past this object's closing brace and the optional
            // separator so the loop lands on the next `{` or the array's `]`.
            r.ws();
            if r.b[r.i] == b'}' {
                r.i += 1;
            }
            r.ws();
            if r.b[r.i] == b',' {
                r.i += 1;
            }
        }
        assert!(cases >= 24, "expected >= 24 ease cases, got {cases}");
        eprintln!("ease parity: {cases} cases, max err {max_err:e}");
    }

    impl<'a> Reader<'a> {
        /// Seek `key` within the current object then read one number.
        fn number_after(&mut self, key: &str) -> f64 {
            self.seek(key);
            self.number()
        }
    }
}
