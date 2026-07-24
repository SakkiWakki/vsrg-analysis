//! The scalar-timeline substrate: columnar (SoA) piecewise channels.
//!
//! This is the lowered form the Schedule IR's `lower()` emits (Ramp /
//! Hold breakpoints); the full schedule fold ports here later. A
//! channel is a slice into three parallel breakpoint arrays plus a
//! rest value; sampling is a binary search + linear ease. Contents are
//! hot-swappable (the lazy sweep replaces breakpoint data, never
//! channel identity), matching the Python SegmentTimeline contract.

/// One channel's slice bounds in the breakpoint arrays.
#[derive(Clone, Copy, Debug)]
struct Span {
    start: u32,
    len: u32,
    rest: f32,
}

#[derive(Default)]
pub struct ChannelTable {
    spans: Vec<Span>,
    /// Breakpoint start times, ascending within a span.
    ts: Vec<f32>,
    /// Value at the breakpoint's START.
    vals: Vec<f32>,
    /// Ramp duration toward the NEXT breakpoint's value (0 = hold).
    durs: Vec<f32>,
}

/// A channel reference: id + the rest served before the first
/// breakpoint (and for the sentinel NONE channel).
#[derive(Clone, Copy, Debug)]
pub struct ChannelRef {
    pub id: u32,
    pub rest: f32,
}

/// Sentinel for "no channel: always the rest value".
pub const NONE: u32 = u32::MAX;

impl ChannelRef {
    pub fn constant(rest: f32) -> Self {
        ChannelRef { id: NONE, rest }
    }
}

impl ChannelTable {
    /// Append a channel from parallel breakpoint arrays; returns its id.
    pub fn push(&mut self, ts: &[f32], vals: &[f32], durs: &[f32], rest: f32) -> u32 {
        debug_assert!(ts.len() == vals.len() && ts.len() == durs.len());
        let id = self.spans.len() as u32;
        self.spans.push(Span {
            start: self.ts.len() as u32,
            len: ts.len() as u32,
            rest,
        });
        self.ts.extend_from_slice(ts);
        self.vals.extend_from_slice(vals);
        self.durs.extend_from_slice(durs);
        id
    }

    pub fn sample(&self, r: ChannelRef, t: f32) -> f32 {
        if r.id == NONE {
            return r.rest;
        }
        let span = self.spans[r.id as usize];
        let (s, e) = (span.start as usize, (span.start + span.len) as usize);
        let (ts, vals, durs) = (&self.ts[s..e], &self.vals[s..e], &self.durs[s..e]);
        if ts.is_empty() || t < ts[0] {
            return if r.rest.is_nan() { span.rest } else { r.rest };
        }
        // Index of the last breakpoint with ts <= t.
        let i = ts.partition_point(|&bt| bt <= t) - 1;
        let (v0, dur) = (vals[i], durs[i]);
        if dur <= 0.0 || i + 1 >= vals.len() {
            return v0;
        }
        let frac = ((t - ts[i]) / dur).clamp(0.0, 1.0);
        v0 + (vals[i + 1] - v0) * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rest_before_first_breakpoint_then_holds_and_ramps() {
        let mut table = ChannelTable::default();
        let id = table.push(&[1.0, 2.0, 3.0], &[10.0, 20.0, 40.0], &[0.0, 1.0, 0.0], 5.0);
        let r = ChannelRef { id, rest: 5.0 };
        assert_eq!(table.sample(r, 0.5), 5.0); // rest
        assert_eq!(table.sample(r, 1.5), 10.0); // hold
        assert_eq!(table.sample(r, 2.5), 30.0); // mid-ramp 20 -> 40
        assert_eq!(table.sample(r, 9.0), 40.0); // held tail
    }

    #[test]
    fn none_channel_serves_rest() {
        let table = ChannelTable::default();
        assert_eq!(table.sample(ChannelRef::constant(7.0), 123.0), 7.0);
    }
}
