//! The Schedule IR: the SM actor tween-queue time algebra, ported from
//! `analysis/player/render/schedule.py`.
//!
//! The engine's tween queue is an exact composition of timed segments
//! (openitg `Actor::UpdateTweening`, Actor.cpp:469): entries drain with
//! exact arithmetic, a command fires when its entry first becomes head,
//! and Sleep/QueueCommand are themselves queue entries. `lower()` runs
//! that fold once at compile with time as a closed variable, producing
//! per-property breakpoint runs (the `ChannelTable::push` shape: parallel
//! `ts`/`vals`/`durs`, dur 0 = hold) plus opaque-effect fire times.
//!
//! Nodes: `Seg` is one queue entry (duration, ease, absolute prop
//! targets, optional effect fired at entry start); `Seq` is the queue;
//! `Hibernate` is the Update-level prefix sleep (leftover-carry);
//! `Loop` is the self-requeue fixpoint, unrolled to the horizon. An
//! effect may itself be a schedule, joining the fold at the queue tail
//! (`Actor::BeginTweening` appends; depth bound 50 at :617).

use std::collections::VecDeque;

const QUEUE_DEPTH_BOUND: u32 = 50;

/// A relative target: resolves to `start + delta` at fold time (the
/// engine's add-onto-dest verbs: addx, addrotationz, ...).
#[derive(Clone, Copy, Debug)]
pub struct Add {
    pub delta: f32,
}

/// One property target within a segment: an absolute value or a relative
/// `Add` onto the property's current state.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    Abs(f32),
    Add(f32),
}

/// A schedule node. `Seg` carries its property targets as `(prop, Target)`
/// pairs (insertion order preserved, matching the Python dict). An optional
/// `effect` fires at the segment's start: a nested `Node` joins the fold at
/// the tail, an opaque effect (identified by `fire_id >= 0`) emits a Fire.
#[derive(Clone, Debug)]
pub enum Node {
    Seg {
        dur: f32,
        ease: i32,
        targets: Vec<(u32, Target)>,
        effect: Option<Box<Node>>,
        fire_id: i64,
    },
    Seq(Vec<Node>),
    Hibernate(f32),
    Loop {
        period: f32,
        body: Box<Node>,
    },
}

impl Node {
    pub fn seg(dur: f32, ease: i32, targets: Vec<(u32, Target)>) -> Node {
        Node::Seg { dur, ease, targets, effect: None, fire_id: -1 }
    }
}

/// One lowered property: the `ChannelTable::push` breakpoint triple. `ts`
/// ascending, `vals` the value at each breakpoint start, `durs` the ramp
/// duration toward the next breakpoint (0 = hold).
#[derive(Clone, Debug, Default)]
pub struct LoweredProp {
    pub ts: Vec<f32>,
    pub vals: Vec<f32>,
    pub durs: Vec<f32>,
}

/// The result of lowering one schedule: per-property breakpoint runs (in
/// first-touched property order) plus effect fire times in time order.
#[derive(Clone, Debug, Default)]
pub struct LoweredOut {
    pub props: Vec<(u32, LoweredProp)>,
    pub fires: Vec<f32>,
}

/// One breakpoint emission mirroring the Python `Ramp`/`Hold` split.
enum Emission {
    Ramp { prop: u32, t0: f32, t1: f32, v0: f32, v1: f32, ease: i32 },
    Hold { prop: u32, t: f32, v: f32 },
}

enum Work {
    Seg { dur: f32, ease: i32, targets: Vec<(u32, Target)>, effect: Option<Box<Node>>, fire_id: i64 },
    Hibernate(f32),
    Loop { period: f32, body: Box<Node> },
}

/// Evaluate one actor chain to its per-property breakpoint runs. `state`
/// seeds property values (the ease-from side of the first segments);
/// `horizon` bounds evaluation and is required when the schedule contains
/// a Loop (a non-positive horizon means unbounded, matching Python's None).
pub fn lower(node: &Node, t0: f32, horizon: f32, state: &[(u32, f32)]) -> LoweredOut {
    let mut prop_state: Vec<(u32, f32)> = state.to_vec();
    let mut emissions: Vec<Emission> = Vec::new();
    let mut fires: Vec<f32> = Vec::new();
    let has_horizon = horizon.is_finite();

    let mut t = t0;
    let mut work: VecDeque<Work> = VecDeque::new();
    push_tail(&mut work, node, 0);
    while let Some(entry) = {
        if has_horizon && t >= horizon {
            None
        } else {
            work.pop_front()
        }
    } {
        match entry {
            Work::Hibernate(dur) => t += dur.max(0.0),
            Work::Loop { period, body } => {
                t = unroll_loop(period, &body, t, horizon, has_horizon, &mut work);
            }
            Work::Seg { dur, ease, targets, effect, fire_id } => {
                t = run_seg(
                    dur, ease, &targets, effect, fire_id, t,
                    &mut prop_state, &mut emissions, &mut fires, &mut work,
                );
            }
        }
    }

    LoweredOut { props: to_props(&emissions), fires }
}

fn state_get(state: &[(u32, f32)], prop: u32) -> Option<f32> {
    state.iter().find(|(k, _)| *k == prop).map(|(_, v)| *v)
}

fn state_set(state: &mut Vec<(u32, f32)>, prop: u32, v: f32) {
    match state.iter_mut().find(|(k, _)| *k == prop) {
        Some(slot) => slot.1 = v,
        None => state.push((prop, v)),
    }
}

fn node_to_work(node: &Node) -> Work {
    match node {
        Node::Seg { dur, ease, targets, effect, fire_id } => Work::Seg {
            dur: *dur,
            ease: *ease,
            targets: targets.clone(),
            effect: effect.clone(),
            fire_id: *fire_id,
        },
        Node::Hibernate(dur) => Work::Hibernate(*dur),
        Node::Loop { period, body } => Work::Loop { period: *period, body: body.clone() },
        Node::Seq(_) => unreachable!("Seq is flattened by push_tail"),
    }
}

fn push_tail(work: &mut VecDeque<Work>, node: &Node, depth: u32) {
    if depth > QUEUE_DEPTH_BOUND {
        panic!(
            "schedule depth exceeds the engine tween bound ({QUEUE_DEPTH_BOUND}); \
             infinitely recursing command?"
        );
    }
    match node {
        Node::Seq(parts) => {
            for part in parts {
                push_tail(work, part, depth + 1);
            }
        }
        Node::Seg { .. } | Node::Hibernate(_) | Node::Loop { .. } => work.push_back(node_to_work(node)),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_seg(
    dur: f32,
    ease: i32,
    targets: &[(u32, Target)],
    effect: Option<Box<Node>>,
    fire_id: i64,
    t: f32,
    state: &mut Vec<(u32, f32)>,
    emissions: &mut Vec<Emission>,
    fires: &mut Vec<f32>,
    work: &mut VecDeque<Work>,
) -> f32 {
    // Entry START: the command fires first (Actor.cpp:484-495), and any
    // entries it queues join at the TAIL, after everything pending.
    if let Some(nested) = effect {
        push_tail(work, &nested, 1);
    } else if fire_id >= 0 {
        fires.push(t);
    }

    let end = t + dur.max(0.0);
    for (prop, target) in targets {
        let v0 = state_get(state, *prop);
        let dest = match target {
            Target::Abs(v) => *v,
            Target::Add(delta) => v0.unwrap_or(0.0) + delta,
        };
        let changed = v0.map_or(true, |v| dest != v);
        if changed && dur > 0.0 {
            emissions.push(Emission::Ramp {
                prop: *prop,
                t0: t,
                t1: end,
                v0: v0.unwrap_or(0.0),
                v1: dest,
                ease,
            });
        } else if changed {
            emissions.push(Emission::Hold { prop: *prop, t, v: dest });
        }
        state_set(state, *prop, dest);
    }
    end
}

fn unroll_loop(
    period: f32,
    body: &Node,
    t: f32,
    horizon: f32,
    has_horizon: bool,
    work: &mut VecDeque<Work>,
) -> f32 {
    // The self-requeue rig re-arms itself each pass, so expansion is lazy
    // and naturally bounded by the horizon. Mirrors the Python
    // `work.extendleft(reversed(head))`: body, then the period-fill
    // hibernate, then the loop itself, all at the queue HEAD.
    if !has_horizon {
        panic!("lowering a Loop requires a horizon");
    }
    if period <= 0.0 {
        panic!("Loop period must be positive");
    }
    if t < horizon {
        let mut head: VecDeque<Work> = VecDeque::new();
        push_tail(&mut head, body, 1);
        head.push_back(Work::Hibernate((period - body_duration(body)).max(0.0)));
        head.push_back(Work::Loop { period, body: Box::new(body.clone()) });
        while let Some(w) = head.pop_back() {
            work.push_front(w);
        }
    }
    t
}

fn body_duration(node: &Node) -> f32 {
    match node {
        Node::Seg { dur, .. } => dur.max(0.0),
        Node::Hibernate(dur) => dur.max(0.0),
        Node::Seq(parts) => parts.iter().map(body_duration).sum(),
        Node::Loop { .. } => 0.0,
    }
}

/// Convert the interleaved Ramp/Hold emissions into per-property
/// `(ts, vals, durs)` breakpoint runs. Emissions for a property arrive in
/// time order; a Ramp becomes a start breakpoint (dur = t1 - t0) plus a
/// terminal hold at its end value, and consecutive emissions that share a
/// breakpoint time+value coalesce (the SegmentTimeline single-writer append
/// leaves no duplicate directory rows).
fn to_props(emissions: &[Emission]) -> Vec<(u32, LoweredProp)> {
    let mut order: Vec<u32> = Vec::new();
    let mut props: Vec<(u32, LoweredProp)> = Vec::new();

    for e in emissions {
        let prop = match e {
            Emission::Ramp { prop, .. } | Emission::Hold { prop, .. } => *prop,
        };
        if !order.contains(&prop) {
            order.push(prop);
            props.push((prop, LoweredProp::default()));
        }
        let lp = &mut props.iter_mut().find(|(k, _)| *k == prop).unwrap().1;
        match e {
            Emission::Ramp { t0, t1, v0, v1, .. } => {
                push_bp(lp, *t0, *v0, (*t1 - *t0).max(0.0));
                push_bp(lp, *t1, *v1, 0.0);
            }
            Emission::Hold { t, v, .. } => push_bp(lp, *t, *v, 0.0),
        }
    }
    props
}

/// Append a breakpoint, coalescing when it repeats the previous one exactly
/// (same start time and value) - the terminal hold of a ramp landing on the
/// start of the next ramp is a single directory row, not two.
fn push_bp(lp: &mut LoweredProp, t: f32, v: f32, dur: f32) {
    if let Some(&last_t) = lp.ts.last() {
        let last_v = *lp.vals.last().unwrap();
        let last_dur = *lp.durs.last().unwrap();
        if last_t == t && last_v == v && last_dur == 0.0 {
            // Overwrite the placeholder terminal hold with the real segment.
            *lp.durs.last_mut().unwrap() = dur;
            return;
        }
    }
    lp.ts.push(t);
    lp.vals.push(v);
    lp.durs.push(dur);
}

#[cfg(test)]
mod tests {
    //! Fixture parity. The generator writes `fixtures/schedule_cases.json`
    //! from the REAL Python `lower()`; we parse it with a tiny dependency-
    //! free JSON reader (the crate carries no serde, and Cargo.toml is not
    //! ours to touch) and assert breakpoint/fire parity to 1e-4.
    use super::*;

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
        fn u32(&self) -> u32 {
            self.f32() as u32
        }
        fn i64(&self) -> i64 {
            match self {
                Json::Num(n) => *n as i64,
                _ => panic!("expected number"),
            }
        }
        fn s(&self) -> &str {
            match self {
                Json::Str(s) => s,
                _ => panic!("expected string"),
            }
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

    fn build(n: &Json) -> Node {
        match n.get("kind").unwrap().s() {
            "Seg" => Node::Seg {
                dur: n.get("dur").unwrap().f32(),
                ease: n.get("ease").unwrap().i64() as i32,
                targets: n
                    .get("targets")
                    .unwrap()
                    .arr()
                    .iter()
                    .map(|t| {
                        let value = t.get("value").unwrap().f32();
                        let target = match t.get("mode").unwrap().s() {
                            "add" => Target::Add(value),
                            _ => Target::Abs(value),
                        };
                        (t.get("prop").unwrap().u32(), target)
                    })
                    .collect(),
                effect: match n.get("effect") {
                    Some(Json::Null) | None => None,
                    Some(e) => Some(Box::new(build(e))),
                },
                fire_id: n.get("fire_id").map_or(-1, |v| v.i64()),
            },
            "Seq" => Node::Seq(n.get("parts").unwrap().arr().iter().map(build).collect()),
            "Hibernate" => Node::Hibernate(n.get("dur").unwrap().f32()),
            "Loop" => Node::Loop {
                period: n.get("period").unwrap().f32(),
                body: Box::new(build(n.get("body").unwrap())),
            },
            other => panic!("unknown node kind {other}"),
        }
    }

    const FIXTURES: &str = include_str!("../fixtures/schedule_cases.json");

    #[test]
    fn parity_against_python_lower() {
        let root = parse(FIXTURES);
        let cases = root.arr();
        assert!(cases.len() >= 25, "expected >= 25 cases, got {}", cases.len());

        let mut max_err = 0.0f32;
        for case in cases {
            let name = case.get("name").unwrap().s();
            let node = build(case.get("node").unwrap());
            let t0 = case.get("t0").unwrap().f32();
            let horizon = match case.get("horizon") {
                Some(Json::Null) | None => f32::INFINITY,
                Some(v) => v.f32(),
            };
            let state: Vec<(u32, f32)> = case
                .get("state")
                .unwrap()
                .arr()
                .iter()
                .map(|p| {
                    let pair = p.arr();
                    (pair[0].u32(), pair[1].f32())
                })
                .collect();

            let out = lower(&node, t0, horizon, &state);
            let want_props = case.get("expected_props").unwrap().arr();
            assert_eq!(out.props.len(), want_props.len(), "case {name}: prop count");

            for (got, want) in out.props.iter().zip(want_props) {
                assert_eq!(got.0, want.get("prop").unwrap().u32(), "case {name}: prop key order");
                let (wts, wvals, wdurs) = (
                    want.get("ts").unwrap().arr(),
                    want.get("vals").unwrap().arr(),
                    want.get("durs").unwrap().arr(),
                );
                assert_eq!(got.1.ts.len(), wts.len(), "case {name}: prop {} len", got.0);
                for i in 0..wts.len() {
                    let de = [
                        (got.1.ts[i] - wts[i].f32()).abs(),
                        (got.1.vals[i] - wvals[i].f32()).abs(),
                        (got.1.durs[i] - wdurs[i].f32()).abs(),
                    ];
                    for d in de {
                        max_err = max_err.max(d);
                        assert!(d < 1e-4, "case {name}: breakpoint {i} diff {d}");
                    }
                }
            }

            let want_fires = case.get("expected_fires").unwrap().arr();
            assert_eq!(out.fires.len(), want_fires.len(), "case {name}: fire count");
            for (g, w) in out.fires.iter().zip(want_fires) {
                let d = (g - w.f32()).abs();
                max_err = max_err.max(d);
                assert!(d < 1e-4, "case {name}: fire time diff {d}");
            }
        }
        eprintln!("schedule parity: {} cases, max err {max_err:e}", cases.len());
    }
}
