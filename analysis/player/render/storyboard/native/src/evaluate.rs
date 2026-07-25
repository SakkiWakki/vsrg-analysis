//! evaluate(doc, t, feeds) -> DrawSchedule: the per-frame fold.
//!
//! Rules (drawable-ir.md): insertion order is truth; SortSpan is the
//! only local reordering (stable, over sampled z); Snapshot copies the
//! in-progress composite at its command index; an item sourcing a
//! drawable that has not yet drawn this frame (or itself) reads the
//! executor's retained content from last frame - feedback via
//! document order, no ping-pong. Non-persistent referenced drawables
//! evaluate before their consumers (dependency order, cycle edges =
//! retained reads); the screen evaluates last.
//!
//! SortSpan interactions: a Snapshot inside a SortSpan is ALLOWED and
//! participates in the sort with z 0 (Snapshot carries no z), landing
//! among z-0 items in document order (stable) - it copies the composite
//! as of its sorted position, not its authored index. Nested SortSpans
//! are FORBIDDEN and rejected at DocBuilder::finish (the engine's
//! SetDrawByZPosition has no such construct), so no inner span reaches
//! this fold.
//!
//! The output is flat fixed-width SoA records (no per-op objects) so
//! the schedule crosses Seam B as two buffers.

use std::collections::HashMap;

use crate::channels::{ChannelRef, ChannelTable};
use crate::doc::{
    Cmd, DrawableDoc, Item, Reaction, Source, PROP_FRAME, PROP_OPACITY, PROP_TINT_B, PROP_TINT_G,
    PROP_TINT_R, PROP_ZOOM, SCREEN,
};
use crate::schedule::{lower, LoweredProp};

/// Op kinds in the u32 record.
pub const OP_BEGIN: u32 = 0;
pub const OP_BLIT: u32 = 1;
pub const OP_COPY: u32 = 2;
pub const OP_END: u32 = 3;

/// Event kinds (the type sheet's `EventKind`): the already-decided input
/// classes a Reaction may trigger on. Values are stable wire constants
/// (Python passes them in the SoA `event_kinds` array; a reaction's
/// `trigger_kind` matches against them).
pub const EV_PRESS: u32 = 0;
pub const EV_RELEASE: u32 = 1;
pub const EV_HIT: u32 = 2;
pub const EV_MISS: u32 = 3;
pub const EV_CLICK: u32 = 4;
pub const EV_FRAMETICK: u32 = 5;

/// Source kinds in the u32 record.
pub const SRC_IMAGE: u32 = 0;
pub const SRC_DRAWABLE: u32 = 1;
pub const SRC_MESH: u32 = 2;
pub const SRC_FILL: u32 = 3;
pub const SRC_LINES: u32 = 4;

/// u32 lanes per op record. Lanes:
///   [0] op kind, [1] a, [2] b, [3] c (BLIT src_aux / frame),
///   [4] blend, [5] shader+1 (0 = none), [6] clip+1 (0 = none),
///   [7] screen_space, [8] uniform_offset, [9] uniform_count.
/// The uniform offset/count index the schedule's third flat buffer
/// (`uf`), holding this BLIT's sampled shader-uniform VALUES as f32.
/// count == 0 => the op binds no uniforms (offset is then unused).
pub const U_STRIDE: usize = 10;
/// f32 lanes per op record: mat3 (row-major) + opacity + tint rgb +
/// crop ltrb + 3 reserved.
pub const F_STRIDE: usize = 20;

#[derive(Default)]
pub struct DrawSchedule {
    pub u: Vec<u32>,
    pub f: Vec<f32>,
    /// Flat uniform-value buffer: each BLIT with bound shader uniforms
    /// appends its sampled f32 values contiguously; the op's lanes 8/9
    /// carry (offset, count) into this buffer. Fixed strides above stay
    /// untouched - variable-length uniform data lives only here.
    pub uf: Vec<f32>,
}

impl DrawSchedule {
    pub fn len(&self) -> usize {
        self.u.len() / U_STRIDE
    }

    fn push(&mut self, u: [u32; U_STRIDE], f: [f32; F_STRIDE]) {
        self.u.extend_from_slice(&u);
        self.f.extend_from_slice(&f);
    }

    fn op(&mut self, kind: u32, a: u32, b: u32) {
        let mut u = [0u32; U_STRIDE];
        (u[0], u[1], u[2]) = (kind, a, b);
        self.push(u, [0.0; F_STRIDE]);
    }
}

/// A dynamic drawable's per-frame command feed (flat SoA; see lib.rs
/// for the buffer layout Python fills).
pub struct Feed<'a> {
    pub drawable: u32,
    pub items: &'a [Item],
}

/// One already-decided input event this frame carries (Press/Hit/... -
/// the authoring layer's gameplay reaction, reduced to data at the
/// boundary). `kind` matches a reaction's trigger; `column` is filtered
/// against the reaction's column_filter (-1 = any); `time` is when it
/// arrived (the lowering `t0`); `strength` is reserved for reactions that
/// scale by hit strength (unused this wave, carried for the frozen SoA).
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub kind: u32,
    pub time: f32,
    pub column: i32,
    pub strength: f32,
}

/// The reaction lowering cache: a fragment lowered once per (reaction id,
/// event time) into its target prop's breakpoint run. `evaluate` shares
/// one across the frame so a re-spliced or repeated reaction pays the
/// fold at most once per distinct arrival time (the Evaluator keeps it
/// across frames - a Press at te=1.0 lowers once, ever). Keyed by
/// (reaction id, event-time bits) so equal float times collide exactly.
pub type ReactionCache = HashMap<(u32, u32), Option<LoweredProp>>;

/// Resolve the reactions active on `item` at time `t` into per-prop
/// value overrides. For each reaction, the latest matching event with
/// `time <= t` (engine queue-append: last one wins) selects an arrival
/// `te`; the fragment lowers from the property's base value at `te`,
/// seeding `state` so the ramp eases off the live base. The lowered run
/// is sampled at `t` and returned keyed by prop. When two reactions on
/// the same item target the same prop, the later-declared one wins
/// (document order, matching insertion-is-truth).
fn reaction_overrides(
    item: &Item,
    events: &[Event],
    cache: &mut ReactionCache,
    t: f32,
    base: &dyn Fn(u32, f32) -> f32,
) -> Vec<(u32, f32)> {
    let mut overrides: Vec<(u32, f32)> = Vec::new();
    for reaction in &item.reactions {
        let Some(te) = latest_matching_event(reaction, events, t) else {
            continue;
        };
        let key = (reaction.id, te.to_bits());
        let lowered = cache
            .entry(key)
            .or_insert_with(|| lower_reaction_prop(reaction, te, base(reaction.prop, te)));
        let Some(prop) = lowered else { continue };
        let value = sample_lowered(prop, t);
        set_override(&mut overrides, reaction.prop, value);
    }
    overrides
}

/// The item's base value for a spliced prop at the event time `te` - the
/// seed the reaction's ramp eases off (engine: the tween starts from the
/// property's live value). opacity folds in the link-chain alpha, tints
/// sample their channels, zoom's base is 1.0 (a multiplier, no scale
/// channel to read), frame reads the image's sheet-frame channel.
fn reaction_base_value(item: &Item, ch: &ChannelTable, prop: u32, te: f32, alpha_mul: f32) -> f32 {
    match prop {
        PROP_OPACITY => ch.sample(item.opacity, te) * alpha_mul,
        PROP_TINT_R => ch.sample(item.tint[0], te),
        PROP_TINT_G => ch.sample(item.tint[1], te),
        PROP_TINT_B => ch.sample(item.tint[2], te),
        PROP_ZOOM => 1.0,
        PROP_FRAME => match item.source {
            Source::Image { frame, .. } => ch.sample(frame, te),
            _ => 0.0,
        },
        _ => 0.0,
    }
}

/// The time of the latest event (largest `time <= t`) matching this
/// reaction's kind and column filter, or None if none has arrived yet.
fn latest_matching_event(reaction: &Reaction, events: &[Event], t: f32) -> Option<f32> {
    events
        .iter()
        .filter(|e| {
            e.kind == reaction.trigger_kind
                && e.time <= t
                && (reaction.column_filter < 0 || reaction.column_filter == e.column)
        })
        .map(|e| e.time)
        .fold(None, |acc, te| Some(acc.map_or(te, |a: f32| a.max(te))))
}

/// Lower a reaction's fragment at `t0 = te`, seeding the target prop's
/// base value, and keep only that prop's breakpoint run (the fragment may
/// touch several; a reaction splices exactly one). Unbounded horizon: a
/// reaction fragment is a finite command tween, never a Loop.
fn lower_reaction_prop(reaction: &Reaction, te: f32, base: f32) -> Option<LoweredProp> {
    let out = lower(&reaction.fragment, te, f32::INFINITY, &[(reaction.prop, base)]);
    out.props
        .into_iter()
        .find(|(k, _)| *k == reaction.prop)
        .map(|(_, prop)| prop)
}

/// Sample a lowered breakpoint run at `t` (the channel-table ease: rest
/// before the first breakpoint = the first value, hold/ramp within).
fn sample_lowered(prop: &LoweredProp, t: f32) -> f32 {
    let (ts, vals, durs) = (&prop.ts, &prop.vals, &prop.durs);
    if ts.is_empty() || t < ts[0] {
        return vals.first().copied().unwrap_or(0.0);
    }
    let i = ts.partition_point(|&bt| bt <= t) - 1;
    let (v0, dur) = (vals[i], durs[i]);
    if dur <= 0.0 || i + 1 >= vals.len() {
        return v0;
    }
    let frac = ((t - ts[i]) / dur).clamp(0.0, 1.0);
    v0 + (vals[i + 1] - v0) * frac
}

/// Insert or replace a prop override (later reaction on the same prop
/// wins - document order).
fn set_override(overrides: &mut Vec<(u32, f32)>, prop: u32, value: f32) {
    match overrides.iter_mut().find(|(k, _)| *k == prop) {
        Some(slot) => slot.1 = value,
        None => overrides.push((prop, value)),
    }
}

fn find_override(overrides: &[(u32, f32)], prop: u32) -> Option<f32> {
    overrides.iter().find(|(k, _)| *k == prop).map(|(_, v)| *v)
}

/// u32 lanes per fed item (FROZEN, drawable-port-wave1.md A5):
///   [source_kind, source_id, frame, flags]
/// flags bit0 = additive, bit1 = screen_space, bit2 = has_z.
pub const FEED_U_STRIDE: usize = 4;
/// f32 lanes per fed item (FEED V2, drawable-port-wave3.md C1): the
/// mat3 crosses NATIVE - column-vector RECORD convention (translation
/// in lanes 2/5, exactly the BLIT record's f-lanes 0..9), so a fed mat3
/// is written to the BLIT record verbatim (homographies included). No
/// affine decomposition, no projective skip.
///   [m00, m01, m02, m10, m11, m12, m20, m21, m22,
///    opacity, r, g, b, crop_l, crop_t, crop_r, crop_b, z]
pub const FEED_F_STRIDE: usize = 18;

const FEED_FLAG_ADDITIVE: u32 = 1;
const FEED_FLAG_SCREEN: u32 = 1 << 1;
const FEED_FLAG_HAS_Z: u32 = 1 << 2;

/// Decode one fed item's flat SoA lanes into an `Item`. Fed values are
/// already sampled scalars, so every channel is a constant (rest-only)
/// ChannelRef - no doc channel lookups happen for fed items. The fed
/// mat3 rides `fed_mat` (already in the record's column-vector layout);
/// emit_item writes it to the BLIT record verbatim.
fn feed_item(u: &[u32], f: &[f32]) -> Item {
    let (source_kind, source_id, frame, flags) = (u[0], u[1], u[2], u[3]);
    let source = match source_kind {
        SRC_IMAGE => Source::Image {
            image: source_id,
            frame: ChannelRef::constant(frame as f32),
        },
        SRC_DRAWABLE => Source::Drawable(source_id),
        SRC_MESH => Source::Mesh(source_id),
        SRC_LINES => Source::Lines(source_id),
        _ => Source::Fill,
    };
    let mut item = Item::of(source);
    let mut mat = [0.0f32; 9];
    mat.copy_from_slice(&f[0..9]);
    item.fed_mat = Some(mat);
    item.opacity = ChannelRef::constant(f[9]);
    item.tint = [
        ChannelRef::constant(f[10]),
        ChannelRef::constant(f[11]),
        ChannelRef::constant(f[12]),
    ];
    item.crop = [
        ChannelRef::constant(f[13]),
        ChannelRef::constant(f[14]),
        ChannelRef::constant(f[15]),
        ChannelRef::constant(f[16]),
    ];
    item.blend = if flags & FEED_FLAG_ADDITIVE != 0 {
        crate::doc::Blend::Additive
    } else {
        crate::doc::Blend::SourceOver
    };
    item.space = if flags & FEED_FLAG_SCREEN != 0 {
        crate::doc::Space::Screen
    } else {
        crate::doc::Space::Scene
    };
    item.z = (flags & FEED_FLAG_HAS_Z != 0).then(|| ChannelRef::constant(f[17]));
    item
}

/// Owned per-feed item storage decoded from the flat SoA byte buffers.
/// `Feed`s borrow from this - keep it alive for the evaluate() call.
pub struct FeedItems {
    pub drawable: u32,
    pub items: Vec<Item>,
}

/// Parse the frozen SoA feed buffers into owned items. `ids` and
/// `item_counts` are parallel (one entry per fed drawable); `u` / `f`
/// are the concatenated fixed-stride lane buffers across all feeds in
/// the same order.
pub fn parse_feeds(
    ids: &[u32],
    item_counts: &[u32],
    u: &[u32],
    f: &[f32],
) -> Vec<FeedItems> {
    let mut feeds = Vec::with_capacity(ids.len());
    let mut ui = 0usize;
    let mut fi = 0usize;
    for (feed, &count) in ids.iter().zip(item_counts) {
        let mut items = Vec::with_capacity(count as usize);
        for _ in 0..count {
            items.push(feed_item(&u[ui..ui + FEED_U_STRIDE], &f[fi..fi + FEED_F_STRIDE]));
            ui += FEED_U_STRIDE;
            fi += FEED_F_STRIDE;
        }
        feeds.push(FeedItems { drawable: *feed, items });
    }
    feeds
}

pub fn evaluate(doc: &DrawableDoc, t: f32, feeds: &[Feed]) -> DrawSchedule {
    let mut cache = ReactionCache::new();
    evaluate_with_events(doc, t, feeds, &[], &mut cache)
}

/// The full per-frame fold with input events. Reactions on any item
/// splice their lowered fragment over the named prop for events that have
/// arrived by `t` (see reaction_overrides); `cache` is the Evaluator's
/// persistent lowering cache so each (reaction, arrival time) folds once.
/// The event-free `evaluate` is the same fold with an empty event slice.
pub fn evaluate_with_events(
    doc: &DrawableDoc,
    t: f32,
    feeds: &[Feed],
    events: &[Event],
    cache: &mut ReactionCache,
) -> DrawSchedule {
    let mut schedule = DrawSchedule::default();
    for id in eval_order(doc, feeds) {
        emit_drawable(doc, id, t, feeds, events, cache, &mut schedule);
    }
    schedule
}

/// Dependency order over Source::Drawable edges: consumers after their
/// (non-persistent, acyclic) sources; the screen last; unreferenced
/// drawables with no commands skipped. Cycle/self edges are simply not
/// followed - the executor's retained texture serves them.
fn eval_order(doc: &DrawableDoc, feeds: &[Feed]) -> Vec<u32> {
    let n = doc.drawables.len();
    let mut order = Vec::with_capacity(n);
    let mut state = vec![0u8; n]; // 0 unvisited / 1 in-stack / 2 done

    fn visit(doc: &DrawableDoc, feeds: &[Feed], id: u32, state: &mut [u8], order: &mut Vec<u32>) {
        match state[id as usize] {
            1 | 2 => return, // cycle edge or already emitted
            _ => {}
        }
        state[id as usize] = 1;
        for cmd in commands_of(doc, feeds, id) {
            if let Cmd::Item(item) = cmd {
                if let Source::Drawable(src) = item.source {
                    // An edge into the screen is a retained read (the
                    // root always evaluates last); snapshots WRITE, so
                    // they carry no dependency edge either.
                    if src != SCREEN {
                        visit(doc, feeds, src, state, order);
                    }
                }
            }
        }
        state[id as usize] = 2;
        order.push(id);
    }

    for id in 1..n as u32 {
        visit(doc, feeds, id, &mut state, &mut order);
    }
    order.push(SCREEN);
    order
}

fn commands_of<'a>(doc: &'a DrawableDoc, feeds: &'a [Feed], id: u32) -> &'a [Cmd] {
    let drawable = &doc.drawables[id as usize];
    if drawable.dynamic {
        // Feeds carry items only; wrap borrowed as a slice of Cmd is
        // not possible without allocation, so dynamic drawables store
        // their fed commands in a scratch built by `emit_drawable`.
        // Here we only need dependency edges; feeds reference other
        // drawables rarely, handled below.
        return &[];
    }
    &drawable.commands
}

#[allow(clippy::too_many_arguments)]
fn emit_drawable(
    doc: &DrawableDoc,
    id: u32,
    t: f32,
    feeds: &[Feed],
    events: &[Event],
    cache: &mut ReactionCache,
    schedule: &mut DrawSchedule,
) {
    let drawable = &doc.drawables[id as usize];
    if drawable.inline {
        // A feed SLOT is never a target of its own; `Cmd::Feed` draws its
        // items into the enclosing drawable at their tree position.
        return;
    }
    let feed_items = feeds
        .iter()
        .find(|f| f.drawable == id)
        .map(|f| f.items)
        .unwrap_or(&[]);
    if drawable.commands.is_empty() && feed_items.is_empty() && id != SCREEN {
        return;
    }

    schedule.op(OP_BEGIN, id, drawable.clear as u32);
    if drawable.dynamic {
        for item in feed_items {
            emit_item(doc, item, t, events, cache, schedule);
        }
    } else {
        emit_commands(doc, &drawable.commands, t, events, cache, schedule, feeds);
    }
    schedule.op(OP_END, id, 0);
}

fn emit_commands(
    doc: &DrawableDoc,
    commands: &[Cmd],
    t: f32,
    events: &[Event],
    cache: &mut ReactionCache,
    schedule: &mut DrawSchedule,
    feeds: &[Feed],
) {
    let mut i = 0usize;
    while i < commands.len() {
        match &commands[i] {
            Cmd::Item(item) => {
                emit_item(doc, item, t, events, cache, schedule);
                i += 1;
            }
            Cmd::Feed {
                slot,
                links,
                flip_base_y,
                visible,
                projection,
            } => {
                // Inline: the slot's fed items draw HERE, as ordinary items of
                // the enclosing drawable, so nothing bounds them but the real
                // render target.
                emit_feed(
                    doc,
                    *slot,
                    links,
                    *flip_base_y,
                    *visible,
                    *projection,
                    t,
                    feeds,
                    events,
                    cache,
                    schedule,
                );
                i += 1;
            }
            Cmd::Snapshot { into } => {
                schedule.op(OP_COPY, *into, 0);
                i += 1;
            }
            Cmd::SortSpan { len } => {
                let span_end = (i + 1 + *len as usize).min(commands.len());
                let mut keyed: Vec<(f32, usize)> = (i + 1..span_end)
                    .map(|j| {
                        let z = match &commands[j] {
                            Cmd::Item(item) => item
                                .z
                                .map(|r| doc.channels.sample(r, t))
                                .unwrap_or(0.0),
                            _ => 0.0,
                        };
                        (z, j)
                    })
                    .collect();
                // Stable: ties keep document order (sort_by is stable).
                keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for (_, j) in keyed {
                    match &commands[j] {
                        Cmd::Item(item) => emit_item(doc, item, t, events, cache, schedule),
                        Cmd::Snapshot { into } => schedule.op(OP_COPY, *into, 0),
                        // A z-sorted span holds only drawing items; a feed
                        // slot inside one has no single z to sort by, so it
                        // draws in place (its own items keep their order).
                        Cmd::Feed {
                            slot,
                            links,
                            flip_base_y,
                            visible,
                            projection,
                        } => emit_feed(
                            doc,
                            *slot,
                            links,
                            *flip_base_y,
                            *visible,
                            *projection,
                            t,
                            feeds,
                            events,
                            cache,
                            schedule,
                        ),
                        // Nested spans are rejected at DocBuilder::finish;
                        // an inner span never reaches this fold. A doc built
                        // outside the validated builder degrades to a no-op.
                        Cmd::SortSpan { .. } => {}
                    }
                }
                i = span_end;
            }
        }
    }
}

/// The item's placement in the BLIT record's column-vector mat3
/// convention, plus an alpha multiplier and an optional overriding crop.
///
/// Linkless items keep the first-cut TRS bit-identically (alpha 1, no crop
/// override). A linked item samples each link's ChannelRefs into a
/// `TransformState`, folds the chain through `transform::compose_links`
/// (returning None when hidden/degenerate/faint, which drops the item),
/// TRANSPOSES the resulting row-vector H into the record's column-vector
/// layout, and carries the chain's alpha + (swapped) leaf crop.
fn item_transform(
    doc: &DrawableDoc,
    item: &Item,
    t: f32,
) -> Option<([f32; 9], f32, Option<[f32; 4]>)> {
    let ch = &doc.channels;
    // A dynamic-feed item carries its mat3 pre-resolved (feed v2): write
    // it verbatim, alpha 1 (the feed's opacity lane already holds the
    // chain alpha), crop from the item's own crop lanes.
    if let Some(mat) = item.fed_mat {
        return Some((mat, 1.0, None));
    }
    if item.links.is_empty() {
        let (x, y) = (ch.sample(item.transform.x, t), ch.sample(item.transform.y, t));
        let (sx, sy) = (
            ch.sample(item.transform.scale_x, t),
            ch.sample(item.transform.scale_y, t),
        );
        let rot = ch.sample(item.transform.rot, t).to_radians();
        let (c, s) = (rot.cos(), rot.sin());
        // mat3, column-vector record convention: translate(x, y) * rotate * scale.
        let mat = [
            c * sx, -s * sy, x, //
            s * sx, c * sy, y, //
            0.0, 0.0, 1.0,
        ];
        return Some((mat, 1.0, None));
    }

    let links: Vec<crate::transform::TransformState> =
        item.links.iter().map(|l| sample_link(ch, l, t)).collect();
    // The camera goes INTO the chain fold, not onto its result: the 4x4s must
    // multiply before the z row/column is dropped (see compose_links).
    let cam = item.projection.map(|c| c.matrix(ch, t));
    let (h, alpha, crop) =
        crate::transform::compose_links(&links, item.flip_base_y, cam.as_ref())?;
    // compose_links returns row-vector H (translation in the bottom row,
    // indices 6/7); the BLIT record is column-vector (M @ p), so transpose.
    let mat = [
        h[0], h[3], h[6], //
        h[1], h[4], h[7], //
        h[2], h[5], h[8],
    ];
    Some((mat, alpha, Some(crop)))
}

/// Sample one link's channel refs into a `transform::TransformState`.
fn sample_link(
    ch: &ChannelTable,
    link: &crate::doc::LinkRef,
    t: f32,
) -> crate::transform::TransformState {
    crate::transform::TransformState {
        x: ch.sample(link.x, t),
        y: ch.sample(link.y, t),
        zoom_x: ch.sample(link.zoom_x, t),
        zoom_y: ch.sample(link.zoom_y, t),
        rot: ch.sample(link.rot, t),
        skew_x: ch.sample(link.skew_x, t),
        skew_y: ch.sample(link.skew_y, t),
        base_scale_x: ch.sample(link.base_scale_x, t),
        base_scale_y: ch.sample(link.base_scale_y, t),
        halign: ch.sample(link.halign, t),
        valign: ch.sample(link.valign, t),
        hidden: ch.sample(link.hidden, t),
        alpha: ch.sample(link.alpha, t),
        crop: [
            ch.sample(link.crop[0], t),
            ch.sample(link.crop[1], t),
            ch.sample(link.crop[2], t),
            ch.sample(link.crop[3], t),
        ],
        natural_w: ch.sample(link.natural_w, t),
        natural_h: ch.sample(link.natural_h, t),
        rotation_x: ch.sample(link.rotation_x, t),
        rotation_y: ch.sample(link.rotation_y, t),
        z: ch.sample(link.z, t),
        scale_z: ch.sample(link.scale_z, t),
        base_scale_z: ch.sample(link.base_scale_z, t),
        rotation_order: link.rotation_order,
    }
}

/// Emit one Feed command: the slot's fed items, gated by the feed's
/// `visible` channel. A non-empty consumer link chain (a proxy/player
/// field) composes over every item - its H over each mat3, its alpha over
/// each opacity - so the consumer RE-RENDERS the same unclipped items
/// instead of blitting a capture-boxed texture (the copy-render rule).
#[allow(clippy::too_many_arguments)]
fn emit_feed(
    doc: &DrawableDoc,
    slot: u32,
    links: &[crate::doc::LinkRef],
    flip_base_y: bool,
    visible: ChannelRef,
    projection: Option<crate::doc::CameraRef>,
    t: f32,
    feeds: &[Feed],
    events: &[Event],
    cache: &mut ReactionCache,
    schedule: &mut DrawSchedule,
) {
    let ch = &doc.channels;
    if ch.sample(visible, t) < 0.5 {
        return;
    }
    let Some(feed) = feeds.iter().find(|f| f.drawable == slot) else {
        return;
    };
    if links.is_empty() {
        for item in feed.items {
            emit_item(doc, item, t, events, cache, schedule);
        }
        return;
    }

    let states: Vec<crate::transform::TransformState> =
        links.iter().map(|l| sample_link(ch, l, t)).collect();
    let cam = projection.map(|c| c.matrix(ch, t));
    let Some((h, alpha, _crop)) =
        crate::transform::compose_links(&states, flip_base_y, cam.as_ref())
    else {
        return; // hidden / degenerate / faint chain drops the whole feed
    };
    // compose_links returns row-vector H; fed mats and the record are
    // column-vector, so transpose once before composing over each item.
    // The chain's crop is not applied: per-item crop fractions are of the
    // item's own box, not the chain's content box.
    let chain = [
        h[0], h[3], h[6], //
        h[1], h[4], h[7], //
        h[2], h[5], h[8],
    ];
    for item in feed.items {
        emit_item_folded(doc, item, t, events, cache, schedule, Some((&chain, alpha)));
    }
}

fn emit_item(
    doc: &DrawableDoc,
    item: &Item,
    t: f32,
    events: &[Event],
    cache: &mut ReactionCache,
    schedule: &mut DrawSchedule,
) {
    emit_item_folded(doc, item, t, events, cache, schedule, None)
}

/// `emit_item` with an optional pre-composed consumer transform: `pre` is
/// a (column-vector chain H, chain alpha) applied over the item's own
/// placement - the linked-Feed path (see `emit_feed`).
#[allow(clippy::too_many_arguments)]
fn emit_item_folded(
    doc: &DrawableDoc,
    item: &Item,
    t: f32,
    events: &[Event],
    cache: &mut ReactionCache,
    schedule: &mut DrawSchedule,
    pre: Option<(&[f32; 9], f32)>,
) {
    let ch: &ChannelTable = &doc.channels;
    if ch.sample(item.visible, t) < 0.5 {
        return;
    }

    // The item's placement: the full leaf-link chain when present (its H,
    // alpha, and crop win), else the first-cut TRS. Both land in the BLIT
    // record's column-vector mat3 convention.
    let (mut mat, mut alpha_mul, link_crop) = match item_transform(doc, item, t) {
        Some(triple) => triple,
        None => return, // hidden / degenerate / faint link chain
    };
    if let Some((chain, chain_alpha)) = pre {
        mat = crate::transform::mul3(chain, &mat);
        alpha_mul *= chain_alpha;
    }

    // Event-driven property splices: a reaction whose event has arrived
    // by `t` overrides its base draw property with its lowered fragment.
    // Base values are the same scalars this emit samples below, evaluated
    // lazily so the reaction's ramp eases off the live base at the event.
    let overrides = if item.reactions.is_empty() {
        Vec::new()
    } else {
        let base = |prop: u32, te: f32| reaction_base_value(item, ch, prop, te, alpha_mul);
        reaction_overrides(item, events, cache, t, &base)
    };
    let over = |prop: u32, base: f32| find_override(&overrides, prop).unwrap_or(base);

    let opacity = over(PROP_OPACITY, ch.sample(item.opacity, t) * alpha_mul);
    if opacity < 1.0 / 255.0 {
        return;
    }

    // A per-item camera projection folds the 2D mat3 onto the z=0 design
    // plane and back to a projective mat3 (row/col collapse; see camera.rs).
    // A spliced zoom is a uniform scale multiplier folded first (pre-
    // projection), so perspective sees the scaled quad.
    if let Some(zoom) = find_override(&overrides, PROP_ZOOM) {
        for m in mat.iter_mut().take(6) {
            *m *= zoom;
        }
    }
    // A LINKED item already projected inside compose_links, where the 4x4
    // still had its Z. Only the linkless TRS path folds here, and there the
    // content is flat by construction so the collapse costs nothing.
    if item.links.is_empty() {
        if let Some(cam) = item.projection {
            let p = cam.matrix(ch, t);
            mat = crate::camera::fold_projection(&mat, &p);
        }
    }

    let (src_kind, src_id, src_aux) = match item.source {
        Source::Image { image, frame } => (
            SRC_IMAGE,
            image,
            over(PROP_FRAME, ch.sample(frame, t)) as u32,
        ),
        Source::Drawable(d) => (SRC_DRAWABLE, d, 0),
        Source::Mesh(m) => (SRC_MESH, m, 0),
        Source::Fill => (SRC_FILL, 0, 0),
        Source::Lines(l) => (SRC_LINES, l, 0),
    };

    let mut f = [0.0f32; F_STRIDE];
    f[..9].copy_from_slice(&mat);
    f[9] = opacity;
    let tint_props = [PROP_TINT_R, PROP_TINT_G, PROP_TINT_B];
    for (lane, tint) in item.tint.iter().enumerate() {
        f[10 + lane] = over(tint_props[lane], ch.sample(*tint, t));
    }
    match link_crop {
        // The link chain's crop (already top/bottom-swapped under flip)
        // overrides the item's own crop lanes.
        Some(crop) => f[13..17].copy_from_slice(&crop),
        None => {
            for (lane, crop) in item.crop.iter().enumerate() {
                f[13 + lane] = ch.sample(*crop, t);
            }
        }
    }

    let (uniform_offset, uniform_count) = if item.shader.is_some() && !item.uniforms.is_empty() {
        let offset = schedule.uf.len() as u32;
        for (_name_idx, r) in &item.uniforms {
            schedule.uf.push(ch.sample(*r, t));
        }
        (offset, item.uniforms.len() as u32)
    } else {
        (0, 0)
    };

    let mut u = [0u32; U_STRIDE];
    u[0] = OP_BLIT;
    u[1] = src_kind;
    u[2] = src_id;
    u[3] = src_aux;
    u[4] = item.blend as u32;
    u[5] = item.shader.map(|s| s + 1).unwrap_or(0);
    u[6] = item.clip.map(|c| c + 1).unwrap_or(0);
    u[7] = matches!(item.space, crate::doc::Space::Screen) as u32;
    u[8] = uniform_offset;
    u[9] = uniform_count;
    schedule.push(u, f);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::ChannelRef;
    use crate::doc::{Cmd, DrawableDoc, Item, Source};

    fn blit_sources(schedule: &DrawSchedule) -> Vec<(u32, u32)> {
        (0..schedule.len())
            .filter(|i| schedule.u[i * U_STRIDE] == OP_BLIT)
            .map(|i| (schedule.u[i * U_STRIDE + 1], schedule.u[i * U_STRIDE + 2]))
            .collect()
    }

    fn op_kinds(schedule: &DrawSchedule) -> Vec<u32> {
        (0..schedule.len()).map(|i| schedule.u[i * U_STRIDE]).collect()
    }

    #[test]
    fn insertion_order_is_truth() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        for image in 0..3 {
            doc.drawables[0]
                .commands
                .push(Cmd::Item(Item::of(Source::Image {
                    image,
                    frame: ChannelRef::constant(0.0),
                })));
        }
        let schedule = evaluate(&doc, 0.0, &[]);
        assert_eq!(
            blit_sources(&schedule),
            vec![(SRC_IMAGE, 0), (SRC_IMAGE, 1), (SRC_IMAGE, 2)]
        );
    }

    #[test]
    fn mesh_source_carries_its_id_into_the_record() {
        use crate::doc::MeshDesc;
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mesh = doc.add_mesh(MeshDesc {
            vertices: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            mode: 0,
            vert_shader: None,
            vert_source: None,
        });
        doc.drawables[0]
            .commands
            .push(Cmd::Item(Item::of(Source::Mesh(mesh))));
        let schedule = evaluate(&doc, 0.0, &[]);
        // The BLIT record carries (SRC_MESH, mesh_id) in lanes 1/2.
        assert_eq!(blit_sources(&schedule), vec![(SRC_MESH, mesh)]);
    }

    #[test]
    fn sort_span_orders_by_sampled_z_stably() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mk = |image, z: Option<f32>| {
            let mut item = Item::of(Source::Image {
                image,
                frame: ChannelRef::constant(0.0),
            });
            item.z = z.map(ChannelRef::constant);
            Cmd::Item(item)
        };
        doc.drawables[0].commands.push(Cmd::SortSpan { len: 3 });
        doc.drawables[0].commands.push(mk(0, Some(5.0)));
        doc.drawables[0].commands.push(mk(1, Some(-5.0)));
        doc.drawables[0].commands.push(mk(2, Some(5.0))); // tie with 0: doc order
        doc.drawables[0].commands.push(mk(3, None)); // outside the span
        let schedule = evaluate(&doc, 0.0, &[]);
        let sources: Vec<u32> = blit_sources(&schedule).iter().map(|s| s.1).collect();
        assert_eq!(sources, vec![1, 0, 2, 3]);
    }

    #[test]
    fn snapshot_emits_copy_at_position() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let slot = doc.add_drawable([640.0, 480.0], true, false);
        doc.drawables[0]
            .commands
            .push(Cmd::Item(Item::of(Source::Image {
                image: 0,
                frame: ChannelRef::constant(0.0),
            })));
        doc.drawables[0].commands.push(Cmd::Snapshot { into: slot });
        doc.drawables[0]
            .commands
            .push(Cmd::Item(Item::of(Source::Drawable(slot))));
        let schedule = evaluate(&doc, 0.0, &[]);
        assert_eq!(op_kinds(&schedule), vec![OP_BEGIN, OP_BLIT, OP_COPY, OP_BLIT, OP_END]);
    }

    #[test]
    fn referenced_drawable_evaluates_before_consumer_screen_last() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let sub = doc.add_drawable([64.0, 64.0], false, false);
        doc.drawables[sub as usize]
            .commands
            .push(Cmd::Item(Item::of(Source::Image {
                image: 7,
                frame: ChannelRef::constant(0.0),
            })));
        doc.drawables[0]
            .commands
            .push(Cmd::Item(Item::of(Source::Drawable(sub))));
        let schedule = evaluate(&doc, 0.0, &[]);
        let begins: Vec<u32> = (0..schedule.len())
            .filter(|i| schedule.u[i * U_STRIDE] == OP_BEGIN)
            .map(|i| schedule.u[i * U_STRIDE + 1])
            .collect();
        assert_eq!(begins, vec![sub, SCREEN]);
    }

    #[test]
    fn self_feedback_does_not_recurse() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let fb = doc.add_drawable([64.0, 64.0], true, false);
        doc.drawables[fb as usize]
            .commands
            .push(Cmd::Item(Item::of(Source::Drawable(fb))));
        doc.drawables[0]
            .commands
            .push(Cmd::Item(Item::of(Source::Drawable(fb))));
        let schedule = evaluate(&doc, 0.0, &[]); // must terminate
        assert!(schedule.len() > 0);
    }

    #[test]
    fn hidden_and_transparent_items_are_dropped() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mut hidden = Item::of(Source::Fill);
        hidden.visible = ChannelRef::constant(0.0);
        let mut clear = Item::of(Source::Fill);
        clear.opacity = ChannelRef::constant(0.0);
        doc.drawables[0].commands.push(Cmd::Item(hidden));
        doc.drawables[0].commands.push(Cmd::Item(clear));
        let schedule = evaluate(&doc, 0.0, &[]);
        assert_eq!(op_kinds(&schedule), vec![OP_BEGIN, OP_END]);
    }

    #[test]
    fn shader_uniform_values_ride_the_side_buffer() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let sh = doc.add_shader(crate::doc::ShaderDesc {
            frag: "void main(){}".into(),
            vert: None,
            uniform_names: vec!["a".into(), "b".into()],
        });
        let mut plain = Item::of(Source::Fill);
        plain.tint = [ChannelRef::constant(1.0); 3];
        let mut shaded = Item::of(Source::Fill);
        shaded.shader = Some(sh);
        shaded.uniforms = vec![
            (0, ChannelRef::constant(2.5)),
            (1, ChannelRef::constant(-7.0)),
        ];
        doc.drawables[0].commands.push(Cmd::Item(plain));
        doc.drawables[0].commands.push(Cmd::Item(shaded));
        let schedule = evaluate(&doc, 0.0, &[]);

        // Two BLITs; the first binds nothing, the second two uniforms.
        let blits: Vec<usize> = (0..schedule.len())
            .filter(|i| schedule.u[i * U_STRIDE] == OP_BLIT)
            .collect();
        assert_eq!(blits.len(), 2);
        let plain_op = blits[0];
        assert_eq!(schedule.u[plain_op * U_STRIDE + 9], 0); // count
        let shaded_op = blits[1];
        assert_eq!(schedule.u[shaded_op * U_STRIDE + 5], sh + 1); // shader+1
        let (off, cnt) = (
            schedule.u[shaded_op * U_STRIDE + 8] as usize,
            schedule.u[shaded_op * U_STRIDE + 9] as usize,
        );
        assert_eq!(cnt, 2);
        assert_eq!(&schedule.uf[off..off + cnt], &[2.5, -7.0]);
    }

    #[test]
    fn clip_id_rides_lane_six() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let clip = doc.add_clip(crate::doc::ClipDesc::Rect([0.0, 0.0, 10.0, 10.0]));
        let mut item = Item::of(Source::Fill);
        item.clip = Some(clip);
        doc.drawables[0].commands.push(Cmd::Item(item));
        let schedule = evaluate(&doc, 0.0, &[]);
        let op = (0..schedule.len())
            .find(|i| schedule.u[i * U_STRIDE] == OP_BLIT)
            .unwrap();
        assert_eq!(schedule.u[op * U_STRIDE + 6], clip + 1);
    }

    #[test]
    fn parse_feeds_decodes_frozen_soa_layout() {
        // Two fed items: an additive image and a screen-space fill w/ z.
        // Feed v2: f32 lanes 0..9 are the mat3 in the record's
        // column-vector layout, written to the BLIT record verbatim.
        let u: Vec<u32> = vec![
            SRC_IMAGE, 5, 3, FEED_FLAG_ADDITIVE, // item 0
            SRC_FILL, 0, 0, FEED_FLAG_SCREEN | FEED_FLAG_HAS_Z, // item 1
        ];
        // item 0 mat3: scale(2, 0.5) translate(10, 20) column-vector; item 1
        // mat3: a projective row (m20/m21 non-zero) proving homographies
        // cross verbatim, not just affine.
        let f: Vec<f32> = vec![
            // m00..m22, opacity, r, g, b, crop l/t/r/b, z
            2.0, 0.0, 10.0, 0.0, 0.5, 20.0, 0.0, 0.0, 1.0, //
            0.75, 1.0, 0.5, 0.25, 0.1, 0.2, 0.3, 0.4, 0.0, // 0
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.001, 0.002, 1.0, //
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 9.0, // 1
        ];
        let feeds = parse_feeds(&[7], &[2], &u, &f);
        assert_eq!(feeds.len(), 1);
        assert_eq!(feeds[0].drawable, 7);
        let items = &feeds[0].items;
        assert_eq!(items.len(), 2);

        match items[0].source {
            Source::Image { image, .. } => assert_eq!(image, 5),
            _ => panic!("item 0 should be an image"),
        }
        assert!(matches!(items[0].blend, crate::doc::Blend::Additive));
        assert!(matches!(items[0].space, crate::doc::Space::Scene));
        assert!(items[0].z.is_none());
        assert_eq!(
            items[0].fed_mat,
            Some([2.0, 0.0, 10.0, 0.0, 0.5, 20.0, 0.0, 0.0, 1.0])
        );

        assert!(matches!(items[1].source, Source::Fill));
        assert!(matches!(items[1].space, crate::doc::Space::Screen));
        assert!(items[1].z.is_some());
        // The projective row rides verbatim through parse into the item.
        assert_eq!(items[1].fed_mat.unwrap()[6], 0.001);
        assert_eq!(items[1].fed_mat.unwrap()[7], 0.002);

        // Fed items feed the same emit path: frame() over a dynamic
        // drawable consuming this feed produces the two blits in order,
        // and item 0's fed mat3 lands in the BLIT record verbatim.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let dyn_id = doc.add_drawable([640.0, 480.0], false, true);
        doc.drawables[0]
            .commands
            .push(Cmd::Item(Item::of(Source::Drawable(dyn_id))));
        let borrowed = [Feed {
            drawable: dyn_id,
            items,
        }];
        let schedule = evaluate(&doc, 0.0, &borrowed);
        let srcs = blit_sources(&schedule);
        assert_eq!(srcs, vec![(SRC_IMAGE, 5), (SRC_FILL, 0), (SRC_DRAWABLE, dyn_id)]);
        // The first BLIT's record mat3 equals item 0's fed mat3.
        let op = (0..schedule.len())
            .find(|i| schedule.u[i * U_STRIDE] == OP_BLIT)
            .unwrap();
        let mut m = [0.0f32; 9];
        m.copy_from_slice(&schedule.f[op * F_STRIDE..op * F_STRIDE + 9]);
        assert_eq!(m, [2.0, 0.0, 10.0, 0.0, 0.5, 20.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn inline_feed_draws_into_the_enclosing_drawable() {
        // A feed SLOT is not a target: its items draw as ordinary items of
        // the enclosing drawable, between that drawable's own commands, so
        // nothing but the real render target bounds them. Rendering them
        // into a box of their own is what clips mod-displaced notes away.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let slot = doc.add_drawable([0.0, 0.0], false, true);
        doc.drawables[slot as usize].inline = true;
        doc.drawables[0]
            .commands
            .push(Cmd::Item(Item::of(Source::Fill)));
        doc.drawables[0].commands.push(Cmd::Feed {
            slot,
            links: Vec::new(),
            flip_base_y: false,
            visible: ChannelRef::constant(1.0),
            projection: None,
        });
        doc.drawables[0]
            .commands
            .push(Cmd::Item(Item::of(Source::Image {
                image: 9,
                frame: ChannelRef::constant(0.0),
            })));
        let fed = vec![Item::of(Source::Image {
            image: 3,
            frame: ChannelRef::constant(0.0),
        })];
        let borrowed = [Feed {
            drawable: slot,
            items: &fed,
        }];
        let schedule = evaluate(&doc, 0.0, &borrowed);

        // The fed item lands BETWEEN the screen's own items, in tree order,
        // and the slot never opens a target of its own.
        assert_eq!(
            blit_sources(&schedule),
            vec![(SRC_FILL, 0), (SRC_IMAGE, 3), (SRC_IMAGE, 9)]
        );
        let begins: Vec<u32> = (0..schedule.len())
            .filter(|i| schedule.u[i * U_STRIDE] == OP_BEGIN)
            .map(|i| schedule.u[i * U_STRIDE + 1])
            .collect();
        assert_eq!(begins, vec![SCREEN], "a feed slot is never a render target");
    }

    #[test]
    fn linked_feed_composes_the_chain_over_each_fed_item() {
        // A Feed carrying a consumer link chain re-renders the fed items
        // under it: each item's record mat equals the chain's H composed
        // over its fed mat, and the chain alpha multiplies its opacity.
        // With an IDENTITY fed mat, the record must match what the same
        // chain produces on an ordinary linked item.
        let mut linked = DrawableDoc::with_screen([640.0, 480.0]);
        let mut item = Item::of(Source::Fill);
        item.links.push(default_link());
        linked.drawables[0].commands.push(Cmd::Item(item));
        let expected = blit_mat(&evaluate(&linked, 0.0, &[]));

        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let slot = doc.add_drawable([0.0, 0.0], false, true);
        doc.drawables[slot as usize].inline = true;
        let mut faded = default_link();
        faded.alpha = ChannelRef::constant(0.5);
        doc.drawables[0].commands.push(Cmd::Feed {
            slot,
            links: vec![faded],
            flip_base_y: false,
            visible: ChannelRef::constant(1.0),
            projection: None,
        });
        let mut fed = Item::of(Source::Fill);
        fed.fed_mat = Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        fed.opacity = ChannelRef::constant(0.6);
        let items = vec![fed];
        let feeds = [Feed {
            drawable: slot,
            items: &items,
        }];
        let schedule = evaluate(&doc, 0.0, &feeds);

        let m = blit_mat(&schedule);
        for lane in 0..9 {
            assert!(
                (m[lane] - expected[lane]).abs() < 1e-4,
                "lane {lane}: {} vs {}",
                m[lane],
                expected[lane]
            );
        }
        let op = (0..schedule.len())
            .find(|i| schedule.u[i * U_STRIDE] == OP_BLIT)
            .unwrap();
        assert!((schedule.f[op * F_STRIDE + 9] - 0.3).abs() < 1e-4);
    }

    #[test]
    fn feed_visibility_gate_and_hidden_chain_drop_all_items() {
        // visible < 0.5 emits nothing; so does a chain whose link is hidden.
        for hide_via_chain in [false, true] {
            let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
            let slot = doc.add_drawable([0.0, 0.0], false, true);
            doc.drawables[slot as usize].inline = true;
            let mut hidden = default_link();
            if hide_via_chain {
                hidden.hidden = ChannelRef::constant(1.0);
            }
            doc.drawables[0].commands.push(Cmd::Feed {
                slot,
                links: vec![hidden],
                flip_base_y: false,
                visible: ChannelRef::constant(if hide_via_chain { 1.0 } else { 0.0 }),
                projection: None,
            });
            let mut fed = Item::of(Source::Fill);
            fed.fed_mat = Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            let items = vec![fed];
            let feeds = [Feed {
                drawable: slot,
                items: &items,
            }];
            let schedule = evaluate(&doc, 0.0, &feeds);
            assert_eq!(op_kinds(&schedule), vec![OP_BEGIN, OP_END]);
        }
    }

    #[test]
    fn dynamic_feed_supplies_items() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let notes = doc.add_drawable([640.0, 480.0], false, true);
        doc.drawables[0]
            .commands
            .push(Cmd::Item(Item::of(Source::Drawable(notes))));
        let fed = vec![
            Item::of(Source::Image {
                image: 1,
                frame: ChannelRef::constant(0.0),
            }),
            Item::of(Source::Image {
                image: 2,
                frame: ChannelRef::constant(0.0),
            }),
        ];
        let feeds = [Feed {
            drawable: notes,
            items: &fed,
        }];
        let schedule = evaluate(&doc, 0.0, &feeds);
        assert_eq!(
            blit_sources(&schedule),
            vec![(SRC_IMAGE, 1), (SRC_IMAGE, 2), (SRC_DRAWABLE, notes)]
        );
    }

    /// The f32 mat3 of the first (and only) BLIT op.
    fn blit_mat(schedule: &DrawSchedule) -> [f32; 9] {
        let op = (0..schedule.len())
            .find(|i| schedule.u[i * U_STRIDE] == OP_BLIT)
            .expect("a blit");
        let mut m = [0.0f32; 9];
        m.copy_from_slice(&schedule.f[op * F_STRIDE..op * F_STRIDE + 9]);
        m
    }

    fn default_link() -> crate::doc::LinkRef {
        let s = crate::transform::TransformState::default();
        let c = ChannelRef::constant;
        crate::doc::LinkRef {
            x: c(s.x),
            y: c(s.y),
            zoom_x: c(s.zoom_x),
            zoom_y: c(s.zoom_y),
            rot: c(s.rot),
            skew_x: c(s.skew_x),
            skew_y: c(s.skew_y),
            base_scale_x: c(s.base_scale_x),
            base_scale_y: c(s.base_scale_y),
            halign: c(s.halign),
            valign: c(s.valign),
            hidden: c(s.hidden),
            alpha: c(s.alpha),
            crop: [c(s.crop[0]), c(s.crop[1]), c(s.crop[2]), c(s.crop[3])],
            natural_w: c(s.natural_w),
            natural_h: c(s.natural_h),
            rotation_x: c(s.rotation_x),
            rotation_y: c(s.rotation_y),
            z: c(s.z),
            scale_z: c(s.scale_z),
            base_scale_z: c(s.base_scale_z),
            rotation_order: s.rotation_order,
        }
    }

    #[test]
    fn linked_item_uses_compose_links_transposed_into_the_record() {
        // A single default link folds through compose_links to the centered
        // _TO_CONTENT translate; the record carries its column-vector
        // transpose (translation in lanes 2/5). compose_links' row-vector H
        // puts -320/-240 in lanes 6/7; transposed they land at 2/5.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mut item = Item::of(Source::Fill);
        item.links.push(default_link());
        doc.drawables[0].commands.push(Cmd::Item(item));
        let schedule = evaluate(&doc, 0.0, &[]);
        let m = blit_mat(&schedule);
        // Column-vector affine: x' = m[0]*u + m[1]*v + m[2].
        assert!((m[0] - 1.0).abs() < 1e-4 && (m[4] - 1.0).abs() < 1e-4);
        assert!((m[2] - -320.0).abs() < 1e-3, "tx {}", m[2]);
        assert!((m[5] - -240.0).abs() < 1e-3, "ty {}", m[5]);
    }

    #[test]
    fn linked_item_alpha_multiplies_and_hidden_drops() {
        // Link alpha 0.5 multiplies the item opacity 0.6 -> 0.3.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mut faded = default_link();
        faded.alpha = ChannelRef::constant(0.5);
        let mut item = Item::of(Source::Fill);
        item.opacity = ChannelRef::constant(0.6);
        item.links.push(faded);
        doc.drawables[0].commands.push(Cmd::Item(item));
        let schedule = evaluate(&doc, 0.0, &[]);
        let op = (0..schedule.len())
            .find(|i| schedule.u[i * U_STRIDE] == OP_BLIT)
            .unwrap();
        assert!((schedule.f[op * F_STRIDE + 9] - 0.3).abs() < 1e-4);

        // A hidden link drops the item entirely.
        let mut doc2 = DrawableDoc::with_screen([640.0, 480.0]);
        let mut hidden = default_link();
        hidden.hidden = ChannelRef::constant(1.0);
        let mut item2 = Item::of(Source::Fill);
        item2.links.push(hidden);
        doc2.drawables[0].commands.push(Cmd::Item(item2));
        let s2 = evaluate(&doc2, 0.0, &[]);
        assert_eq!(op_kinds(&s2), vec![OP_BEGIN, OP_END]);
    }

    #[test]
    fn linked_item_carries_swapped_leaf_crop() {
        // A flipped leaf swaps top/bottom crop; the record crop lanes 13/16
        // reflect the swap. Set top=0.1, bottom=0.3 -> flipped top=0.3.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mut link = default_link();
        link.crop = [
            ChannelRef::constant(0.0),
            ChannelRef::constant(0.1),
            ChannelRef::constant(0.0),
            ChannelRef::constant(0.3),
        ];
        let mut item = Item::of(Source::Fill);
        item.links.push(link);
        item.flip_base_y = true;
        doc.drawables[0].commands.push(Cmd::Item(item));
        let schedule = evaluate(&doc, 0.0, &[]);
        let op = (0..schedule.len())
            .find(|i| schedule.u[i * U_STRIDE] == OP_BLIT)
            .unwrap();
        // crop lanes: [13] l, [14] t, [15] r, [16] b.
        assert!((schedule.f[op * F_STRIDE + 14] - 0.3).abs() < 1e-4);
        assert!((schedule.f[op * F_STRIDE + 16] - 0.1).abs() < 1e-4);
    }

    #[test]
    fn projection_on_untransformed_fullscreen_leaves_the_mat_unchanged() {
        // A centered fov-45 projection folds to identity on a fullscreen
        // linkless item, so the record mat3 stays the first-cut identity.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let far = crate::camera::eye_distance(45.0, 640.0) + crate::camera::FAR_SLACK;
        let mut item = Item::of(Source::Fill);
        item.projection = Some(crate::doc::CameraRef {
            fov_deg: ChannelRef::constant(45.0),
            vanish_x: ChannelRef::constant(320.0),
            vanish_y: ChannelRef::constant(240.0),
            far: ChannelRef::constant(far),
            w: 640.0,
            h: 480.0,
        });
        doc.drawables[0].commands.push(Cmd::Item(item));
        let schedule = evaluate(&doc, 0.0, &[]);
        let m = blit_mat(&schedule);
        let s = m[8];
        let want = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for i in 0..9 {
            assert!((m[i] / s - want[i]).abs() < 1e-4, "mat[{i}] = {}", m[i]);
        }
    }

    #[test]
    fn linkless_item_stays_bit_identical() {
        // The first-cut TRS record must be untouched by the wiring: a plain
        // translate(10, 20) scale(2, 3) item yields the exact same mat3 the
        // pre-wave-2 emit produced.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mut item = Item::of(Source::Fill);
        item.transform = crate::doc::TransformRef {
            x: ChannelRef::constant(10.0),
            y: ChannelRef::constant(20.0),
            scale_x: ChannelRef::constant(2.0),
            scale_y: ChannelRef::constant(3.0),
            rot: ChannelRef::constant(0.0),
        };
        doc.drawables[0].commands.push(Cmd::Item(item));
        let schedule = evaluate(&doc, 0.0, &[]);
        let m = blit_mat(&schedule);
        assert_eq!(m, [2.0, 0.0, 10.0, 0.0, 3.0, 20.0, 0.0, 0.0, 1.0]);
    }

    // --- wave 4: event-driven reaction splices -------------------------

    use crate::doc::{Reaction, PROP_OPACITY, PROP_ZOOM};
    use crate::schedule::{Node, Target};

    /// A one-segment ramp fragment on `prop`: 0..value linearly over `dur`.
    fn ramp_reaction(id: u32, kind: u32, column_filter: i32, prop: u32, dur: f32, value: f32) -> Reaction {
        Reaction {
            id,
            trigger_kind: kind,
            column_filter,
            fragment: Node::seg(dur, 0, vec![(prop, Target::Abs(value))]),
            prop,
        }
    }

    fn opacity_of(schedule: &DrawSchedule) -> f32 {
        let op = (0..schedule.len())
            .find(|i| schedule.u[i * U_STRIDE] == OP_BLIT)
            .expect("a blit");
        schedule.f[op * F_STRIDE + 9]
    }

    #[test]
    fn reaction_base_channel_rules_before_the_event() {
        // An opacity reaction ramps 0 -> 1 over 1s on a Press. Before the
        // press arrives, the item's base opacity (0.4) rules unchanged.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mut item = Item::of(Source::Fill);
        item.opacity = ChannelRef::constant(0.4);
        item.reactions.push(ramp_reaction(0, EV_PRESS, -1, PROP_OPACITY, 1.0, 1.0));
        doc.drawables[0].commands.push(Cmd::Item(item));

        let mut cache = ReactionCache::new();
        // No events: base rules.
        let s = evaluate_with_events(&doc, 0.5, &[], &[], &mut cache);
        assert!((opacity_of(&s) - 0.4).abs() < 1e-4);
        // Event at te=1.0 but sampled at t=0.5 (before it): base still rules.
        let ev = [Event { kind: EV_PRESS, time: 1.0, column: -1, strength: 1.0 }];
        let s = evaluate_with_events(&doc, 0.5, &[], &ev, &mut cache);
        assert!((opacity_of(&s) - 0.4).abs() < 1e-4);
    }

    #[test]
    fn reaction_ramps_the_prop_after_the_event() {
        // Press at te=1.0; the 0->1-over-1s ramp eases off the base (0.4)
        // at te and reaches 1.0 by te+1. Midpoint te+0.5 = 0.7.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mut item = Item::of(Source::Fill);
        item.opacity = ChannelRef::constant(0.4);
        item.reactions.push(ramp_reaction(0, EV_PRESS, -1, PROP_OPACITY, 1.0, 1.0));
        doc.drawables[0].commands.push(Cmd::Item(item));

        let mut cache = ReactionCache::new();
        let ev = [Event { kind: EV_PRESS, time: 1.0, column: -1, strength: 1.0 }];
        let s = evaluate_with_events(&doc, 1.5, &[], &ev, &mut cache);
        assert!((opacity_of(&s) - 0.7).abs() < 1e-4, "mid-ramp {}", opacity_of(&s));
        let s = evaluate_with_events(&doc, 2.5, &[], &ev, &mut cache);
        assert!((opacity_of(&s) - 1.0).abs() < 1e-4, "held {}", opacity_of(&s));
    }

    #[test]
    fn second_event_resplices_from_the_new_time() {
        // A later Press re-splices: the ramp restarts from te2, easing off
        // the base again. At te2+0.5 the fresh ramp reads 0.7, not the held
        // 1.0 of the first press.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mut item = Item::of(Source::Fill);
        item.opacity = ChannelRef::constant(0.4);
        item.reactions.push(ramp_reaction(0, EV_PRESS, -1, PROP_OPACITY, 1.0, 1.0));
        doc.drawables[0].commands.push(Cmd::Item(item));

        let mut cache = ReactionCache::new();
        let ev = [
            Event { kind: EV_PRESS, time: 1.0, column: -1, strength: 1.0 },
            Event { kind: EV_PRESS, time: 5.0, column: -1, strength: 1.0 },
        ];
        // Latest press (te=5) wins; at t=5.5 the fresh ramp reads 0.7.
        let s = evaluate_with_events(&doc, 5.5, &[], &ev, &mut cache);
        assert!((opacity_of(&s) - 0.7).abs() < 1e-4, "resplice {}", opacity_of(&s));
    }

    #[test]
    fn column_filter_gates_the_reaction() {
        // The reaction only fires on column 3; a column-1 press is ignored.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mut item = Item::of(Source::Fill);
        item.opacity = ChannelRef::constant(0.4);
        item.reactions.push(ramp_reaction(0, EV_PRESS, 3, PROP_OPACITY, 1.0, 1.0));
        doc.drawables[0].commands.push(Cmd::Item(item));

        let mut cache = ReactionCache::new();
        let wrong = [Event { kind: EV_PRESS, time: 1.0, column: 1, strength: 1.0 }];
        let s = evaluate_with_events(&doc, 2.0, &[], &wrong, &mut cache);
        assert!((opacity_of(&s) - 0.4).abs() < 1e-4, "filtered out");
        let right = [Event { kind: EV_PRESS, time: 1.0, column: 3, strength: 1.0 }];
        let s = evaluate_with_events(&doc, 2.0, &[], &right, &mut cache);
        assert!((opacity_of(&s) - 1.0).abs() < 1e-4, "matched");
    }

    #[test]
    fn zoom_reaction_scales_the_mat3() {
        // A zoom reaction (base 1.0) ramps to 2.0 over 1s; at te+1 the mat3
        // scale lanes are doubled. Item is scale-2, so 2 * 2 = 4.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let mut item = Item::of(Source::Fill);
        item.transform.scale_x = ChannelRef::constant(2.0);
        item.transform.scale_y = ChannelRef::constant(2.0);
        item.reactions.push(ramp_reaction(0, EV_HIT, -1, PROP_ZOOM, 1.0, 2.0));
        doc.drawables[0].commands.push(Cmd::Item(item));

        let mut cache = ReactionCache::new();
        let ev = [Event { kind: EV_HIT, time: 0.0, column: -1, strength: 1.0 }];
        let s = evaluate_with_events(&doc, 1.0, &[], &ev, &mut cache);
        let m = blit_mat(&s);
        assert!((m[0] - 4.0).abs() < 1e-4, "sx {}", m[0]);
        assert!((m[4] - 4.0).abs() < 1e-4, "sy {}", m[4]);
    }
}
