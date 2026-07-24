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
//! The output is flat fixed-width SoA records (no per-op objects) so
//! the schedule crosses Seam B as two buffers.

use crate::channels::{ChannelRef, ChannelTable};
use crate::doc::{Cmd, DrawableDoc, Item, Source, SCREEN};

/// Op kinds in the u32 record.
pub const OP_BEGIN: u32 = 0;
pub const OP_BLIT: u32 = 1;
pub const OP_COPY: u32 = 2;
pub const OP_END: u32 = 3;

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

/// u32 lanes per fed item (FROZEN, drawable-port-wave1.md A5):
///   [source_kind, source_id, frame, flags]
/// flags bit0 = additive, bit1 = screen_space, bit2 = has_z.
pub const FEED_U_STRIDE: usize = 4;
/// f32 lanes per fed item (FROZEN):
///   [x, y, sx, sy, rot, opacity, r, g, b, crop_l, crop_t, crop_r,
///    crop_b, z]
pub const FEED_F_STRIDE: usize = 14;

const FEED_FLAG_ADDITIVE: u32 = 1;
const FEED_FLAG_SCREEN: u32 = 1 << 1;
const FEED_FLAG_HAS_Z: u32 = 1 << 2;

/// Decode one fed item's flat SoA lanes into an `Item`. Fed values are
/// already sampled scalars, so every channel is a constant (rest-only)
/// ChannelRef - no doc channel lookups happen for fed items.
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
    item.transform = crate::doc::TransformRef {
        x: ChannelRef::constant(f[0]),
        y: ChannelRef::constant(f[1]),
        scale_x: ChannelRef::constant(f[2]),
        scale_y: ChannelRef::constant(f[3]),
        rot: ChannelRef::constant(f[4]),
    };
    item.opacity = ChannelRef::constant(f[5]);
    item.tint = [
        ChannelRef::constant(f[6]),
        ChannelRef::constant(f[7]),
        ChannelRef::constant(f[8]),
    ];
    item.crop = [
        ChannelRef::constant(f[9]),
        ChannelRef::constant(f[10]),
        ChannelRef::constant(f[11]),
        ChannelRef::constant(f[12]),
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
    item.z = (flags & FEED_FLAG_HAS_Z != 0).then(|| ChannelRef::constant(f[13]));
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
    let mut schedule = DrawSchedule::default();
    for id in eval_order(doc, feeds) {
        emit_drawable(doc, id, t, feeds, &mut schedule);
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

fn emit_drawable(
    doc: &DrawableDoc,
    id: u32,
    t: f32,
    feeds: &[Feed],
    schedule: &mut DrawSchedule,
) {
    let drawable = &doc.drawables[id as usize];
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
            emit_item(doc, item, t, schedule);
        }
    } else {
        emit_commands(doc, &drawable.commands, t, schedule);
    }
    schedule.op(OP_END, id, 0);
}

fn emit_commands(doc: &DrawableDoc, commands: &[Cmd], t: f32, schedule: &mut DrawSchedule) {
    let mut i = 0usize;
    while i < commands.len() {
        match &commands[i] {
            Cmd::Item(item) => {
                emit_item(doc, item, t, schedule);
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
                        Cmd::Item(item) => emit_item(doc, item, t, schedule),
                        Cmd::Snapshot { into } => schedule.op(OP_COPY, *into, 0),
                        Cmd::SortSpan { .. } => {} // nested spans: forbid for now
                    }
                }
                i = span_end;
            }
        }
    }
}

fn emit_item(doc: &DrawableDoc, item: &Item, t: f32, schedule: &mut DrawSchedule) {
    let ch: &ChannelTable = &doc.channels;
    if ch.sample(item.visible, t) < 0.5 {
        return;
    }
    let opacity = ch.sample(item.opacity, t);
    if opacity < 1.0 / 255.0 {
        return;
    }

    let (src_kind, src_id, src_aux) = match item.source {
        Source::Image { image, frame } => (SRC_IMAGE, image, ch.sample(frame, t) as u32),
        Source::Drawable(d) => (SRC_DRAWABLE, d, 0),
        Source::Mesh(m) => (SRC_MESH, m, 0),
        Source::Fill => (SRC_FILL, 0, 0),
        Source::Lines(l) => (SRC_LINES, l, 0),
    };

    let (x, y) = (ch.sample(item.transform.x, t), ch.sample(item.transform.y, t));
    let (sx, sy) = (
        ch.sample(item.transform.scale_x, t),
        ch.sample(item.transform.scale_y, t),
    );
    let rot = ch.sample(item.transform.rot, t).to_radians();
    let (c, s) = (rot.cos(), rot.sin());
    // mat3, row-major: translate(x, y) * rotate * scale.
    let mat = [
        c * sx, -s * sy, x, //
        s * sx, c * sy, y, //
        0.0, 0.0, 1.0,
    ];

    let mut f = [0.0f32; F_STRIDE];
    f[..9].copy_from_slice(&mat);
    f[9] = opacity;
    for (lane, tint) in item.tint.iter().enumerate() {
        f[10 + lane] = ch.sample(*tint, t);
    }
    for (lane, crop) in item.crop.iter().enumerate() {
        f[13 + lane] = ch.sample(*crop, t);
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
        let u: Vec<u32> = vec![
            SRC_IMAGE, 5, 3, FEED_FLAG_ADDITIVE, // item 0
            SRC_FILL, 0, 0, FEED_FLAG_SCREEN | FEED_FLAG_HAS_Z, // item 1
        ];
        let f: Vec<f32> = vec![
            10.0, 20.0, 2.0, 0.5, 45.0, 0.75, 1.0, 0.5, 0.25, 0.1, 0.2, 0.3, 0.4, 0.0, // 0
            0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 9.0, // 1
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

        assert!(matches!(items[1].source, Source::Fill));
        assert!(matches!(items[1].space, crate::doc::Space::Screen));
        assert!(items[1].z.is_some());

        // Fed items feed the same emit path: frame() over a dynamic
        // drawable consuming this feed produces the two blits in order.
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
}
