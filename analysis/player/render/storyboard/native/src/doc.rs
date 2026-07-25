//! The Drawable document: Seam A's compiled form.
//!
//! Types follow .claude/plans/drawable-ir.md draft 3. Per-object
//! construction is allowed here (once per chart); nothing per-frame
//! touches these structures except reads.

use crate::camera::Mat4;
use crate::channels::{ChannelRef, ChannelTable};
use crate::schedule::Node;

pub const SCREEN: u32 = 0; // drawables[0] is always the screen (root)

/// The draw properties a `Reaction` may splice onto (this wave). Each
/// names one sampled scalar in `emit_item`: opacity multiplies the BLIT
/// record opacity lane, the tint channels multiply the tint lanes, zoom
/// is a uniform scale multiplier folded onto the item's mat3, and frame
/// overrides the image src_aux (sheet frame) lane. The u32 value is the
/// `prop` key the reaction's Schedule fragment targets and is keyed off
/// by evaluate.rs when it looks up the lowered run.
pub const PROP_OPACITY: u32 = 0;
pub const PROP_TINT_R: u32 = 1;
pub const PROP_TINT_G: u32 = 2;
pub const PROP_TINT_B: u32 = 3;
pub const PROP_ZOOM: u32 = 4;
pub const PROP_FRAME: u32 = 5;

/// One event-driven drawing response attached to an `Item`. On a frame
/// carrying an event whose kind == `trigger_kind` and whose column
/// passes `column_filter` (-1 = any), the `fragment` Schedule lowers at
/// `t0 = event time` and its `prop` breakpoint run splices over the
/// item's base property value for t >= that event time. The latest
/// matching event wins (engine queue-append semantics). `id` is a stable
/// per-reaction key (assigned at build) the evaluator caches lowerings
/// against, paired with the event time.
#[derive(Clone, Debug)]
pub struct Reaction {
    pub id: u32,
    pub trigger_kind: u32,
    pub column_filter: i32,
    pub fragment: Node,
    pub prop: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClearMode {
    TransparentBlack,
    OpaqueBlack,
    Retain, // persistent targets: no clear, content carries across frames
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    SourceOver,
    Additive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    Scene,
    Screen,
}

/// What an item draws. Geometry-bearing sources (mesh vertices, line
/// strips) reference doc tables or arrive via dynamic feeds.
#[derive(Clone, Copy, Debug)]
pub enum Source {
    Image { image: u32, frame: ChannelRef },
    Drawable(u32),
    Mesh(u32),
    Fill,
    Lines(u32),
}

/// A shader program the doc can attach to items (attach point #1) or
/// to composed-drawable blits (attach point #2). `uniform_names`
/// indexes the per-item `uniforms` list: a `(name_idx, ChannelRef)`
/// binds a uniform by its position in this vector.
#[derive(Clone, Debug)]
pub struct ShaderDesc {
    pub frag: String,
    pub vert: Option<String>,
    pub uniform_names: Vec<String>,
}

/// A registered mesh (drawable-ir.md MeshDesc): flat interleaved
/// `[x, y, u, v]` vertices, a primitive `mode`, and an optional vertex
/// shader (id into `DrawableDoc.shaders`, or inline source). The GL DRAW
/// of these vertices is the executor tier; the core only registers the
/// data and lets `Source::Mesh(id)` reference it into a record lane.
#[derive(Clone, Debug)]
pub struct MeshDesc {
    /// Interleaved vertex attributes, 4 floats per vertex: x, y, u, v.
    pub vertices: Vec<f32>,
    /// Primitive assembly mode (executor-defined enum: e.g. triangles /
    /// strip / fan); carried opaquely through the core.
    pub mode: u32,
    /// Optional vertex-shader program id (into `shaders`) or inline source
    /// for the crumple.vert path; None = the executor's default mesh vert.
    pub vert_shader: Option<u32>,
    pub vert_source: Option<String>,
}

/// A clip shape in the target drawable's logical units (Item.clip).
/// Extensible per the type sheet's ClipDesc vocabulary; Mesh clips are
/// deferred until the mesh tier lands.
#[derive(Clone, Debug)]
pub enum ClipDesc {
    Rect([f32; 4]), // l, t, r, b
    Polygon(Vec<[f32; 2]>),
}

/// Channel-backed 2D placement. First cut: translate/scale/rotate
/// about the item origin; the full TransformChannel semantics
/// (halign/valign anchors, crop-under-flip, base-scale flip cancel)
/// port on top of these same refs.
#[derive(Clone, Copy, Debug)]
pub struct TransformRef {
    pub x: ChannelRef,
    pub y: ChannelRef,
    pub scale_x: ChannelRef,
    pub scale_y: ChannelRef,
    pub rot: ChannelRef, // degrees, engine convention
}

impl TransformRef {
    pub fn identity() -> Self {
        TransformRef {
            x: ChannelRef::constant(0.0),
            y: ChannelRef::constant(0.0),
            scale_x: ChannelRef::constant(1.0),
            scale_y: ChannelRef::constant(1.0),
            rot: ChannelRef::constant(0.0),
        }
    }
}

/// One channel-backed link of the full leaf-link transform chain: the
/// 17 scalars `transform::TransformState` folds, each a `ChannelRef`
/// sampled per frame. An `Item` carrying a non-empty `links` list uses
/// the full `transform::compose_links` path (halign/valign anchors,
/// base-scale flip cancel, crop-under-flip, multi-link parent
/// composition) in place of the first-cut TRS. Field order tracks
/// field_compose._LINK_RESTS / TransformState.
#[derive(Clone, Copy, Debug)]
pub struct LinkRef {
    pub x: ChannelRef,
    pub y: ChannelRef,
    pub zoom_x: ChannelRef,
    pub zoom_y: ChannelRef,
    pub rot: ChannelRef,
    pub skew_x: ChannelRef,
    pub skew_y: ChannelRef,
    pub base_scale_x: ChannelRef,
    pub base_scale_y: ChannelRef,
    pub halign: ChannelRef,
    pub valign: ChannelRef,
    pub hidden: ChannelRef,
    pub alpha: ChannelRef,
    pub crop: [ChannelRef; 4], // l, t, r, b
    pub natural_w: ChannelRef,
    pub natural_h: ChannelRef,
    /// Out-of-plane terms. All rest at engine identity, so a link that
    /// never sets them composes through the exact 2D path.
    pub rotation_x: ChannelRef,
    pub rotation_y: ChannelRef,
    pub z: ChannelRef,
    pub scale_z: ChannelRef,
    pub base_scale_z: ChannelRef,
    /// Euler order for `rotation_x/y/rot`; constant per link (a token, not
    /// an animatable scalar), so it rides the LinkRef directly.
    pub rotation_order: crate::camera::RotOrder,
}

/// A channel-backed camera projection attached to an item (Item.projection
/// in the type sheet): the fov / vanish / far a `camera::design_projection`
/// is built from each frame. The resulting mat4 folds the item's 2D
/// transform onto the z=0 design plane and back to a projective mat3
/// homography written to the BLIT record (see evaluate.rs::fold_projection).
#[derive(Clone, Copy, Debug)]
pub struct CameraRef {
    pub fov_deg: ChannelRef,
    pub vanish_x: ChannelRef,
    pub vanish_y: ChannelRef,
    pub far: ChannelRef,
    pub w: f32,
    pub h: f32,
}

impl CameraRef {
    /// Build the design projection mat4 for the sampled fov/vanish/far.
    pub fn matrix(&self, table: &ChannelTable, t: f32) -> Mat4 {
        let fov = table.sample(self.fov_deg, t);
        let vanish = [table.sample(self.vanish_x, t), table.sample(self.vanish_y, t)];
        let far = table.sample(self.far, t);
        crate::camera::design_projection(fov, self.w, self.h, vanish, far)
    }
}

#[derive(Clone, Debug)]
pub struct Item {
    pub source: Source,
    pub transform: TransformRef,
    /// Full leaf-link transform chain (root-first). Empty = use the
    /// first-cut `transform` TRS bit-identically; non-empty routes through
    /// `transform::compose_links`, whose H replaces the TRS mat3, whose
    /// alpha multiplies opacity, and whose crop supplies the crop lanes.
    pub links: Vec<LinkRef>,
    /// Mirror the leaf link's vertical source axis (AFT capture
    /// compensation); only consulted when `links` is non-empty.
    pub flip_base_y: bool,
    /// A dynamic-feed item's pre-sampled mat3, already in the BLIT
    /// record's column-vector layout (feed v2). When present it is
    /// written to the record verbatim - the game side (bridge) has
    /// already resolved the full homography, so no TRS/link folding
    /// happens. Static (doc-built) items leave this None and go through
    /// the TRS or link chain.
    pub fed_mat: Option<[f32; 9]>,
    /// Optional per-item camera projection folded onto the item's mat3
    /// (None = the parent's/no projection, a plain 2D blit).
    pub projection: Option<CameraRef>,
    pub space: Space,
    pub opacity: ChannelRef,
    pub tint: [ChannelRef; 3],
    pub blend: Blend,
    pub crop: [ChannelRef; 4], // l, t, r, b fractions
    pub visible: ChannelRef,   // >= 0.5 draws
    pub shader: Option<u32>,   // ShaderId into DrawableDoc.shaders
    /// Per-item shader uniform bindings: (uniform_names index, channel).
    /// Sampled each frame; the values ride the schedule's uniform buffer
    /// (see evaluate.rs). Empty when no shader / no bound uniforms.
    pub uniforms: Vec<(u16, ChannelRef)>,
    pub clip: Option<u32>,     // ClipId into DrawableDoc.clips
    pub z: Option<ChannelRef>, // local sort key, read inside SortSpan only
    /// Event-driven drawing responses: on a matching event, a reaction's
    /// Schedule fragment lowers at the event time and splices over the
    /// named draw property. Empty for items with no reactions (the common
    /// case); never consulted unless the frame carries events.
    pub reactions: Vec<Reaction>,
}

impl Item {
    pub fn of(source: Source) -> Self {
        Item {
            source,
            transform: TransformRef::identity(),
            links: Vec::new(),
            flip_base_y: false,
            fed_mat: None,
            projection: None,
            space: Space::Scene,
            opacity: ChannelRef::constant(1.0),
            tint: [ChannelRef::constant(1.0); 3],
            blend: Blend::SourceOver,
            crop: [ChannelRef::constant(0.0); 4],
            visible: ChannelRef::constant(1.0),
            shader: None,
            uniforms: Vec::new(),
            clip: None,
            z: None,
            reactions: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Cmd {
    Item(Item),
    /// At-position capture: copy this drawable's in-progress composite
    /// into another drawable now (AFT capture semantics).
    Snapshot {
        into: u32,
        /// The capture NODE's own link chain. The engine captures only while
        /// the node DRAWS (EnablePreserveTexture holds the last capture across
        /// hidden frames), so a hidden or faint node must leave the slot
        /// untouched - that retained image IS the freeze a still-frames rig
        /// relies on. An empty chain means "always capture".
        links: Vec<LinkRef>,
        flip_base_y: bool,
    },
    /// The next `len` commands re-sort stably by their sampled `z`
    /// before drawing (SetDrawByZPosition).
    SortSpan { len: u32 },
    /// Emit `slot`'s fed items INLINE, at this tree position, into the
    /// enclosing drawable - no target of its own.
    ///
    /// This is how per-frame content (notes, receptors) draws as ordinary
    /// items on the screen rather than into an intermediate box. Rendering
    /// them into their own drawable would clip everything past that box,
    /// which is precisely what cuts mod-displaced notes off; drawn inline,
    /// each item is placed by its own mat3 and only the real render target
    /// bounds it.
    ///
    /// A non-empty `links` chain is a CONSUMER transform (a proxy/player
    /// field re-render): the chain's H composes over each fed item's mat3
    /// and its alpha multiplies each item's opacity, so the consumer shows
    /// the same unclipped items instead of a capture-boxed texture. The
    /// chain's crop does not apply - per-item crop fractions are of the
    /// item's own box, not the chain's content box (documented limitation).
    /// `visible` gates the whole feed (< 0.5 emits nothing).
    Feed {
        slot: u32,
        links: Vec<LinkRef>,
        flip_base_y: bool,
        visible: ChannelRef,
        /// The consumer chain's camera, folded onto the composed chain before
        /// it composes over each fed item (see evaluate.rs::emit_feed). A
        /// linked feed is a field copy re-rendering the notes, so its chain
        /// needs the same perspective divide an Item's does - without it a
        /// rotated field is a flat squash, not a 3D turn.
        projection: Option<CameraRef>,
    },
}

pub struct Drawable {
    /// Logical size: the single source-space authority for this
    /// target's items and crops. The executor maps logical to device
    /// pixels exactly once.
    pub size: [f32; 2],
    pub persistent: bool,
    pub clear: ClearMode,
    /// Insertion order IS engine tree order.
    pub commands: Vec<Cmd>,
    /// Dynamic: commands arrive per frame via feeds (notes,
    /// travelpaths); the static list above stays empty.
    pub dynamic: bool,
    /// Inline: this drawable is a FEED SLOT, never a render target. It is
    /// skipped by the per-drawable walk; a `Cmd::Feed` emits its items into
    /// whichever drawable holds that command (see Cmd::Feed).
    pub inline: bool,
}

#[derive(Default)]
pub struct DrawableDoc {
    pub drawables: Vec<Drawable>,
    pub shaders: Vec<ShaderDesc>,
    pub meshes: Vec<MeshDesc>,
    pub clips: Vec<ClipDesc>,
    pub channels: ChannelTable,
    /// Monotonic counter minting stable per-reaction ids (the lowering
    /// cache key); bumped once per `item_reaction` at build.
    pub reaction_count: u32,
}

impl DrawableDoc {
    pub fn with_screen(size: [f32; 2]) -> Self {
        let mut doc = DrawableDoc::default();
        doc.drawables.push(Drawable {
            size,
            persistent: false,
            clear: ClearMode::OpaqueBlack,
            commands: Vec::new(),
            dynamic: false,
            inline: false,
        });
        doc
    }

    pub fn add_shader(&mut self, shader: ShaderDesc) -> u32 {
        let id = self.shaders.len() as u32;
        self.shaders.push(shader);
        id
    }

    pub fn add_mesh(&mut self, mesh: MeshDesc) -> u32 {
        let id = self.meshes.len() as u32;
        self.meshes.push(mesh);
        id
    }

    pub fn add_clip(&mut self, clip: ClipDesc) -> u32 {
        let id = self.clips.len() as u32;
        self.clips.push(clip);
        id
    }

    /// Mint the next stable reaction id.
    pub fn next_reaction_id(&mut self) -> u32 {
        let id = self.reaction_count;
        self.reaction_count += 1;
        id
    }

    /// Structural check: a SortSpan re-sorts the next `len` commands by
    /// sampled z; the engine's SetDrawByZPosition has no notion of a span
    /// living inside another span's window, so this is forbidden. Returns
    /// `Err((drawable, outer_index, inner_index))` for the first nested
    /// span found (Snapshot-inside-SortSpan is fine - it sorts with the
    /// span). `Ok(())` when every span's window is span-free.
    pub fn find_nested_sort_span(&self) -> Result<(), (usize, usize, usize)> {
        for (d, drawable) in self.drawables.iter().enumerate() {
            let cmds = &drawable.commands;
            for (i, cmd) in cmds.iter().enumerate() {
                let Cmd::SortSpan { len } = cmd else { continue };
                let end = (i + 1 + *len as usize).min(cmds.len());
                for (j, inner) in cmds.iter().enumerate().take(end).skip(i + 1) {
                    if matches!(inner, Cmd::SortSpan { .. }) {
                        return Err((d, i, j));
                    }
                }
            }
        }
        Ok(())
    }

    /// Mint a drawable. A plain drawable is a sprite/actor surface: it is
    /// only as big as its content and composites transparent - it must never
    /// contribute an opaque region it does not own. Persistent drawables
    /// (engine AFT capture semantics) retain content across frames. A
    /// producer whose surface genuinely owns a backing (an AFT rig that
    /// blacks out) opts in via `set_drawable_clear`.
    pub fn add_drawable(&mut self, size: [f32; 2], persistent: bool, dynamic: bool) -> u32 {
        let id = self.drawables.len() as u32;
        self.drawables.push(Drawable {
            size,
            persistent,
            clear: if persistent {
                ClearMode::Retain
            } else {
                ClearMode::TransparentBlack
            },
            commands: Vec::new(),
            dynamic,
            inline: false,
        });
        id
    }

    pub fn set_drawable_clear(&mut self, id: u32, clear: ClearMode) {
        if let Some(d) = self.drawables.get_mut(id as usize) {
            d.clear = clear;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_item() -> Cmd {
        Cmd::Item(Item::of(Source::Image {
            image: 0,
            frame: ChannelRef::constant(0.0),
        }))
    }

    #[test]
    fn nested_sort_span_is_detected() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let cmds = &mut doc.drawables[0].commands;
        cmds.push(Cmd::SortSpan { len: 3 });
        cmds.push(image_item());
        cmds.push(Cmd::SortSpan { len: 1 }); // inner span inside the outer window
        cmds.push(image_item());
        assert_eq!(doc.find_nested_sort_span(), Err((0, 0, 2)));
    }

    #[test]
    fn adjacent_sort_spans_are_not_nested() {
        // A second span beginning AFTER the first's window is fine.
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let cmds = &mut doc.drawables[0].commands;
        cmds.push(Cmd::SortSpan { len: 1 });
        cmds.push(image_item());
        cmds.push(Cmd::SortSpan { len: 1 });
        cmds.push(image_item());
        assert!(doc.find_nested_sort_span().is_ok());
    }

    #[test]
    fn snapshot_inside_sort_span_is_allowed() {
        let mut doc = DrawableDoc::with_screen([640.0, 480.0]);
        let slot = doc.add_drawable([640.0, 480.0], true, false);
        let cmds = &mut doc.drawables[0].commands;
        cmds.push(Cmd::SortSpan { len: 2 });
        cmds.push(image_item());
        cmds.push(Cmd::Snapshot {
            into: slot,
            links: Vec::new(),
            flip_base_y: false,
        });
        assert!(doc.find_nested_sort_span().is_ok());
    }
}
