//! The Drawable document: Seam A's compiled form.
//!
//! Types follow .claude/plans/drawable-ir.md draft 3. Per-object
//! construction is allowed here (once per chart); nothing per-frame
//! touches these structures except reads.

use crate::camera::Mat4;
use crate::channels::{ChannelRef, ChannelTable};

pub const SCREEN: u32 = 0; // drawables[0] is always the screen (root)

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
}

impl Item {
    pub fn of(source: Source) -> Self {
        Item {
            source,
            transform: TransformRef::identity(),
            links: Vec::new(),
            flip_base_y: false,
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
        }
    }
}

#[derive(Clone, Debug)]
pub enum Cmd {
    Item(Item),
    /// At-position capture: copy this drawable's in-progress composite
    /// into another drawable now (AFT capture semantics).
    Snapshot { into: u32 },
    /// The next `len` commands re-sort stably by their sampled `z`
    /// before drawing (SetDrawByZPosition).
    SortSpan { len: u32 },
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
}

#[derive(Default)]
pub struct DrawableDoc {
    pub drawables: Vec<Drawable>,
    pub shaders: Vec<ShaderDesc>,
    pub clips: Vec<ClipDesc>,
    pub channels: ChannelTable,
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
        });
        doc
    }

    pub fn add_shader(&mut self, shader: ShaderDesc) -> u32 {
        let id = self.shaders.len() as u32;
        self.shaders.push(shader);
        id
    }

    pub fn add_clip(&mut self, clip: ClipDesc) -> u32 {
        let id = self.clips.len() as u32;
        self.clips.push(clip);
        id
    }

    pub fn add_drawable(&mut self, size: [f32; 2], persistent: bool, dynamic: bool) -> u32 {
        let id = self.drawables.len() as u32;
        self.drawables.push(Drawable {
            size,
            persistent,
            clear: if persistent {
                ClearMode::Retain
            } else {
                ClearMode::OpaqueBlack
            },
            commands: Vec::new(),
            dynamic,
        });
        id
    }
}
