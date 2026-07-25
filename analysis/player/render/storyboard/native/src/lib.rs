//! PyO3 extension: the game-agnostic Drawable core.
//!
//! Seam A: `DocBuilder` - a per-object builder Python drives ONCE per
//! chart (channels, drawables, commands). Seam B: `Evaluator.frame(t)`
//! returns the DrawSchedule as two flat buffers (u32 records + f32
//! records, fixed stride) - no per-op objects cross the boundary (the
//! no-heavy-marshalling rule). Dynamic feeds arrive as flat arrays.
//!
//! ## Shader-uniform buffer encoding (Seam B)
//!
//! `frame()` / `frame_with_feeds()` return a THIRD flat f32 buffer of
//! shader-uniform VALUES alongside the fixed-stride u32/f32 op records.
//! A BLIT op with a bound shader appends its sampled uniform values to
//! this buffer; u32 op lanes 8/9 are `(uniform_offset, uniform_count)`
//! indexing it. count == 0 means the op binds no uniforms. The fixed op
//! strides are NEVER widened by uniform data - variable-length uniforms
//! live only in this side buffer, so reshape(n, u_stride) /
//! reshape(n, f_stride) over the first two buffers stays valid. Uniform
//! value j maps to the item's j-th `(uniform_names index, channel)`
//! binding; the executor pairs it with `shaders[id].uniform_names[k]`
//! (names travel once at Seam A).
//!
//! ## Feed SoA layout (per-frame in)
//!
//! `frame_with_feeds` takes flat SoA feed buffers (FROZEN strides in
//! evaluate.rs): u32 stride 4 `[source_kind, source_id, frame, flags]`
//! (flags bit0 additive, bit1 screen_space, bit2 has_z); f32 stride 18
//! (feed v2) `[m00, m01, m02, m10, m11, m12, m20, m21, m22, opacity,
//! r, g, b, crop_l, crop_t, crop_r, crop_b, z]` - the mat3 in the BLIT
//! record's column-vector layout, written to the record verbatim
//! (homographies included; the game side resolves the full homography).
//! `feed_ids` / `feed_item_counts` are parallel; the lane buffers
//! concatenate items in that same order.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod camera;
mod channels;
mod doc;
mod ease;
mod evaluate;
mod schedule;
mod transform;

use crate::channels::ChannelRef;
use crate::doc::{
    CameraRef, ClipDesc, Cmd, DrawableDoc, Item, LinkRef, MeshDesc, ShaderDesc, Source,
    Space, TransformRef,
};
use crate::doc::Reaction;
use crate::evaluate::{
    evaluate_with_events, parse_feeds, Event, Feed, ReactionCache, FEED_F_STRIDE, FEED_U_STRIDE,
    F_STRIDE, U_STRIDE,
};
use crate::schedule::{lower, LoweredProp, Node, Target};
use std::cell::RefCell;

/// Decode a (id, rest) pair from Python: id < 0 means "no channel".
fn chan(id: i64, rest: f32) -> ChannelRef {
    if id < 0 {
        ChannelRef::constant(rest)
    } else {
        ChannelRef { id: id as u32, rest }
    }
}

/// Build a schedule `Node` from a Python mapping in the A2 fixture shape:
///   {"kind": "Seg"|"Seq"|"Hibernate"|"Loop", ...}
///     Seg:       dur, ease, targets:[{prop, mode:"abs"|"add", value}],
///                effect: <node>|None, fire_id: int (default -1)
///     Seq:       parts: [<node>, ...]
///     Hibernate: dur
///     Loop:      period, body: <node>
fn node_from_py(node: &Bound<'_, PyAny>) -> PyResult<Node> {
    let kind: String = node.get_item("kind")?.extract()?;
    match kind.as_str() {
        "Seg" => {
            let dur: f32 = node.get_item("dur")?.extract()?;
            let ease: i32 = node.get_item("ease")?.extract()?;
            let targets_any = node.get_item("targets")?;
            let mut targets = Vec::new();
            for tgt in targets_any.try_iter()? {
                let tgt = tgt?;
                let prop: u32 = tgt.get_item("prop")?.extract()?;
                let value: f32 = tgt.get_item("value")?.extract()?;
                let mode: String = tgt.get_item("mode")?.extract()?;
                let target = match mode.as_str() {
                    "add" => Target::Add(value),
                    _ => Target::Abs(value),
                };
                targets.push((prop, target));
            }
            let effect = match node.get_item("effect") {
                Ok(e) if !e.is_none() => Some(Box::new(node_from_py(&e)?)),
                _ => None,
            };
            let fire_id: i64 = match node.get_item("fire_id") {
                Ok(v) if !v.is_none() => v.extract()?,
                _ => -1,
            };
            Ok(Node::Seg { dur, ease, targets, effect, fire_id })
        }
        "Seq" => {
            let parts_any = node.get_item("parts")?;
            let mut parts = Vec::new();
            for part in parts_any.try_iter()? {
                parts.push(node_from_py(&part?)?);
            }
            Ok(Node::Seq(parts))
        }
        "Hibernate" => Ok(Node::Hibernate(node.get_item("dur")?.extract()?)),
        "Loop" => Ok(Node::Loop {
            period: node.get_item("period")?.extract()?,
            body: Box::new(node_from_py(&node.get_item("body")?)?),
        }),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown schedule node kind {other}"
        ))),
    }
}

#[pyclass(unsendable)]
struct DocBuilder {
    doc: Option<DrawableDoc>,
}

#[pymethods]
impl DocBuilder {
    #[new]
    fn new(screen_w: f32, screen_h: f32) -> Self {
        DocBuilder {
            doc: Some(DrawableDoc::with_screen([screen_w, screen_h])),
        }
    }

    /// Append a channel from parallel breakpoint lists; returns its id.
    /// `eases` (optional) carries one ease id per breakpoint shaping the
    /// ramp toward the next value (see `ease.rs`); omitted / empty means
    /// all-linear, which is bit-identical to the pre-ease behavior. A
    /// non-empty `eases` must run parallel to `ts`.
    #[pyo3(signature = (ts, vals, durs, rest, eases=Vec::new()))]
    fn channel(
        &mut self,
        ts: Vec<f32>,
        vals: Vec<f32>,
        durs: Vec<f32>,
        rest: f32,
        eases: Vec<i32>,
    ) -> u32 {
        let channels = &mut self
            .doc
            .as_mut()
            .expect("builder already finished")
            .channels;
        if eases.is_empty() {
            channels.push(&ts, &vals, &durs, rest)
        } else {
            channels.push_eased(&ts, &vals, &durs, &eases, rest)
        }
    }

    #[pyo3(signature = (w, h, persistent, dynamic, clear=None))]
    fn drawable(
        &mut self,
        w: f32,
        h: f32,
        persistent: bool,
        dynamic: bool,
        clear: Option<&str>,
    ) -> u32 {
        let doc = self.doc.as_mut().expect("builder already finished");
        let id = doc.add_drawable([w, h], persistent, dynamic);
        // Explicit backing override ('transparent' | 'opaque' | 'retain');
        // None keeps the derived default (persistent -> Retain, else
        // TransparentBlack - a plain drawable owns no backing).
        if let Some(mode) = clear.and_then(|s| match s {
            "transparent" => Some(crate::doc::ClearMode::TransparentBlack),
            "opaque" => Some(crate::doc::ClearMode::OpaqueBlack),
            "retain" => Some(crate::doc::ClearMode::Retain),
            _ => None,
        }) {
            doc.set_drawable_clear(id, mode);
        }
        id
    }

    /// Mint a FEED SLOT: a drawable that is never a render target of its
    /// own. Its per-frame items draw inline wherever `feed_inline` places
    /// it, so they are bounded only by the real target - the whole point
    /// for notes, which mods push outside any fixed box.
    fn feed_slot(&mut self) -> u32 {
        let doc = self.doc.as_mut().expect("builder already finished");
        let id = doc.add_drawable([0.0, 0.0], false, true);
        doc.drawables[id as usize].inline = true;
        id
    }

    /// Draw `slot`'s fed items at this point in `target`'s command stream.
    /// `visible_id`/`visible_rest` gate the whole feed (an item `visible`
    /// channel: < 0.5 emits nothing). Follow with `item_link` calls to
    /// attach a consumer transform chain - the chain's H then composes
    /// over each fed item's mat3 (a proxy/player field re-render).
    #[pyo3(signature = (target, slot, visible_id=-1, visible_rest=1.0,
                        z_id=-1, z_rest=0.0, has_z=false))]
    fn feed_inline(&mut self, target: u32, slot: u32, visible_id: i64, visible_rest: f32,
                   z_id: i64, z_rest: f32, has_z: bool) {
        let doc = self.doc.as_mut().expect("builder already finished");
        doc.drawables[target as usize]
            .commands
            .push(crate::doc::Cmd::Feed {
                slot,
                links: Vec::new(),
                flip_base_y: false,
                visible: chan(visible_id, visible_rest),
                projection: None,
                z: has_z.then(|| chan(z_id, z_rest)),
            });
    }

    /// Register a shader; returns its id. `uniform_names` orders the
    /// per-item uniform bindings (see `item_uniform`).
    #[pyo3(signature = (frag, vert=None, uniform_names=Vec::new()))]
    fn shader(&mut self, frag: String, vert: Option<String>, uniform_names: Vec<String>) -> u32 {
        self.doc
            .as_mut()
            .expect("builder already finished")
            .add_shader(ShaderDesc {
                frag,
                vert,
                uniform_names,
            })
    }

    /// Register a mesh; returns its id (for `Source::Mesh` items via
    /// `item(..., source_kind=SRC_MESH, source_id=<this id>, ...)`).
    /// `vertices` is flat interleaved [x, y, u, v] (4 floats/vertex);
    /// `mode` is the executor's primitive mode; `vert_shader_id` (< 0 =
    /// none) and `vert_source` (None = none) are the optional vertex-shader
    /// program. The GL DRAW of the mesh is the executor tier.
    #[pyo3(signature = (vertices, mode=0, vert_shader_id=-1, vert_source=None))]
    fn mesh(
        &mut self,
        vertices: Vec<f32>,
        mode: u32,
        vert_shader_id: i64,
        vert_source: Option<String>,
    ) -> u32 {
        self.doc
            .as_mut()
            .expect("builder already finished")
            .add_mesh(MeshDesc {
                vertices,
                mode,
                vert_shader: (vert_shader_id >= 0).then_some(vert_shader_id as u32),
                vert_source,
            })
    }

    /// Register a rectangle clip (l, t, r, b, logical units); returns id.
    fn clip_rect(&mut self, l: f32, t: f32, r: f32, b: f32) -> u32 {
        self.doc
            .as_mut()
            .expect("builder already finished")
            .add_clip(ClipDesc::Rect([l, t, r, b]))
    }

    /// Register a polygon clip from flat [x0,y0,x1,y1,...]; returns id.
    fn clip_polygon(&mut self, points: Vec<f32>) -> u32 {
        let verts = points.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        self.doc
            .as_mut()
            .expect("builder already finished")
            .add_clip(ClipDesc::Polygon(verts))
    }

    /// Attach a shader (and optional clip) to the item most recently
    /// pushed onto `target`. Panics if the last command isn't an Item.
    #[pyo3(signature = (target, shader_id, clip_id=-1))]
    fn item_shader(&mut self, target: u32, shader_id: i64, clip_id: i64) {
        let item = self.last_item(target);
        item.shader = Some(shader_id as u32);
        if clip_id >= 0 {
            item.clip = Some(clip_id as u32);
        }
    }

    /// Bind one shader uniform on the item most recently pushed onto
    /// `target`: `name_idx` into the shader's uniform_names, sampled from
    /// the (id, rest) channel each frame. Values ride the uniform buffer
    /// in binding order (see the module docstring).
    #[pyo3(signature = (target, name_idx, ch_id=-1, ch_rest=0.0))]
    fn item_uniform(&mut self, target: u32, name_idx: u16, ch_id: i64, ch_rest: f32) {
        let ref_ = chan(ch_id, ch_rest);
        self.last_item(target).uniforms.push((name_idx, ref_));
    }

    /// Set a clip on the last-pushed item without a shader.
    fn item_clip(&mut self, target: u32, clip_id: u32) {
        self.last_item(target).clip = Some(clip_id);
    }

    /// Tag the item most recently pushed onto `target` with an opaque id
    /// written to a record lane. Diagnostics only - it never affects drawing.
    ///
    /// A differential tool needs to pair a per-frame record back to the item
    /// that produced it, and build order cannot do that: visibility gates and
    /// time windows cull items, so the Nth record is not the Nth item.
    fn item_tag(&mut self, target: u32, tag: u32) {
        self.last_item(target).tag = tag;
    }

    /// Draw-box origin and absolute size on the item most recently pushed
    /// onto `target`.
    ///
    /// `origin` is a fraction of the item's own drawn size, subtracted from
    /// the quad before the transform - SM's `translate(-origin*w, -origin*h)`.
    /// It rests at the TOP-LEFT, so an actor the chart centres (the SM
    /// default, origin 0.5/0.5) must set it or it draws half its size off.
    ///
    /// `size_*` overrides the source's natural box per axis, sampled each
    /// frame; negative keeps the natural size. `zoomto`/`setsize` REPLACE the
    /// basis that scale then multiplies, so this is not another scale lane.
    #[pyo3(signature = (target, origin_x=0.0, origin_y=0.0,
                        size_x_id=-1, size_x_rest=-1.0,
                        size_y_id=-1, size_y_rest=-1.0))]
    fn item_box(
        &mut self,
        target: u32,
        origin_x: f32,
        origin_y: f32,
        size_x_id: i64,
        size_x_rest: f32,
        size_y_id: i64,
        size_y_rest: f32,
    ) {
        let item = self.last_item(target);
        item.origin = [origin_x, origin_y];
        item.size = [chan(size_x_id, size_x_rest), chan(size_y_id, size_y_rest)];
    }

    /// SM SetFadeLeft/Right/Top/Bottom on the item most recently pushed onto
    /// `target`: each a fraction of the drawn box over which alpha ramps from
    /// 0 at that edge up to 1 inward. All rest at 0 (a hard edge), and the
    /// ramps multiply where they overlap.
    #[pyo3(signature = (target, l_id=-1, l_rest=0.0, r_id=-1, r_rest=0.0,
                        t_id=-1, t_rest=0.0, b_id=-1, b_rest=0.0))]
    fn item_fade(
        &mut self,
        target: u32,
        l_id: i64,
        l_rest: f32,
        r_id: i64,
        r_rest: f32,
        t_id: i64,
        t_rest: f32,
        b_id: i64,
        b_rest: f32,
    ) {
        self.last_item(target).fade = [
            chan(l_id, l_rest),
            chan(r_id, r_rest),
            chan(t_id, t_rest),
            chan(b_id, b_rest),
        ];
    }

    /// ScaleToCover / ScaleToFitInside on the item most recently pushed onto
    /// `target`: the uniform zoom that fits the source's natural box to a
    /// recorded rect. `mode` < 0.5 is off, 1.0 is cover, else fit-inside.
    ///
    /// Only the rect's EXTENT is sent - the zoom is `rect/natural` per axis
    /// and then the larger or smaller of the two, so the corners never
    /// matter. The natural size lives in the executor, which is why this
    /// cannot fold into `item_box`'s absolute size.
    #[pyo3(signature = (target, mode_id=-1, mode_rest=0.0, w_id=-1, w_rest=0.0,
                        h_id=-1, h_rest=0.0))]
    fn item_fit(
        &mut self,
        target: u32,
        mode_id: i64,
        mode_rest: f32,
        w_id: i64,
        w_rest: f32,
        h_id: i64,
        h_rest: f32,
    ) {
        self.last_item(target).fit = [
            chan(mode_id, mode_rest),
            chan(w_id, w_rest),
            chan(h_id, h_rest),
        ];
    }

    /// Additive-blend gate on the item most recently pushed onto `target`,
    /// sampled every frame (>= 0.5 additive). Use instead of `item`'s
    /// `additive=` flag whenever the chart can change blending at runtime -
    /// the flag bakes one mode for the whole chart.
    #[pyo3(signature = (target, ch_id=-1, ch_rest=0.0))]
    fn item_blend(&mut self, target: u32, ch_id: i64, ch_rest: f32) {
        self.last_item(target).blend_add = chan(ch_id, ch_rest);
    }

    /// Per-channel diffuse tint on the item most recently pushed onto
    /// `target`, sampled every frame. Rests at white, so an item that
    /// never calls this composes untinted.
    ///
    /// A Quad's diffuse IS its colour in the engine, and the AFT-rig idiom
    /// puts a `diffuse,0,0,0,1` curtain between a capture node and its
    /// samplers - untinted that curtain composes WHITE and paints over the
    /// rig instead of masking it.
    #[pyo3(signature = (target, r_id=-1, r_rest=1.0, g_id=-1, g_rest=1.0,
                        b_id=-1, b_rest=1.0))]
    fn item_tint(
        &mut self,
        target: u32,
        r_id: i64,
        r_rest: f32,
        g_id: i64,
        g_rest: f32,
        b_id: i64,
        b_rest: f32,
    ) {
        self.last_item(target).tint =
            [chan(r_id, r_rest), chan(g_id, g_rest), chan(b_id, b_rest)];
    }

    /// Attach an event-driven drawing reaction to the item most recently
    /// pushed onto `target`. On a frame whose events include one of kind
    /// `trigger_kind` passing `column_filter` (-1 = any column), the
    /// Schedule fragment `node` (A2 fixture shape) lowers at the event
    /// time and splices its `prop` breakpoint run over the item's base
    /// draw property for t >= that time (latest matching event wins).
    /// `prop` is one of the PROP_* constants (opacity / tint_r,g,b / zoom
    /// / frame). Panics if the last command isn't an Item.
    #[pyo3(signature = (target, trigger_kind, column_filter, node, prop))]
    fn item_reaction(
        &mut self,
        target: u32,
        trigger_kind: u32,
        column_filter: i32,
        node: &Bound<'_, PyAny>,
        prop: u32,
    ) -> PyResult<()> {
        let fragment = node_from_py(node)?;
        let id = self
            .doc
            .as_mut()
            .expect("builder already finished")
            .next_reaction_id();
        self.last_item(target).reactions.push(Reaction {
            id,
            trigger_kind,
            column_filter,
            fragment,
            prop,
        });
        Ok(())
    }

    /// Append one link of the full leaf-link transform chain (root-first)
    /// to the item most recently pushed onto `target`. Each scalar is an
    /// (id, rest) channel pair (id<0 = constant); defaults are the SM link
    /// rests an untouched link carries. An item with >= 1 link uses the
    /// full `transform::compose_links` path instead of the first-cut TRS.
    /// `flip_base_y` mirrors the LEAF link's vertical source axis (AFT
    /// capture compensation) - the last item_link call's value wins.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (target,
                        x_id=-1, x_rest=0.0, y_id=-1, y_rest=0.0,
                        zoom_x_id=-1, zoom_x_rest=1.0, zoom_y_id=-1, zoom_y_rest=1.0,
                        rot_id=-1, rot_rest=0.0,
                        skew_x_id=-1, skew_x_rest=0.0, skew_y_id=-1, skew_y_rest=0.0,
                        base_scale_x_id=-1, base_scale_x_rest=1.0,
                        base_scale_y_id=-1, base_scale_y_rest=1.0,
                        halign_id=-1, halign_rest=0.5, valign_id=-1, valign_rest=0.5,
                        hidden_id=-1, hidden_rest=0.0, alpha_id=-1, alpha_rest=1.0,
                        crop_l_id=-1, crop_l_rest=0.0, crop_t_id=-1, crop_t_rest=0.0,
                        crop_r_id=-1, crop_r_rest=0.0, crop_b_id=-1, crop_b_rest=0.0,
                        natural_w_id=-1, natural_w_rest=640.0,
                        natural_h_id=-1, natural_h_rest=480.0,
                        flip_base_y=false,
                        rotation_x_id=-1, rotation_x_rest=0.0,
                        rotation_y_id=-1, rotation_y_rest=0.0,
                        z_id=-1, z_rest=0.0,
                        scale_z_id=-1, scale_z_rest=1.0,
                        base_scale_z_id=-1, base_scale_z_rest=1.0,
                        rotation_order="xyz"))]
    fn item_link(
        &mut self,
        target: u32,
        x_id: i64,
        x_rest: f32,
        y_id: i64,
        y_rest: f32,
        zoom_x_id: i64,
        zoom_x_rest: f32,
        zoom_y_id: i64,
        zoom_y_rest: f32,
        rot_id: i64,
        rot_rest: f32,
        skew_x_id: i64,
        skew_x_rest: f32,
        skew_y_id: i64,
        skew_y_rest: f32,
        base_scale_x_id: i64,
        base_scale_x_rest: f32,
        base_scale_y_id: i64,
        base_scale_y_rest: f32,
        halign_id: i64,
        halign_rest: f32,
        valign_id: i64,
        valign_rest: f32,
        hidden_id: i64,
        hidden_rest: f32,
        alpha_id: i64,
        alpha_rest: f32,
        crop_l_id: i64,
        crop_l_rest: f32,
        crop_t_id: i64,
        crop_t_rest: f32,
        crop_r_id: i64,
        crop_r_rest: f32,
        crop_b_id: i64,
        crop_b_rest: f32,
        natural_w_id: i64,
        natural_w_rest: f32,
        natural_h_id: i64,
        natural_h_rest: f32,
        flip_base_y: bool,
        rotation_x_id: i64,
        rotation_x_rest: f32,
        rotation_y_id: i64,
        rotation_y_rest: f32,
        z_id: i64,
        z_rest: f32,
        scale_z_id: i64,
        scale_z_rest: f32,
        base_scale_z_id: i64,
        base_scale_z_rest: f32,
        rotation_order: &str,
    ) {
        let link = LinkRef {
            x: chan(x_id, x_rest),
            y: chan(y_id, y_rest),
            zoom_x: chan(zoom_x_id, zoom_x_rest),
            zoom_y: chan(zoom_y_id, zoom_y_rest),
            rot: chan(rot_id, rot_rest),
            skew_x: chan(skew_x_id, skew_x_rest),
            skew_y: chan(skew_y_id, skew_y_rest),
            base_scale_x: chan(base_scale_x_id, base_scale_x_rest),
            base_scale_y: chan(base_scale_y_id, base_scale_y_rest),
            halign: chan(halign_id, halign_rest),
            valign: chan(valign_id, valign_rest),
            hidden: chan(hidden_id, hidden_rest),
            alpha: chan(alpha_id, alpha_rest),
            crop: [
                chan(crop_l_id, crop_l_rest),
                chan(crop_t_id, crop_t_rest),
                chan(crop_r_id, crop_r_rest),
                chan(crop_b_id, crop_b_rest),
            ],
            natural_w: chan(natural_w_id, natural_w_rest),
            natural_h: chan(natural_h_id, natural_h_rest),
            rotation_x: chan(rotation_x_id, rotation_x_rest),
            rotation_y: chan(rotation_y_id, rotation_y_rest),
            z: chan(z_id, z_rest),
            scale_z: chan(scale_z_id, scale_z_rest),
            base_scale_z: chan(base_scale_z_id, base_scale_z_rest),
            // An unrecognized token falls back to the stock RageMatrix order
            // rather than failing the build (parity with the Python reader).
            rotation_order: crate::camera::RotOrder::from_str(rotation_order)
                .unwrap_or(crate::camera::RotOrder::Xyz),
        };
        let (links, flip) = self.last_links(target);
        links.push(link);
        *flip = flip_base_y;
    }

    /// Attach a channel-backed camera projection to the item most recently
    /// pushed onto `target`: fov (degrees), vanishing point (px), and far
    /// plane, each an (id, rest) channel. `w`/`h` are the design viewport
    /// (default SM 640x480). The projection folds onto the item's 2D mat3
    /// each frame (row/col collapse -> a projective homography).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (target, fov_id=-1, fov_rest=45.0,
                        vanish_x_id=-1, vanish_x_rest=320.0,
                        vanish_y_id=-1, vanish_y_rest=240.0,
                        far_id=-1, far_rest=1772.7,
                        w=640.0, h=480.0))]
    fn item_projection(
        &mut self,
        target: u32,
        fov_id: i64,
        fov_rest: f32,
        vanish_x_id: i64,
        vanish_x_rest: f32,
        vanish_y_id: i64,
        vanish_y_rest: f32,
        far_id: i64,
        far_rest: f32,
        w: f32,
        h: f32,
    ) {
        let camera = Some(CameraRef {
            fov_deg: chan(fov_id, fov_rest),
            vanish_x: chan(vanish_x_id, vanish_x_rest),
            vanish_y: chan(vanish_y_id, vanish_y_rest),
            far: chan(far_id, far_rest),
            w,
            h,
        });
        *self.last_projection(target) = camera;
    }

    /// Append one item command. Channel args are (id, rest) with id<0
    /// meaning constant. `source_kind`/`source_id` follow the schedule
    /// record encoding; `z_id` gates SortSpan participation.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (target, source_kind, source_id, frame_id=-1, frame_rest=0.0,
                        x_id=-1, x_rest=0.0, y_id=-1, y_rest=0.0,
                        sx_id=-1, sx_rest=1.0, sy_id=-1, sy_rest=1.0,
                        rot_id=-1, rot_rest=0.0,
                        opacity_id=-1, opacity_rest=1.0,
                        visible_id=-1, visible_rest=1.0,
                        additive=false, screen_space=false,
                        z_id=-1, z_rest=0.0, has_z=false))]
    fn item(
        &mut self,
        target: u32,
        source_kind: u32,
        source_id: u32,
        frame_id: i64,
        frame_rest: f32,
        x_id: i64,
        x_rest: f32,
        y_id: i64,
        y_rest: f32,
        sx_id: i64,
        sx_rest: f32,
        sy_id: i64,
        sy_rest: f32,
        rot_id: i64,
        rot_rest: f32,
        opacity_id: i64,
        opacity_rest: f32,
        visible_id: i64,
        visible_rest: f32,
        additive: bool,
        screen_space: bool,
        z_id: i64,
        z_rest: f32,
        has_z: bool,
    ) {
        let source = match source_kind {
            evaluate::SRC_IMAGE => Source::Image {
                image: source_id,
                frame: chan(frame_id, frame_rest),
            },
            evaluate::SRC_DRAWABLE => Source::Drawable(source_id),
            evaluate::SRC_MESH => Source::Mesh(source_id),
            evaluate::SRC_LINES => Source::Lines(source_id),
            _ => Source::Fill,
        };
        let mut item = Item::of(source);
        item.transform = TransformRef {
            x: chan(x_id, x_rest),
            y: chan(y_id, y_rest),
            scale_x: chan(sx_id, sx_rest),
            scale_y: chan(sy_id, sy_rest),
            rot: chan(rot_id, rot_rest),
        };
        item.opacity = chan(opacity_id, opacity_rest);
        item.visible = chan(visible_id, visible_rest);
        item.blend_add = ChannelRef::constant(additive as u32 as f32);
        item.space = if screen_space { Space::Screen } else { Space::Scene };
        item.z = has_z.then(|| chan(z_id, z_rest));
        self.push(target, Cmd::Item(item));
    }

    /// Lower a schedule tree (the A2 fixture JSON shape, passed as a Python
    /// dict) and push property `prop_key`'s breakpoint run as a doc channel;
    /// returns its id, or None when the schedule never touches that prop.
    /// `horizon <= 0` (or non-finite) means unbounded, matching the Python
    /// `None` horizon; a Loop requires a finite horizon.
    #[pyo3(signature = (node, t0, horizon, prop_key, state=Vec::new()))]
    fn schedule_channel(
        &mut self,
        node: &Bound<'_, PyAny>,
        t0: f32,
        horizon: f32,
        prop_key: u32,
        state: Vec<(u32, f32)>,
    ) -> PyResult<Option<u32>> {
        let root = node_from_py(node)?;
        let horizon = if horizon > 0.0 { horizon } else { f32::INFINITY };
        let out = lower(&root, t0, horizon, &state);
        let Some((_, prop)) = out.props.into_iter().find(|(k, _)| *k == prop_key) else {
            return Ok(None);
        };
        let LoweredProp { ts, vals, durs, eases } = prop;
        let rest = vals.first().copied().unwrap_or(0.0);
        let id = self
            .doc
            .as_mut()
            .expect("builder already finished")
            .channels
            .push_eased(&ts, &vals, &durs, &eases, rest);
        Ok(Some(id))
    }

    /// Lower a schedule tree (A2 fixture shape) and return its effect fire
    /// times in time order. Reactions consume these in a later wave.
    #[pyo3(signature = (node, t0, horizon, state=Vec::new()))]
    fn schedule_fires(
        &self,
        node: &Bound<'_, PyAny>,
        t0: f32,
        horizon: f32,
        state: Vec<(u32, f32)>,
    ) -> PyResult<Vec<f32>> {
        let root = node_from_py(node)?;
        let horizon = if horizon > 0.0 { horizon } else { f32::INFINITY };
        Ok(lower(&root, t0, horizon, &state).fires)
    }

    #[pyo3(signature = (target, into, z_id=-1, z_rest=0.0, has_z=false))]
    fn snapshot(&mut self, target: u32, into: u32, z_id: i64, z_rest: f32, has_z: bool) {
        self.push(target, Cmd::Snapshot {
            into,
            links: Vec::new(),
            flip_base_y: false,
            z: has_z.then(|| chan(z_id, z_rest)),
        });
    }

    fn sort_span(&mut self, target: u32, len: u32) {
        self.push(target, Cmd::SortSpan { len });
    }

    /// Hand the finished doc to an evaluator (the builder empties). Rejects
    /// structurally invalid docs (nested SortSpans - the engine has no such
    /// construct) before the doc can reach the per-frame evaluator.
    fn finish(&mut self) -> PyResult<Evaluator> {
        let doc = self.doc.take().expect("builder already finished");
        if let Err((d, outer, inner)) = doc.find_nested_sort_span() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "nested SortSpan on drawable {d}: the span at command {outer} covers \
                 another SortSpan at command {inner} (the engine has no such construct)"
            )));
        }
        Ok(Evaluator {
            doc,
            reaction_cache: RefCell::new(ReactionCache::new()),
        })
    }
}

impl DocBuilder {
    fn push(&mut self, target: u32, cmd: Cmd) {
        self.doc.as_mut().expect("builder already finished").drawables[target as usize]
            .commands
            .push(cmd);
    }

    /// The Item at the tail of `target`'s command list, for post-hoc
    /// attachment (shader/clip/uniforms). Panics if the tail isn't one.
    fn last_item(&mut self, target: u32) -> &mut Item {
        let commands =
            &mut self.doc.as_mut().expect("builder already finished").drawables[target as usize].commands;
        match commands.last_mut() {
            Some(Cmd::Item(item)) => item,
            _ => panic!("last command on target {target} is not an Item"),
        }
    }

    /// The camera slot of the tail command - an Item or a Feed. A linked
    /// feed is a field copy re-rendering the notes, so its chain takes a
    /// perspective fold exactly as an item's does; restricting this to Items
    /// left the inline-notes path (the default) unable to carry one at all.
    /// Panics if the tail is neither.
    fn last_projection(&mut self, target: u32) -> &mut Option<CameraRef> {
        let commands =
            &mut self.doc.as_mut().expect("builder already finished").drawables[target as usize].commands;
        match commands.last_mut() {
            Some(Cmd::Item(item)) => &mut item.projection,
            Some(Cmd::Feed { projection, .. }) => projection,
            _ => panic!("last command on target {target} takes no projection"),
        }
    }

    /// The link chain + flip slot of the tail command - an Item or a Feed
    /// (`item_link` attaches to either). Panics if the tail is neither.
    fn last_links(&mut self, target: u32) -> (&mut Vec<LinkRef>, &mut bool) {
        let commands =
            &mut self.doc.as_mut().expect("builder already finished").drawables[target as usize].commands;
        match commands.last_mut() {
            Some(Cmd::Item(item)) => (&mut item.links, &mut item.flip_base_y),
            Some(Cmd::Feed { links, flip_base_y, .. }) => (links, flip_base_y),
            Some(Cmd::Snapshot { links, flip_base_y, .. }) => (links, flip_base_y),
            _ => panic!("last command on target {target} takes no links"),
        }
    }
}

#[pyclass(unsendable)]
struct Evaluator {
    doc: DrawableDoc,
    /// Persistent reaction lowering cache: a (reaction id, event-time)
    /// fragment lowers once, ever (a Press at te=1.0 folds a single time
    /// across all frames that see it). RefCell because `frame` is `&self`.
    reaction_cache: RefCell<ReactionCache>,
}

#[pymethods]
impl Evaluator {
    /// Evaluate one frame; returns (u32_records_bytes, f32_records_bytes,
    /// uniform_values_bytes, op_count). Fixed strides: U_STRIDE u32 lanes
    /// + F_STRIDE f32 lanes per op - view with
    /// numpy.frombuffer(...).reshape(n, stride). The third buffer is the
    /// flat f32 shader-uniform values, indexed by op lanes 8/9 (see the
    /// module docstring).
    fn frame<'py>(
        &self,
        py: Python<'py>,
        t: f32,
    ) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize) {
        self.emit(py, t, &[], &[])
    }

    /// Same as `frame`, but carries this frame's input events (SoA
    /// parallel arrays; additive alongside frame/frame_with_feeds). An
    /// item's reactions splice their lowered Schedule fragment over the
    /// named prop when a matching event (kind + column filter) has arrived
    /// by `t`, latest wins. `event_kinds[i]` / `event_times[i]` /
    /// `event_columns[i]` / `event_strengths[i]` are the i-th event's
    /// fields (column -1 = no column; strength reserved this wave).
    #[pyo3(signature = (t, event_kinds, event_times, event_columns, event_strengths))]
    fn frame_with_events<'py>(
        &self,
        py: Python<'py>,
        t: f32,
        event_kinds: Vec<u32>,
        event_times: Vec<f32>,
        event_columns: Vec<i32>,
        event_strengths: Vec<f32>,
    ) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize) {
        let events = build_events(&event_kinds, &event_times, &event_columns, &event_strengths);
        self.emit(py, t, &[], &events)
    }

    /// Same as `frame`, but injects dynamic-drawable command feeds parsed
    /// from flat SoA buffers (frozen strides FEED_U_STRIDE / FEED_F_STRIDE).
    /// `feed_ids[i]` is the target drawable, `feed_item_counts[i]` its item
    /// count; `feed_u` / `feed_f` are the concatenated lane buffers.
    #[pyo3(signature = (t, feed_ids, feed_item_counts, feed_u, feed_f))]
    fn frame_with_feeds<'py>(
        &self,
        py: Python<'py>,
        t: f32,
        feed_ids: Vec<u32>,
        feed_item_counts: Vec<u32>,
        feed_u: &[u8],
        feed_f: &[u8],
    ) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize) {
        let u32s = cast_bytes_u32(feed_u);
        let f32s = cast_bytes_f32(feed_f);
        let owned = parse_feeds(&feed_ids, &feed_item_counts, &u32s, &f32s);
        let feeds: Vec<Feed> = owned
            .iter()
            .map(|fi| Feed {
                drawable: fi.drawable,
                items: &fi.items,
            })
            .collect();
        self.emit(py, t, &feeds, &[])
    }

    #[getter]
    fn u_stride(&self) -> usize {
        U_STRIDE
    }

    #[getter]
    fn f_stride(&self) -> usize {
        F_STRIDE
    }

    #[getter]
    fn feed_u_stride(&self) -> usize {
        FEED_U_STRIDE
    }

    #[getter]
    fn feed_f_stride(&self) -> usize {
        FEED_F_STRIDE
    }

    fn drawable_count(&self) -> usize {
        self.doc.drawables.len()
    }

    fn mesh_count(&self) -> usize {
        self.doc.meshes.len()
    }
}

impl Evaluator {
    fn emit<'py>(
        &self,
        py: Python<'py>,
        t: f32,
        feeds: &[Feed],
        events: &[Event],
    ) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize) {
        let mut cache = self.reaction_cache.borrow_mut();
        let schedule = evaluate_with_events(&self.doc, t, feeds, events, &mut cache);
        let n = schedule.len();
        let u_bytes = PyBytes::new(py, bytemuck_cast_u32(&schedule.u));
        let f_bytes = PyBytes::new(py, bytemuck_cast_f32(&schedule.f));
        let uf_bytes = PyBytes::new(py, bytemuck_cast_f32(&schedule.uf));
        (u_bytes, f_bytes, uf_bytes, n)
    }
}

fn bytemuck_cast_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytemuck_cast_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// Reinterpret an incoming byte buffer as u32 lanes (copies, since the
/// Python bytes may be unaligned to 4). Length must be a multiple of 4.
fn cast_bytes_u32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cast_bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Zip the SoA event arrays into `Event`s. The four arrays are parallel;
/// the shortest bounds the count so a short array never reads past its
/// end (defensive at the Python boundary).
fn build_events(kinds: &[u32], times: &[f32], columns: &[i32], strengths: &[f32]) -> Vec<Event> {
    let n = kinds
        .len()
        .min(times.len())
        .min(columns.len())
        .min(strengths.len());
    (0..n)
        .map(|i| Event {
            kind: kinds[i],
            time: times[i],
            column: columns[i],
            strength: strengths[i],
        })
        .collect()
}

#[pymodule]
fn storyboard_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DocBuilder>()?;
    m.add_class::<Evaluator>()?;
    m.add("SRC_IMAGE", evaluate::SRC_IMAGE)?;
    m.add("SRC_DRAWABLE", evaluate::SRC_DRAWABLE)?;
    m.add("SRC_MESH", evaluate::SRC_MESH)?;
    m.add("SRC_FILL", evaluate::SRC_FILL)?;
    m.add("SRC_LINES", evaluate::SRC_LINES)?;
    m.add("OP_BEGIN", evaluate::OP_BEGIN)?;
    m.add("OP_BLIT", evaluate::OP_BLIT)?;
    m.add("OP_COPY", evaluate::OP_COPY)?;
    m.add("OP_END", evaluate::OP_END)?;
    m.add("EV_PRESS", evaluate::EV_PRESS)?;
    m.add("EV_RELEASE", evaluate::EV_RELEASE)?;
    m.add("EV_HIT", evaluate::EV_HIT)?;
    m.add("EV_MISS", evaluate::EV_MISS)?;
    m.add("EV_CLICK", evaluate::EV_CLICK)?;
    m.add("EV_FRAMETICK", evaluate::EV_FRAMETICK)?;
    m.add("PROP_OPACITY", doc::PROP_OPACITY)?;
    m.add("PROP_TINT_R", doc::PROP_TINT_R)?;
    m.add("PROP_TINT_G", doc::PROP_TINT_G)?;
    m.add("PROP_TINT_B", doc::PROP_TINT_B)?;
    m.add("PROP_ZOOM", doc::PROP_ZOOM)?;
    m.add("PROP_FRAME", doc::PROP_FRAME)?;
    Ok(())
}
