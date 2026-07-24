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
//! (flags bit0 additive, bit1 screen_space, bit2 has_z); f32 stride 14
//! `[x, y, sx, sy, rot, opacity, r, g, b, crop_l, crop_t, crop_r,
//! crop_b, z]`. `feed_ids` / `feed_item_counts` are parallel; the lane
//! buffers concatenate items in that same order.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod camera;
mod channels;
mod doc;
mod evaluate;
mod schedule;
mod transform;

use crate::channels::ChannelRef;
use crate::doc::{
    Blend, CameraRef, ClipDesc, Cmd, DrawableDoc, Item, LinkRef, ShaderDesc, Source, Space,
    TransformRef,
};
use crate::evaluate::{
    evaluate, parse_feeds, Feed, FEED_F_STRIDE, FEED_U_STRIDE, F_STRIDE, U_STRIDE,
};
use crate::schedule::{lower, LoweredProp, Node, Target};

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
    fn channel(&mut self, ts: Vec<f32>, vals: Vec<f32>, durs: Vec<f32>, rest: f32) -> u32 {
        self.doc
            .as_mut()
            .expect("builder already finished")
            .channels
            .push(&ts, &vals, &durs, rest)
    }

    fn drawable(&mut self, w: f32, h: f32, persistent: bool, dynamic: bool) -> u32 {
        self.doc
            .as_mut()
            .expect("builder already finished")
            .add_drawable([w, h], persistent, dynamic)
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
                        flip_base_y=false))]
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
        };
        let item = self.last_item(target);
        item.links.push(link);
        item.flip_base_y = flip_base_y;
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
        self.last_item(target).projection = Some(CameraRef {
            fov_deg: chan(fov_id, fov_rest),
            vanish_x: chan(vanish_x_id, vanish_x_rest),
            vanish_y: chan(vanish_y_id, vanish_y_rest),
            far: chan(far_id, far_rest),
            w,
            h,
        });
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
        item.blend = if additive { Blend::Additive } else { Blend::SourceOver };
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
        let LoweredProp { ts, vals, durs } = prop;
        let rest = vals.first().copied().unwrap_or(0.0);
        let id = self
            .doc
            .as_mut()
            .expect("builder already finished")
            .channels
            .push(&ts, &vals, &durs, rest);
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

    fn snapshot(&mut self, target: u32, into: u32) {
        self.push(target, Cmd::Snapshot { into });
    }

    fn sort_span(&mut self, target: u32, len: u32) {
        self.push(target, Cmd::SortSpan { len });
    }

    /// Hand the finished doc to an evaluator (the builder empties).
    fn finish(&mut self) -> Evaluator {
        Evaluator {
            doc: self.doc.take().expect("builder already finished"),
        }
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
}

#[pyclass(unsendable)]
struct Evaluator {
    doc: DrawableDoc,
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
        self.emit(py, t, &[])
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
        self.emit(py, t, &feeds)
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
}

impl Evaluator {
    fn emit<'py>(
        &self,
        py: Python<'py>,
        t: f32,
        feeds: &[Feed],
    ) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize) {
        let schedule = evaluate(&self.doc, t, feeds);
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
    Ok(())
}
