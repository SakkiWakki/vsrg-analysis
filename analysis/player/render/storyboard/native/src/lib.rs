//! PyO3 extension: the game-agnostic Drawable core.
//!
//! Seam A: `DocBuilder` - a per-object builder Python drives ONCE per
//! chart (channels, drawables, commands). Seam B: `Evaluator.frame(t)`
//! returns the DrawSchedule as two flat buffers (u32 records + f32
//! records, fixed stride) - no per-op objects cross the boundary (the
//! no-heavy-marshalling rule). Dynamic feeds arrive as flat arrays.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod channels;
mod doc;
mod evaluate;

use crate::channels::ChannelRef;
use crate::doc::{Blend, Cmd, DrawableDoc, Item, Source, Space, TransformRef};
use crate::evaluate::{evaluate, Feed, F_STRIDE, U_STRIDE};

/// Decode a (id, rest) pair from Python: id < 0 means "no channel".
fn chan(id: i64, rest: f32) -> ChannelRef {
    if id < 0 {
        ChannelRef::constant(rest)
    } else {
        ChannelRef { id: id as u32, rest }
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
}

#[pyclass(unsendable)]
struct Evaluator {
    doc: DrawableDoc,
}

#[pymethods]
impl Evaluator {
    /// Evaluate one frame; returns (u32_records_bytes, f32_records_bytes,
    /// op_count). Fixed strides: U_STRIDE u32 lanes + F_STRIDE f32
    /// lanes per op - view with numpy.frombuffer(...).reshape(n, stride).
    fn frame<'py>(&self, py: Python<'py>, t: f32) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize) {
        let schedule = evaluate(&self.doc, t, &[]);
        let n = schedule.len();
        let u_bytes = PyBytes::new(py, bytemuck_cast_u32(&schedule.u));
        let f_bytes = PyBytes::new(py, bytemuck_cast_f32(&schedule.f));
        (u_bytes, f_bytes, n)
    }

    #[getter]
    fn u_stride(&self) -> usize {
        U_STRIDE
    }

    #[getter]
    fn f_stride(&self) -> usize {
        F_STRIDE
    }

    fn drawable_count(&self) -> usize {
        self.doc.drawables.len()
    }
}

fn bytemuck_cast_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytemuck_cast_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
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
