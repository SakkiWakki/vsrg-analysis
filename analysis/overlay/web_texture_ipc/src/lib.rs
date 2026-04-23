//! Python-facing entry point for the web-texture side-channel.
//!
//! See module docs in ``socket.rs``, ``egl.rs``, and ``protocol.rs``
//! for the per-piece design. This file only marshals between Python
//! and the crate's pure-Rust internals.

#![allow(clippy::missing_safety_doc)]

mod error;
mod protocol;
mod socket;
mod egl;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

// Re-export constants Python code wants to reference by name.
#[pyfunction] fn magic()        -> u32 { protocol::MAGIC }
#[pyfunction] fn version()      -> u32 { protocol::VERSION }
#[pyfunction] fn kind_publish() -> u32 { protocol::KIND_PUBLISH }
#[pyfunction] fn kind_release() -> u32 { protocol::KIND_RELEASE }
#[pyfunction] fn format_argb8888() -> u32 { protocol::FORMAT_ARGB8888 }
#[pyfunction] fn format_xrgb8888() -> u32 { protocol::FORMAT_XRGB8888 }
#[pyfunction] fn format_abgr8888() -> u32 { protocol::FORMAT_ABGR8888 }

/// Producer-side channel. Opens the side socket and exports GL
/// textures as dmabuf fds on demand.
///
/// ``path`` overrides the default ``/tmp/vsrg_overlay_web.sock`` for
/// tests that want to point at a socketpair.
#[pyclass]
struct WebTextureChannel {
    socket: Option<socket::FrameSocket>,
}

#[pymethods]
impl WebTextureChannel {
    #[new]
    #[pyo3(signature = (path = None))]
    fn new(path: Option<&str>) -> PyResult<Self> {
        let p = path.unwrap_or(protocol::SOCKET_PATH);
        let sock = socket::FrameSocket::connect(p)?;
        Ok(Self { socket: Some(sock) })
    }

    /// Export ``gl_texture_id`` from the calling thread's current GL
    /// context as a dmabuf and send it over the socket. The caller
    /// must stamp a matching shm widget with the same
    /// ``(channel_id, generation)``.
    ///
    /// Returns ``True`` on success; ``False`` if the send would block
    /// (overlay is behind -- we drop rather than queue, per the
    /// "latest frame wins" model). Raises on EGL errors, missing
    /// extensions, or other fatal failures.
    ///
    /// The exported format comes from the driver (the GL texture's
    /// internal format decides the FourCC it exports as). We
    /// propagate whatever the driver reports into the wire format so
    /// the overlay's EGL import uses the right attribute list.
    #[pyo3(signature = (channel_id, generation, gl_texture_id, width, height))]
    fn publish_from_gl_texture(
        &self,
        channel_id: u32,
        generation: u32,
        gl_texture_id: u32,
        width: u32,
        height: u32,
    ) -> PyResult<bool> {
        let sock = self.socket.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("channel closed"))?;

        let exported = egl::export_gl_texture(gl_texture_id)?;

        let mut frame = protocol::Frame::new_publish(
            channel_id, generation, width, height,
            exported.fourcc, exported.modifier);
        frame.offsets[0] = exported.offset;
        frame.strides[0] = exported.stride;

        let sent_ok = sock.send(&frame, Some(exported.fd))?;
        // Kernel duped the fd; our copy is now owned by the sendmsg
        // path on success, or we still hold it on EAGAIN. Close here
        // either way to avoid leaking: on success the kernel has the
        // dup; on drop we skip this frame and don't need to keep fd.
        unsafe { libc::close(exported.fd) };
        Ok(sent_ok)
    }

    /// Tell the overlay it can drop any cached textures for this
    /// channel. Best-effort; safe to call multiple times.
    fn release(&self, channel_id: u32) -> PyResult<bool> {
        let sock = self.socket.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("channel closed"))?;
        let frame = protocol::Frame::new_release(channel_id);
        Ok(sock.send(&frame, None)?)
    }

    fn close(&mut self) {
        self.socket = None;
    }
}

/// Quick probe used by the PAL dispatcher to decide whether to
/// register the dmabuf backend. Returns True iff:
///   - a GL/EGL context is current on the calling thread,
///   - the driver advertises ``EGL_MESA_image_dma_buf_export``.
///
/// Intended usage: the Python backend's ``is_available()`` ensures a
/// throwaway GL context is current, calls this, then tears down. If
/// it returns False, the PAL skips this backend and the PAL's
/// qpixmap fallback handles the load.
#[pyfunction]
fn egl_supports_export() -> bool {
    // We can't fail softly from here because egl::export_gl_texture
    // needs a real texture. Instead probe by attempting to resolve
    // the symbols + querying the extension string. We side-step by
    // doing a no-op import of ``egl.rs``'s ``syms`` -- done by
    // triggering a query on a fake tex id 0 and catching the specific
    // error variant.
    match egl::export_gl_texture(0) {
        // tex_id=0 is always rejected up front as InvalidArg *after*
        // the extension check, so we get InvalidArg if support is
        // present and MissingEglExtension otherwise.
        Err(error::Error::InvalidArg(_)) => true,
        Err(error::Error::MissingEglExtension(_)) => false,
        // Any other error (no context current etc.) means we can't
        // tell; be conservative and say unsupported.
        _ => false,
    }
}

#[pymodule]
fn web_texture_ipc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WebTextureChannel>()?;
    m.add_function(wrap_pyfunction!(egl_supports_export, m)?)?;
    m.add_function(wrap_pyfunction!(magic, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(kind_publish, m)?)?;
    m.add_function(wrap_pyfunction!(kind_release, m)?)?;
    m.add_function(wrap_pyfunction!(format_argb8888, m)?)?;
    m.add_function(wrap_pyfunction!(format_xrgb8888, m)?)?;
    m.add_function(wrap_pyfunction!(format_abgr8888, m)?)?;
    m.add("SOCKET_PATH", protocol::SOCKET_PATH)?;
    Ok(())
}
