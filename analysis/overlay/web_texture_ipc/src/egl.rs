//! EGL helpers: initialise, import a GL texture into an EGLImage,
//! export the image as a dmabuf.
//!
//! This module is Linux/MESA-specific. NVIDIA's proprietary driver
//! exposes a different dmabuf export path (``EGL_EXT_device_query_drm``)
//! that we don't support here; on unsupported drivers `prepare_export`
//! returns [`Error::MissingEglExtension`] and the PAL dispatcher should
//! fall back to the qpixmap backend.
//!
//! The flow:
//!
//!   1. Grab the current EGL display + context from whoever owns the
//!      GL context (Qt, in our case). We don't create a display
//!      ourselves -- we borrow whatever the caller's GL code already
//!      has.
//!
//!   2. `eglCreateImage(display, ctx, EGL_GL_TEXTURE_2D, tex_id, ...)`
//!      wraps a GL texture in an EGLImage.
//!
//!   3. `eglExportDMABUFImageQueryMESA` + `eglExportDMABUFImageMESA`
//!      give us (fd, stride, offset, modifier) for the one plane.
//!
//!   4. Caller sends the fd over the socket and dup()-closes it.
//!
//! Feature detection runs once on first use; the cached extension
//! pointers are stashed in a module-level `OnceCell`.

use std::os::fd::RawFd;
use std::os::raw::{c_int, c_void};
use std::sync::OnceLock;

use libloading::{Library, Symbol};

use crate::error::{Error, Result};

// ── EGL type + constant decls ─────────────────────────────────────
// We declare just the subset we use; pulling in the full
// ``khronos-egl`` crate for the handful of constants below is
// overkill. These values come straight from the Khronos registry and
// are ABI-stable.

type EGLDisplay = *mut c_void;
type EGLContext = *mut c_void;
type EGLImage   = *mut c_void;
type EGLBoolean = c_int;
type EGLenum    = u32;
type EGLint     = i32;
type EGLClientBuffer = *mut c_void;

const EGL_NO_IMAGE: EGLImage = std::ptr::null_mut();
const EGL_GL_TEXTURE_2D: EGLenum = 0x30B1;
const EGL_NONE: EGLint = 0x3038;
const EGL_TRUE: EGLBoolean = 1;

// Function-pointer types for the EGL bits we touch.
type FnGetCurrentDisplay = unsafe extern "C" fn() -> EGLDisplay;
type FnGetCurrentContext = unsafe extern "C" fn() -> EGLContext;
type FnCreateImage = unsafe extern "C" fn(
    EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, *const EGLint,
) -> EGLImage;
type FnDestroyImage = unsafe extern "C" fn(EGLDisplay, EGLImage) -> EGLBoolean;
type FnQueryString = unsafe extern "C" fn(EGLDisplay, EGLint) -> *const i8;
type FnGetProcAddress =
    unsafe extern "C" fn(*const i8) -> Option<unsafe extern "C" fn()>;

// MESA dmabuf-export extension.
type FnExportDMABUFImageQuery = unsafe extern "C" fn(
    EGLDisplay, EGLImage,
    *mut c_int,       // out fourcc
    *mut c_int,       // out num_planes
    *mut u64,         // out modifier
) -> EGLBoolean;

type FnExportDMABUFImage = unsafe extern "C" fn(
    EGLDisplay, EGLImage,
    *mut c_int,       // out fds[n_planes]
    *mut c_int,       // out strides[n_planes]
    *mut c_int,       // out offsets[n_planes]
) -> EGLBoolean;

const EGL_EXTENSIONS: EGLint = 0x3055;

// ── Cached symbol table ─────────────────────────────────────────

struct EglSyms {
    get_current_display: FnGetCurrentDisplay,
    get_current_context: FnGetCurrentContext,
    create_image: FnCreateImage,
    destroy_image: FnDestroyImage,
    query_string: FnQueryString,
    export_query: FnExportDMABUFImageQuery,
    export_image: FnExportDMABUFImage,
    // Keep the Library handle alive -- if it drops, the function
    // pointers dangle.
    _lib: Library,
}

static SYMS: OnceLock<std::result::Result<EglSyms, String>> = OnceLock::new();

unsafe fn load_egl() -> std::result::Result<EglSyms, String> {
    // libEGL.so.1 is the SONAME on every Linux distro; even if only
    // libEGL.so is present, Qt will have loaded it already so our
    // dlopen hits the cached resident copy.
    let lib = unsafe { Library::new("libEGL.so.1") }
        .or_else(|_| unsafe { Library::new("libEGL.so") })
        .map_err(|e| format!("failed to load libEGL: {e}"))?;

    macro_rules! sym {
        ($name:literal, $ty:ty) => {{
            let s: Symbol<$ty> = lib.get($name)
                .map_err(|e| format!("missing EGL symbol {}: {}",
                                     std::str::from_utf8($name)
                                         .unwrap_or("?"), e))?;
            *s
        }};
    }

    let get_current_display = unsafe { sym!(b"eglGetCurrentDisplay\0", FnGetCurrentDisplay) };
    let get_current_context = unsafe { sym!(b"eglGetCurrentContext\0", FnGetCurrentContext) };
    let create_image        = unsafe { sym!(b"eglCreateImage\0",       FnCreateImage) };
    let destroy_image       = unsafe { sym!(b"eglDestroyImage\0",      FnDestroyImage) };
    let query_string        = unsafe { sym!(b"eglQueryString\0",       FnQueryString) };
    let get_proc_addr: FnGetProcAddress =
        unsafe { sym!(b"eglGetProcAddress\0", FnGetProcAddress) };

    // Resolve MESA extensions via eglGetProcAddress (they're not in
    // the core lib). Missing here = driver doesn't support export.
    unsafe fn proc_or_err(
        getter: FnGetProcAddress,
        name: &'static [u8],
    ) -> std::result::Result<unsafe extern "C" fn(), String> {
        let p = unsafe { getter(name.as_ptr() as *const i8) };
        p.ok_or_else(|| format!(
            "eglGetProcAddress returned null for {}",
            std::str::from_utf8(name).unwrap_or("?").trim_end_matches('\0')))
    }

    let q = unsafe { proc_or_err(get_proc_addr, b"eglExportDMABUFImageQueryMESA\0")? };
    let x = unsafe { proc_or_err(get_proc_addr, b"eglExportDMABUFImageMESA\0")? };
    let export_query: FnExportDMABUFImageQuery = unsafe {
        std::mem::transmute(q)
    };
    let export_image: FnExportDMABUFImage = unsafe {
        std::mem::transmute(x)
    };

    Ok(EglSyms {
        get_current_display,
        get_current_context,
        create_image,
        destroy_image,
        query_string,
        export_query,
        export_image,
        _lib: lib,
    })
}

fn syms() -> Result<&'static EglSyms> {
    let entry = SYMS.get_or_init(|| unsafe { load_egl() });
    entry.as_ref().map_err(|s| Error::Egl(s.clone()))
}

/// One exported dmabuf plane. The caller owns ``fd`` and is responsible
/// for closing it after sending over the socket (the kernel dup()s on
/// sendmsg; both ends must ultimately close their copies).
#[derive(Debug)]
pub struct Exported {
    pub fd: RawFd,
    pub stride: u32,
    pub offset: u32,
    pub modifier: u64,
    pub fourcc: u32,
}

/// Export a live GL texture (current context's ``GL_TEXTURE_2D`` with
/// name ``tex_id``) as a single-plane dmabuf.
///
/// The texture must belong to the GL context that's current when this
/// is called (eglGetCurrentContext returns its EGLContext). The
/// exported fd is independent of that GL context lifecycle -- the
/// overlay can hold on to the dmabuf after we delete the texture on
/// our side.
///
/// Returns [`Error::MissingEglExtension`] on drivers that don't support
/// the MESA dmabuf-export path (NVIDIA proprietary is the main case).
pub fn export_gl_texture(tex_id: u32) -> Result<Exported> {
    let s = syms()?;
    let display = unsafe { (s.get_current_display)() };
    if display.is_null() {
        return Err(Error::Egl("no current EGL display".into()));
    }
    let context = unsafe { (s.get_current_context)() };
    if context.is_null() {
        return Err(Error::Egl("no current EGL context".into()));
    }

    // Validate the driver actually ships the extensions we use. Done
    // here (not at load time) because eglGetProcAddress can return a
    // non-null pointer for a stub on some drivers; the extension
    // string is authoritative.
    if !has_extension(s, display, "EGL_MESA_image_dma_buf_export") {
        return Err(Error::MissingEglExtension("EGL_MESA_image_dma_buf_export"));
    }

    // EGLClientBuffer is an opaque pointer-sized handle. For the
    // EGL_GL_TEXTURE_2D target it's the GL texture name cast to
    // uintptr_t. A zero here would mean "no texture".
    if tex_id == 0 {
        return Err(Error::InvalidArg("gl texture id is 0"));
    }
    let client_buf: EGLClientBuffer = tex_id as usize as *mut c_void;

    let attrs = [EGL_NONE];
    let image = unsafe {
        (s.create_image)(display, context, EGL_GL_TEXTURE_2D,
                         client_buf, attrs.as_ptr())
    };
    if image == EGL_NO_IMAGE {
        return Err(Error::Egl("eglCreateImage returned EGL_NO_IMAGE".into()));
    }

    // Query shape.
    let mut fourcc: c_int = 0;
    let mut n_planes: c_int = 0;
    let mut modifier: u64 = 0;
    let ok = unsafe {
        (s.export_query)(display, image,
                         &mut fourcc, &mut n_planes, &mut modifier)
    };
    if ok != EGL_TRUE || n_planes < 1 {
        unsafe { (s.destroy_image)(display, image); }
        return Err(Error::EmptyExport);
    }

    // We only handle single-plane RGB for now. If n_planes > 1 the
    // producer would need to extend the protocol and the overlay
    // would have to import multi-plane; skip with a clear error so
    // the PAL can fall back.
    if n_planes != 1 {
        unsafe { (s.destroy_image)(display, image); }
        return Err(Error::InvalidArg("multi-plane dmabuf export not supported"));
    }

    // Export fd + stride + offset.
    let mut fd: c_int = -1;
    let mut stride: c_int = 0;
    let mut offset: c_int = 0;
    let ok = unsafe {
        (s.export_image)(display, image, &mut fd, &mut stride, &mut offset)
    };
    // eglExportDMABUFImageMESA leaves the EGLImage bound; destroy it
    // now. The fd survives -- it's a dup of the underlying buffer.
    unsafe { (s.destroy_image)(display, image); }
    if ok != EGL_TRUE || fd < 0 {
        return Err(Error::EmptyExport);
    }

    Ok(Exported {
        fd,
        stride: stride as u32,
        offset: offset as u32,
        modifier,
        fourcc: fourcc as u32,
    })
}

fn has_extension(s: &EglSyms, display: EGLDisplay, name: &str) -> bool {
    let ptr = unsafe { (s.query_string)(display, EGL_EXTENSIONS) };
    if ptr.is_null() {
        return false;
    }
    let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
    match cstr.to_str() {
        Ok(list) => list.split_whitespace().any(|ext| ext == name),
        Err(_) => false,
    }
}
