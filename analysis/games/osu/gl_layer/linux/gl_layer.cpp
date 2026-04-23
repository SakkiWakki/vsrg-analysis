// LD_PRELOAD probe for osu!stable's Linux GL path.
//
// Stable under Wine does not necessarily touch Vulkan at all. In the
// observed osu-winello log it loads OpenGL/EGL libraries, so this layer
// starts with the lowest-risk proof points: intercept buffer-present
// calls and log the first hit. Later this can grow a renderer in the
// same place, just before calling the real swap function.

#include <EGL/egl.h>
#include <GL/glx.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>

// The NanoVG-backed shim is 64-bit only for now — building NanoVG in
// 32-bit requires multilib GL/fontstash dependencies we don't ship.
// Step 1 lights up the 64-bit path (which is what osu!stable uses on
// current Wine builds); the 32-bit .so stays a logging-only stub.
#if defined(VSRG_GL_LAYER_HAS_RENDERER)
extern "C" {
#include "render.h"
#include "widgets.h"
#include "input.h"
#include "shm_consumer.h"
}
#endif

extern "C" {
EGLBoolean eglSwapBuffersWithDamageKHR(EGLDisplay dpy,
                                       EGLSurface surface,
                                       const EGLint* rects,
                                       EGLint n_rects);
EGLBoolean eglSwapBuffersWithDamageEXT(EGLDisplay dpy,
                                       EGLSurface surface,
                                       const EGLint* rects,
                                       EGLint n_rects);
int64_t glXSwapBuffersMscOML(Display* dpy,
                             GLXDrawable drawable,
                             int64_t target_msc,
                             int64_t divisor,
                             int64_t remainder);
void glXCopySubBufferMESA(Display* dpy,
                          GLXDrawable drawable,
                          int x,
                          int y,
                          int width,
                          int height);
}

namespace {

using PfnEglGetProcAddress =
    __eglMustCastToProperFunctionPointerType (*)(const char*);
using PfnEglQuerySurface =
    EGLBoolean (*)(EGLDisplay, EGLSurface, EGLint, EGLint*);
using PfnEglSwapBuffers = EGLBoolean (*)(EGLDisplay, EGLSurface);
using PfnEglSwapBuffersWithDamage =
    EGLBoolean (*)(EGLDisplay, EGLSurface, const EGLint*, EGLint);

using PfnGlXGetProcAddress = __GLXextFuncPtr (*)(const GLubyte*);
using PfnGlXQueryDrawable = void (*)(Display*, GLXDrawable, int, unsigned int*);
using PfnGlXSwapBuffers = void (*)(Display*, GLXDrawable);
using PfnGlXSwapBuffersMscOML =
    int64_t (*)(Display*, GLXDrawable, int64_t, int64_t, int64_t);
using PfnGlXCopySubBufferMESA =
    void (*)(Display*, GLXDrawable, int, int, int, int);
using PfnDlsym = void* (*)(void*, const char*);

std::atomic_bool g_logged_egl_swap{false};
std::atomic_bool g_logged_egl_damage{false};
std::atomic_bool g_logged_glx_swap{false};
std::atomic_bool g_logged_glx_msc{false};
std::atomic_bool g_logged_glx_copy{false};
std::atomic_bool g_logged_dlsym_swap{false};
std::atomic_bool g_logged_missing{false};

bool enabled() {
    const char* v = std::getenv("VSRG_GL_OVERLAY");
    return v && std::strcmp(v, "0") != 0;
}

template <typename T>
T next_symbol(const char* name) {
    static PfnDlsym real_dlsym =
        reinterpret_cast<PfnDlsym>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
    if (!real_dlsym) return nullptr;
    return reinterpret_cast<T>(real_dlsym(RTLD_NEXT, name));
}

void log_missing_once(const char* name) {
    if (!enabled()) return;
    if (!g_logged_missing.exchange(true)) {
        std::fprintf(stderr, "[vsrg-gl] missing real %s via RTLD_NEXT\n", name);
    }
}

void egl_size(EGLDisplay dpy, EGLSurface surface, int* out_w, int* out_h) {
    *out_w = 0;
    *out_h = 0;
    auto query = next_symbol<PfnEglQuerySurface>("eglQuerySurface");
    if (!query) return;

    EGLint w = 0;
    EGLint h = 0;
    if (query(dpy, surface, EGL_WIDTH, &w) == EGL_TRUE &&
        query(dpy, surface, EGL_HEIGHT, &h) == EGL_TRUE) {
        *out_w = static_cast<int>(w);
        *out_h = static_cast<int>(h);
    }
}

void glx_size(Display* dpy, GLXDrawable drawable, int* out_w, int* out_h) {
    *out_w = 0;
    *out_h = 0;
    auto query = next_symbol<PfnGlXQueryDrawable>("glXQueryDrawable");
    if (!query) return;

    unsigned int w = 0;
    unsigned int h = 0;
    query(dpy, drawable, GLX_WIDTH, &w);
    query(dpy, drawable, GLX_HEIGHT, &h);
    *out_w = static_cast<int>(w);
    *out_h = static_cast<int>(h);
}

void log_first(std::atomic_bool& flag, const char* hook, int w, int h) {
    if (!enabled()) return;
    if (!flag.exchange(true)) {
        std::fprintf(stderr,
                     "[vsrg-gl] first %s intercepted (%dx%d)\n",
                     hook, w, h);
    }
}

#if defined(VSRG_GL_LAYER_HAS_RENDERER)
// NanoVG init is deferred until a GL context is current on our
// thread — that doesn't happen until the game's first swap. After
// that, we draw a sanity-check rect per-frame bracketed by full GL
// state save/restore so the game's renderer is unaffected.

std::atomic_bool g_render_init_tried{false};
std::atomic_bool g_render_ready{false};
std::atomic_bool g_input_init_tried{false};
std::atomic_bool g_input_ready{false};
VsrgDragState    g_drag_state{};   // zero-init == clean, no edit, no drag

// Saves the GL state NanoVG's GL2 backend touches: program, bound
// buffers, active texture, scissor test, blend state, depth test,
// cull. We use the legacy glPush/PopAttrib path since NanoVG GL2
// already expects a compatibility profile. That's the same context
// the existing gamescope overlay runs in, so this has been exercised.
struct GlStateGuard {
    GlStateGuard() {
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
    }
    ~GlStateGuard() {
        glPopClientAttrib();
        glPopAttrib();
    }
};

void draw_overlay_frame(int w, int h, uintptr_t surface_handle) {
    if (!g_render_init_tried.exchange(true)) {
        bool ok = render_init() != 0;
        g_render_ready.store(ok);
        std::fprintf(stderr,
                     "[vsrg-gl] render_init %s\n",
                     ok ? "succeeded" : "FAILED");
    }
    if (!g_render_ready.load()) return;
    if (w <= 0 || h <= 0) return;

    // Input is non-fatal — if it fails (no DISPLAY reachable) we
    // still draw widgets, we just can't enter edit mode.
    if (!g_input_init_tried.exchange(true)) {
        bool ok = vsrg_input_init() != 0;
        g_input_ready.store(ok);
        std::fprintf(stderr,
                     "[vsrg-gl] input_init %s\n",
                     ok ? "succeeded" : "FAILED");
    }

    // Attach to the publisher's shm lazily — the feed may come up
    // after the game window, and we should survive that. On any
    // given frame it's fine to have no shm yet; we just draw nothing.
    bool have_shm = shm_consumer_ensure() != 0;
    VsrgOverlayShm snap;
    bool have_snap = have_shm && (shm_consumer_read(&snap) != 0);

    int hover_idx = -1;
    if (g_input_ready.load()) {
        // Tell the backend which X window hosts our render surface
        // so cursor coords arrive in client pixels. We refresh it
        // every frame because in principle the game could reparent.
        vsrg_input_set_surface(surface_handle);

        VsrgInputState in;
        vsrg_input_poll(w, h, &in);

        VsrgOverlayShm *mut = have_snap ? shm_consumer_writable() : nullptr;
        hover_idx = vsrg_drag_tick(&g_drag_state, &in,
                                   have_snap ? &snap : nullptr,
                                   mut, w, h);

        // Re-read the snapshot after drag — we may have just
        // written new widget positions into shm, and the publisher
        // may also be racing us with a new frame. vsrg_draw_widgets
        // reads from our local ``snap`` so keep it current.
        if (g_drag_state.edit_mode && g_drag_state.drag_idx >= 0
                && have_shm) {
            have_snap = (shm_consumer_read(&snap) != 0);
        }
    }

    GlStateGuard guard;
    render_begin_frame(w, h);
    if (have_snap) {
        vsrg_draw_widgets(&snap, w, h);
    }
    // Draw edit decorations regardless of have_snap: on menus the
    // publisher may not have any widgets yet, but the user still needs
    // to see that shift+tab registered. vsrg_draw_edit_decorations
    // tolerates a null/empty widget list — it only iterates widgets
    // when painting outlines, and the screen-dim + help-bar draws
    // unconditionally.
    if (g_drag_state.edit_mode) {
        static const VsrgOverlayShm k_empty{};
        vsrg_draw_edit_decorations(have_snap ? &snap : &k_empty,
                                   w, h, hover_idx);
    }
    render_end_frame();
}
#endif  // VSRG_GL_LAYER_HAS_RENDERER

bool proc_is(const char* got, const char* want) {
    return got && std::strcmp(got, want) == 0;
}

bool proc_is(const GLubyte* got, const char* want) {
    return got && std::strcmp(reinterpret_cast<const char*>(got), want) == 0;
}

void* maybe_hook_proc(const char* name) {
    if (proc_is(name, "eglSwapBuffers")) {
        return reinterpret_cast<void*>(&eglSwapBuffers);
    }
    if (proc_is(name, "eglSwapBuffersWithDamageKHR")) {
        return reinterpret_cast<void*>(&eglSwapBuffersWithDamageKHR);
    }
    if (proc_is(name, "eglSwapBuffersWithDamageEXT")) {
        return reinterpret_cast<void*>(&eglSwapBuffersWithDamageEXT);
    }
    if (proc_is(name, "eglGetProcAddress")) {
        return reinterpret_cast<void*>(&eglGetProcAddress);
    }
    if (proc_is(name, "glXSwapBuffers")) {
        return reinterpret_cast<void*>(&glXSwapBuffers);
    }
    if (proc_is(name, "glXSwapBuffersMscOML")) {
        return reinterpret_cast<void*>(&glXSwapBuffersMscOML);
    }
    if (proc_is(name, "glXCopySubBufferMESA")) {
        return reinterpret_cast<void*>(&glXCopySubBufferMESA);
    }
    if (proc_is(name, "glXGetProcAddress")) {
        return reinterpret_cast<void*>(&glXGetProcAddress);
    }
    if (proc_is(name, "glXGetProcAddressARB")) {
        return reinterpret_cast<void*>(&glXGetProcAddressARB);
    }
    return nullptr;
}

}  // namespace

extern "C" {

__attribute__((constructor))
static void vsrg_gl_loaded() {
    if (!enabled()) return;

    char exe[256] = {};
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n < 0) {
        std::snprintf(exe, sizeof(exe), "?");
    } else {
        exe[n] = '\0';
    }
    std::fprintf(stderr, "[vsrg-gl] loaded pid=%ld exe=%s\n",
                 static_cast<long>(getpid()), exe);
}

__attribute__((visibility("default")))
void* dlsym(void* handle, const char* symbol) {
    static PfnDlsym real_dlsym =
        reinterpret_cast<PfnDlsym>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));

    if (void* hook = maybe_hook_proc(symbol)) {
        if (enabled() && !g_logged_dlsym_swap.exchange(true)) {
            std::fprintf(stderr, "[vsrg-gl] dlsym intercepted %s\n", symbol);
        }
        return hook;
    }

    if (!real_dlsym) return nullptr;
    return real_dlsym(handle, symbol);
}

__attribute__((visibility("default")))
EGLBoolean eglSwapBuffers(EGLDisplay dpy, EGLSurface surface) {
    int w = 0;
    int h = 0;
    egl_size(dpy, surface, &w, &h);
    log_first(g_logged_egl_swap, "eglSwapBuffers", w, h);

    auto real = next_symbol<PfnEglSwapBuffers>("eglSwapBuffers");
    if (!real) {
        log_missing_once("eglSwapBuffers");
        return EGL_FALSE;
    }
    return real(dpy, surface);
}

__attribute__((visibility("default")))
EGLBoolean eglSwapBuffersWithDamageKHR(EGLDisplay dpy,
                                       EGLSurface surface,
                                       const EGLint* rects,
                                       EGLint n_rects) {
    int w = 0;
    int h = 0;
    egl_size(dpy, surface, &w, &h);
    log_first(g_logged_egl_damage, "eglSwapBuffersWithDamageKHR", w, h);

    auto real = next_symbol<PfnEglSwapBuffersWithDamage>(
        "eglSwapBuffersWithDamageKHR");
    if (real) return real(dpy, surface, rects, n_rects);

    auto swap = next_symbol<PfnEglSwapBuffers>("eglSwapBuffers");
    if (!swap) {
        log_missing_once("eglSwapBuffersWithDamageKHR/eglSwapBuffers");
        return EGL_FALSE;
    }
    return swap(dpy, surface);
}

__attribute__((visibility("default")))
EGLBoolean eglSwapBuffersWithDamageEXT(EGLDisplay dpy,
                                       EGLSurface surface,
                                       const EGLint* rects,
                                       EGLint n_rects) {
    int w = 0;
    int h = 0;
    egl_size(dpy, surface, &w, &h);
    log_first(g_logged_egl_damage, "eglSwapBuffersWithDamageEXT", w, h);

    auto real = next_symbol<PfnEglSwapBuffersWithDamage>(
        "eglSwapBuffersWithDamageEXT");
    if (real) return real(dpy, surface, rects, n_rects);

    auto swap = next_symbol<PfnEglSwapBuffers>("eglSwapBuffers");
    if (!swap) {
        log_missing_once("eglSwapBuffersWithDamageEXT/eglSwapBuffers");
        return EGL_FALSE;
    }
    return swap(dpy, surface);
}

__attribute__((visibility("default")))
__eglMustCastToProperFunctionPointerType eglGetProcAddress(
    const char* procname) {
    if (proc_is(procname, "eglSwapBuffers")) {
        return reinterpret_cast<__eglMustCastToProperFunctionPointerType>(
            &eglSwapBuffers);
    }
    if (proc_is(procname, "eglSwapBuffersWithDamageKHR")) {
        return reinterpret_cast<__eglMustCastToProperFunctionPointerType>(
            &eglSwapBuffersWithDamageKHR);
    }
    if (proc_is(procname, "eglSwapBuffersWithDamageEXT")) {
        return reinterpret_cast<__eglMustCastToProperFunctionPointerType>(
            &eglSwapBuffersWithDamageEXT);
    }

    auto real = next_symbol<PfnEglGetProcAddress>("eglGetProcAddress");
    if (!real) {
        log_missing_once("eglGetProcAddress");
        return nullptr;
    }
    return real(procname);
}

__attribute__((visibility("default")))
void glXSwapBuffers(Display* dpy, GLXDrawable drawable) {
    int w = 0;
    int h = 0;
    glx_size(dpy, drawable, &w, &h);
    log_first(g_logged_glx_swap, "glXSwapBuffers", w, h);

#if defined(VSRG_GL_LAYER_HAS_RENDERER)
    // GLXDrawable is an XID — for standard windowed/fullscreen GLX
    // (what osu!stable uses) it's the Window itself, which is what
    // the X11 input backend needs. If osu! ever used a GLXPbuffer
    // or GLXPixmap here, this would need XGetGeometry to find the
    // parent Window — left as a TODO.
    if (enabled()) {
        draw_overlay_frame(w, h,
                           static_cast<uintptr_t>(drawable));
    }
#endif

    auto real = next_symbol<PfnGlXSwapBuffers>("glXSwapBuffers");
    if (!real) {
        log_missing_once("glXSwapBuffers");
        return;
    }
    real(dpy, drawable);
}

__attribute__((visibility("default")))
int64_t glXSwapBuffersMscOML(Display* dpy,
                             GLXDrawable drawable,
                             int64_t target_msc,
                             int64_t divisor,
                             int64_t remainder) {
    int w = 0;
    int h = 0;
    glx_size(dpy, drawable, &w, &h);
    log_first(g_logged_glx_msc, "glXSwapBuffersMscOML", w, h);

    auto real = next_symbol<PfnGlXSwapBuffersMscOML>(
        "glXSwapBuffersMscOML");
    if (!real) {
        log_missing_once("glXSwapBuffersMscOML");
        return 0;
    }
    return real(dpy, drawable, target_msc, divisor, remainder);
}

__attribute__((visibility("default")))
void glXCopySubBufferMESA(Display* dpy,
                          GLXDrawable drawable,
                          int x,
                          int y,
                          int width,
                          int height) {
    log_first(g_logged_glx_copy, "glXCopySubBufferMESA", width, height);

    auto real = next_symbol<PfnGlXCopySubBufferMESA>("glXCopySubBufferMESA");
    if (!real) {
        log_missing_once("glXCopySubBufferMESA");
        return;
    }
    real(dpy, drawable, x, y, width, height);
}

__attribute__((visibility("default")))
__GLXextFuncPtr glXGetProcAddress(const GLubyte* procname) {
    if (proc_is(procname, "glXSwapBuffers")) {
        return reinterpret_cast<__GLXextFuncPtr>(&glXSwapBuffers);
    }
    if (proc_is(procname, "glXSwapBuffersMscOML")) {
        return reinterpret_cast<__GLXextFuncPtr>(&glXSwapBuffersMscOML);
    }
    if (proc_is(procname, "glXCopySubBufferMESA")) {
        return reinterpret_cast<__GLXextFuncPtr>(&glXCopySubBufferMESA);
    }

    auto real = next_symbol<PfnGlXGetProcAddress>("glXGetProcAddress");
    if (!real) {
        log_missing_once("glXGetProcAddress");
        return nullptr;
    }
    return real(procname);
}

__attribute__((visibility("default")))
__GLXextFuncPtr glXGetProcAddressARB(const GLubyte* procname) {
    if (proc_is(procname, "glXSwapBuffers")) {
        return reinterpret_cast<__GLXextFuncPtr>(&glXSwapBuffers);
    }
    if (proc_is(procname, "glXSwapBuffersMscOML")) {
        return reinterpret_cast<__GLXextFuncPtr>(&glXSwapBuffersMscOML);
    }
    if (proc_is(procname, "glXCopySubBufferMESA")) {
        return reinterpret_cast<__GLXextFuncPtr>(&glXCopySubBufferMESA);
    }

    auto real = next_symbol<PfnGlXGetProcAddress>("glXGetProcAddressARB");
    if (!real) {
        log_missing_once("glXGetProcAddressARB");
        return nullptr;
    }
    return real(procname);
}

}  // extern "C"
