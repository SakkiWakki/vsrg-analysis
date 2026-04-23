#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../../../../third_party/minhook/include/MinHook.h"
#include "win_log.h"

extern "C" {
#include "win_gl_loader.h"
#include "win_shm_consumer.h"
#include "../../../../overlay/renderer/render.h"
#include "../../../../overlay/widgets/widgets.h"
#include "../../../../overlay/input/input.h"
}

namespace {

using PfnWglSwapBuffers = BOOL (WINAPI *)(HDC);
PfnWglSwapBuffers g_real_wgl_swap = nullptr;

std::atomic_bool g_render_init_tried{false};
std::atomic_bool g_render_ready{false};
std::atomic_bool g_input_init_tried{false};
std::atomic_bool g_input_ready{false};
VsrgDragState    g_drag_state{};

// Latches so log lines fire on transitions, not every frame.
std::atomic_bool g_shm_open_logged{false};
std::atomic_bool g_first_snap_logged{false};
std::atomic<uint32_t> g_last_n_widgets{UINT32_MAX};
std::atomic<uint64_t> g_frames_since_status{0};

bool enabled() {
    const char *v = std::getenv("VSRG_GL_OVERLAY");
    return v && std::strcmp(v, "0") != 0;
}

struct GlStateGuard {
    GlStateGuard()  { glPushAttrib(GL_ALL_ATTRIB_BITS); glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS); }
    ~GlStateGuard() { glPopClientAttrib(); glPopAttrib(); }
};

void draw_overlay_frame(int w, int h, HWND hwnd) {
    if (!g_render_init_tried.exchange(true)) {
        if (!vsrg_gl_load_extensions()) {
            VSRG_LOG("[vsrg-gl] extension load failed");
        }
        bool ok = render_init() != 0;
        g_render_ready.store(ok);
        VSRG_LOG("[vsrg-gl] render_init %s (canvas %dx%d)",
                 ok ? "ok" : "FAILED", w, h);
    }
    if (!g_render_ready.load()) return;
    if (w <= 0 || h <= 0) return;

    if (!g_input_init_tried.exchange(true)) {
        bool ok = vsrg_input_init() != 0;
        g_input_ready.store(ok);
        VSRG_LOG("[vsrg-gl] input_init %s", ok ? "ok" : "FAILED");
    }

    bool have_shm  = shm_consumer_ensure() != 0;

    if (!g_shm_open_logged.load()) {
        if (have_shm) {
            VSRG_LOG("[vsrg-gl] shm opened (\"vsrg_overlay\")");
            g_shm_open_logged.store(true);
        } else {
            // ~once/sec at 60 fps.
            uint64_t c = g_frames_since_status.fetch_add(1) + 1;
            if (c % 60 == 1) {
                VSRG_LOG("[vsrg-gl] shm not available yet "
                         "(OpenFileMapping(\"vsrg_overlay\") returned NULL) "
                         "— publisher running?");
            }
        }
    }

    VsrgOverlayShm snap;
    bool have_snap = have_shm && (shm_consumer_read(&snap) != 0);

    if (have_snap) {
        if (!g_first_snap_logged.exchange(true)) {
            VSRG_LOG("[vsrg-gl] first snapshot: magic=0x%08x version=%u "
                     "n_widgets=%u seq=%u",
                     (unsigned)snap.magic, (unsigned)snap.version,
                     (unsigned)snap.n_widgets, (unsigned)snap.seq);
        }
        uint32_t nw_prev = g_last_n_widgets.exchange(snap.n_widgets);
        if (nw_prev != snap.n_widgets) {
            VSRG_LOG("[vsrg-gl] widget count %u → %u",
                     (unsigned)nw_prev, (unsigned)snap.n_widgets);
        }
    } else if (have_shm && !g_first_snap_logged.load()) {
        // Mapped but seqlock read failing: magic/version mismatch or torn.
        uint64_t c = g_frames_since_status.fetch_add(1) + 1;
        if (c % 60 == 1) {
            VSRG_LOG("[vsrg-gl] shm attached but shm_consumer_read failed "
                     "(magic/version mismatch or unstable seqlock)");
        }
    }

    int hover_idx = -1;
    if (g_input_ready.load()) {
        vsrg_input_set_surface(reinterpret_cast<uintptr_t>(hwnd));

        VsrgInputState in;
        vsrg_input_poll(w, h, &in);

        VsrgOverlayShm *mut = have_snap ? shm_consumer_writable() : nullptr;
        hover_idx = vsrg_drag_tick(&g_drag_state, &in,
                                   have_snap ? &snap : nullptr,
                                   mut, w, h);

        if (g_drag_state.edit_mode && g_drag_state.drag_idx >= 0 && have_shm)
            have_snap = (shm_consumer_read(&snap) != 0);
    }

    GlStateGuard guard;
    render_begin_frame(w, h);
    if (have_snap)
        vsrg_draw_widgets(&snap, w, h);
    if (g_drag_state.edit_mode) {
        static const VsrgOverlayShm k_empty{};
        vsrg_draw_edit_decorations(have_snap ? &snap : &k_empty, w, h, hover_idx);
    }
    render_end_frame();
}

BOOL WINAPI hooked_wgl_swap_buffers(HDC hdc) {
    if (enabled()) {
        HWND hwnd = WindowFromDC(hdc);
        RECT rect{};
        if (hwnd) GetClientRect(hwnd, &rect);
        int w = rect.right  - rect.left;
        int h = rect.bottom - rect.top;
        draw_overlay_frame(w, h, hwnd);
    }
    return g_real_wgl_swap(hdc);
}

void install_hook() {
    HMODULE ogl = GetModuleHandleA("opengl32.dll");
    if (!ogl) {
        VSRG_LOG("[vsrg-gl] opengl32.dll not loaded yet");
        return;
    }
    void *target = reinterpret_cast<void *>(
        GetProcAddress(ogl, "wglSwapBuffers"));
    if (!target) {
        VSRG_LOG("[vsrg-gl] wglSwapBuffers not found");
        return;
    }

    MH_Initialize();
    MH_STATUS st = MH_CreateHook(target,
                                 reinterpret_cast<void *>(&hooked_wgl_swap_buffers),
                                 reinterpret_cast<void **>(&g_real_wgl_swap));
    if (st != MH_OK) {
        VSRG_LOG("[vsrg-gl] MH_CreateHook failed: %d", st);
        return;
    }
    MH_EnableHook(target);
    VSRG_LOG("[vsrg-gl] wglSwapBuffers hooked");
}

void remove_hook() {
    MH_DisableHook(MH_ALL_HOOKS);
    MH_Uninitialize();
    render_shutdown();
    vsrg_input_shutdown();
}

} // namespace

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        if (enabled()) {
            VSRG_LOG("[vsrg-gl] DllMain attach — installing hook");
            install_hook();
        } else {
            VSRG_LOG("[vsrg-gl] DllMain attach — VSRG_GL_OVERLAY unset, "
                     "not hooking");
        }
    } else if (reason == DLL_PROCESS_DETACH) {
        VSRG_LOG("[vsrg-gl] DllMain detach");
        remove_hook();
    }
    return TRUE;
}
