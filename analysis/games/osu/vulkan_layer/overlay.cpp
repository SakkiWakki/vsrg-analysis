// HUD driver for the Vulkan layer.
//
// Reads the same /dev/shm/vsrg_overlay segment the gamescope overlay
// binary reads (single contract, two consumers), and replays each
// widget as an ImGui background-drawlist primitive. We intentionally
// do NOT use any ImGui widgets (Begin/Button/etc.) ; the HUD is just
// raw rects + text, same shape as our NanoVG/gamescope renderer, so
// it won't look like a debug menu.
//
// This file is Vulkan-free. It sees ImGui + shm structs + plain X
// for input. layer.cpp owns the Vulkan state and calls overlay_tick()
// once per present under its own mutex.

#include "overlay.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "imgui/imgui.h"

extern "C" {
#include "../../../overlay/widgets/overlay_shm.h"
}

namespace vsrg {

namespace {

constexpr const char* SHM_PATH = "/dev/shm/vsrg_overlay";

// px_scale (bitmap-era: 1.0 = 8 px tall) → pixel height for TTF text.
// Matches the gamescope C renderer's VSRG_TEXT_HEIGHT_PER_PX_SCALE so
// the publisher's layout constants don't need to know which renderer
// is active.
constexpr float PX_HEIGHT_PER_SCALE = 8.0f;

VsrgOverlayShm* g_shm = nullptr;
bool g_edit_mode = false;

ImFont* g_font = nullptr;

// Seqlock read: spin up to 16 times while the publisher is mid-write
// (odd seq), then memcpy into the caller's buffer. Returns true on a
// consistent read with matching magic+version.
bool shm_read(VsrgOverlayShm* out) {
    if (!g_shm) return false;
    for (int tries = 0; tries < 16; ++tries) {
        uint32_t s0 = __atomic_load_n(&g_shm->seq, __ATOMIC_ACQUIRE);
        if (s0 & 1u) continue;
        std::memcpy(out, (const void*)g_shm, sizeof(*out));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        uint32_t s1 = __atomic_load_n(&g_shm->seq, __ATOMIC_ACQUIRE);
        if (s0 == s1) {
            return out->magic == VSRG_OVERLAY_MAGIC
                && out->version == VSRG_OVERLAY_VERSION;
        }
    }
    return false;
}

// Resolve a widget's normalized position + anchor to a pixel
// top-left. Mirrors resolve_box() in gamescope_overlay/osu_overlay.c
// so both renderers place widgets identically.
struct Box { float x, y, w, h; };

Box resolve_box(const VsrgOverlayWidget& w, int canvas_w, int canvas_h,
                float text_w_px, float text_h_px) {
    float pw, ph;
    if (w.kind == VSRG_OVERLAY_KIND_TEXT) {
        pw = text_w_px;
        ph = text_h_px;
    } else {
        pw = w.w * canvas_w;
        ph = w.h * canvas_h;
    }
    float px = w.x * canvas_w;
    float py = w.y * canvas_h;
    switch (w.anchor) {
        case VSRG_OVERLAY_ANCHOR_TR: px = canvas_w - px - pw; break;
        case VSRG_OVERLAY_ANCHOR_BL: py = canvas_h - py - ph; break;
        case VSRG_OVERLAY_ANCHOR_BR:
            px = canvas_w - px - pw;
            py = canvas_h - py - ph;
            break;
        case VSRG_OVERLAY_ANCHOR_C:
            px += canvas_w * 0.5f - pw * 0.5f;
            py += canvas_h * 0.5f - ph * 0.5f;
            break;
        default: break;  // TL: already top-left.
    }
    return { px, py, pw, ph };
}

// Measure text at a given pixel height using the loaded font (or the
// default). ImFont::CalcTextSizeA takes the requested size and wraps
// internally; we don't wrap, so wrap_width=0.
ImVec2 measure_text(const char* s, float px_height) {
    ImFont* font = g_font ? g_font : ImGui::GetFont();
    return font->CalcTextSizeA(px_height, FLT_MAX, 0.0f, s);
}

// Emit one widget into the drawlist. Colors are the publisher's
// RGBA32 layout (byte 0 = R). ImGui's IM_COL32 is also byte 0 = R,
// so we can pass the u32 through unchanged.
void draw_widget(ImDrawList* dl, const VsrgOverlayWidget& w,
                 int canvas_w, int canvas_h) {
    float text_w = 0.0f, text_h = 0.0f;
    float px_height = w.px_scale * PX_HEIGHT_PER_SCALE;
    if (w.kind == VSRG_OVERLAY_KIND_TEXT) {
        ImVec2 sz = measure_text(w.text, px_height);
        text_w = sz.x;
        text_h = sz.y;
    }
    Box b = resolve_box(w, canvas_w, canvas_h, text_w, text_h);

    if (w.kind == VSRG_OVERLAY_KIND_RECT) {
        dl->AddRectFilled(ImVec2(b.x, b.y),
                          ImVec2(b.x + b.w, b.y + b.h),
                          w.color);
    } else if (w.kind == VSRG_OVERLAY_KIND_TEXT) {
        ImFont* font = g_font ? g_font : ImGui::GetFont();
        dl->AddText(font, px_height, ImVec2(b.x, b.y), w.color, w.text);
    }
}

}  // namespace

bool shm_ensure(void) {
    if (g_shm) return true;
    int fd = ::open(SHM_PATH, O_CREAT | O_RDWR, 0600);
    if (fd < 0) return false;
    struct stat st;
    if (::fstat(fd, &st) < 0) { ::close(fd); return false; }
    if (st.st_size < (off_t)sizeof(VsrgOverlayShm)) {
        // Don't truncate here ; if the publisher hasn't run yet, an
        // empty file won't have VSRG_OVERLAY_MAGIC so shm_read will
        // just fail benignly.
        if (::ftruncate(fd, sizeof(VsrgOverlayShm)) < 0) {
            ::close(fd); return false;
        }
    }
    void* p = ::mmap(nullptr, sizeof(VsrgOverlayShm),
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) return false;
    g_shm = reinterpret_cast<VsrgOverlayShm*>(p);
    return true;
}

bool overlay_edit_mode(void) { return g_edit_mode; }

void overlay_set_edit_mode(bool on) {
    g_edit_mode = on;
    if (g_shm) {
        __atomic_store_n(&g_shm->edit_mode, on ? 1u : 0u, __ATOMIC_RELEASE);
    }
}

void font_load(void) {
    ImGuiIO& io = ImGui::GetIO();
    const char* path = std::getenv("VSRG_OVERLAY_FONT");
    if (!path || !*path) path = "/usr/share/fonts/TTF/DejaVuSansMono.ttf";
    // Size 18 as the atlas size; ImGui scales per-call in AddText so
    // one atlas covers all text heights the HUD uses. 18 is big enough
    // that even the largest HUD text (px_scale=2.5 → 20 px) stays
    // readable without per-size atlases.
    ImFontConfig cfg;
    cfg.OversampleH = 2;
    cfg.OversampleV = 2;
    ImFont* f = io.Fonts->AddFontFromFileTTF(path, 18.0f, &cfg);
    if (!f) {
        std::fprintf(stderr, "[vsrg-layer] failed to load font '%s', "
                             "falling back to ImGui default\n", path);
        io.Fonts->AddFontDefault();
        return;
    }
    g_font = f;
}

int overlay_tick(ImDrawList* dl, int canvas_w, int canvas_h) {
    if (!shm_ensure()) return 0;
    VsrgOverlayShm snap;
    if (!shm_read(&snap)) return 0;

    uint32_t n = snap.n_widgets;
    if (n > VSRG_OVERLAY_MAX_WIDGETS) n = VSRG_OVERLAY_MAX_WIDGETS;

    // Edit-mode dim under the widgets.
    if (g_edit_mode) {
        dl->AddRectFilled(ImVec2(0, 0),
                          ImVec2((float)canvas_w, (float)canvas_h),
                          IM_COL32(0, 0, 0, 115));
    }

    int drawn = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const VsrgOverlayWidget& w = snap.widgets[i];
        if (w.kind == VSRG_OVERLAY_KIND_UNUSED) continue;
        draw_widget(dl, w, canvas_w, canvas_h);
        ++drawn;
    }

    // Edit-mode help strip + outlines.
    if (g_edit_mode) {
        dl->AddRectFilled(ImVec2(0, 0),
                          ImVec2((float)canvas_w, 28.0f),
                          IM_COL32(0, 0, 0, 204));
        ImFont* font = g_font ? g_font : ImGui::GetFont();
        dl->AddText(font, 14.0f, ImVec2(12.0f, 6.0f),
                    IM_COL32(255, 255, 255, 255),
                    "EDIT MODE  SHIFT+TAB TO EXIT");
        for (uint32_t i = 0; i < n; ++i) {
            const VsrgOverlayWidget& w = snap.widgets[i];
            if (w.kind == VSRG_OVERLAY_KIND_UNUSED) continue;
            float text_w = 0.0f, text_h = 0.0f;
            if (w.kind == VSRG_OVERLAY_KIND_TEXT) {
                ImVec2 sz = measure_text(w.text, w.px_scale * PX_HEIGHT_PER_SCALE);
                text_w = sz.x;
                text_h = sz.y;
            }
            Box b = resolve_box(w, canvas_w, canvas_h, text_w, text_h);
            dl->AddRect(ImVec2(b.x - 2.0f, b.y - 2.0f),
                        ImVec2(b.x + b.w + 2.0f, b.y + b.h + 2.0f),
                        IM_COL32(255, 255, 255, 166),
                        0.0f, 0, 1.0f);
        }
    }

    return drawn;
}

}  // namespace vsrg
