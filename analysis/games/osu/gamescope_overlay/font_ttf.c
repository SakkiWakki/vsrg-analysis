// TTF glyph renderer — see font_ttf.h.
//
// One 512x512 ALPHA atlas is baked from DejaVu Sans Mono at 48 px (the
// "design height"). Draws emit textured quads scaled to whatever height
// the caller asks for; bilinear filtering gives us crisp UI text at the
// small sizes the overlay uses (16–48 px) without a multi-size atlas.
//
// Atlas glyphs are ASCII 32..126 only — enough for the HUD's digits,
// colon, percent sign, and 'X'. Glyphs outside that range draw nothing
// and advance a space width so we never blow past the atlas.
//
// Why GL_ALPHA and not GL_RED: the overlay uses fixed-function GL (no
// shaders). GL_ALPHA modulates glColor4f's alpha by the texel, which is
// exactly "tint a coverage mask with the caller's colour" — what we
// want. GL_RED would require a shader or texture-env fiddling.

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include <GL/gl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "font_ttf.h"

// System path — DejaVu Sans Mono ships with most distros and is
// Bitstream-Vera-licensed (permissive, bundled redistribution OK).
// Override via env for headless CI / other distros.
#define DEFAULT_FONT_PATH "/usr/share/fonts/TTF/DejaVuSansMono.ttf"

#define ATLAS_W     512
#define ATLAS_H     512
#define DESIGN_PX   48.0f
#define FIRST_CHAR  32
#define CHAR_COUNT  95  // 32..126 inclusive

static int              g_ready = 0;
static GLuint           g_tex   = 0;
static stbtt_bakedchar  g_chars[CHAR_COUNT];
// Scale factor from design px to 1 unit — used so callers can pass a
// target pixel height and we figure out the glyph quad size.
// advance width at design px is stored implicitly via stbtt_bakedchar.

static int load_file(const char *path, unsigned char **out_buf, long *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return 0; }
    unsigned char *buf = (unsigned char *)malloc((size_t)n);
    if (!buf) { fclose(f); return 0; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf); fclose(f); return 0;
    }
    fclose(f);
    *out_buf = buf;
    *out_n   = n;
    return 1;
}

int font_ttf_init(void) {
    if (g_ready) return 1;

    const char *path = getenv("VSRG_OVERLAY_FONT");
    if (!path || !*path) path = DEFAULT_FONT_PATH;

    unsigned char *ttf = NULL;
    long ttf_n = 0;
    if (!load_file(path, &ttf, &ttf_n)) {
        fprintf(stderr, "[font_ttf] could not read '%s'; "
                        "overlay will fall back to bitmap font\n", path);
        return 0;
    }

    unsigned char *atlas = (unsigned char *)calloc(ATLAS_W * ATLAS_H, 1);
    if (!atlas) { free(ttf); return 0; }

    int baked = stbtt_BakeFontBitmap(ttf, 0, DESIGN_PX,
                                     atlas, ATLAS_W, ATLAS_H,
                                     FIRST_CHAR, CHAR_COUNT,
                                     g_chars);
    free(ttf);
    if (baked <= 0) {
        fprintf(stderr, "[font_ttf] stbtt_BakeFontBitmap failed (%d); "
                        "atlas too small or font invalid\n", baked);
        free(atlas);
        return 0;
    }

    glGenTextures(1, &g_tex);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA,
                 ATLAS_W, ATLAS_H, 0,
                 GL_ALPHA, GL_UNSIGNED_BYTE, atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    free(atlas);
    g_ready = 1;
    printf("[font_ttf] baked %d glyphs from %s at design %.0fpx\n",
           baked, path, DESIGN_PX);
    return 1;
}

int font_ttf_ready(void) { return g_ready; }

// Advance width in pixels for one character at the design size.
static float advance_design_px(char c) {
    int idx = (unsigned char)c - FIRST_CHAR;
    if (idx < 0 || idx >= CHAR_COUNT) {
        // Fall back to space width for unknown chars.
        idx = ' ' - FIRST_CHAR;
        if (idx < 0 || idx >= CHAR_COUNT) return 0.0f;
    }
    return g_chars[idx].xadvance;
}

float font_ttf_measure(const char *s, float px_height) {
    if (!g_ready || !s) return 0.0f;
    float scale = px_height / DESIGN_PX;
    float w = 0.0f;
    for (; *s; s++) {
        w += advance_design_px(*s) * scale;
    }
    return w;
}

void font_ttf_draw(const char *s, float x, float y, float px_height) {
    if (!g_ready || !s) return;

    // stbtt_GetBakedQuad returns a quad in *design* pixels relative to
    // an advancing (pen_x, pen_y) baseline. To scale to the caller's
    // target height we do the bake at DESIGN_PX, let stbtt emit the
    // quad at that size, then scale the resulting x/y around the
    // caller's origin.
    //
    // Baseline placement: the caller gives us the top-left of the
    // text box at px_height. DejaVu Sans Mono's ascent at 48 px is
    // ~37 px; we approximate via 'M'.y0 (negative = pixels above
    // baseline) so y0 lands on the box top.
    float scale = px_height / DESIGN_PX;
    float ascent_design = 0.0f;
    {
        int m_idx = 'M' - FIRST_CHAR;
        if (m_idx >= 0 && m_idx < CHAR_COUNT) {
            ascent_design = -g_chars[m_idx].y0;
        }
    }

    float pen_x_design = 0.0f;
    float pen_y_design = ascent_design;

    glBindTexture(GL_TEXTURE_2D, g_tex);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    for (; *s; s++) {
        int idx = (unsigned char)*s - FIRST_CHAR;
        if (idx < 0 || idx >= CHAR_COUNT) {
            pen_x_design += advance_design_px(' ');
            continue;
        }
        stbtt_aligned_quad q;
        stbtt_GetBakedQuad(g_chars, ATLAS_W, ATLAS_H,
                           idx, &pen_x_design, &pen_y_design, &q, 1);

        float gx0 = x + q.x0 * scale;
        float gy0 = y + q.y0 * scale;
        float gx1 = x + q.x1 * scale;
        float gy1 = y + q.y1 * scale;

        glTexCoord2f(q.s0, q.t0); glVertex2f(gx0, gy0);
        glTexCoord2f(q.s1, q.t0); glVertex2f(gx1, gy0);
        glTexCoord2f(q.s1, q.t1); glVertex2f(gx1, gy1);
        glTexCoord2f(q.s0, q.t1); glVertex2f(gx0, gy1);
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);
}
