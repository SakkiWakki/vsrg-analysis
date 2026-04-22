// TTF glyph renderer built on stb_truetype.
//
// Replaces the 8x8 bitmap font. One atlas is baked at a fixed design
// size from a system-installed TTF (DejaVu Sans Mono by default); text
// draws as textured quads scaled to the caller's requested height.
//
// The atlas is built once at init and never resized, so all rendering
// is alloc-free and thread-unsafe-but-single-threaded (the overlay
// binary only draws from main()).
//
// API mirrors the shapes the bitmap renderer exposed:
//   - font_ttf_init(): load + bake. Returns 1 on success, 0 on failure.
//                      On failure the overlay falls back to bitmap text.
//   - font_ttf_draw(s, x, y, px_height, rgba): emit textured quads.
//                      Colour is baked via glColor4f before each glyph.
//                      Caller must have GL_BLEND on; we manage
//                      GL_TEXTURE_2D state internally.
//   - font_ttf_measure(s, px_height): advance width in pixels.
//
// Anchoring: (x, y) is the top-left of the text's bounding box at the
// requested height, matching draw_text's old contract so callers don't
// need layout changes. Baseline is derived from font metrics.

#ifndef VSRG_FONT_TTF_H
#define VSRG_FONT_TTF_H

#include <stdint.h>

int  font_ttf_init(void);
int  font_ttf_ready(void);
void font_ttf_draw(const char *s, float x, float y, float px_height);
float font_ttf_measure(const char *s, float px_height);

#endif
