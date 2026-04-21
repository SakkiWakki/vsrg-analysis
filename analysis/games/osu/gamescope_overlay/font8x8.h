// Minimal 8x8 bitmap font — a hand-curated subset covering the
// glyphs we need for the HUD (digits, percent, dot, x, space, and
// the letters U / R / C / M / etc. used in the short labels).
//
// Each glyph is 8 rows; each row is one byte, MSB = leftmost pixel.
// We draw with GL_POINTS scaled to make the font block-legible at
// 1080p; no texture atlas, no FreeType, no font hinting — that
// keeps the overlay binary ~40 KB.
#ifndef OSU_OVERLAY_FONT8X8_H
#define OSU_OVERLAY_FONT8X8_H

#include <stdint.h>

typedef uint8_t Glyph8x8[8];

// Returns the glyph bitmap for an ASCII character, or a blank
// glyph for characters we don't have. Only the glyphs we actually
// use on the HUD are populated; everything else comes back blank.
static const Glyph8x8 *font_glyph(char c);

static const Glyph8x8 _font_blank = {0,0,0,0,0,0,0,0};

// ─ digits ───────────────────────────────────────────────────────
static const Glyph8x8 _font_0 = {
    0b00111100,
    0b01100110,
    0b01101110,
    0b01110110,
    0b01100110,
    0b01100110,
    0b00111100,
    0b00000000 };
static const Glyph8x8 _font_1 = {
    0b00011000,
    0b00111000,
    0b00011000,
    0b00011000,
    0b00011000,
    0b00011000,
    0b01111110,
    0b00000000 };
static const Glyph8x8 _font_2 = {
    0b00111100,
    0b01100110,
    0b00000110,
    0b00001100,
    0b00011000,
    0b00110000,
    0b01111110,
    0b00000000 };
static const Glyph8x8 _font_3 = {
    0b00111100,
    0b01100110,
    0b00000110,
    0b00011100,
    0b00000110,
    0b01100110,
    0b00111100,
    0b00000000 };
static const Glyph8x8 _font_4 = {
    0b00001100,
    0b00011100,
    0b00111100,
    0b01101100,
    0b01111110,
    0b00001100,
    0b00001100,
    0b00000000 };
static const Glyph8x8 _font_5 = {
    0b01111110,
    0b01100000,
    0b01111100,
    0b00000110,
    0b00000110,
    0b01100110,
    0b00111100,
    0b00000000 };
static const Glyph8x8 _font_6 = {
    0b00111100,
    0b01100000,
    0b01100000,
    0b01111100,
    0b01100110,
    0b01100110,
    0b00111100,
    0b00000000 };
static const Glyph8x8 _font_7 = {
    0b01111110,
    0b00000110,
    0b00001100,
    0b00011000,
    0b00110000,
    0b00110000,
    0b00110000,
    0b00000000 };
static const Glyph8x8 _font_8 = {
    0b00111100,
    0b01100110,
    0b01100110,
    0b00111100,
    0b01100110,
    0b01100110,
    0b00111100,
    0b00000000 };
static const Glyph8x8 _font_9 = {
    0b00111100,
    0b01100110,
    0b01100110,
    0b00111110,
    0b00000110,
    0b01100110,
    0b00111100,
    0b00000000 };

// ─ punctuation ──────────────────────────────────────────────────
static const Glyph8x8 _font_dot = {
    0,0,0,0,0,0b00011000,0b00011000,0 };
static const Glyph8x8 _font_percent = {
    0b01100010,
    0b01100110,
    0b00001100,
    0b00011000,
    0b00110000,
    0b01100110,
    0b01000110,
    0 };
static const Glyph8x8 _font_x = {
    0,0,
    0b01000010,
    0b00100100,
    0b00011000,
    0b00100100,
    0b01000010,
    0 };
static const Glyph8x8 _font_minus = {
    0,0,0,
    0b01111110,
    0,0,0,0 };
static const Glyph8x8 _font_colon = {
    0,
    0b00011000,
    0b00011000,
    0,0,
    0b00011000,
    0b00011000,
    0 };

// ─ letters (only the ones we print) ─────────────────────────────
static const Glyph8x8 _font_U = {
    0b01100110,
    0b01100110,
    0b01100110,
    0b01100110,
    0b01100110,
    0b01100110,
    0b00111100,
    0 };
static const Glyph8x8 _font_R = {
    0b01111100,
    0b01100110,
    0b01100110,
    0b01111100,
    0b01111000,
    0b01101100,
    0b01100110,
    0 };
static const Glyph8x8 _font_O = {
    0b00111100,
    0b01100110,
    0b01100110,
    0b01100110,
    0b01100110,
    0b01100110,
    0b00111100,
    0b00000000 };
static const Glyph8x8 _font_S = {
    0b00111110,
    0b01100000,
    0b01100000,
    0b00111100,
    0b00000110,
    0b00000110,
    0b01111100,
    0 };
static const Glyph8x8 _font_B = {
    0b01111100,
    0b01100110,
    0b01100110,
    0b01111100,
    0b01100110,
    0b01100110,
    0b01111100,
    0 };
static const Glyph8x8 _font_Y = {
    0b01100110,
    0b01100110,
    0b01100110,
    0b00111100,
    0b00011000,
    0b00011000,
    0b00011000,
    0 };
static const Glyph8x8 _font_M = {
    0b01100110,
    0b01110110,
    0b01111110,
    0b01111110,
    0b01101110,
    0b01100110,
    0b01100110,
    0 };
static const Glyph8x8 _font_C = {
    0b00111110,
    0b01100000,
    0b01100000,
    0b01100000,
    0b01100000,
    0b01100000,
    0b00111110,
    0 };
static const Glyph8x8 _font_A = {
    0b00011000,
    0b00111100,
    0b01100110,
    0b01100110,
    0b01111110,
    0b01100110,
    0b01100110,
    0 };
static const Glyph8x8 _font_T = {
    0b01111110,
    0b00011000,
    0b00011000,
    0b00011000,
    0b00011000,
    0b00011000,
    0b00011000,
    0 };
static const Glyph8x8 _font_I = {
    0b00111100,
    0b00011000,
    0b00011000,
    0b00011000,
    0b00011000,
    0b00011000,
    0b00111100,
    0 };
static const Glyph8x8 _font_V = {
    0b01100110,
    0b01100110,
    0b01100110,
    0b01100110,
    0b00111100,
    0b00011000,
    0b00011000,
    0 };
static const Glyph8x8 _font_G = {
    0b00111100,
    0b01100110,
    0b01100000,
    0b01101110,
    0b01100110,
    0b01100110,
    0b00111100,
    0 };
static const Glyph8x8 _font_E = {
    0b01111110,
    0b01100000,
    0b01100000,
    0b01111100,
    0b01100000,
    0b01100000,
    0b01111110,
    0 };
static const Glyph8x8 _font_N = {
    0b01100110,
    0b01110110,
    0b01111110,
    0b01111110,
    0b01101110,
    0b01100110,
    0b01100110,
    0 };
static const Glyph8x8 _font_L = {
    0b01100000,
    0b01100000,
    0b01100000,
    0b01100000,
    0b01100000,
    0b01100000,
    0b01111110,
    0 };
static const Glyph8x8 _font_F = {
    0b01111110,
    0b01100000,
    0b01100000,
    0b01111100,
    0b01100000,
    0b01100000,
    0b01100000,
    0 };
static const Glyph8x8 _font_P = {
    0b01111100,
    0b01100110,
    0b01100110,
    0b01111100,
    0b01100000,
    0b01100000,
    0b01100000,
    0 };
static const Glyph8x8 _font_K = {
    0b01100110,
    0b01101100,
    0b01111000,
    0b01110000,
    0b01111000,
    0b01101100,
    0b01100110,
    0 };

static const Glyph8x8 *font_glyph(char c) {
    switch (c) {
        case '0': return &_font_0;
        case '1': return &_font_1;
        case '2': return &_font_2;
        case '3': return &_font_3;
        case '4': return &_font_4;
        case '5': return &_font_5;
        case '6': return &_font_6;
        case '7': return &_font_7;
        case '8': return &_font_8;
        case '9': return &_font_9;
        case '.': return &_font_dot;
        case '%': return &_font_percent;
        case 'x': case 'X': return &_font_x;
        case '-': return &_font_minus;
        case ':': return &_font_colon;
        case 'U': case 'u': return &_font_U;
        case 'R': case 'r': return &_font_R;
        case 'O': case 'o': return &_font_O;
        case 'S': case 's': return &_font_S;
        case 'B': case 'b': return &_font_B;
        case 'Y': case 'y': return &_font_Y;
        case 'M': case 'm': return &_font_M;
        case 'C': case 'c': return &_font_C;
        case 'A': case 'a': return &_font_A;
        case 'T': case 't': return &_font_T;
        case 'I': case 'i': return &_font_I;
        case 'V': case 'v': return &_font_V;
        case 'G': case 'g': return &_font_G;
        case 'E': case 'e': return &_font_E;
        case 'N': case 'n': return &_font_N;
        case 'L': case 'l': return &_font_L;
        case 'F': case 'f': return &_font_F;
        case 'P': case 'p': return &_font_P;
        case 'K': case 'k': return &_font_K;
        default:  return &_font_blank;
    }
}

#endif
