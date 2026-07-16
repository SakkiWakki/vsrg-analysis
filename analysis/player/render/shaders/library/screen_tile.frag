#version 150

// NotITG built-in "tile" screen flag as a fullscreen pass: the frame is
// repeated into an NxN grid of mirror-flipped copies (a kaleidoscope
// fold), matching the gat reference's tiled-playfield look. u_strength.x
// blends identity -> tiled so strength 0 is an identity map (the library
// contract). u_strength.y sets the tile count per axis (>= 2 tiles; the
// default of 0 falls back to a 2x2 grid).

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

void main(void) {
    float strength = clamp(u_strength.x, 0.0, 1.0);
    float tiles = max(2.0, trunc(u_strength.y + 0.5));
    if (u_strength.y < 1.0) tiles = 2.0;

    vec2 uv = gl_FragCoord.xy / u_resolution;

    // Fold each tile so neighbours mirror at their shared edge: scale
    // into tile space, then reflect the odd tiles (triangle wave).
    vec2 scaled = uv * tiles;
    vec2 cell = mod(scaled, 2.0);
    vec2 folded = 1.0 - abs(cell - 1.0);

    o_colour = texture(u_tex, mix(uv, folded, strength));
}
