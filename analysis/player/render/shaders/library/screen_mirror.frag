#version 150

// NotITG built-in "mirror" screen flag as a fullscreen pass. Reflects
// the frame about its center; u_strength.x blends identity -> mirrored so
// the effect is an identity map at strength 0 (the library contract).
// u_strength.y selects the axis (>= 0.5 = vertical mirror / left-right,
// else horizontal / top-bottom).

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

void main(void) {
    float strength = clamp(u_strength.x, 0.0, 1.0);
    bool vertical = u_strength.y >= 0.5;

    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 mirrored = vertical ? vec2(1.0 - uv.x, uv.y)
                             : vec2(uv.x, 1.0 - uv.y);

    // Mirror only the far half so the near half stays original; at full
    // strength this reads as a folded reflection about the center line.
    float coord = vertical ? uv.x : uv.y;
    vec2 uvOut = (coord > 0.5) ? mix(uv, mirrored, strength) : uv;
    o_colour = texture(u_tex, uvOut);
}
