#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

void main(void) {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec4 colour = texture(u_tex, uv);

    vec2 offset = vec2(0.001, 0.0) * u_strength.x;
    colour.r = texture(u_tex, uv + offset).r;
    colour.b = texture(u_tex, uv - offset).b;

    o_colour = colour;
}
