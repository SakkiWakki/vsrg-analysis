#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

void main(void) {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec4 colour = texture(u_tex, uv);

    vec3 inverted = 1.0 - colour.rgb;
    o_colour = vec4(mix(colour.rgb, inverted, u_strength.x), colour.a);
}
