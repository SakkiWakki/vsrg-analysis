#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

void main(void) {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec4 colour = texture(u_tex, uv);
    float strength = u_strength.x;

    uv = uv * (vec2(1.0) - uv.yx);
    float vig = uv.x * uv.y * (1.0 - strength);
    vig = pow(vig, strength);
    o_colour = mix(colour, vec4(0.0, 0.0, 0.0, 1.0), 1.0 - vig);
}
