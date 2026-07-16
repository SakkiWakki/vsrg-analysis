#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

void main(void) {
    float strength = u_strength.x;
    vec2 splits = vec2(max(1.0, trunc(u_strength.y)),
                       max(1.0, trunc(u_strength.z)));
    vec2 splitInv = 1.0 / splits;

    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 quad = trunc(uv * splits);

    vec2 zoom = mix(vec2(1.0), splitInv, vec2(strength));
    vec2 offset = mix(vec2(0.0), quad * splitInv, vec2(strength));

    vec2 uvQuad = (uv - offset) / zoom;
    o_colour = texture(u_tex, uvQuad);
}
