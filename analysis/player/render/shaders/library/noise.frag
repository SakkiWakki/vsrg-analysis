#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform float u_time;
uniform vec3 u_strength;

out vec4 o_colour;

float random(vec2 st, float size) {
    st = floor(st * size) / size;

    // avoid a column of non-changing pixels
    if (st.x == 0.0)
        st = vec2(0.4, st.y);

    return fract(sin(dot(st.xy, vec2(u_time, 78.233))) * 43758.5453123);
}

void main(void) {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec4 colour = texture(u_tex, uv);

    float ratio = u_resolution.x / u_resolution.y;

    float rng = random(vec2(uv.x * ratio, uv.y), u_resolution.y / 2.0);
    rng *= 0.75;

    o_colour = vec4(mix(colour.rgb, vec3(rng), u_strength.x), colour.a);
}
