#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

#define PI 3.141593

vec2 FishEyeUV(vec2 uv) {
    vec2 center = vec2(0.5);
    float corner = length(center);
    vec2 d = uv - 0.5;
    float r = length(d);
    float strength = u_strength.x;

    if (strength > 0.0) {
        float fac = strength * PI * 0.5;
        uv = vec2(0.5) + normalize(d) * tan(r * fac) * corner / tan(corner * fac);
    } else {
        float fac = tan(strength) * PI * 2.0;
        uv = vec2(0.5) + normalize(d) * atan(r * -fac) * 0.5 / atan(0.5 * -fac);
    }
    return uv;
}

void main(void) {
    vec2 uv = gl_FragCoord.xy / u_resolution;

    o_colour = texture(u_tex, FishEyeUV(uv));
}
