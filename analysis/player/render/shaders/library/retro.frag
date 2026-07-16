#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

// https://www.shadertoy.com/view/WsVSzV

void main(void) {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float strength = u_strength.x;

    vec2 dc = abs(vec2(0.5) - uv);
    dc *= dc;

    uv.x -= 0.5; uv.x *= 1.0 + (dc.y * (0.3 * strength)); uv.x += 0.5;
    uv.y -= 0.5; uv.y *= 1.0 + (dc.x * (0.4 * strength)); uv.y += 0.5;

    if (uv.y > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0) {
        o_colour = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        float apply = abs(sin(gl_FragCoord.y) * 0.5 * strength);
        o_colour = vec4(mix(texture(u_tex, uv).rgb, vec3(0.0), apply), 1.0);
    }
}
