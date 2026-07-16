#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

const float maxSamples = 10.0;

void main(void) {
    float strength = u_strength.x;
    vec2 scaleFac = u_strength.y + vec2(1.0);

    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 toCenter = uv - vec2(0.5);

    vec3 color = vec3(0.0);
    vec2 scale = vec2(1.0);
    float sampleStrength = 1.0;

    // fluXis's early-out for weak strengths shadows its own loop bound,
    // so the shipped behavior is a fixed 10 samples; match that.
    for (float i = 0.0; i < maxSamples; i++) {
        vec2 sampleUV = toCenter / scale + vec2(0.5);
        vec4 sampled = texture(u_tex, sampleUV);
        color += sampleStrength * sampled.w * sampled.xyz;
        scale *= scaleFac;
        sampleStrength *= strength;
    }

    o_colour = vec4(color, 1.0);
}
