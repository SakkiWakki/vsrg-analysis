#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform float u_time;
uniform vec3 u_strength;

out vec4 o_colour;

float random(vec2 st, float seed) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233) + seed)) * 43758.5453123);
}

void main(void) {
    // fluXis feeds this shader strength/10 on both axes and its clock
    // in milliseconds wrapped at 10s; u_strength/u_time carry the raw
    // event values and seconds, so rescale here.
    float strengthX = u_strength.x / 10.0;
    float strengthY = u_strength.y / 10.0;
    float blockSize = u_strength.z;
    float time = mod(u_time * 1000.0, 10000.0);

    vec2 uv = gl_FragCoord.xy / u_resolution;

    float blockSizeInPixels = mix(1.0, min(u_resolution.x, u_resolution.y),
                                  blockSize);
    vec2 blockUV = floor(uv * blockSizeInPixels) / blockSizeInPixels;

    float randomShiftX = (random(blockUV, time) - 0.5) * strengthX;
    float randomShiftY = (random(blockUV + vec2(5.0), time) - 0.5) * strengthY;

    vec2 fixedUV = uv + vec2(randomShiftX, randomShiftY);

    o_colour = textureLod(u_tex, fixedUV, 0.0);
}
