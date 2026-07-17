#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

const float INV_SQRT_2PI = 0.39894;
const int MAX_RADIUS = 200;

float computeGauss(float x, float sigma) {
    return INV_SQRT_2PI * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

// fluXis picks the kernel radius on the CPU via Blur.KernelSize(sigma):
// the first even offset whose gaussian weight drops below 0.1x the peak.
// Recomputed here so all param massaging stays in the shader.
int kernelRadius(float sigma) {
    if (sigma == 0.0) return 0;
    float threshold = 0.1 * computeGauss(0.0, sigma);
    for (int i = 0; i < MAX_RADIUS; i++) {
        if (computeGauss(float(i), sigma) < threshold) {
            return max(i - 1, 0);
        }
    }
    return MAX_RADIUS;
}

vec4 blur(int radius, vec2 direction, vec2 texCoord, vec2 texSize, float sigma) {
    float factor = computeGauss(0.0, sigma);
    vec4 sum = textureLod(u_tex, texCoord, 0.0) * factor;
    float totalFactor = factor;

    for (int i = 2; i <= MAX_RADIUS; i += 2) {
        float x = float(i) - 0.5;
        factor = computeGauss(x, sigma) * 2.0;
        totalFactor += 2.0 * factor;
        sum += textureLod(u_tex, texCoord + direction * x / texSize, 0.0) * factor;
        sum += textureLod(u_tex, texCoord - direction * x / texSize, 0.0) * factor;
        if (i >= radius) break;
    }

    return sum / totalFactor;
}

void main(void) {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float sigma = 20.0 * u_strength.x;
    if (sigma == 0.0) {
        o_colour = textureLod(u_tex, uv, 0.0);
        return;
    }
    o_colour = blur(kernelRadius(sigma), vec2(0.0, 1.0), uv, u_resolution, sigma);
}
