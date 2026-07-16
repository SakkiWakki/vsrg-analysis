#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

void main(void) {
    vec2 uv = gl_FragCoord.xy / u_resolution;

    float pixelSizeFactor = mix(1.0, min(u_resolution.x, u_resolution.y),
                                1.0 - u_strength.x);
    vec2 pixelSize = vec2(pixelSizeFactor,
                          pixelSizeFactor * (u_resolution.y / u_resolution.x));
    vec2 pixelatedUV = (floor(uv * pixelSize) + 0.5) / pixelSize;

    o_colour = textureLod(u_tex, pixelatedUV, 0.0);
}
