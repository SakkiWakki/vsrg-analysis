#version 150

uniform sampler2D u_tex;    // blurred glow (vertical blur pass output)
uniform sampler2D u_tex2;   // original scene (pre-chain capture, unit 1)
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

void main(void) {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec3 scene = textureLod(u_tex2, uv, 0.0).rgb;
    vec3 glow = textureLod(u_tex, uv, 0.0).rgb;
    o_colour = vec4(scene + glow * u_strength.x, 1.0);
}
