#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform vec3 u_strength;

out vec4 o_colour;

void main(void) {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec4 colour = texture(u_tex, uv);

    float grey = dot(colour.rgb, vec3(0.299, 0.587, 0.114));
    o_colour = mix(colour, vec4(vec3(grey), colour.a), u_strength.x);
}
