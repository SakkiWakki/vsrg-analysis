#version 150

uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform float u_time;
uniform vec3 u_strength;

out vec4 o_colour;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 4.1414))) * 43758.5453123);
}

float randomRange(vec2 st, float lo, float hi) {
    return lo + random(st) * (hi - lo);
}

float insideRange(float v, float b, float t) {
    return step(b, v) - step(t, v);
}

void main(void) {
    // fluXis feeds Amount = strength, Speed = strength2, and its clock in
    // seconds; u_strength/u_time carry the raw event values already.
    float amount = u_strength.x;
    float speed = u_strength.y;
    float time = floor(u_time * speed * 60.0);

    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec4 colour = textureLod(u_tex, uv, 0.0);

    float maxOff = amount / 2.0;
    // fluXis slices `for (i = 0; i < 10 * Amount; i++)`; a constant loop
    // bound (max Amount seen is 1 -> ceil(10)) keeps ES/desktop compilers
    // happy, with a break once i passes the per-frame slice count.
    float sliceCount = 10.0 * amount;
    for (int i = 0; i < 10; i++) {
        if (float(i) >= sliceCount) break;
        float fi = float(i);
        float slcY = random(vec2(time, 2345.0 + fi));
        float slcH = random(vec2(time, 9035.0 + fi)) * 0.25;
        float hOff = randomRange(vec2(time, 9625.0 + fi), -maxOff, maxOff);

        vec2 uvOff = uv;
        uvOff.x += hOff;

        if (insideRange(uv.y, slcY, fract(slcY + slcH)) == 1.0) {
            colour = textureLod(u_tex, uvOff, 0.0);
        }
    }

    float maxColOff = amount / 6.0;
    float rnd = random(vec2(time, 9545.0));
    vec2 colOffset = vec2(
        randomRange(vec2(time, 9545.0), -maxColOff, maxColOff),
        randomRange(vec2(time, 7205.0), -maxColOff, maxColOff));

    if (rnd < 0.33) {
        colour.r = textureLod(u_tex, uv + colOffset, 0.0).r;
    } else if (rnd < 0.66) {
        colour.g = textureLod(u_tex, uv + colOffset, 0.0).g;
    } else {
        colour.b = textureLod(u_tex, uv + colOffset, 0.0).b;
    }

    o_colour = colour;
}
