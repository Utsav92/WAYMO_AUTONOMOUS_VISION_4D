// WAYMO AUTONOMOUS VISION 4D - road / terrain colouring (GLSL MAT, pixel).
//  * base: camera colour where cameras saw the cell, else the base colour texture / asphalt tone
//  * lane paint & retro-reflective markings: lit from the paint channel (neon cyan-white)
//  * distance fog into the sky horizon colour; soft coverage edge (no stair-steps)
uniform sampler2D sGround;
uniform sampler2D sColor;
uniform float uRgbMix;
uniform float uMark;
uniform float uCinematic;
uniform float uExtent;
uniform float uMaxDist;

in vec2  vUV;
in float vDist;
in vec4  vG;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

void main()
{
    TDCheckDiscard();
    vec4 g = texture(sGround, vUV);
    if (g.b < 0.5) discard;
    vec3 base = texture(sColor, vUV).rgb;
    float haveCam = clamp(g.a * 1.5, 0.0, 1.0);
    vec3 asphalt = vec3(0.030, 0.050, 0.062);
    vec3 col = mix(asphalt, base, clamp(uRgbMix, 0.0, 1.0) * max(haveCam, step(0.001, dot(base, vec3(1.0)))));
    col = max(col, vec3(0.030, 0.045, 0.055));                    // shadows stay dark, never pure black
    float l = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(l), col, 1.30) * 1.04;                      // a little saturation / lift
    float mark = smoothstep(0.40, 0.70, g.g) * uMark;
    vec3 neon = vec3(0.45, 0.95, 1.0);
    col = mix(col, neon, mark * 0.9) + neon * mark * 0.30 * (0.4 + uCinematic);
    float fog = smoothstep(0.35 * uMaxDist, 0.95 * uMaxDist, vDist);
    vec3 fogc = vec3(0.030, 0.095, 0.130);
    col = mix(col, fogc, fog);
    vec4 color = vec4(col, 1.0);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
