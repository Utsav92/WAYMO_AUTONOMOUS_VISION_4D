// WAYMO AUTONOMOUS VISION 4D - timestamp-faded history layer (GLSL MAT, pixel). Pairs with lidar.vert.
// vAge is REAL elapsed seconds between the point's frame and the current frame (from frame timestamps), so uneven
// frame spacing fades correctly.  recent = cyan, intermediate = purple, oldest = translucent white.
uniform float uAgeMax;
uniform float uOpacity;
uniform float uCinematic;

in float vRange;
in float vInt;
in float vH;
in float vAge;
in float vFade;
in vec4  vRgb;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

void main()
{
    TDCheckDiscard();
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = length(c);
    if (r > 1.0) discard;
    float u = clamp(vAge / max(uAgeMax, 0.05), 0.0, 1.0);
    vec3 cyan = vec3(0.0, 0.85, 1.0), purple = vec3(0.62, 0.2, 1.0), white = vec3(0.9, 0.9, 1.0);
    vec3 col = u < 0.5 ? mix(cyan, purple, u * 2.0) : mix(purple, white, (u - 0.5) * 2.0);
    float halo = pow(1.0 - r, 2.0);
    float a = (1.0 - 0.85 * u) * (0.4 + 0.6 * halo) * uOpacity * vFade;
    // emissive: additive blending (one, one) => colour is pre-multiplied by its fade
    vec4 color = vec4(col * (1.0 + 0.5 * uCinematic) * a * 2.0, a);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
