// WAYMO AUTONOMOUS VISION 4D - camera-coloured LiDAR layer (GLSL MAT, pixel). Pairs with lidar.vert.
// Only points that a calibrated camera actually saw (a = 1) are drawn; no colour is invented for the rest.
uniform float uOpacity;
uniform float uCinematic;
uniform float uGain;

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
    if (vRgb.a < 0.5) discard;
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = length(c);
    if (r > 1.0) discard;
    float halo = pow(1.0 - r, 2.0);
    float soft = mix(1.0, 0.55 + 0.8 * halo, uCinematic);
    float a = uOpacity * vFade;
    vec4 color = vec4(vRgb.rgb * soft * uGain, a);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
