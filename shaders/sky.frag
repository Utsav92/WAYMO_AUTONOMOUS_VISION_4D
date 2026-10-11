// WAYMO AUTONOMOUS VISION 4D - sky dome colour: deep navy zenith, teal horizon glow, dark ground fog below.
// The horizon colour equals the fog colour used by ground.frag so the road dissolves into the sky.
in vec3 vDir;
layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

void main()
{
    float h = vDir.y;
    vec3 horizon = vec3(0.030, 0.095, 0.130);
    vec3 zenith  = vec3(0.004, 0.012, 0.034);
    vec3 below   = vec3(0.012, 0.035, 0.050);
    vec3 col = h > 0.0 ? mix(horizon, zenith, pow(clamp(h, 0.0, 1.0), 0.45))
                       : mix(horizon, below, clamp(-h * 4.0, 0.0, 1.0));
    col += horizon * 0.9 * exp(-abs(h) * 22.0);                 // soft glow band on the horizon
    vec4 color = vec4(col, 1.0);
    fragColor[0] = TDOutputSwizzle(color);
}
