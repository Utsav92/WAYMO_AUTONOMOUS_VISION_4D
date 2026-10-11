// WAYMO AUTONOMOUS VISION 4D - LiDAR point colouring (GLSL MAT, pixel).
// uColorMode: 0 orange sensor colour (#FF7500), 1 range, 2 intensity, 3 height, 4 camera RGB (orange where unseen)
// uCinematic: 0 = scientific (flat, measurement-faithful), 1 = soft halo / additive glow
uniform float uColorMode;
uniform float uOpacity;
uniform float uCinematic;
uniform float uGain;

in float vRange;
in float vInt;
in float vH;
in float vAge;
in float vSem;
in float vFade;
in vec4  vRgb;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

vec3 ramp(float x)   // blue -> cyan -> green -> yellow -> red
{
    x = clamp(x, 0.0, 1.0);
    return clamp(vec3(1.5 * x - 0.5, 1.0 - abs(2.0 * x - 1.0) * 1.2 + 0.1, 1.1 - 1.5 * x), 0.0, 1.0);
}

// Waymo lidar_segmentation taxonomy (0 undefined ... 22 sidewalk)
vec3 semColor(float c)
{
    int k = int(c + 0.5);
    if (k == 1)  return vec3(0.15, 1.0, 0.35);   // car
    if (k == 2)  return vec3(0.0, 0.75, 0.4);    // truck
    if (k == 3)  return vec3(0.4, 1.0, 0.0);     // bus
    if (k == 4)  return vec3(0.2, 0.9, 0.6);     // other vehicle
    if (k == 5 || k == 6 || k == 13) return vec3(0.0, 0.85, 1.0);  // motorcyclist / bicyclist / motorcycle
    if (k == 7)  return vec3(1.0, 0.85, 0.0);    // pedestrian
    if (k == 8)  return vec3(1.0, 0.27, 0.83);   // sign
    if (k == 9)  return vec3(1.0, 0.1, 0.1);    // traffic light
    if (k == 10) return vec3(0.85, 0.9, 1.0);    // pole
    if (k == 11) return vec3(1.0, 0.45, 0.0);    // cone
    if (k == 12) return vec3(0.0, 0.7, 1.0);     // bicycle
    if (k == 14) return vec3(0.55, 0.3, 1.0);    // building
    if (k == 15) return vec3(0.1, 0.65, 0.15);   // vegetation
    if (k == 16) return vec3(0.5, 0.35, 0.1);    // tree trunk
    if (k == 17) return vec3(0.0, 1.0, 1.0);     // curb
    if (k == 18) return vec3(0.2, 0.35, 0.9);    // road
    if (k == 19) return vec3(1.0, 1.0, 1.0);     // lane marker
    if (k == 20) return vec3(0.4, 0.45, 0.7);    // other ground
    if (k == 21) return vec3(0.8, 0.5, 0.8);    // walkable
    if (k == 22) return vec3(0.0, 0.9, 0.8);     // sidewalk
    return vec3(0.35, 0.35, 0.4);                 // undefined
}

void main()
{
    TDCheckDiscard();
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = length(c);
    if (r > 1.0) discard;

    vec3 orange = vec3(1.0, 0.4588, 0.0);
    vec3 col = orange;
    if (uColorMode > 0.5 && uColorMode < 1.5) col = ramp(vRange / 60.0);
    else if (uColorMode > 1.5 && uColorMode < 2.5) col = mix(vec3(0.15, 0.1, 0.5), vec3(1.0, 0.95, 0.7), clamp(vInt * 1.3, 0.0, 1.0));
    else if (uColorMode > 2.5 && uColorMode < 3.5) col = ramp(vH);
    else if (uColorMode > 3.5 && uColorMode < 4.5) col = mix(orange, vRgb.rgb, vRgb.a);
    else if (uColorMode > 4.5) col = semColor(vSem);

    float halo = pow(1.0 - r, 2.0);
    float core = 1.0 - smoothstep(0.0, 0.55, r);
    float soft = mix(1.0, 0.45 + 0.9 * halo, uCinematic);
    vec3 rgb = col * soft * uGain + vec3(1.0, 0.9, 0.8) * core * 0.25 * uCinematic;
    float a = mix(1.0, clamp(0.35 + 0.65 * halo + core, 0.0, 1.0), uCinematic) * uOpacity * vFade;
    vec4 color = vec4(rgb, a * mix(0.92, 1.0, uCinematic));   // alpha blending (sa / omsa)
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
