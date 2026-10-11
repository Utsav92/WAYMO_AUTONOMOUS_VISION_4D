// WAYMO AUTONOMOUS VISION 4D - LiDAR / RGB / history point sprites (GLSL MAT, vertex). One draw call, no per-point objects.
// Geometry : POINT Grid SOP 512 x 256 = 131,072 slots, uv 0..1 (point-sprite particles).
// Samplers : sPos   RGBA32F  rgb = TD-space position (x fwd, y up, z = -left), a = measured range m (0 = empty slot)
//            sAttr  RGBA32F  r = intensity, g = elongation, b = sensor id (1 TOP..5 REAR), a = age s (0 = current frame)
//            sRgb   RGBA32F  rgb = camera colour 0..1, a = 1 when a camera saw the point
// Vectors  : uPointSize px @ uRefDist m, uMaxRange, uDensity 0..1, uAgeMax s, uSensorMask (bitmask over sensor ids),
//            uHeightLo/Hi (TD y), uCamPos
// Geometry is NEVER altered by the render mode: positions come straight from sPos.
uniform sampler2D sPos;
uniform sampler2D sAttr;
uniform sampler2D sRgb;
uniform float uPointSize;
uniform float uRefDist;
uniform float uMaxRange;
uniform float uDensity;
uniform float uAgeMax;
uniform float uSensorMask;
uniform float uHeightLo;
uniform float uHeightHi;
uniform vec3  uCamPos;

out float vRange;
out float vInt;
out float vH;
out float vAge;
out float vSem;
out float vFade;
out vec4  vRgb;

void main()
{
    ivec2 sz = textureSize(sPos, 0);
    ivec2 px = clamp(ivec2(floor(uv[0].st * vec2(sz - 1) + 0.5)), ivec2(0), sz - 1);
    vec4 p  = texelFetch(sPos, px, 0);
    vec4 at = texelFetch(sAttr, px, 0);
    vec4 c  = texelFetch(sRgb, px, 0);

    vec3 pos = p.xyz;
    float dist = distance(pos, uCamPos);
    float h = fract(sin(dot(vec2(px), vec2(12.9898, 78.233))) * 43758.5453);
    float sid = floor(at.b + 0.5);
    float bit = mod(floor(uSensorMask / exp2(sid)), 2.0);
    bool visible = (p.a > 0.0) && (p.a <= uMaxRange) && (h <= uDensity) && (at.a <= uAgeMax) && (bit > 0.5);

    vec4 worldPos = TDDeform(pos);
    gl_Position = visible ? TDWorldToProj(worldPos) : vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = visible ? clamp(uPointSize * uRefDist / max(dist, 1.0), 1.2, 2.6 * uPointSize + 1.0) : 0.0;

    vRange = p.a;
    vInt = at.r;
    vH = clamp((pos.y - uHeightLo) / max(uHeightHi - uHeightLo, 1e-3), 0.0, 1.0);
    vAge = at.a;
    vSem = at.g;
    vFade = 1.0 - smoothstep(0.8 * uMaxRange, uMaxRange, p.a);
    vRgb = c;
}
