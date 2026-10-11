// WAYMO AUTONOMOUS VISION 4D - road / terrain surface (GLSL MAT, vertex).
// Height field built per frame in frame_loader.build_ground (Perception: lowest LiDAR returns, Motion: map raster,
// E2E: driving corridor).  A 512x512 vertex grid samples the 256x256 texture bilinearly, so edges and slopes are smooth.
// Samplers : sGround r = height (TD y, m)  g = lane-paint / intensity  b = coverage 0..1  a = camera-colour fraction
//            sColor  rgb = base colour of the cell
// Vectors  : uCenter (TD x,z of the grid centre), uExtent (m), uDrop (m)
uniform sampler2D sGround;
uniform vec2  uCenter;
uniform float uExtent;
uniform float uDrop;

out vec2  vUV;
out float vDist;
out vec4  vG;

void main()
{
    vec2 uvv = uv[0].st;
    vec4 g = texture(sGround, uvv);
    vec3 p;
    p.x = uCenter.x + (uvv.x - 0.5) * uExtent;
    p.z = uCenter.y - (uvv.y - 0.5) * uExtent;        // TD z = -Waymo y
    p.y = g.r - uDrop;
    gl_Position = TDWorldToProj(TDDeform(p));
    vUV = uvv;
    vG = g;
    vDist = length(vec2(uvv - 0.5) * uExtent);
}
