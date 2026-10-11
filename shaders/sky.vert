// WAYMO AUTONOMOUS VISION 4D - sky dome (GLSL MAT, vertex).  A large sphere centred on the camera; direction = vertex.
out vec3 vDir;

void main()
{
    vDir = normalize(P);
    gl_Position = TDWorldToProj(TDDeform(P));
}
