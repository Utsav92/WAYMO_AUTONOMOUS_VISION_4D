// WAYMO AUTONOMOUS VISION 4D - glow composite (GLSL TOP).  Input 0 = scene, input 1 = blurred scene.
// uGlow = 0 gives the untouched scientific image.
uniform float uGlow;
uniform float uVignette;
out vec4 fragColor;

void main()
{
    vec4 scene = texture(sTD2DInputs[0], vUV.st);
    vec4 blur  = texture(sTD2DInputs[1], vUV.st);
    vec3 rgb = scene.rgb + blur.rgb * uGlow;
    vec2 d = vUV.st - 0.5;
    rgb *= 1.0 - uVignette * smoothstep(0.35, 0.85, length(d));
    fragColor = TDOutputSwizzle(vec4(rgb, 1.0));
}
