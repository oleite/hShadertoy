#define T iTime

float4 smoke(float2 u) {
    float i, d, s, t = iTime;
    float3 p = iResolution;
    float4 o = (float4)(0);
    for (; ++i < 64.f; )     for (; s < 4.f; s += s)     ;
    return o / 2e3f;
}

float4 fire(float2 u) {
    float i, d, s, n;
    float3 p;
    float4 o = (float4)(0);
    for (; ++i < 64.f; ) {
        p = (float3)(u * d, d);
        p += GLSL_cos(p.z + T + p.yzx * .5f) * .6f;
        s = p.y - 2.f;
        p.yz = GLSL_mul( p.yz, (mat2)(GLSL_cos((float4)(.3f) * (float4)(T) + (float4)(0, 33, 11, 0))));
        for (n = 1.6f; n < 32.f; n += n)         s += GLSL_abs(GLSL_dot(GLSL_sin(p.z + T + p * n), (float3)(2.5f))) / n;
        d += s = .01f + GLSL_abs(s) * .1f;
        o += 1.f / s;
    }
    return (float4)(6, 2, 1, 1) * o * o / d / 2e5f;
}

