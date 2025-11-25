float random(float x) {
    return GLSL_fract(1e4f * GLSL_sin((x) * 23.27f));
}

void random2(float2 uv, __private float2* noise2) {
    float a = GLSL_fract(1e4f * GLSL_sin((uv.x) * 541.17f));
    float b = GLSL_fract(1e4f * GLSL_sin((uv.y) * 321.46f));
    *noise2 = (float2)(a, b);
}

