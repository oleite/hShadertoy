// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord.xy / iResolution.xy;
    float angle = GLSL_sin(iTime) * 2.0f;
    mat2 rotation = (mat2)(GLSL_cos(angle), -GLSL_sin(angle), GLSL_sin(angle), GLSL_cos(angle));
    float2 rotatedUV = GLSL_mul(rotation, uv);
    fragColor = (float4)(rotatedUV, 0.5f + 0.5f * GLSL_sin(iTime), 1.0f);
// ---- SHADERTOY CODE END ----