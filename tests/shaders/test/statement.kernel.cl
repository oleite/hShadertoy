// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord / iResolution.xy;
    float3 col;
    if (uv.x > 0.7f) {
        col = (float3)(1.f, 0.f, 0.f);
    }
    else if (uv.x > 0.4f) {
        col = (float3)(0.f, 1.f, 0.f);
    }
    else {
        col = (float3)(0.f, 0.f, 1.f);
    }
    for (int i = 0; i < 16; ++i) {
        if (i < 8)     }
    return col = uv.y > 0.5f ? col * 0.5f : col;
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----