// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float fov = 1.8f;
    float3 E;
    float3 I;
    float px = 1.0f * (2.0f / iResolution.y) * 1.0f / fov;
    float t = -(1.0f + E.y) / I.y;
    float3 col = (float3)(px, t, 0.0f);
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----