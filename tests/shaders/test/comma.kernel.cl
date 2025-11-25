// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float foo;
    float3 bar;
    float x, y, z;
    int a = 10, b = 20;
    float3 position, normal, tangent;
    x = (float)(b);
    y = 0.5f;
    z = 0.0f;
    float3 col = (float3)(x, y, z) + normal;
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----