// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
typedef struct {
        float3 a;
        float b, c;
} Foo;
    Foo _bar = {(float3)(0), 1.0f, 0.5f};
    Hit _hit = {(float3)(0), 1.0f, 0.5f};
    _bar.a = _hit.p;
    float x = _hit.t;
    float3 col = _cam.p;
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----