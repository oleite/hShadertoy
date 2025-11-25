// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float noise = random(fragCoord.x + fragCoord.y);
    float2 pixelnoise;
    random2(fragCoord, &pixelnoise);
    float3 col = (float3)(pixelnoise, noise);
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----