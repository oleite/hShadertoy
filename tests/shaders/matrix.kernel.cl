// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord / iResolution.xy;
    float2 V2 = (float2)(1.0f, 0.0f);
    float3 V3 = (float3)(1.0f, 0.0f, 0.0f);
    float4 V4 = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
    matrix2x2 M2 = GLSL_mat2(1.0f, 0.0f, 0.0f, 1.0f);
    matrix3x3 M3 = GLSL_mat3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    matrix4x4 M4 = GLSL_mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    float2 op1 = GLSL_mul_vec2_mat2(V2, M2);
    float2 op2 = GLSL_mul_vec2_mat2(GLSL_mul_vec2_mat2(V2, M2), M2);
    op2 = GLSL_mul_vec2_mat2(op2, M2);
    op2 = GLSL_mul_vec2_mat2(op2, foo(3.14f));
    float3 op3 = GLSL_mul_vec3_mat3(V3, M3);
    float3 op4 = GLSL_mul_vec3_mat3(GLSL_mul_vec3_mat3(V3, M3), M3);
    op4 = GLSL_mul_vec3_mat3(op4, M3);
    op4 = GLSL_mul_vec3_mat3(op4, bar((float3)(1.0f, 0.0f, 0.0f), 3.14f));
    float3 op5 = GLSL_mul_mat3_vec3(M3, GLSL_normalize((float3)(V2, 1.0f)));
    float3 op6 = GLSL_mul_vec3_mat3(GLSL_cross(GLSL_mul_vec3_mat3(GLSL_normalize(V3), M3), GLSL_mul_mat3_vec3(M3, (float3)(1.0f, V2))), M3);
    op6 = GLSL_mul_vec3_mat3(op6, translate(V2));
    op6 = tonemap(V3);
    float4 o7 = GLSL_mul_vec4_mat4(V4, M4);
    float4 op8 = GLSL_mul_vec4_mat4(GLSL_mul_vec4_mat4(V4, M4), M4);
    op8 = GLSL_mul_vec4_mat4(op8, M4);
    op8 = GLSL_mul_vec4_mat4(V4, foobar(V3, 3.14f));
    matrix2x2 xf1 = GLSL_transpose(M2);
    matrix2x2 xf2 = GLSL_inverse(M2);
    matrix3x3 xf3 = GLSL_transpose_mat3(M3);
    matrix3x3 xf4 = GLSL_inverse_mat3(M3);
    matrix4x4 xf5 = GLSL_transpose_mat4(M4);
    matrix4x4 xf6 = GLSL_inverse_mat4(M4);
    fragColor = (float4)(uv, 0.0f, 1.0f);
// ---- SHADERTOY CODE END ----