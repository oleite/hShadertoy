// Moving Voronoi Cells
// https://www.shadertoy.com/view/lccyD8
// Hash Functions
float hash(float2 p) {
    float3 p3 = gol_fract3((float3)(p.xyx) * 0.1031f);
    p3 += dot(p3, p3.yzx + 33.33f);
    return gol_fract((p3.x + p3.y) * p3.z);
}

float2 hash2(float2 p) {
    float3 p3 = gol_fract3((float3)(p.xyx) * 0.1031f);
    p3 += dot(p3, p3.yzx + 33.33f);
    return gol_fract2((float2)((p3.x + p3.y) * p3.z, (p3.y + p3.z) * p3.x));
}

float3 hash3(float2 p) {
    float3 p3 = gol_fract3((float3)(p.xyx + (float3)(1.0f, 2.0f, 3.0f)) * 0.1031f);
    p3 += dot(p3, p3.yzx + 33.33f);
    return gol_fract((p3 + p3.yzx) * p3.zxy);
}

// Main