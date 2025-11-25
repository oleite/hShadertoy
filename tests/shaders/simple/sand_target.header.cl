// Sand - Random Gradient Noise 
// https://www.shadertoy.com/view/3ccGRH
const float2 HASH_VECTOR = (float2)(127.1f, 311.7f);
const float HASH_SCALE = 43758.5453123f;
const float GAMMA = 0.45f;

float hash(float2 p) { return GLSL_fract(sin(dot(p, HASH_VECTOR)) * HASH_SCALE); }

float noise(float2 p) {
    float2 i = GLSL_floor(p);
    float2 f = GLSL_fract(p);
    float a = hash(i);
    float b = hash(i + (float2)(1.0f, 0.0f));
    float c = hash(i + (float2)(0.0f, 1.0f));
    float d = hash(i + (float2)(1.0f, 1.0f));
    float2 u = f * f * (3.0f - 2.0f * f);

    return GLSL_mix(a, b, u.x) + (c - a) * u.y * (1.0f - u.x) + (d - b) * u.x * u.y;
}