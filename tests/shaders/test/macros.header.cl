#define random(x) GLSL_fract(1e4f*GLSL_sin((x)*541.17f))

#define PI 3.14159265f

#define DIRECTION_X

#define ANIMATE

float somefunc(float x) {
    float2 o = GLSL_mix(x, 1.0f, 0.5f);
#ifdef ANIMATE // If ANIMATE is defined
        o = GLSL_mix( o + iTime, x, 0.5f);
    #else // Otherwise
        o = GLSL_mix( o, x, 0.5f);
    #endif
    return o.x;
}

