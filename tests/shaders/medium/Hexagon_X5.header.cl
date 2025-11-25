// Hexagon X5
// https://www.shadertoy.com/view/4cVfWG
/** 

    License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0f Unported License
    Playing in 2D for something I want to make in 3D/Raymarching. 
    Then got caught up in the polar coords thing, and said - thats 
    a neat design!
    
    Hexagon X5
    12/17/2024  @byt3_m3chanic
    
*/

#define R     iResolution
#define T     iTime
#define M     iMouse

#define PI    3.141592653f
#define PI2   6.283185307f

const float N = 3.f;
const float s4 = .577350f, s3 = .288683f, s2 = .866025f;
const float2 s = (float2)(1.732f,1);

float3 clr, trm;
float tk, ln;
mat2 r2,r3;

mat2 rot(float g) { return mat2(cos(g), sin(g),-sin(g), cos(g)); }
float hash21(float2 p) { 
    p.x = gol_mod(p.x,3.f*N);
    return gol_fract(sin(dot(p,(float2)(26.37f,45.93f)))*4374.23f); 
}

// Hexagon grid system, can be simplified but
// written out long-form for readability. 
// return float2 uv and float2 id
float4 hexgrid(float2 uv) {
    float2 p1 = floor(uv/(float2)(1.732f,1))+.5f,
         p2 = floor((uv-(float2)(1,.5f))/(float2)(1.732f,1))+.5f;
    float2 h1 = uv- p1*(float2)(1.732f,1),
         h2 = uv-(p2+.5f)*(float2)(1.732f,1);
    return dot(h1,h1) < dot(h2,h2) ? (float4)(h1,p1) : (float4)(h2,p2+.5f);
}

void draw(float d, float px, inout float3 C) {
    float b = abs(d)-tk;
    C = gol_mix(C,C*.25f,gol_smoothstep(.1f+px,-px,b-.01f) );
    C = gol_mix(C,clr,gol_smoothstep(px,-px,b ));
    C = gol_mix(C,clamp(C+.2f,C,(float3)(.95f)),gol_smoothstep(.01f+px,-px, b+.1f ));
    C = gol_mix(C,trm,gol_smoothstep(px,-px,abs(b)-ln ));
}