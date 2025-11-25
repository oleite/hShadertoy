// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
r2 = rot( 1.047f);
    r3 = rot(-1.047f);
    
    float2 uv = (2.f*fragCoord-R.xy)/max(R.x,R.y);
  
    uv = -(float2)(log(length(uv)),atan(uv.y,uv.x))-((2.f*M.xy-R.xy)/R.xy);
    uv /= 3.628f;
    uv *= N;
        
    uv.y += T*.05f;
    uv.x += T*.15f;
    float2 mv=uv;
    float sc = 3.f, px = 0.01f;

    float4 H = hexgrid(uv.yx*sc);
    float2 p = H.xy, id = H.zw;

    float hs = hash21(id);
    
    if(hs<.5f) p *= hs < .25f ? r3 : r2;

    float2 p0 = p - (float2)(-s3, .5f),
         p1 = p - (float2)( s4,  0),
         p2 = p - (float2)(-s3,-.5f);

    float3 d3 = (float3)(length(p0), length(p1), length(p2));
    float2 pp = (float2)(0);

    if(d3.x>d3.y) pp = p1;
    if(d3.y>d3.z) pp = p2;
    if(d3.z>d3.x && d3.y>d3.x) pp = p0;
     
    ln = .015f;
    tk = .14f+.1f*sin(uv.x*5.f+T);
    
    float3 C = (float3)(0);
    
    // tile background
    float d = max(abs(p.x)*.866025f + abs(p.y)/2.f, abs(p.y))-(.5f-ln);
    C = gol_mix3((float3)(.0125f),texture(iChannel0,p*2.f).rgb*(float3)(0.906f,0.282f,0.075f),gol_smoothstep(px,-px,d) );
    C = gol_mix3(C,C+.1f,gol_mix(gol_smoothstep(px,-px,d+.035f),0.f,clamp(1.f-(H.y+.15f),0.f,1.f)) );
    C = gol_mix3(C,C*.1f,gol_mix(gol_smoothstep(px,-px,d+.025f),0.f,clamp(1.f-(H.x+.5f),0.f,1.f)) );
    
    // base tile and empty vars
    float b = length(pp)-s3;
    float t = 1e5, g = 1e5;
    float tg= 1.f;
    
    hs = gol_fract(hs*53.71f);

    // alternate tiles
    if(hs>.95f) {
        float2 p4 = p*r3, p5 = p*r2;
        
        b = length((float2)(p.x,abs(p.y)-.5f));
        g = length(p5.x);
        t = length(p4.x);
        tg= 0.f;
    }else if(hs>.65f) {
        b = length(p.x);
        g = min(length(p1)-s3,length(p1+(float2)(1.155f,0))-s3);
        
        tg= 0.f;
    } else if(hs<.15f) {
        float2 p4 = p*r3, p5 = p*r2;
        
        t = length(p.x);
        b = length(p5.x);
        g = length(p4.x);
        
        tg= 0.f;
    } else if(hs<.22f) {
        b = length((float2)(p.x,abs(p.y)-.5f));
        g = min(length(p1)-s3,length(p1+(float2)(1.155f,0))-s3);

    }
    
    clr = (float3)(0.420f,0.278f,0.043f);
    trm = (float3)(.0f);
    
    // draw segments
    draw(t,px,C);
    draw(g,px,C);
    draw(b,px,C);
    // solid balls
    if(tg>0.f){
        float v = length(p)-.25f;
        C = gol_mix3(C,C*.25f,gol_smoothstep(.1f+px,-px,v-.01f) );
        C = gol_mix3(C,clr,gol_smoothstep(px,-px,v ));
        C = gol_mix3(C,clamp(C+.2f,C,(float3)(.95f)),gol_smoothstep(.01f+px,-px, v+.1f ));
        C = gol_mix3(C,trm,gol_smoothstep(px,-px,abs(v)-ln ));
    
    }
    
    C = pow(C,(float3)(.4545f));
    fragColor = (float4)(C,1);
// ---- SHADERTOY CODE END ----