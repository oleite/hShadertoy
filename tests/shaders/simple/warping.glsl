// https://www.shadertoy.com/view/3XlfWH

float hash21(vec2 p) // replace this by something better
{
    p  = 50.0*fract( p*0.3183099 + vec2(0.71,0.113));
    return -1.0+2.0*fract( p.x*p.y*(p.x+p.y) );
}

vec2 hash22( vec2 p ) // replace this by something better
{
    p = vec2(dot(p,vec2(127.1,311.7)),
             dot(p,vec2(269.5,183.3)));
    return fract(sin(p)*18.5453);
}

float noise( in vec2 x )
{
    vec2 i = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float a = hash21(i+vec2(0,0));
	float b = hash21(i+vec2(1,0));
	float c = hash21(i+vec2(0,1));
	float d = hash21(i+vec2(1,1));
    return mix(mix( a, b,f.x), mix( c, d,f.x),f.y);
}

float voronoi( in vec2 p )
{
	vec2 i = floor(p);
	vec2 f = fract(p);
	float d = 10.0; 
	for( int n=-1; n<=1; n++ )
    for( int m=-1; m<=1; m++ )
    {
        vec2 b = vec2(m, n);
        vec2 r = b - f + hash22(i+b);
        d = min(d,dot(r,r));
    }
	return d;
}

float fbmNoise( vec2 p, int oct, float r )
{
    const mat2 m = mat2( 0.80,  0.60, -0.60,  0.80 );

    float f = 0.0;
    float s = 0.5;
    float t = 0.0;
    for( int i=0; i<oct; i++ )
    {
        f += s*noise( p );
        t += s;
        p = m*p*2.01;
        s *= r;
    }
    return f/t;
}

float fbmVoronoi(in vec2 p, int oct)
{
    float f = 1.0;
    float s = 1.0;
    for( int i=0; i<oct; i++ )
    {
        float v = voronoi(p);
        f = min(f,v*s);
        p *= 2.0;
        s *= 1.4;
    }
    return 3.0*f;
}

vec2 fbm2Noise( vec2 p, int o, float r )
{
    return vec2(fbmNoise(p.xy+vec2(0.0,0.0),o,r), 
                fbmNoise(p.yx+vec2(0.7,1.3),o,r));
}

//====================================================================

// distortion
vec2 dis( vec2 p, float t )
{
    t += 0.1*sin(t);
    p.x -= 0.2*t;
    
    vec2 op = p;

    const float a = 0.7;

    p += a*0.5000*sin(p.yx*1.4+0.0+t);
    p += a*0.2500*sin(p.yx*2.3+1.0+t);
    p += a*0.1250*sin(p.yx*4.2+2.0+t);
    p += a*0.0625*sin(p.yx*8.1+3.0+t);
    
    p += 0.4*fbm2Noise( 0.5*p-0.9*t*vec2(1.0,0.2), 2, 0.5 );

    return p;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 p = (2.0*fragCoord-iResolution.xy)/iResolution.y;

    // distortion (and its velocity)
    const float dt = 0.01;
    vec2  q = dis(p,iTime);
    vec2 oq = dis(p,iTime-dt);
    float vel = length(q-oq)/dt;

    // wave
    float f = q.y-0.2*sin(1.57*q.x-0.7*iTime*0.0);    
    
    // circles
    f -= 0.5*vel*vel*(0.5-fbmVoronoi(q,8));

    // colorize
    f = 0.5 + 1.5*fbmNoise( vec2(2.5*f,0.0), 12,0.5 );
    vec3 col = mix(vec3(0.0,0.25,0.6),vec3(1.0),f);

    // vignetting
    col *= 1.0 - 0.1*dot(p,p);
    
    fragColor = vec4( col, 1.0 );
}
