// simple hexagonal tiles
// https://www.shadertoy.com/view/MlXyDl
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 u = 8.*fragCoord/iResolution.x;
    
    vec2 s = vec2(1.,1.732);
    vec2 a = mod(u     ,s)*2.-s;
    vec2 b = mod(u+s*.5,s)*2.-s;
    
	fragColor = vec4(.5*min(dot(a,a),dot(b,b)));
}