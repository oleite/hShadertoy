// https://www.shadertoy.com/view/l3lGWj
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord.xy / iResolution.xy;
    
    // Apply rotation matrix transformation
    float angle = sin(iTime) * 2.0;
    mat2 rotation = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    vec2 rotatedUV = rotation * uv;
    
    // Output color
    fragColor = vec4(rotatedUV, 0.5 + 0.5 * sin(iTime), 1.0);
}
