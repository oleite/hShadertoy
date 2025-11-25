void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // if-else statement
    vec3 col;
    if(uv.x > 0.7){
        col = vec3(1., 0., 0.);
    }else if(uv.x > 0.4){
        col = vec3(0., 1., 0.);
    }else{
        col = vec3(0., 0., 1.);
    }

    if(uv.x > 0.9){
        discard;
    }
    col = uv.y > 0.5 ? col * 0.5 : col;

    fragColor = vec4(col,1.0);
}