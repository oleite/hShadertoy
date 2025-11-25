void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    vec3 col;

    // break
    for (int i = 0; i < 8; ++i) {
        if (i < 8) break;
    }

    // continue
    for (int i = 0; i < 8; ++i) {
        if (uv.x < uv.y) {
            continue;
        }    
    } 

    // while
    int i = 0;
    while (i < 10) {
        // Code to be executed
        i++;
    }

    // do
    int j = 0;
    do {
        // Code to be executed
        j++;
    } while (j < 10);

    col = uv.y > 0.5 ? vec3(.5) * 0.5 : vec3(.2) ;

    fragColor = vec4(col,1.0);
}