float random( in float x ){
    return fract(1e4*sin((x)*23.27));
}
void random2( in vec2 uv, out vec2 noise2){
    float a = fract(1e4*sin((uv.x)*541.17));
    float b = fract(1e4*sin((uv.y)*321.46));
    noise2 = vec2( a, b );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    float noise = random(fragCoord.x + fragCoord.y);   
    vec2 pixelnoise;
    random2( fragCoord, pixelnoise );

    vec3 col = vec3(pixelnoise,noise);

    // Output to screen
    fragColor = vec4(col,1.0);
}