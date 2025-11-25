void mainImage( out vec4 fragColor, in vec2 fragCoord ) {

    float fov = 1.8;
    vec3 E;
    vec3 I;

    float px = 1.0*(2.0/iResolution.y)*(1.0/fov);
    float t = -(1.0+E.y)/I.y;

    vec3 col = vec3(px,t,0.0);
    fragColor = vec4(col,1.0);
}