void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float foo;
    vec3 bar;
    float x, y, z; // Declares three float variables: x, y, and z
    int a = 10, b = 20; // Declares and initializes two int variables: a and b
    vec3 position, normal, tangent; // Declares three vec3 variables
    
    x = float(b);
    y = 0.5;
    z = 0.0;
    
    vec3 col = vec3(x, y, z) + normal;

    // Output to screen
    fragColor = vec4(col,1.0);
}