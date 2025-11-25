void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float foo; // Undefined float variable
    vec2 bar; // Undefined vec3 variable
    float x, y, z; // Declares three float variables: x, y, and z
    int a = 10, b, c; // Declares and initializes two int variables: a and b
    vec3 position, normal, tangent; // Declares three vec3 variables

    //arrays
    float myarray1[5];
    vec3 myarray3[5];
    
    vec3 col = vec3(x, y, z) + normal;

    // Output to screen
    fragColor = vec4(col,1.0);
}