
mat3 rot(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(c, s, 0.0,
                -s, c, 0.0,
                0.0, 0.0, 1.0);
}

// Construct a 3x3 matrix for a 2D axis-aligned scale
mat3 scale(vec2 s) {
    return mat3(s.x, 0.0, 0.0,
                0.0, s.y, 0.0,
                0.0, 0.0, 1.0);
}

// Construct a 3x3 matrix for a 2D perspective distortion
mat3 distort(vec2 k) {
    return mat3(1.0, 0.0, k.x,
                0.0, 1.0, k.y,
                0.0, 0.0, 1.0);
                
}

// Construct a 3x3 matrix for a 2D translation
mat3 translate(vec2 t) {
    return mat3(1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                t.x, t.y, 1.0);
}

mat2 foo(float a) {
	float c = cos(a), s = sin(a);
    return mat2(
        c, s, // column 1
        -s, c // column 2
    );
}

mat3 bar(vec3 axis, float angle) {
  float s = sin(angle);
  float c = cos(angle);
  float oc = 1.0 - c;
  return mat3(
    oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, 
    oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, 
    oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c
  );
}

mat4 foobar(vec3 axis, float angle) {
  float s = sin(angle);
  float c = cos(angle);
  float oc = 1.0 - c;
  return mat4(
    oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, 0., 
    oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, 0., 
    oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c, 0.,
    0., 0., 0., 1.
  );
}

vec3 tonemap(vec3 color) {  
  mat3 m1 = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
  );
  mat3 m2 = mat3(
    1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
  );
  vec3 v = m1 * color;  
  vec3 a = v * (v + 0.0245786) - 0.000090537;
  vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));  
}



void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // constructed vectors and matrices for testing
    vec2 V2 =vec2(1.0, 0.0);
    vec3 V3 =vec3(1.0, 0.0, 0.0);
    vec4 V4 =vec4(1.0, 0.0, 0.0, 0.0);

    mat2 M2 = mat2(1.0, 0.0, 0.0, 1.0 );
	mat3 M3 = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
	mat4 M4 = mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    
    // common matrix operations
    vec2 op1 = V2 * M2; // 2D vector transform
    vec2 op2 = V2 * M2 * M2; // 2D vector transform x2
    op2 *= M2; // another common transform operation
    op2 = op2 * foo(3.14);

    vec3 op3 = V3 * M3;  // 3D vector transform using 3x3 transform
    vec3 op4 = V3 * M3 * M3; // sequential, eg Model to World to Screen
    op4 *= M3; // // another common transform operation
    op4 = op4 * bar(vec3(1.0, 0.0, 0.0), 3.14) ;
    vec3 op5 = M3 * normalize( vec3(V2, 1.0));
    vec3 op6 = cross( normalize(V3) * M3, M3 * vec3(1.0, V2) ) * M3;
    op6 = op6 * translate(V2);
    op6 = tonemap(V3);

    vec4 o7 = V4 * M4; // 3D point transorm using 4x4 matrix
    vec4 op8 = V4 * M4 * M4; // sequential, eg Model to World to Screen
    op8 *= M4;
    op8 = V4 * foobar(V3, 3.14);

    // operations isolated vector components
    V2 = V3.xy * M2;
    V4.xy *= M2;
  

    // common matrix functions
    mat2 xf1 = transpose(M2);
    mat2 xf2 = inverse(M2);	
    mat3 xf3 = transpose(M3);
    mat3 xf4 = inverse(M3);
    mat4 xf5 = transpose(M4);
    mat4 xf6 = inverse(M4);	


    fragColor = vec4(uv, 0.0, 1.0);
}