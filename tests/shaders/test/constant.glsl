const float foo = 0.5;
const int i = 1;
const vec3 v3 = vec3(0.);
// const mat3 m3 = mat3(0.); // constant matrix not supported yet - requires matrix_ops.h redesign

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    const float bar = 0.5;
    const vec2 v2 = vec2(0.);
    // mat2 m2 = mat2(0.);  // constant matrix not supported yet - requires matrix_ops.h redesign
    vec3 col = vec3(foo,bar,float(i));
    fragColor = vec4(col,1.0);
}