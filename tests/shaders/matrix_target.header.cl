matrix3x3 createRotationMatrixAxisAngle(float3 axis, float angle) {
    float s = GLSL_sin(angle);
    float c = GLSL_cos(angle);
    float oc = 1.0f - c;
    return GLSL_mat3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c);
}

matrix2x2 rotationMatrix2D(float a) {
    float c = GLSL_cos(a), s = GLSL_sin(a);
    return GLSL_mat2(c, s, -s, c);
}

