// Standalone matrix library test - no Shadertoy infrastructure

// Define fpreal types
#ifndef fpreal
#define fpreal float
#endif

typedef fpreal fpreal2 __attribute__((ext_vector_type(2)));
typedef fpreal fpreal3 __attribute__((ext_vector_type(3)));
typedef fpreal fpreal4 __attribute__((ext_vector_type(4)));
typedef fpreal fpreal16 __attribute__((ext_vector_type(16)));

// Define old Houdini matrix types (needed by glslHelpers.h temporarily)
typedef fpreal3 mat3[3];
typedef fpreal4 mat2;
typedef fpreal16 mat4;

// Include new matrix library
#include "../../houdini/ocl/include/matrix_types.h"
#include "../../houdini/ocl/include/matrix_ops.h"

// Test kernel
kernel void test_matrix_operations(global float* output) {
    int idx = 0;

    // Test matrix3x3 diagonal constructor
    matrix3x3 m3_diag = GLSL_matrix3x3_diagonal(3.0f);
    output[idx++] = m3_diag.cols[0].x;  // Should be 3.0
    output[idx++] = m3_diag.cols[1].y;  // Should be 3.0
    output[idx++] = m3_diag.cols[2].z;  // Should be 3.0

    // Test full constructor
    matrix3x3 m3_full = GLSL_mat3(
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    );
    output[idx++] = m3_full.cols[0].x;  // Should be 1.0
    output[idx++] = m3_full.cols[1].y;  // Should be 5.0
    output[idx++] = m3_full.cols[2].z;  // Should be 9.0

    // Test matrix-vector multiplication
    float3 v = (float3)(1.0f, 2.0f, 3.0f);
    float3 result_mv = GLSL_mul_mat3_vec3(m3_full, v);
    output[idx++] = result_mv.x;
    output[idx++] = result_mv.y;
    output[idx++] = result_mv.z;

    // Test transpose
    matrix3x3 m3_t = GLSL_transpose_mat3(m3_full);
    output[idx++] = m3_t.cols[0].x;  // Should be 1.0
    output[idx++] = m3_t.cols[0].y;  // Should be 4.0
    output[idx++] = m3_t.cols[0].z;  // Should be 7.0

    // Test matrix2x2
    matrix2x2 m2 = GLSL_mat2(1.0f, 2.0f, 3.0f, 4.0f);
    output[idx++] = m2.cols[0].x;
    output[idx++] = m2.cols[1].y;

    // Test determinant mat2
    float det2 = GLSL_determinant(m2);
    output[idx++] = det2;

    // Test inverse mat2
    matrix2x2 m2_inv = GLSL_inverse(m2);
    output[idx++] = m2_inv.cols[0].x;

    // Test matrix4x4
    matrix4x4 m4_diag = GLSL_matrix4x4_diagonal(2.0f);
    output[idx++] = m4_diag.cols[0].x;
    output[idx++] = m4_diag.cols[3].w;

    // Success marker
    output[idx++] = 999.0f;
}
