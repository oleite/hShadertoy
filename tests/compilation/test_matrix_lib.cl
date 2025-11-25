/*
 * Test file for matrix library (matrix_types.h + matrix_ops.h)
 *
 * This file tests each operation in isolation to ensure the library compiles.
 */

#include "../../houdini/ocl/include/glslHelpers.h"

kernel void test_matrix_library(global float* output) {
    int idx = 0;

    // Test matrix2x2 diagonal constructor
    matrix2x2 m2_diag = GLSL_matrix2x2_diagonal(2.0f);
    output[idx++] = m2_diag.cols[0].x;

    // Test matrix3x3 diagonal constructor
    matrix3x3 m3_diag = GLSL_matrix3x3_diagonal(3.0f);
    output[idx++] = m3_diag.cols[1].y;

    // Test matrix4x4 diagonal constructor
    matrix4x4 m4_diag = GLSL_matrix4x4_diagonal(4.0f);
    output[idx++] = m4_diag.cols[2].z;

    // Test full constructors
    matrix2x2 m2_full = GLSL_mat2(1.0f, 2.0f, 3.0f, 4.0f);
    output[idx++] = m2_full.cols[0].x;

    matrix3x3 m3_full = GLSL_mat3(
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    );
    output[idx++] = m3_full.cols[1].y;

    matrix4x4 m4_full = GLSL_mat4(
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    );
    output[idx++] = m4_full.cols[2].z;

    // Test column constructors
    matrix2x2 m2_cols = GLSL_mat2_cols((float2)(1.0f, 2.0f), (float2)(3.0f, 4.0f));
    output[idx++] = m2_cols.cols[0].x;

    matrix3x3 m3_cols = GLSL_mat3_cols(
        (float3)(1.0f, 2.0f, 3.0f),
        (float3)(4.0f, 5.0f, 6.0f),
        (float3)(7.0f, 8.0f, 9.0f)
    );
    output[idx++] = m3_cols.cols[1].y;

    matrix4x4 m4_cols = GLSL_mat4_cols(
        (float4)(1.0f, 2.0f, 3.0f, 4.0f),
        (float4)(5.0f, 6.0f, 7.0f, 8.0f),
        (float4)(9.0f, 10.0f, 11.0f, 12.0f),
        (float4)(13.0f, 14.0f, 15.0f, 16.0f)
    );
    output[idx++] = m4_cols.cols[2].z;

    // Test type casting
    matrix3x3 m3_from_m4 = GLSL_mat3_from_mat4(m4_full);
    output[idx++] = m3_from_m4.cols[0].x;

    matrix4x4 m4_from_m3 = GLSL_mat4_from_mat3(m3_full);
    output[idx++] = m4_from_m3.cols[1].y;

    // Test matrix-vector multiplication
    float2 v2 = (float2)(1.0f, 2.0f);
    float2 r2_mv = GLSL_mul_mat2_vec2(m2_full, v2);
    output[idx++] = r2_mv.x;

    float3 v3 = (float3)(1.0f, 2.0f, 3.0f);
    float3 r3_mv = GLSL_mul_mat3_vec3(m3_full, v3);
    output[idx++] = r3_mv.y;

    float4 v4 = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
    float4 r4_mv = GLSL_mul_mat4_vec4(m4_full, v4);
    output[idx++] = r4_mv.z;

    // Test vector-matrix multiplication
    float2 r2_vm = GLSL_mul_vec2_mat2(v2, m2_full);
    output[idx++] = r2_vm.x;

    float3 r3_vm = GLSL_mul_vec3_mat3(v3, m3_full);
    output[idx++] = r3_vm.y;

    float4 r4_vm = GLSL_mul_vec4_mat4(v4, m4_full);
    output[idx++] = r4_vm.z;

    // Test matrix-matrix multiplication
    matrix2x2 m2_mm = GLSL_mul_mat2_mat2(m2_full, m2_cols);
    output[idx++] = m2_mm.cols[0].x;

    matrix3x3 m3_mm = GLSL_mul_mat3_mat3(m3_full, m3_cols);
    output[idx++] = m3_mm.cols[1].y;

    matrix4x4 m4_mm = GLSL_mul_mat4_mat4(m4_full, m4_cols);
    output[idx++] = m4_mm.cols[2].z;

    // Test transpose
    matrix2x2 m2_t = GLSL_transpose(m2_full);
    output[idx++] = m2_t.cols[0].x;

    matrix3x3 m3_t = GLSL_transpose_mat3(m3_full);
    output[idx++] = m3_t.cols[1].y;

    matrix4x4 m4_t = GLSL_transpose_mat4(m4_full);
    output[idx++] = m4_t.cols[2].z;

    // Test determinant
    float det2 = GLSL_determinant(m2_full);
    output[idx++] = det2;

    float det3 = GLSL_determinant_mat3(m3_full);
    output[idx++] = det3;

    float det4 = GLSL_determinant_mat4(m4_full);
    output[idx++] = det4;

    // Test inverse
    matrix2x2 m2_inv = GLSL_inverse(m2_full);
    output[idx++] = m2_inv.cols[0].x;

    matrix3x3 m3_inv = GLSL_inverse_mat3(m3_full);
    output[idx++] = m3_inv.cols[1].y;

    matrix4x4 m4_inv = GLSL_inverse_mat4(m4_full);
    output[idx++] = m4_inv.cols[2].z;

    // Test matrixCompMult
    matrix2x2 m2_comp = GLSL_matrixCompMult(m2_full, m2_cols);
    output[idx++] = m2_comp.cols[0].x;

    matrix3x3 m3_comp = GLSL_matrixCompMult_mat3(m3_full, m3_cols);
    output[idx++] = m3_comp.cols[1].y;

    matrix4x4 m4_comp = GLSL_matrixCompMult_mat4(m4_full, m4_cols);
    output[idx++] = m4_comp.cols[2].z;
/* Closing brace added by compilecl.py */
