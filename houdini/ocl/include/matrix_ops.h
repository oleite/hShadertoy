#ifndef __MATRIX_OPS_H__
#define __MATRIX_OPS_H__

#include "matrix_types.h"

/*
 * Matrix operations for GLSL to OpenCL transpiler
 *
 * All functions use GLSL_ prefix to match existing convention.
 * Column-major layout throughout (GLSL standard).
 */

/* ============================================================
 * MATRIX CONSTRUCTORS - Diagonal (single scalar)
 * ============================================================ */

inline matrix2x2 GLSL_matrix2x2_diagonal(float s) {
    matrix2x2 m;
    m.cols[0] = (float2)(s, 0.0f);
    m.cols[1] = (float2)(0.0f, s);
    return m;
}

inline matrix3x3 GLSL_matrix3x3_diagonal(float s) {
    matrix3x3 m;
    m.cols[0] = (float3)(s, 0.0f, 0.0f);
    m.cols[1] = (float3)(0.0f, s, 0.0f);
    m.cols[2] = (float3)(0.0f, 0.0f, s);
    return m;
}

inline matrix4x4 GLSL_matrix4x4_diagonal(float s) {
    matrix4x4 m;
    m.cols[0] = (float4)(s, 0.0f, 0.0f, 0.0f);
    m.cols[1] = (float4)(0.0f, s, 0.0f, 0.0f);
    m.cols[2] = (float4)(0.0f, 0.0f, s, 0.0f);
    m.cols[3] = (float4)(0.0f, 0.0f, 0.0f, s);
    return m;
}

/* ============================================================
 * MATRIX CONSTRUCTORS - Full (all elements)
 * Column-major order: GLSL mat3(c0r0, c0r1, c0r2, c1r0, ...)
 * ============================================================ */

inline matrix2x2 GLSL_mat2(float m00, float m10,
                            float m01, float m11) {
    matrix2x2 m;
    m.cols[0] = (float2)(m00, m10);
    m.cols[1] = (float2)(m01, m11);
    return m;
}

inline matrix3x3 GLSL_mat3(float m00, float m10, float m20,
                            float m01, float m11, float m21,
                            float m02, float m12, float m22) {
    matrix3x3 m;
    m.cols[0] = (float3)(m00, m10, m20);
    m.cols[1] = (float3)(m01, m11, m21);
    m.cols[2] = (float3)(m02, m12, m22);
    return m;
}

inline matrix4x4 GLSL_mat4(float m00, float m10, float m20, float m30,
                            float m01, float m11, float m21, float m31,
                            float m02, float m12, float m22, float m32,
                            float m03, float m13, float m23, float m33) {
    matrix4x4 m;
    m.cols[0] = (float4)(m00, m10, m20, m30);
    m.cols[1] = (float4)(m01, m11, m21, m31);
    m.cols[2] = (float4)(m02, m12, m22, m32);
    m.cols[3] = (float4)(m03, m13, m23, m33);
    return m;
}

/* ============================================================
 * MATRIX CONSTRUCTORS - From column vectors
 * ============================================================ */

inline matrix2x2 GLSL_mat2_cols(float2 col0, float2 col1) {
    matrix2x2 m;
    m.cols[0] = col0;
    m.cols[1] = col1;
    return m;
}

inline matrix3x3 GLSL_mat3_cols(float3 col0, float3 col1, float3 col2) {
    matrix3x3 m;
    m.cols[0] = col0;
    m.cols[1] = col1;
    m.cols[2] = col2;
    return m;
}

inline matrix4x4 GLSL_mat4_cols(float4 col0, float4 col1, float4 col2, float4 col3) {
    matrix4x4 m;
    m.cols[0] = col0;
    m.cols[1] = col1;
    m.cols[2] = col2;
    m.cols[3] = col3;
    return m;
}

/* ============================================================
 * MATRIX CONSTRUCTORS - Type casting
 * ============================================================ */

inline matrix3x3 GLSL_mat3_from_mat4(matrix4x4 m) {
    matrix3x3 result;
    result.cols[0] = m.cols[0].xyz;
    result.cols[1] = m.cols[1].xyz;
    result.cols[2] = m.cols[2].xyz;
    return result;
}

inline matrix4x4 GLSL_mat4_from_mat3(matrix3x3 m) {
    matrix4x4 result;
    result.cols[0] = (float4)(m.cols[0], 0.0f);
    result.cols[1] = (float4)(m.cols[1], 0.0f);
    result.cols[2] = (float4)(m.cols[2], 0.0f);
    result.cols[3] = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    return result;
}

/* ============================================================
 * MATRIX-VECTOR MULTIPLICATION
 * M * v (column vector)
 * ============================================================ */

inline float2 GLSL_mul_mat2_vec2(matrix2x2 M, float2 v) {
    return (float2)(
        dot(M.cols[0], v),
        dot(M.cols[1], v)
    );
}

inline float3 GLSL_mul_mat3_vec3(matrix3x3 M, float3 v) {
    return (float3)(
        dot(M.cols[0], v),
        dot(M.cols[1], v),
        dot(M.cols[2], v)
    );
}

inline float4 GLSL_mul_mat4_vec4(matrix4x4 M, float4 v) {
    return (float4)(
        dot(M.cols[0], v),
        dot(M.cols[1], v),
        dot(M.cols[2], v),
        dot(M.cols[3], v)
    );
}

/* ============================================================
 * VECTOR-MATRIX MULTIPLICATION
 * v * M (row vector)
 * ============================================================ */

inline float2 GLSL_mul_vec2_mat2(float2 v, matrix2x2 M) {
    return v.x * M.cols[0] + v.y * M.cols[1];
}

inline float3 GLSL_mul_vec3_mat3(float3 v, matrix3x3 M) {
    return v.x * M.cols[0] + v.y * M.cols[1] + v.z * M.cols[2];
}

inline float4 GLSL_mul_vec4_mat4(float4 v, matrix4x4 M) {
    return v.x * M.cols[0] + v.y * M.cols[1] + v.z * M.cols[2] + v.w * M.cols[3];
}

/* ============================================================
 * MATRIX-MATRIX MULTIPLICATION
 * A * B
 * ============================================================ */

inline matrix2x2 GLSL_mul_mat2_mat2(matrix2x2 A, matrix2x2 B) {
    matrix2x2 result;
    result.cols[0] = GLSL_mul_mat2_vec2(A, B.cols[0]);
    result.cols[1] = GLSL_mul_mat2_vec2(A, B.cols[1]);
    return result;
}

inline matrix3x3 GLSL_mul_mat3_mat3(matrix3x3 A, matrix3x3 B) {
    matrix3x3 result;
    result.cols[0] = GLSL_mul_mat3_vec3(A, B.cols[0]);
    result.cols[1] = GLSL_mul_mat3_vec3(A, B.cols[1]);
    result.cols[2] = GLSL_mul_mat3_vec3(A, B.cols[2]);
    return result;
}

inline matrix4x4 GLSL_mul_mat4_mat4(matrix4x4 A, matrix4x4 B) {
    matrix4x4 result;
    result.cols[0] = GLSL_mul_mat4_vec4(A, B.cols[0]);
    result.cols[1] = GLSL_mul_mat4_vec4(A, B.cols[1]);
    result.cols[2] = GLSL_mul_mat4_vec4(A, B.cols[2]);
    result.cols[3] = GLSL_mul_mat4_vec4(A, B.cols[3]);
    return result;
}

/* ============================================================
 * TRANSPOSE
 * ============================================================ */

inline matrix2x2 GLSL_transpose(matrix2x2 M) {
    matrix2x2 result;
    result.cols[0] = (float2)(M.cols[0].x, M.cols[1].x);
    result.cols[1] = (float2)(M.cols[0].y, M.cols[1].y);
    return result;
}

inline matrix3x3 GLSL_transpose_mat3(matrix3x3 M) {
    matrix3x3 result;
    result.cols[0] = (float3)(M.cols[0].x, M.cols[1].x, M.cols[2].x);
    result.cols[1] = (float3)(M.cols[0].y, M.cols[1].y, M.cols[2].y);
    result.cols[2] = (float3)(M.cols[0].z, M.cols[1].z, M.cols[2].z);
    return result;
}

inline matrix4x4 GLSL_transpose_mat4(matrix4x4 M) {
    matrix4x4 result;
    result.cols[0] = (float4)(M.cols[0].x, M.cols[1].x, M.cols[2].x, M.cols[3].x);
    result.cols[1] = (float4)(M.cols[0].y, M.cols[1].y, M.cols[2].y, M.cols[3].y);
    result.cols[2] = (float4)(M.cols[0].z, M.cols[1].z, M.cols[2].z, M.cols[3].z);
    result.cols[3] = (float4)(M.cols[0].w, M.cols[1].w, M.cols[2].w, M.cols[3].w);
    return result;
}

/* ============================================================
 * DETERMINANT
 * ============================================================ */

inline float GLSL_determinant(matrix2x2 M) {
    return M.cols[0].x * M.cols[1].y - M.cols[0].y * M.cols[1].x;
}

inline float GLSL_determinant_mat3(matrix3x3 M) {
    float a = M.cols[0].x;
    float b = M.cols[1].x;
    float c = M.cols[2].x;
    float d = M.cols[0].y;
    float e = M.cols[1].y;
    float f = M.cols[2].y;
    float g = M.cols[0].z;
    float h = M.cols[1].z;
    float i = M.cols[2].z;

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

inline float GLSL_determinant_mat4(matrix4x4 M) {
    float a = M.cols[0].x, b = M.cols[1].x, c = M.cols[2].x, d = M.cols[3].x;
    float e = M.cols[0].y, f = M.cols[1].y, g = M.cols[2].y, h = M.cols[3].y;
    float i = M.cols[0].z, j = M.cols[1].z, k = M.cols[2].z, l = M.cols[3].z;
    float m = M.cols[0].w, n = M.cols[1].w, o = M.cols[2].w, p = M.cols[3].w;

    float kp_lo = k * p - l * o;
    float jp_ln = j * p - l * n;
    float jo_kn = j * o - k * n;
    float ip_lm = i * p - l * m;
    float io_km = i * o - k * m;
    float in_jm = i * n - j * m;

    return a * (f * kp_lo - g * jp_ln + h * jo_kn) -
           b * (e * kp_lo - g * ip_lm + h * io_km) +
           c * (e * jp_ln - f * ip_lm + h * in_jm) -
           d * (e * jo_kn - f * io_km + g * in_jm);
}

/* ============================================================
 * INVERSE
 * ============================================================ */

inline matrix2x2 GLSL_inverse(matrix2x2 M) {
    float det = GLSL_determinant(M);
    float invDet = 1.0f / det;

    matrix2x2 result;
    result.cols[0] = (float2)( M.cols[1].y * invDet, -M.cols[0].y * invDet);
    result.cols[1] = (float2)(-M.cols[1].x * invDet,  M.cols[0].x * invDet);
    return result;
}

inline matrix3x3 GLSL_inverse_mat3(matrix3x3 M) {
    float a = M.cols[0].x, b = M.cols[1].x, c = M.cols[2].x;
    float d = M.cols[0].y, e = M.cols[1].y, f = M.cols[2].y;
    float g = M.cols[0].z, h = M.cols[1].z, i = M.cols[2].z;

    float A = e * i - f * h;
    float B = -(d * i - f * g);
    float C = d * h - e * g;
    float D = -(b * i - c * h);
    float E = a * i - c * g;
    float F = -(a * h - b * g);
    float G = b * f - c * e;
    float H = -(a * f - c * d);
    float I = a * e - b * d;

    float det = a * A + b * B + c * C;
    float invDet = 1.0f / det;

    matrix3x3 result;
    result.cols[0] = (float3)(A * invDet, B * invDet, C * invDet);
    result.cols[1] = (float3)(D * invDet, E * invDet, F * invDet);
    result.cols[2] = (float3)(G * invDet, H * invDet, I * invDet);
    return result;
}

inline matrix4x4 GLSL_inverse_mat4(matrix4x4 M) {
    float a = M.cols[0].x, b = M.cols[1].x, c = M.cols[2].x, d = M.cols[3].x;
    float e = M.cols[0].y, f = M.cols[1].y, g = M.cols[2].y, h = M.cols[3].y;
    float i = M.cols[0].z, j = M.cols[1].z, k = M.cols[2].z, l = M.cols[3].z;
    float m = M.cols[0].w, n = M.cols[1].w, o = M.cols[2].w, p = M.cols[3].w;

    float kp_lo = k * p - l * o;
    float jp_ln = j * p - l * n;
    float jo_kn = j * o - k * n;
    float ip_lm = i * p - l * m;
    float io_km = i * o - k * m;
    float in_jm = i * n - j * m;

    float A =  (f * kp_lo - g * jp_ln + h * jo_kn);
    float B = -(e * kp_lo - g * ip_lm + h * io_km);
    float C =  (e * jp_ln - f * ip_lm + h * in_jm);
    float D = -(e * jo_kn - f * io_km + g * in_jm);

    float det = a * A + b * B + c * C + d * D;
    float invDet = 1.0f / det;

    float gp_ho = g * p - h * o;
    float fp_hn = f * p - h * n;
    float fo_gn = f * o - g * n;
    float ep_hm = e * p - h * m;
    float eo_gm = e * o - g * m;
    float en_fm = e * n - f * m;

    float gl_hk = g * l - h * k;
    float fl_hj = f * l - h * j;
    float fk_gj = f * k - g * j;
    float el_hi = e * l - h * i;
    float ek_gi = e * k - g * i;
    float ej_fi = e * j - f * i;

    matrix4x4 result;
    result.cols[0] = (float4)(
         A * invDet,
         B * invDet,
         C * invDet,
         D * invDet
    );
    result.cols[1] = (float4)(
        -(b * kp_lo - c * jp_ln + d * jo_kn) * invDet,
         (a * kp_lo - c * ip_lm + d * io_km) * invDet,
        -(a * jp_ln - b * ip_lm + d * in_jm) * invDet,
         (a * jo_kn - b * io_km + c * in_jm) * invDet
    );
    result.cols[2] = (float4)(
         (b * gp_ho - c * fp_hn + d * fo_gn) * invDet,
        -(a * gp_ho - c * ep_hm + d * eo_gm) * invDet,
         (a * fp_hn - b * ep_hm + d * en_fm) * invDet,
        -(a * fo_gn - b * eo_gm + c * en_fm) * invDet
    );
    result.cols[3] = (float4)(
        -(b * gl_hk - c * fl_hj + d * fk_gj) * invDet,
         (a * gl_hk - c * el_hi + d * ek_gi) * invDet,
        -(a * fl_hj - b * el_hi + d * ej_fi) * invDet,
         (a * fk_gj - b * ek_gi + c * ej_fi) * invDet
    );

    return result;
}

/* ============================================================
 * COMPONENT-WISE MULTIPLICATION
 * ============================================================ */

inline matrix2x2 GLSL_matrixCompMult(matrix2x2 A, matrix2x2 B) {
    matrix2x2 result;
    result.cols[0] = A.cols[0] * B.cols[0];
    result.cols[1] = A.cols[1] * B.cols[1];
    return result;
}

inline matrix3x3 GLSL_matrixCompMult_mat3(matrix3x3 A, matrix3x3 B) {
    matrix3x3 result;
    result.cols[0] = A.cols[0] * B.cols[0];
    result.cols[1] = A.cols[1] * B.cols[1];
    result.cols[2] = A.cols[2] * B.cols[2];
    return result;
}

inline matrix4x4 GLSL_matrixCompMult_mat4(matrix4x4 A, matrix4x4 B) {
    matrix4x4 result;
    result.cols[0] = A.cols[0] * B.cols[0];
    result.cols[1] = A.cols[1] * B.cols[1];
    result.cols[2] = A.cols[2] * B.cols[2];
    result.cols[3] = A.cols[3] * B.cols[3];
    return result;
}

#endif /* __MATRIX_OPS_H__ */
