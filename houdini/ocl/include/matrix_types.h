#ifndef __MATRIX_TYPES_H__
#define __MATRIX_TYPES_H__

/*
 * Matrix struct types for GLSL to OpenCL transpiler
 *
 * These types replace Houdini's array-based matrix types (mat3[3])
 * with struct-based types that CAN be returned by value from functions.
 *
 * Layout: Column-major (matches GLSL specification)
 * Access: M.cols[col][row] matches GLSL M[col][row]
 */

typedef struct {
    float2 cols[2];  /* Two columns of float2 vectors */
} matrix2x2;

typedef struct {
    float3 cols[3];  /* Three columns of float3 vectors */
} matrix3x3;

typedef struct {
    float4 cols[4];  /* Four columns of float4 vectors */
} matrix4x4;

#endif /* __MATRIX_TYPES_H__ */
