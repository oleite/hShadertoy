// Minimal header for matrix library testing

// Define fpreal types (from Houdini's matrix.h)
#ifndef fpreal
#define fpreal float
#endif

typedef fpreal fpreal2 __attribute__((ext_vector_type(2)));
typedef fpreal fpreal3 __attribute__((ext_vector_type(3)));
typedef fpreal fpreal4 __attribute__((ext_vector_type(4)));
typedef fpreal fpreal8 __attribute__((ext_vector_type(8)));
typedef fpreal fpreal16 __attribute__((ext_vector_type(16)));

// Define Houdini's old matrix types (needed by glslHelpers.h)
typedef fpreal3 mat3[3];
typedef fpreal4 mat2;
typedef fpreal16 mat4;

// Include new matrix library
#include "../../houdini/ocl/include/glslHelpers.h"
