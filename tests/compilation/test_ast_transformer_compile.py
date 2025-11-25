"""
Compilation tests for AST transformer.

Tests that transformed GLSL compiles to valid OpenCL using PyOpenCL.
"""

import pytest
import pyopencl as cl
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def cl_context():
    """Create OpenCL context for testing."""
    try:
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        ctx = cl.Context([device])
        return ctx
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")


def compile_opencl(opencl_code: str, cl_context) -> bool:
    """
    Try to compile OpenCL code.

    Returns:
        True if compilation succeeds, False otherwise
    """
    # Add Houdini matrix.h types for testing
    header = """
// Minimal Houdini type definitions for testing
typedef float fpreal;
typedef float2 fpreal2;
typedef float3 fpreal3;
typedef float4 fpreal4;
typedef float16 fpreal16;

// Matrix types (simplified for testing)
typedef fpreal4 mat2;      // 2x2 matrix as float4
typedef fpreal3 mat3[3];   // 3x3 matrix as array of float3
typedef fpreal16 mat4;     // 4x4 matrix as float16

// Minimal GLSL helper function declarations
__attribute__((overloadable)) float GLSL_sin(float x) { return sin(x); }
__attribute__((overloadable)) float GLSL_cos(float x) { return cos(x); }
__attribute__((overloadable)) float GLSL_tan(float x) { return tan(x); }
__attribute__((overloadable)) float GLSL_mod(float x, float y) { return x - y * floor(x / y); }
__attribute__((overloadable)) float GLSL_clamp(float x, float a, float b) { return clamp(x, a, b); }
__attribute__((overloadable)) float GLSL_mix(float a, float b, float t) { return mix(a, b, t); }
__attribute__((overloadable)) float GLSL_length(float2 v) { return length(v); }
__attribute__((overloadable)) float GLSL_length(float3 v) { return length(v); }
__attribute__((overloadable)) float GLSL_length(float4 v) { return length(v); }
__attribute__((overloadable)) float GLSL_dot(float2 a, float2 b) { return dot(a, b); }
__attribute__((overloadable)) float GLSL_dot(float3 a, float3 b) { return dot(a, b); }
__attribute__((overloadable)) float GLSL_dot(float4 a, float4 b) { return dot(a, b); }
__attribute__((overloadable)) float2 GLSL_normalize(float2 v) { return normalize(v); }
__attribute__((overloadable)) float3 GLSL_normalize(float3 v) { return normalize(v); }
__attribute__((overloadable)) float4 GLSL_normalize(float4 v) { return normalize(v); }

"""
    full_code = header + opencl_code

    try:
        program = cl.Program(cl_context, full_code).build()
        return True
    except cl.RuntimeError as e:
        print(f"Compilation failed:\n{e}")
        print(f"\nGenerated code:\n{full_code}")
        return False


def transform_and_compile(glsl_code: str, cl_context) -> bool:
    """Helper: Parse, transform, and compile GLSL code."""
    parser = GLSLParser()
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    transformer = ASTTransformer(type_checker)
    emitter = CodeEmitter()

    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    opencl = emitter.emit(transformed)

    return compile_opencl(opencl, cl_context)


# ============================================================================
# Basic Syntax Compilation Tests (10 tests)
# ============================================================================

def test_compile_empty_function(cl_context):
    """Test empty function compiles."""
    glsl = "void test() { }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_float_declaration(cl_context):
    """Test float variable declaration compiles."""
    glsl = "void test() { float x = 1.0; }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_vec2_declaration(cl_context):
    """Test vec2 declaration compiles."""
    glsl = "void test() { vec2 v = vec2(1.0, 2.0); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_vec3_declaration(cl_context):
    """Test vec3 declaration compiles."""
    glsl = "void test() { vec3 v = vec3(1.0, 2.0, 3.0); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_vec4_declaration(cl_context):
    """Test vec4 declaration compiles."""
    glsl = "void test() { vec4 v = vec4(1.0, 2.0, 3.0, 4.0); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_arithmetic_expression(cl_context):
    """Test arithmetic expression compiles."""
    glsl = """
    void test() {
        float a = 1.0;
        float b = 2.0;
        float c = a + b * 3.0 - 4.0 / 5.0;
    }
    """
    assert transform_and_compile(glsl, cl_context)


def test_compile_vector_arithmetic(cl_context):
    """Test vector arithmetic compiles."""
    glsl = """
    void test() {
        vec2 a = vec2(1.0, 2.0);
        vec2 b = vec2(3.0, 4.0);
        vec2 c = a + b;
        vec2 d = a * 2.0;
    }
    """
    assert transform_and_compile(glsl, cl_context)


def test_compile_if_statement(cl_context):
    """Test if statement compiles."""
    glsl = """
    void test() {
        float x = 1.0;
        if (x > 0.0) {
            x = 2.0;
        }
    }
    """
    assert transform_and_compile(glsl, cl_context)


def test_compile_for_loop(cl_context):
    """Test for loop compiles."""
    glsl = """
    void test() {
        float sum = 0.0;
        for (int i = 0; i < 10; i++) {
            sum = sum + 1.0;
        }
    }
    """
    assert transform_and_compile(glsl, cl_context)


def test_compile_function_with_return(cl_context):
    """Test function with return value compiles."""
    glsl = """
    float identity(float x) {
        return x;
    }

    void test() {
        float y = identity(1.0);
    }
    """
    assert transform_and_compile(glsl, cl_context)


# ============================================================================
# Built-in Function Compilation Tests (10 tests)
# ============================================================================

def test_compile_sin_function(cl_context):
    """Test sin() function call compiles."""
    glsl = "void test() { float x = sin(1.0); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_cos_function(cl_context):
    """Test cos() function call compiles."""
    glsl = "void test() { float x = cos(1.0); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_mod_function(cl_context):
    """Test mod() function call compiles."""
    glsl = "void test() { float x = mod(5.0, 3.0); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_clamp_function(cl_context):
    """Test clamp() function call compiles."""
    glsl = "void test() { float x = clamp(5.0, 0.0, 1.0); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_mix_function(cl_context):
    """Test mix() function call compiles."""
    glsl = "void test() { float x = mix(1.0, 2.0, 0.5); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_length_function(cl_context):
    """Test length() function call compiles."""
    glsl = "void test() { float x = length(vec2(1.0, 2.0)); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_dot_function(cl_context):
    """Test dot() function call compiles."""
    glsl = "void test() { float x = dot(vec2(1.0, 2.0), vec2(3.0, 4.0)); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_normalize_function(cl_context):
    """Test normalize() function call compiles."""
    glsl = "void test() { vec2 v = normalize(vec2(1.0, 2.0)); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_nested_functions(cl_context):
    """Test nested function calls compile."""
    glsl = "void test() { float x = sin(cos(1.0)); }"
    assert transform_and_compile(glsl, cl_context)


def test_compile_complex_expression(cl_context):
    """Test complex expression with multiple functions compiles."""
    glsl = """
    void test() {
        vec2 a = vec2(1.0, 2.0);
        vec2 b = vec2(3.0, 4.0);
        float x = dot(normalize(a), b) * length(a + b);
    }
    """
    assert transform_and_compile(glsl, cl_context)


# ============================================================================
# Swizzling Compilation Tests (5 tests)
# ============================================================================

def test_compile_swizzle_xy(cl_context):
    """Test .xy swizzle compiles."""
    glsl = """
    void test() {
        vec3 v = vec3(1.0, 2.0, 3.0);
        vec2 xy = v.xy;
    }
    """
    assert transform_and_compile(glsl, cl_context)


def test_compile_swizzle_rgb(cl_context):
    """Test .rgb swizzle compiles."""
    glsl = """
    void test() {
        vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
        vec3 rgb = v.rgb;
    }
    """
    assert transform_and_compile(glsl, cl_context)


def test_compile_swizzle_single_component(cl_context):
    """Test single component swizzle compiles."""
    glsl = """
    void test() {
        vec3 v = vec3(1.0, 2.0, 3.0);
        float x = v.x;
    }
    """
    assert transform_and_compile(glsl, cl_context)


def test_compile_swizzle_in_expression(cl_context):
    """Test swizzle in expression compiles."""
    glsl = """
    void test() {
        vec3 a = vec3(1.0, 2.0, 3.0);
        vec3 b = vec3(4.0, 5.0, 6.0);
        vec2 c = a.xy + b.xy;
    }
    """
    assert transform_and_compile(glsl, cl_context)


def test_compile_nested_swizzle(cl_context):
    """Test nested/chained swizzle compiles."""
    glsl = """
    void test() {
        vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
        float x = v.xyz.x;
    }
    """
    assert transform_and_compile(glsl, cl_context)
