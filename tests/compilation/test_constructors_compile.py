"""
Compilation tests for constructor transformations (Session 2).

Tests that constructor transformations produce valid, compilable OpenCL code.
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
    Try to compile OpenCL code with matrix constructor helpers.

    Returns:
        True if compilation succeeds, False otherwise
    """
    # Add Houdini types and matrix constructor helpers
    header = """
// Minimal Houdini type definitions
typedef float fpreal;
typedef float2 fpreal2;
typedef float3 fpreal3;
typedef float4 fpreal4;
typedef float16 fpreal16;

// Matrix types
typedef fpreal4 mat2;       // 2x2 matrix as float4
typedef fpreal3 mat3[3];    // 3x3 matrix as array of float3
typedef fpreal16 mat4;      // 4x4 matrix as float16

// Matrix diagonal constructors
__attribute__((overloadable))
mat2 GLSL_mat2_diagonal(float s){
    return (mat2)(s, 0.0f, 0.0f, s);
}

__attribute__((overloadable))
void GLSL_mat3_diagonal(float s, __private mat3 m){
    m[0] = (float3)(s, 0.0f, 0.0f);
    m[1] = (float3)(0.0f, s, 0.0f);
    m[2] = (float3)(0.0f, 0.0f, s);
}

__attribute__((overloadable))
mat4 GLSL_mat4_diagonal(float s){
    return (mat4)(
        s, 0.0f, 0.0f, 0.0f,
        0.0f, s, 0.0f, 0.0f,
        0.0f, 0.0f, s, 0.0f,
        0.0f, 0.0f, 0.0f, s
    );
}

// Matrix type casting
__attribute__((overloadable))
mat4 GLSL_mat4_from_mat3(const mat3 a){
    mat4 m;
    m.lo.lo = (fpreal4)(a[0].x, a[0].y, a[0].z, 0.0f);
    m.lo.hi = (fpreal4)(a[1].x, a[1].y, a[1].z, 0.0f);
    m.hi.lo = (fpreal4)(a[2].x, a[2].y, a[2].z, 0.0f);
    m.hi.hi = (fpreal4)(0.0f, 0.0f, 0.0f, 1.0f);
    return m;
}

__attribute__((overloadable))
void GLSL_mat3_from_mat4(const mat4 a, __private mat3 out){
    out[0] = (float3)(a.s0, a.s1, a.s2);
    out[1] = (float3)(a.s4, a.s5, a.s6);
    out[2] = (float3)(a.s8, a.s9, a.sa);
}

"""
    full_code = header + opencl_code

    try:
        import warnings
        # Suppress compiler warnings (they don't indicate failure)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=cl.CompilerWarning)
            program = cl.Program(cl_context, full_code).build()
        return True
    except cl.RuntimeError as e:
        print(f"Compilation failed:\n{e}")
        print(f"\nGenerated code:\n{full_code}")
        return False


def transform_and_compile(glsl_code, cl_context):
    """Helper: transform GLSL to OpenCL and compile."""
    parser = GLSLParser()
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    transformer = ASTTransformer(type_checker)
    emitter = CodeEmitter()

    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    opencl_code = emitter.emit(transformed)

    # Compile the OpenCL code
    success = compile_opencl(opencl_code, cl_context)
    return success, opencl_code


# ============================================================================
# Diagonal Matrix Constructors
# ============================================================================

def test_compile_mat2_diagonal(cl_context):
    """Test mat2 diagonal constructor compiles."""
    glsl = """
    void test() {
        mat2 M = mat2(1.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mat2 diagonal constructor:\\n{opencl}"


def test_compile_mat3_diagonal(cl_context):
    """Test mat3 diagonal constructor compiles."""
    glsl = """
    void test() {
        mat3 M = mat3(2.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mat3 diagonal constructor:\\n{opencl}"


def test_compile_mat4_diagonal(cl_context):
    """Test mat4 diagonal constructor compiles."""
    glsl = """
    void test() {
        mat4 M = mat4(1.5);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mat4 diagonal constructor:\\n{opencl}"


def test_compile_multiple_diagonal_matrices(cl_context):
    """Test multiple diagonal matrix constructors."""
    glsl = """
    void test() {
        mat2 M2 = mat2(1.0);
        mat3 M3 = mat3(2.0);
        mat4 M4 = mat4(3.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile multiple diagonal constructors:\\n{opencl}"


# ============================================================================
# Full Matrix Constructors
# ============================================================================

def test_compile_mat2_full(cl_context):
    """Test mat2 full constructor compiles."""
    glsl = """
    void test() {
        mat2 M = mat2(1.0, 2.0, 3.0, 4.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mat2 full constructor:\\n{opencl}"


def test_compile_mat3_full(cl_context):
    """Test mat3 full constructor compiles (double-brace syntax)."""
    glsl = """
    void test() {
        mat3 M = mat3(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        );
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mat3 full constructor:\\n{opencl}"


def test_compile_mat4_full(cl_context):
    """Test mat4 full constructor compiles."""
    glsl = """
    void test() {
        mat4 M = mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mat4 full constructor:\\n{opencl}"


# ============================================================================
# Matrix Type Casting
# ============================================================================

def test_compile_mat4_from_mat3(cl_context):
    """Test mat4(mat3) type casting compiles."""
    glsl = """
    void test() {
        mat3 M3 = mat3(1.0);
        mat4 M4 = mat4(M3);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mat4(mat3) cast:\\n{opencl}"


def test_compile_mat3_from_mat4(cl_context):
    """Test mat3(mat4) type casting compiles."""
    glsl = """
    void test() {
        mat4 M4 = mat4(1.0);
        mat3 M3 = mat3(M4);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mat3(mat4) cast:\\n{opencl}"


# ============================================================================
# Advanced Vector Constructors
# ============================================================================

def test_compile_vec4_from_vec3_scalar(cl_context):
    """Test vec4(vec3, scalar) compiles."""
    glsl = """
    void test() {
        vec3 v3 = vec3(1.0, 2.0, 3.0);
        vec4 v4 = vec4(v3, 4.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile vec4(vec3, scalar):\\n{opencl}"


def test_compile_vec4_from_two_vec2(cl_context):
    """Test vec4(vec2, vec2) compiles."""
    glsl = """
    void test() {
        vec2 a = vec2(1.0, 2.0);
        vec2 b = vec2(3.0, 4.0);
        vec4 v = vec4(a, b);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile vec4(vec2, vec2):\\n{opencl}"


def test_compile_vec3_from_vec2_scalar(cl_context):
    """Test vec3(vec2, scalar) compiles."""
    glsl = """
    void test() {
        vec2 v2 = vec2(1.0, 2.0);
        vec3 v3 = vec3(v2, 3.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile vec3(vec2, scalar):\\n{opencl}"


# ============================================================================
# Complex Scenarios
# ============================================================================

def test_compile_matrix_arithmetic(cl_context):
    """Test matrix constructors in arithmetic expressions."""
    glsl = """
    void test() {
        mat2 A = mat2(1.0);
        mat2 B = mat2(2.0);
        mat2 C = A;
        float s = 2.0;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile matrix arithmetic:\\n{opencl}"


def test_compile_nested_constructors(cl_context):
    """Test nested vector/matrix constructors."""
    glsl = """
    void test() {
        vec2 v = vec2(1.0, 2.0);
        vec4 v4 = vec4(vec3(v, 3.0), 4.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile nested constructors:\\n{opencl}"
