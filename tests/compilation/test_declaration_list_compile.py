"""
Compilation tests for comma-separated variable declarations (Session 7).

Tests that comma-separated declarations produce valid, compilable OpenCL code.
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
    # Add minimal type definitions
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

// Minimal GLSL functions
float GLSL_sin(float x) { return sin(x); }
float GLSL_sqrt(float x) { return sqrt(x); }
float GLSL_fract(float x) { return x - floor(x); }

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
# Simple Declaration Compilation Tests
# ============================================================================

def test_compile_float_comma_simple(cl_context):
    """Test float x, y, z; compiles."""
    glsl = """
    void test() {
        float x, y, z;
        x = 1.0;
        y = 2.0;
        z = 3.0;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile float comma declaration:\n{opencl}"


def test_compile_int_comma_with_init(cl_context):
    """Test int a = 10, b = 20; compiles."""
    glsl = """
    void test() {
        int a = 10, b = 20;
        int result = a + b;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile int comma with init:\n{opencl}"


def test_compile_vec2_comma(cl_context):
    """Test vec2 a, b; compiles."""
    glsl = """
    void test() {
        vec2 a, b;
        a = vec2(1.0, 2.0);
        b = a * 2.0;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile vec2 comma:\n{opencl}"


def test_compile_vec3_comma(cl_context):
    """Test vec3 p, n, t; compiles."""
    glsl = """
    void test() {
        vec3 p, n, t;
        p = vec3(1.0, 0.0, 0.0);
        n = vec3(0.0, 1.0, 0.0);
        t = vec3(0.0, 0.0, 1.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile vec3 comma:\n{opencl}"


def test_compile_vec4_comma(cl_context):
    """Test vec4 color1, color2; compiles."""
    glsl = """
    void test() {
        vec4 color1, color2;
        color1 = vec4(1.0, 0.0, 0.0, 1.0);
        color2 = vec4(0.0, 1.0, 0.0, 1.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile vec4 comma:\n{opencl}"


# ============================================================================
# Type Transformation Compilation Tests
# ============================================================================

def test_compile_ivec_uvec_comma(cl_context):
    """Test ivec and uvec comma declarations compile."""
    glsl = """
    void test() {
        ivec2 a, b;
        uvec3 c, d;
        a = ivec2(1, 2);
        c = uvec3(3, 4, 5);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile ivec/uvec comma:\n{opencl}"


def test_compile_mat2_comma(cl_context):
    """Test mat2 M1, M2; compiles."""
    glsl = """
    void test() {
        mat2 M1, M2;
        M1 = mat2(1.0, 0.0, 0.0, 1.0);
        M2 = M1;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mat2 comma:\n{opencl}"


def test_compile_mat4_comma(cl_context):
    """Test mat4 M1, M2; compiles."""
    glsl = """
    void test() {
        mat4 M1, M2;
        M1 = mat4(1.0);
        M2 = M1;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mat4 comma:\n{opencl}"


def test_compile_mixed_initializers(cl_context):
    """Test mixed initialization patterns compile."""
    glsl = """
    void test() {
        float x = 1.0, y, z = 3.0;
        int a, b = 5, c;
        y = 2.0;
        a = 4;
        c = 6;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mixed initializers:\n{opencl}"


def test_compile_complex_expressions(cl_context):
    """Test complex expressions in initializers compile."""
    glsl = """
    void test() {
        float x = 1.0 + 2.0, y = sin(0.5);
        float z = sqrt(4.0), w;
        vec2 a = vec2(1.0, 2.0), b = a * 2.0;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile complex expressions:\n{opencl}"


# ============================================================================
# Complex Pattern Compilation Tests
# ============================================================================

def test_compile_global_comma_declarations(cl_context):
    """Test global comma-separated declarations compile."""
    glsl = """
    float globalX, globalY, globalZ;

    void test() {
        globalX = 1.0;
        globalY = 2.0;
        globalZ = 3.0;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile global comma declarations:\n{opencl}"


def test_compile_for_loop_comma_init(cl_context):
    """Test for loop comma initialization compiles."""
    glsl = """
    void test() {
        int sum = 0;
        for(int i = 0, j = 0; i < 10; i++) {
            j = i * 2;
            sum += j;
        }
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile for loop comma init:\n{opencl}"


def test_compile_multiple_comma_declarations(cl_context):
    """Test multiple comma declarations in same function compile."""
    glsl = """
    void test() {
        float x, y, z;
        int a, b, c;
        vec2 p, q;
        vec3 v1, v2, v3;

        x = 1.0;
        a = 10;
        p = vec2(1.0, 2.0);
        v1 = vec3(0.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile multiple comma declarations:\n{opencl}"


def test_compile_realistic_shader_pattern(cl_context):
    """Test realistic shader pattern with comma declarations."""
    glsl = """
    vec3 calculateLighting(vec3 position, vec3 normal) {
        vec3 lightDir, viewDir, halfDir;
        float diffuse, specular;

        lightDir = vec3(1.0, 1.0, 1.0);
        viewDir = vec3(0.0, 0.0, 1.0);
        halfDir = lightDir + viewDir;

        diffuse = max(0.0, dot(normal, lightDir));
        specular = pow(max(0.0, dot(normal, halfDir)), 32.0);

        return vec3(diffuse + specular);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile realistic shader pattern:\n{opencl}"


def test_compile_all_types_combined(cl_context):
    """Test all supported types in comma declarations compile."""
    glsl = """
    void test() {
        float f1, f2, f3;
        int i1, i2;
        bool b1, b2;
        vec2 v2a, v2b;
        vec3 v3a, v3b;
        vec4 v4a, v4b;
        ivec2 iv2a, iv2b;
        uvec3 uv3a, uv3b;
        mat2 m2a, m2b;
        mat4 m4a, m4b;

        f1 = 1.0;
        i1 = 10;
        b1 = true;
        v2a = vec2(1.0, 2.0);
        v3a = vec3(1.0, 2.0, 3.0);
        v4a = vec4(1.0, 2.0, 3.0, 4.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile all types combined:\n{opencl}"
