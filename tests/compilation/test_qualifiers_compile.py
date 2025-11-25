"""
Compilation tests for function parameter qualifiers (Session 8).

Tests that qualifier transformations produce valid, compilable OpenCL code.
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
    Try to compile OpenCL code with matrix helpers.

    Returns:
        True if compilation succeeds, False otherwise
    """
    # Add Houdini types and GLSL helpers
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

// Minimal GLSL functions
__attribute__((overloadable))
float GLSL_sin(float x) { return sin(x); }
__attribute__((overloadable))
float GLSL_cos(float x) { return cos(x); }
__attribute__((overloadable))
float GLSL_fract(float x) { return x - floor(x); }
__attribute__((overloadable))
float GLSL_sqrt(float x) { return sqrt(x); }
__attribute__((overloadable))
float GLSL_pow(float x, float y) { return pow(x, y); }
__attribute__((overloadable))
float GLSL_max(float x, float y) { return max(x, y); }
__attribute__((overloadable))
float2 GLSL_normalize(float2 v) { return normalize(v); }
__attribute__((overloadable))
float3 GLSL_normalize(float3 v) { return normalize(v); }
__attribute__((overloadable))
float GLSL_dot(float2 a, float2 b) { return dot(a, b); }
__attribute__((overloadable))
float GLSL_dot(float3 a, float3 b) { return dot(a, b); }

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
# All Qualifier Combinations Tests (8 tests)
# ============================================================================

def test_compile_in_qualifier(cl_context):
    """Test in qualifier compiles (removed in OpenCL)."""
    glsl = """
    float test(in float x) {
        return x * 2.0;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile in qualifier:\n{opencl}"


def test_compile_out_scalar(cl_context):
    """Test out scalar parameter compiles."""
    glsl = """
    void test(out float x) {
        x = 1.0;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile out scalar:\n{opencl}"


def test_compile_out_vec2(cl_context):
    """Test out vec2 parameter compiles."""
    glsl = """
    void test(out vec2 v) {
        v = vec2(1.0, 2.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile out vec2:\n{opencl}"


def test_compile_out_vec3(cl_context):
    """Test out vec3 parameter compiles."""
    glsl = """
    void test(out vec3 v) {
        v = vec3(1.0, 2.0, 3.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile out vec3:\n{opencl}"


def test_compile_out_vec4(cl_context):
    """Test out vec4 parameter compiles."""
    glsl = """
    void test(out vec4 v) {
        v = vec4(1.0, 2.0, 3.0, 4.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile out vec4:\n{opencl}"


def test_compile_out_mat2(cl_context):
    """Test out mat2 parameter compiles."""
    glsl = """
    void test(out mat2 M) {
        M = mat2(1.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile out mat2:\n{opencl}"


def test_compile_out_mat3(cl_context):
    """Test out mat3 parameter compiles (simple declaration)."""
    glsl = """
    void test(out mat3 M) {
        M[0] = vec3(1.0, 0.0, 0.0);
        M[1] = vec3(0.0, 1.0, 0.0);
        M[2] = vec3(0.0, 0.0, 1.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile out mat3:\n{opencl}"


def test_compile_out_mat4(cl_context):
    """Test out mat4 parameter compiles."""
    glsl = """
    void test(out mat4 M) {
        M = mat4(1.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile out mat4:\n{opencl}"


# ============================================================================
# Inout and Mixed Qualifiers Tests (4 tests)
# ============================================================================

def test_compile_inout_parameters(cl_context):
    """Test inout parameters compile."""
    glsl = """
    void test(inout float x, inout vec2 v) {
        x = 2.0;
        v = vec2(1.0, 2.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile inout parameters:\n{opencl}"


def test_compile_const_parameters(cl_context):
    """Test const parameters compile."""
    glsl = """
    float test(const float x, const vec2 v) {
        return x + v.x;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile const parameters:\n{opencl}"


def test_compile_mixed_qualifiers(cl_context):
    """Test mixed qualifiers in same function compile."""
    glsl = """
    void test(in float a, out float b, inout float c, const float d) {
        b = a * 2.0;
        c = d + 1.0;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mixed qualifiers:\n{opencl}"


def test_compile_no_qualifiers(cl_context):
    """Test parameters with no qualifiers compile."""
    glsl = """
    float test(float x, vec2 v) {
        return x + v.x;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile no qualifiers:\n{opencl}"


# ============================================================================
# Dereference and Address-Of Tests (4 tests)
# ============================================================================

def test_compile_dereference_assignments(cl_context):
    """Test dereference assignments compile."""
    glsl = """
    void test(out float x, out vec2 v) {
        x = 1.0;
        v = vec2(2.0, 3.0);
        x = 2.0;
        v = vec2(4.0, 6.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile dereference assignments:\n{opencl}"


def test_compile_addressof_calls(cl_context):
    """Test address-of at call sites compiles."""
    glsl = """
    void helper(out float x, out vec2 v) {
        x = 1.0;
        v = vec2(2.0, 3.0);
    }
    void test() {
        float a;
        vec2 b;
        helper(a, b);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile address-of calls:\n{opencl}"


def test_compile_nested_calls(cl_context):
    """Test nested function calls with out parameters compile."""
    glsl = """
    void inner(out float x) {
        x = 1.0;
    }
    void outer(out float y) {
        inner(y);
    }
    void test() {
        float z;
        outer(z);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile nested calls:\n{opencl}"


def test_compile_compound_assignments(cl_context):
    """Test compound assignments with out parameters compile."""
    glsl = """
    void test(inout float x, inout vec2 v) {
        x += 1.0;
        v *= 2.0;
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile compound assignments:\n{opencl}"


# ============================================================================
# Realistic Shader Patterns Tests (4 tests)
# ============================================================================

def test_compile_complex_function(cl_context):
    """Test complex function with multiple out parameters."""
    glsl = """
    void calculateLighting(vec3 normal, vec3 lightDir,
                          out float diffuse, out float specular) {
        float d = max(0.0, dot(normal, lightDir));
        diffuse = d;
        specular = pow(d, 32.0);
    }
    void test() {
        float d, s;
        calculateLighting(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), d, s);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile complex function:\n{opencl}"


def test_compile_realistic_random_function(cl_context):
    """Test realistic random function with out parameter."""
    glsl = """
    void random2(vec2 uv, out vec2 noise2) {
        float a = fract(1e4 * sin(uv.x * 541.17));
        float b = fract(1e4 * sin(uv.y * 321.46));
        noise2 = vec2(a, b);
    }
    void test() {
        vec2 pixelnoise;
        random2(vec2(1.0, 2.0), pixelnoise);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile random function:\n{opencl}"


def test_compile_realistic_noise_function(cl_context):
    """Test realistic noise function with inout parameter."""
    glsl = """
    void accumulateNoise(vec2 uv, inout float noise) {
        noise += fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
    }
    void test() {
        float n = 0.0;
        accumulateNoise(vec2(1.0, 2.0), n);
        accumulateNoise(vec2(3.0, 4.0), n);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile noise function:\n{opencl}"


def test_compile_mainimage_pattern(cl_context):
    """Test mainImage pattern with out parameter."""
    glsl = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        vec2 uv = fragCoord / vec2(512.0, 288.0);
        fragColor = vec4(uv, 0.5, 1.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile mainImage pattern:\n{opencl}"


# ============================================================================
# All Qualifiers Combined Test
# ============================================================================

def test_compile_all_qualifiers_combined(cl_context):
    """Test comprehensive shader with all qualifier types."""
    glsl = """
    void helper1(in float x, out float y) {
        y = x * 2.0;
    }
    void helper2(inout vec2 v, const float scale) {
        v = vec2(scale, scale);
    }
    void helper3(out mat2 M) {
        M = mat2(1.0);
    }
    void mainFunction(out vec4 result) {
        float a = 1.0;
        float b;
        helper1(a, b);

        vec2 v = vec2(2.0, 3.0);
        helper2(v, 2.0);

        mat2 M;
        helper3(M);

        result = vec4(b, v.x, v.y, 1.0);
    }
    """
    success, opencl = transform_and_compile(glsl, cl_context)
    assert success, f"Failed to compile all qualifiers combined:\n{opencl}"
