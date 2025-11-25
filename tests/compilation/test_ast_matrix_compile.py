"""
Compilation tests for matrix operation transformations (Session 4).

Tests that matrix operation transformations produce valid, compilable OpenCL code.
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
    Try to compile OpenCL code with Houdini headers.

    Returns:
        True if compilation succeeds, False otherwise
    """
    # Note: glslHelpers.h has GLSL_mul but requires undefined helpers (mat2mul, etc.)
    # For unit test compilation, use self-contained inline definitions
    header = """
// Matrix type definitions
typedef float fpreal;
typedef float2 fpreal2;
typedef float3 fpreal3;
typedef float4 fpreal4;
typedef float16 fpreal16;

typedef fpreal4 mat2;       // 2x2 matrix as float4
typedef fpreal3 mat3[3];    // 3x3 matrix as array of float3
typedef fpreal16 mat4;      // 4x4 matrix as float16

// GLSL function wrappers (minimal set for testing)
__attribute__((overloadable))
float GLSL_cos(float x) { return cos(x); }

__attribute__((overloadable))
float GLSL_sin(float x) { return sin(x); }

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

// Matrix * Vector multiplication
__attribute__((overloadable))
float2 GLSL_mul(const mat2 A, float2 v){
    return (float2)(dot(A.lo, v), dot(A.hi, v));
}

__attribute__((overloadable))
float3 GLSL_mul(const mat3 A, float3 v){
    return (float3)(
        dot(A[0], v),
        dot(A[1], v),
        dot(A[2], v)
    );
}

__attribute__((overloadable))
float4 GLSL_mul(const mat4 A, float4 v){
    return (float4)(
        dot(A.lo.lo, v),
        dot(A.lo.hi, v),
        dot(A.hi.lo, v),
        dot(A.hi.hi, v)
    );
}

// Vector * Matrix multiplication
__attribute__((overloadable))
float2 GLSL_mul(float2 v, const mat2 A){
    return (float2)(
        v.x * A.s0 + v.y * A.s1,
        v.x * A.s2 + v.y * A.s3
    );
}

__attribute__((overloadable))
float3 GLSL_mul(float3 v, const mat3 A){
    return (float3)(
        v.x * A[0].x + v.y * A[1].x + v.z * A[2].x,
        v.x * A[0].y + v.y * A[1].y + v.z * A[2].y,
        v.x * A[0].z + v.y * A[1].z + v.z * A[2].z
    );
}

__attribute__((overloadable))
float4 GLSL_mul(float4 v, const mat4 A){
    return (float4)(
        v.x * A.s0 + v.y * A.s1 + v.z * A.s2 + v.w * A.s3,
        v.x * A.s4 + v.y * A.s5 + v.z * A.s6 + v.w * A.s7,
        v.x * A.s8 + v.y * A.s9 + v.z * A.sa + v.w * A.sb,
        v.x * A.sc + v.y * A.sd + v.z * A.se + v.w * A.sf
    );
}

// Matrix * Matrix multiplication
__attribute__((overloadable))
mat2 GLSL_mul(const mat2 A, const mat2 B){
    return (mat2)(
        A.s0 * B.s0 + A.s2 * B.s1,
        A.s1 * B.s0 + A.s3 * B.s1,
        A.s0 * B.s2 + A.s2 * B.s3,
        A.s1 * B.s2 + A.s3 * B.s3
    );
}

__attribute__((overloadable))
void GLSL_mul(const mat3 A, const mat3 B, __private mat3 C){
    C[0] = (float3)(
        A[0].x * B[0].x + A[1].x * B[0].y + A[2].x * B[0].z,
        A[0].y * B[0].x + A[1].y * B[0].y + A[2].y * B[0].z,
        A[0].z * B[0].x + A[1].z * B[0].y + A[2].z * B[0].z
    );
    C[1] = (float3)(
        A[0].x * B[1].x + A[1].x * B[1].y + A[2].x * B[1].z,
        A[0].y * B[1].x + A[1].y * B[1].y + A[2].y * B[1].z,
        A[0].z * B[1].x + A[1].z * B[1].y + A[2].z * B[1].z
    );
    C[2] = (float3)(
        A[0].x * B[2].x + A[1].x * B[2].y + A[2].x * B[2].z,
        A[0].y * B[2].x + A[1].y * B[2].y + A[2].y * B[2].z,
        A[0].z * B[2].x + A[1].z * B[2].y + A[2].z * B[2].z
    );
}

__attribute__((overloadable))
mat4 GLSL_mul(const mat4 A, const mat4 B){
    mat4 result;
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            float sum = 0.0f;
            for(int k = 0; k < 4; k++){
                sum += ((__private float*)&A)[i + k*4] * ((__private float*)&B)[k + j*4];
            }
            ((__private float*)&result)[i + j*4] = sum;
        }
    }
    return result;
}
"""

    full_code = header + "\n" + opencl_code

    try:
        import warnings
        # Suppress PyOpenCL CompilerWarning (warnings are OK, only fail on actual errors)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=cl.CompilerWarning)
            cl.Program(cl_context, full_code).build()
        return True
    except cl.RuntimeError as e:
        print(f"Compilation failed: {e}")
        return False


# ============================================================================
# mat2 Compilation Tests
# ============================================================================

def test_compile_mat2_vector_multiply(cl_context):
    """Test mat2 * vec2 compiles."""
    glsl = """
    kernel void test() {
        mat2 M = GLSL_mat2_diagonal(1.0f);
        float2 v = (float2)(1.0f, 0.0f);
        float2 result = GLSL_mul(M, v);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_vec2_matrix_multiply(cl_context):
    """Test vec2 * mat2 compiles."""
    glsl = """
    kernel void test() {
        float2 v = (float2)(1.0f, 0.0f);
        mat2 M = GLSL_mat2_diagonal(1.0f);
        float2 result = GLSL_mul(v, M);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_mat2_mat2_multiply(cl_context):
    """Test mat2 * mat2 compiles."""
    glsl = """
    kernel void test() {
        mat2 M1 = GLSL_mat2_diagonal(1.0f);
        mat2 M2 = GLSL_mat2_diagonal(2.0f);
        mat2 result = GLSL_mul(M1, M2);
    }
    """
    assert compile_opencl(glsl, cl_context)


# ============================================================================
# mat3 Compilation Tests
# ============================================================================

def test_compile_mat3_vector_multiply(cl_context):
    """Test mat3 * vec3 compiles."""
    glsl = """
    kernel void test() {
        mat3 M;
        GLSL_mat3_diagonal(1.0f, &M);
        float3 v = (float3)(1.0f, 0.0f, 0.0f);
        float3 result = GLSL_mul(M, v);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_vec3_matrix_multiply(cl_context):
    """Test vec3 * mat3 compiles."""
    glsl = """
    kernel void test() {
        float3 v = (float3)(1.0f, 0.0f, 0.0f);
        mat3 M;
        GLSL_mat3_diagonal(1.0f, &M);
        float3 result = GLSL_mul(v, M);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_mat3_mat3_multiply(cl_context):
    """Test mat3 * mat3 compiles (out-parameter)."""
    glsl = """
    kernel void test() {
        mat3 M1, M2, result;
        GLSL_mat3_diagonal(1.0f, &M1);
        GLSL_mat3_diagonal(2.0f, &M2);
        GLSL_mul(M1, M2, &result);
    }
    """
    assert compile_opencl(glsl, cl_context)


# ============================================================================
# mat4 Compilation Tests
# ============================================================================

def test_compile_mat4_vector_multiply(cl_context):
    """Test mat4 * vec4 compiles."""
    glsl = """
    kernel void test() {
        mat4 M = GLSL_mat4_diagonal(1.0f);
        float4 v = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
        float4 result = GLSL_mul(M, v);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_vec4_matrix_multiply(cl_context):
    """Test vec4 * mat4 compiles."""
    glsl = """
    kernel void test() {
        float4 v = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
        mat4 M = GLSL_mat4_diagonal(1.0f);
        float4 result = GLSL_mul(v, M);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_mat4_mat4_multiply(cl_context):
    """Test mat4 * mat4 compiles."""
    glsl = """
    kernel void test() {
        mat4 M1 = GLSL_mat4_diagonal(1.0f);
        mat4 M2 = GLSL_mat4_diagonal(2.0f);
        mat4 result = GLSL_mul(M1, M2);
    }
    """
    assert compile_opencl(glsl, cl_context)


# ============================================================================
# Integrated Compilation Tests
# ============================================================================

def test_compile_chained_mat2_operations(cl_context):
    """Test chained mat2 operations compile."""
    glsl = """
    kernel void test() {
        float2 v = (float2)(1.0f, 0.0f);
        mat2 M1 = GLSL_mat2_diagonal(1.0f);
        mat2 M2 = GLSL_mat2_diagonal(2.0f);
        float2 temp = GLSL_mul(v, M1);
        float2 result = GLSL_mul(temp, M2);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_mat2_full_constructor_multiply(cl_context):
    """Test mat2 full constructor with multiplication compiles."""
    glsl = """
    kernel void test() {
        mat2 M = (mat2)(1.0f, 0.0f, 0.0f, 1.0f);
        float2 v = (float2)(1.0f, 0.0f);
        float2 result = GLSL_mul(M, v);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_mat3_full_constructor_multiply(cl_context):
    """Test mat3 full constructor with multiplication compiles."""
    glsl = """
    kernel void test() {
        mat3 M = {(float3)(1.0f, 0.0f, 0.0f), (float3)(0.0f, 1.0f, 0.0f), (float3)(0.0f, 0.0f, 1.0f)};
        float3 v = (float3)(1.0f, 0.0f, 0.0f);
        float3 result = GLSL_mul(M, v);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_rotation_shader_pattern(cl_context):
    """Test typical rotation shader pattern compiles."""
    glsl = """
    kernel void test() {
        float angle = 1.0f;
        mat2 rotation = (mat2)(cos(angle), -sin(angle), sin(angle), cos(angle));
        float2 uv = (float2)(0.5f, 0.5f);
        float2 rotatedUV = GLSL_mul(rotation, uv);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_multiple_matrix_operations(cl_context):
    """Test multiple independent matrix operations compile."""
    glsl = """
    kernel void test() {
        mat2 M = GLSL_mat2_diagonal(1.0f);
        float2 v1 = (float2)(1.0f, 0.0f);
        float2 v2 = (float2)(0.0f, 1.0f);
        float2 r1 = GLSL_mul(M, v1);
        float2 r2 = GLSL_mul(v2, M);
    }
    """
    assert compile_opencl(glsl, cl_context)


def test_compile_transformed_shader(cl_context):
    """Test a realistic transformed shader compiles (extracts mainImage body for kernel)."""
    parser = GLSLParser()
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    ast_transformer = ASTTransformer(type_checker)
    emitter = CodeEmitter()

    glsl = """
    void mainImage(vec2 fragCoord) {
        vec2 uv = fragCoord;
        float angle = 1.0;
        mat2 rotation = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
        vec2 rotatedUV = rotation * uv;
    }
    """

    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)

    # Extract mainImage function body (hShadertoy architecture: body becomes kernel)
    # Find mainImage function in the transformed AST
    mainImage_func = None
    for decl in transformed.declarations:
        if hasattr(decl, 'name') and decl.name == 'mainImage':
            mainImage_func = decl
            break

    assert mainImage_func is not None, "mainImage function not found"

    # Emit just the body statements (not the function definition)
    body_statements = ""
    for stmt in mainImage_func.body.statements:
        body_statements += emitter.emit(stmt)

    # Wrap body in kernel for compilation (matching hShadertoy architecture)
    kernel_code = f"""
    kernel void test(float2 fragCoord) {{
{body_statements}    }}
    """

    assert compile_opencl(kernel_code, cl_context)
