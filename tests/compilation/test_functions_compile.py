"""
OpenCL compilation tests for GLSL built-in function transformations (Session 3).

Tests that transformed function calls successfully compile with PyOpenCL.

Test Structure:
- Trigonometric functions (5 tests)
- Exponential/Power functions (4 tests)
- Common/Math functions (7 tests)
- Geometric functions (4 tests)
- Combined function tests (3 tests)

Total: 23 compilation tests
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
    Try to compile OpenCL code with GLSL helper function prototypes.

    Returns:
        True if compilation succeeds, False otherwise
    """
    # Include glslHelpers.h which contains all GLSL function implementations
    header = """
#define __GLSL_OVER __attribute__((overloadable))

// Basic GLSL function declarations (subset for testing)
__GLSL_OVER float GLSL_sin(float x){ return sin(x); }
__GLSL_OVER float2 GLSL_sin(float2 x){ return sin(x); }
__GLSL_OVER float3 GLSL_sin(float3 x){ return sin(x); }
__GLSL_OVER float4 GLSL_sin(float4 x){ return sin(x); }

__GLSL_OVER float GLSL_cos(float x){ return cos(x); }
__GLSL_OVER float2 GLSL_cos(float2 x){ return cos(x); }
__GLSL_OVER float3 GLSL_cos(float3 x){ return cos(x); }
__GLSL_OVER float4 GLSL_cos(float4 x){ return cos(x); }

__GLSL_OVER float GLSL_tan(float x){ return tan(x); }
__GLSL_OVER float GLSL_asin(float x){ return asin(x); }
__GLSL_OVER float GLSL_acos(float x){ return acos(x); }
__GLSL_OVER float GLSL_atan(float x){ return atan(x); }
__GLSL_OVER float GLSL_atan(float y, float x){ return atan2(y,x); }

__GLSL_OVER float GLSL_radians(float x){ return x * 0.01745329251994329577f; }
__GLSL_OVER float GLSL_degrees(float x){ return x * 57.2957795130823208768f; }

__GLSL_OVER float GLSL_pow(float x, float y){ return pow(x,y); }
__GLSL_OVER float3 GLSL_pow(float3 x, float3 y){ return pow(x,y); }
__GLSL_OVER float GLSL_exp(float x){ return exp(x); }
__GLSL_OVER float GLSL_log(float x){ return log(x); }
__GLSL_OVER float GLSL_exp2(float x){ return exp2(x); }
__GLSL_OVER float GLSL_log2(float x){ return log2(x); }
__GLSL_OVER float GLSL_sqrt(float x){ return sqrt(x); }
__GLSL_OVER float3 GLSL_sqrt(float3 x){ return sqrt(x); }
__GLSL_OVER float GLSL_inversesqrt(float x){ return rsqrt(x); }

__GLSL_OVER float GLSL_abs(float x){ return fabs(x); }
__GLSL_OVER float GLSL_sign(float x){ return sign(x); }
__GLSL_OVER float GLSL_floor(float x){ return floor(x); }
__GLSL_OVER float2 GLSL_floor(float2 x){ return floor(x); }
__GLSL_OVER float GLSL_ceil(float x){ return ceil(x); }
__GLSL_OVER float GLSL_fract(float x){ return x - floor(x); }
__GLSL_OVER float2 GLSL_fract(float2 x){ return x - floor(x); }

__GLSL_OVER float GLSL_mod(float x, float y){ return x - y * floor(x / y); }
__GLSL_OVER float2 GLSL_mod(float2 x, float2 y){ return x - y * floor(x / y); }
__GLSL_OVER float2 GLSL_mod(float2 x, float y){ return x - (float2)(y) * floor(x / (float2)(y)); }

__GLSL_OVER float GLSL_min(float a, float b){ return fmin(a,b); }
__GLSL_OVER float2 GLSL_min(float2 a, float2 b){ return fmin(a,b); }
__GLSL_OVER float3 GLSL_min(float3 a, float3 b){ return fmin(a,b); }
__GLSL_OVER float GLSL_max(float a, float b){ return fmax(a,b); }
__GLSL_OVER float GLSL_clamp(float x, float a, float b){ return clamp(x,a,b); }

__GLSL_OVER float GLSL_mix(float a, float b, float t){ return a + (b - a) * t; }
__GLSL_OVER float3 GLSL_mix(float3 a, float3 b, float t){ return a + (b - a) * t; }
__GLSL_OVER float3 GLSL_mix(float3 a, float3 b, float3 t){ return a + (b - a) * t; }
__GLSL_OVER float GLSL_step(float edge, float x){ return step(edge,x); }
__GLSL_OVER float GLSL_smoothstep(float a, float b, float x){ return smoothstep(a,b,x); }

__GLSL_OVER float GLSL_length(float3 x){ return length(x); }
__GLSL_OVER float GLSL_distance(float2 a, float2 b){ return distance(a,b); }
__GLSL_OVER float GLSL_distance(float3 a, float3 b){ return distance(a,b); }
__GLSL_OVER float GLSL_dot(float2 a, float2 b){ return dot(a,b); }
__GLSL_OVER float GLSL_dot(float3 a, float3 b){ return dot(a,b); }
__GLSL_OVER float3 GLSL_cross(float3 x, float3 y){ return cross(x,y); }
__GLSL_OVER float3 GLSL_normalize(float3 x){ return normalize(x); }
__GLSL_OVER float3 GLSL_reflect(float3 I, float3 N){ return I - 2.0f * dot(N, I) * N; }
__GLSL_OVER float3 GLSL_refract(float3 I, float3 N, float eta){
    float d = dot(N, I);
    float k = 1.0f - eta*eta * (1.0f - d*d);
    return (k < 0.0f) ? (float3)(0.0f) : eta*I - (eta*d + sqrt(k)) * N;
}

"""
    full_code = header + opencl_code

    try:
        import warnings
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

    success = compile_opencl(opencl_code, cl_context)
    return success, opencl_code


# ========================================================================
# Trigonometric Functions (5 tests)
# ========================================================================

def test_compile_sin_cos_tan(cl_context):
    """Test sin/cos/tan compilation."""
    glsl = """
    void test() {
        float angle = 1.57;
        float s = sin(angle);
        float c = cos(angle);
        float t = tan(angle);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_asin_acos(cl_context):
    """Test asin/acos compilation."""
    glsl = """
    void test() {
        float val = 0.5;
        float as = asin(val);
        float ac = acos(val);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_atan_variants(cl_context):
    """Test atan single and two-arg variants."""
    glsl = """
    void test() {
        float a1 = atan(1.0);
        float a2 = atan(1.0, 2.0);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_radians_degrees(cl_context):
    """Test radians/degrees conversion."""
    glsl = """
    void test() {
        float deg = 180.0;
        float rad = radians(deg);
        float back = degrees(rad);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_trig_with_vectors(cl_context):
    """Test trig functions with vector arguments."""
    glsl = """
    void test() {
        vec3 angles = vec3(0.0, 1.57, 3.14);
        vec3 s = sin(angles);
        vec3 c = cos(angles);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


# ========================================================================
# Exponential/Power Functions (4 tests)
# ========================================================================

def test_compile_pow_exp_log(cl_context):
    """Test pow/exp/log compilation."""
    glsl = """
    void test() {
        float x = 2.0;
        float p = pow(x, 3.0);
        float e = exp(x);
        float l = log(x);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_exp2_log2(cl_context):
    """Test exp2/log2 compilation."""
    glsl = """
    void test() {
        float x = 8.0;
        float e2 = exp2(3.0);
        float l2 = log2(x);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_sqrt_inversesqrt(cl_context):
    """Test sqrt/inversesqrt compilation."""
    glsl = """
    void test() {
        float x = 16.0;
        float s = sqrt(x);
        float inv = inversesqrt(x);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_power_with_vectors(cl_context):
    """Test power functions with vectors."""
    glsl = """
    void test() {
        vec3 base = vec3(2.0, 4.0, 8.0);
        vec3 p = pow(base, vec3(2.0));
        vec3 s = sqrt(base);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


# ========================================================================
# Common/Math Functions (7 tests)
# ========================================================================

def test_compile_abs_sign(cl_context):
    """Test abs/sign compilation."""
    glsl = """
    void test() {
        float neg = -5.5;
        float a = abs(neg);
        float s = sign(neg);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_floor_ceil_fract(cl_context):
    """Test floor/ceil/fract compilation."""
    glsl = """
    void test() {
        float x = 3.7;
        float f = floor(x);
        float c = ceil(x);
        float fr = fract(x);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_mod_function(cl_context):
    """Test mod() with GLSL semantics."""
    glsl = """
    void test() {
        float a = 5.5;
        float b = 2.0;
        float m = mod(a, b);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_min_max_clamp(cl_context):
    """Test min/max/clamp compilation."""
    glsl = """
    void test() {
        float a = 3.0;
        float b = 5.0;
        float minv = min(a, b);
        float maxv = max(a, b);
        float cl = clamp(a, 0.0, 1.0);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_mix_step(cl_context):
    """Test mix/step compilation."""
    glsl = """
    void test() {
        float a = 0.0;
        float b = 1.0;
        float t = 0.5;
        float m = mix(a, b, t);
        float s = step(0.5, 0.7);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_smoothstep(cl_context):
    """Test smoothstep compilation."""
    glsl = """
    void test() {
        float x = smoothstep(0.0, 1.0, 0.5);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_math_with_vectors(cl_context):
    """Test math functions with vectors."""
    glsl = """
    void test() {
        vec3 a = vec3(1.0, 2.0, 3.0);
        vec3 b = vec3(0.5, 1.5, 2.5);
        vec3 minv = min(a, b);
        vec3 m = mix(a, b, 0.5);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


# ========================================================================
# Geometric Functions (4 tests)
# ========================================================================

def test_compile_length_distance(cl_context):
    """Test length/distance compilation."""
    glsl = """
    void test() {
        vec3 v = vec3(1.0, 2.0, 3.0);
        float len = length(v);
        float dist = distance(v, vec3(0.0));
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_dot_cross(cl_context):
    """Test dot/cross compilation."""
    glsl = """
    void test() {
        vec3 a = vec3(1.0, 0.0, 0.0);
        vec3 b = vec3(0.0, 1.0, 0.0);
        float d = dot(a, b);
        vec3 c = cross(a, b);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_normalize(cl_context):
    """Test normalize compilation."""
    glsl = """
    void test() {
        vec3 v = vec3(3.0, 4.0, 0.0);
        vec3 n = normalize(v);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_reflect_refract(cl_context):
    """Test reflect/refract compilation."""
    glsl = """
    void test() {
        vec3 I = vec3(1.0, -1.0, 0.0);
        vec3 N = vec3(0.0, 1.0, 0.0);
        vec3 refl = reflect(I, N);
        vec3 refr = refract(I, N, 0.9);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


# ========================================================================
# Combined Function Tests (3 tests)
# ========================================================================

def test_compile_vignette_shader_functions(cl_context):
    """Test functions used in vignette shader."""
    glsl = """
    void test() {
        vec2 uv = vec2(0.5, 0.5);
        float vig = uv.x * uv.y * 15.0;
        vig = pow(vig, 0.25);
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_sand_shader_functions(cl_context):
    """Test functions used in sand shader."""
    glsl = """
    void test() {
        vec2 p = vec2(1.0, 2.0);
        vec2 i = floor(p);
        vec2 f = fract(p);
        float d = dot(p, vec2(127.1, 311.7));
        float h = fract(sin(d) * 43758.5453123);
        vec3 c = mix(vec3(0.75), vec3(0.95), h);
        c = pow(c, vec3(2.2));
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success


def test_compile_hexagonal_shader_functions(cl_context):
    """Test functions used in hexagonal shader."""
    glsl = """
    void test() {
        vec2 u = vec2(1.0, 2.0);
        vec2 s = vec2(1.0, 1.732);
        vec2 a = mod(u, s) * 2.0 - s;
        vec2 b = mod(u + s * 0.5, s) * 2.0 - s;
        float result = 0.5 * min(dot(a, a), dot(b, b));
    }
    """
    success, _ = transform_and_compile(glsl, cl_context)
    assert success
