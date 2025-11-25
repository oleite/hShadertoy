"""
Compilation tests for Session 9: Preprocessor Directives.

Tests that transpiled GLSL with preprocessor directives compiles successfully
with PyOpenCL.

Test coverage:
- Basic macros compile (5 tests)
- Complex macro bodies compile (5 tests)
- Conditional compilation (5 tests)
Total: 15 tests
"""

import pytest
import pyopencl as cl
from tests.transpile import transpile


# Note: These tests verify that preprocessor transformations don't break
# transpilation. They don't compile the final OpenCL code because that requires
# the full Houdini kernel wrapper, which is tested separately by compilecl.py.


def transpile_glsl_shader(glsl_code: str) -> bool:
    """
    Transpile GLSL to OpenCL and check it succeeds.

    For preprocessor tests, we just verify that:
    1. Preprocessing transformations don't break transpilation
    2. No exceptions are raised during transpilation

    Args:
        glsl_code: GLSL source code

    Returns:
        True if transpilation succeeds, False otherwise
    """
    try:
        # Transpile GLSL to OpenCL
        result = transpile(glsl_code)
        header = result.get_header()
        kernel = result.get_kernel()

        # Check that we got output
        assert kernel, "Kernel output is empty"

        # Check that preprocessor directives were transformed
        # (macros should be in the header)
        if "#define" in glsl_code:
            # Header should contain transformed defines
            assert "#define" in header, "Header should contain transformed #define directives"

        return True
    except Exception as e:
        print(f"Transpilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Basic Macros Compile (5 tests)
# ============================================================================

def test_compile_simple_float_macro():
    """Test simple float macro compiles."""
    glsl = """
#define PI 3.14159265

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float x = PI * 2.0;
    fragColor = vec4(x, 0.0, 0.0, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_function_like_macro():
    """Test function-like macro compiles."""
    glsl = """
#define SQUARE(x) ((x)*(x))

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float x = SQUARE(2.0);
    fragColor = vec4(x, 0.0, 0.0, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_macro_with_glsl_function():
    """Test macro with GLSL function compiles."""
    glsl = """
#define SINE(x) sin(x)

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float x = SINE(fragCoord.x);
    fragColor = vec4(x, 0.0, 0.0, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_multiple_macros():
    """Test multiple macros compile."""
    glsl = """
#define PI 3.14159265
#define TWO_PI (PI * 2.0)
#define HALF_PI (PI * 0.5)

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float x = TWO_PI;
    fragColor = vec4(x, HALF_PI, 0.0, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_empty_macro():
    """Test empty macro compiles."""
    glsl = """
#define USE_FEATURE

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


# ============================================================================
# Complex Macro Bodies Compile (5 tests)
# ============================================================================

def test_compile_macro_with_nested_functions():
    """Test macro with nested function calls compiles."""
    glsl = """
#define NOISE(x) fract(sin(x * 12.9898) * 43758.5453)

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float n = NOISE(fragCoord.x);
    fragColor = vec4(n, n, n, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_macro_with_vector_ops():
    """Test macro with vector operations compiles."""
    glsl = """
#define DOT2(a,b) ((a).x*(b).x + (a).y*(b).y)

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 v1 = vec2(1.0, 0.0);
    vec2 v2 = vec2(0.0, 1.0);
    float d = DOT2(v1, v2);
    fragColor = vec4(d, 0.0, 0.0, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_macro_with_ternary():
    """Test macro with ternary operator compiles."""
    glsl = """
#define MAX(a,b) ((a) > (b) ? (a) : (b))

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float m = MAX(fragCoord.x, fragCoord.y);
    fragColor = vec4(m, 0.0, 0.0, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_macro_with_exponent():
    """Test macro with exponential notation compiles."""
    glsl = """
#define LARGE 1e4
#define SMALL 1.5e-3

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float x = LARGE * SMALL;
    fragColor = vec4(x, 0.0, 0.0, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_random_macro():
    """Test complex random macro from macros.glsl compiles."""
    glsl = """
#define random(x) fract(1e4*sin((x)*541.17))

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float noise = random(fragCoord.x);
    fragColor = vec4(noise, noise, noise, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


# ============================================================================
# Conditional Compilation (5 tests)
# ============================================================================

def test_compile_ifdef_defined():
    """Test #ifdef with defined macro compiles."""
    glsl = """
#define USE_RED

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
#ifdef USE_RED
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
#else
    fragColor = vec4(0.0, 1.0, 0.0, 1.0);
#endif
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_ifdef_undefined():
    """Test #ifdef with undefined macro compiles."""
    glsl = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
#ifdef UNDEFINED_MACRO
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
#else
    fragColor = vec4(0.0, 1.0, 0.0, 1.0);
#endif
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_ifndef():
    """Test #ifndef directive compiles."""
    glsl = """
#define FEATURE_ENABLED

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
#ifndef FEATURE_DISABLED
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
#else
    fragColor = vec4(0.0, 1.0, 0.0, 1.0);
#endif
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_nested_conditionals():
    """Test nested conditional directives compile."""
    glsl = """
#define OUTER
#define INNER

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
#ifdef OUTER
    #ifdef INNER
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    #else
        fragColor = vec4(0.0, 1.0, 0.0, 1.0);
    #endif
#else
    fragColor = vec4(0.0, 0.0, 1.0, 1.0);
#endif
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_conditionals_with_macros():
    """Test conditional directives with macros (full integration)."""
    glsl = """
#define random(x) fract(1e4*sin((x)*541.17))
#define PI 3.14159265
#define DIRECTION_X

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
#ifdef DIRECTION_X
    float noise = random(fragCoord.x);
#else
    float noise = random(fragCoord.y);
#endif
    vec3 col = vec3(noise);
    fragColor = vec4(col, 1.0);
}
"""
    assert transpile_glsl_shader(glsl)


def test_compile_ifdef_in_helper_function():
    """Test #ifdef blocks inside helper functions (regression test for bug fix).

    This test verifies that code inside #ifdef blocks in helper functions
    (not just mainImage) gets properly transpiled.

    Bug context: Previously, code inside #ifdef blocks in the header section
    was not being post-processed, causing GLSL functions to remain untransformed.
    """
    glsl = """
#define ANIMATE

float somefunc(float x) {
    vec2 o = mix(x, 1.0, 0.5);
    #ifdef ANIMATE
        o = mix(o + 1.0, x, 0.5);
    #else
        o = mix(o, x, 0.5);
    #endif
    return o.x;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float result = somefunc(fragCoord.x);
    fragColor = vec4(result, 0.0, 0.0, 1.0);
}
"""
    result = transpile(glsl)
    header = result.get_header()

    # Verify that code inside #ifdef blocks was transpiled
    # - mix() should be transformed to GLSL_mix()
    # - vec2 should be transformed to float2
    # - float literals should have 'f' suffix
    assert "GLSL_mix" in header, "mix() inside #ifdef should be transformed to GLSL_mix()"
    assert "float2" in header, "vec2 should be transformed to float2"
    assert "0.5f" in header or "1.0f" in header, "Float literals should have 'f' suffix"

    # Make sure the #ifdef directives are still present
    assert "#ifdef ANIMATE" in header
    assert "#else" in header
    assert "#endif" in header
