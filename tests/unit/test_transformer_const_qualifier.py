"""
Unit tests for const qualifier on variable declarations.

Tests transformation of GLSL const qualifier to OpenCL const qualifier.

Examples:
    GLSL: const float foo = 0.5;
    OpenCL: const float foo = 0.5f;

    GLSL: const vec3 v3 = vec3(0.);
    OpenCL: const float3 v3 = (float3)(0.f);
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter


@pytest.fixture
def parser():
    """Create GLSL parser."""
    return GLSLParser()


@pytest.fixture
def transformer():
    """Create transformer with type checker."""
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    return ASTTransformer(type_checker)


@pytest.fixture
def emitter():
    """Create code emitter."""
    return CodeEmitter()


def transform_and_emit(glsl_code, parser, transformer, emitter):
    """Helper: parse, transform, and emit code."""
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    opencl = emitter.emit(transformed)
    return opencl


# ============================================================================
# 1. Const Qualifier on Scalar Declarations
# ============================================================================

def test_const_float_global(parser, transformer, emitter):
    """Test global const float declaration."""
    glsl = """
    const float foo = 0.5;
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'const float foo = 0.5f;' in opencl


def test_const_int_global(parser, transformer, emitter):
    """Test global const int declaration."""
    glsl = """
    const int i = 1;
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'const int i = 1;' in opencl


def test_const_float_local(parser, transformer, emitter):
    """Test local const float declaration."""
    glsl = """
    void mainImage() {
        const float bar = 0.5;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'const float bar = 0.5f;' in opencl


# ============================================================================
# 2. Const Qualifier on Vector Declarations
# ============================================================================

def test_const_vec2_global(parser, transformer, emitter):
    """Test global const vec2 declaration."""
    glsl = """
    const vec2 v2 = vec2(0.);
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'const float2 v2 = (float2)(0.f);' in opencl


def test_const_vec3_global(parser, transformer, emitter):
    """Test global const vec3 declaration."""
    glsl = """
    const vec3 v3 = vec3(0.);
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'const float3 v3 = (float3)(0.f);' in opencl


def test_const_vec4_global(parser, transformer, emitter):
    """Test global const vec4 declaration."""
    glsl = """
    const vec4 v4 = vec4(1., 2., 3., 4.);
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'const float4 v4 = (float4)(1.f, 2.f, 3.f, 4.f);' in opencl


def test_const_vec2_local(parser, transformer, emitter):
    """Test local const vec2 declaration."""
    glsl = """
    void mainImage() {
        const vec2 v2 = vec2(0.);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'const float2 v2 = (float2)(0.f);' in opencl


# ============================================================================
# 3. Const Qualifier with Comma-Separated Declarations
# ============================================================================

def test_const_comma_separated_float(parser, transformer, emitter):
    """Test const qualifier on comma-separated float declarations."""
    glsl = """
    const float x = 1.0, y = 2.0;
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'const float x = 1.0f, y = 2.0f;' in opencl


def test_const_comma_separated_int(parser, transformer, emitter):
    """Test const qualifier on comma-separated int declarations."""
    glsl = """
    const int a = 10, b = 20, c = 30;
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'const int a = 10, b = 20, c = 30;' in opencl


# ============================================================================
# 4. Non-Const Declarations (ensure no regression)
# ============================================================================

def test_non_const_float(parser, transformer, emitter):
    """Test non-const float declaration (no const qualifier)."""
    glsl = """
    float x = 1.0;
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float x = 1.0f;' in opencl
    # Ensure const is not added
    assert 'const float' not in opencl


def test_non_const_vec3(parser, transformer, emitter):
    """Test non-const vec3 declaration (no const qualifier)."""
    glsl = """
    vec3 v = vec3(0.);
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float3 v = (float3)(0.f);' in opencl
    # Ensure const is not added
    assert 'const float3' not in opencl


# ============================================================================
# 5. Const Qualifier in Function Context
# ============================================================================

def test_const_in_function_body(parser, transformer, emitter):
    """Test const declarations inside function body."""
    glsl = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        const float foo = 0.5;
        const int i = 1;
        const vec3 v3 = vec3(0.);
        vec3 col = vec3(foo, foo, float(i));
        fragColor = vec4(col, 1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'const float foo = 0.5f;' in opencl
    assert 'const int i = 1;' in opencl
    assert 'const float3 v3 = (float3)(0.f);' in opencl
    # Ensure non-const declarations don't have const
    assert opencl.count('const') == 3  # Only the 3 const declarations


# ============================================================================
# 6. Complex const usage from example shader
# ============================================================================

def test_const_example_shader(parser, transformer, emitter):
    """Test full example shader with const qualifiers."""
    glsl = """
    const float foo = 0.5;
    const int i = 1;
    const vec3 v3 = vec3(0.);

    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        const float bar = 0.5;
        const vec2 v2 = vec2(0.);
        vec3 col = vec3(foo, bar, float(i));
        fragColor = vec4(col, 1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)

    # Check global const declarations
    assert 'const float foo = 0.5f;' in opencl
    assert 'const int i = 1;' in opencl
    assert 'const float3 v3 = (float3)(0.f);' in opencl

    # Check local const declarations in function
    assert 'const float bar = 0.5f;' in opencl
    assert 'const vec2 v2 = vec2(0.);' in opencl or 'const float2 v2 = (float2)(0.f);' in opencl

    # Check non-const declarations don't have const
    assert 'float3 col =' in opencl  # col should not be const
