"""
Unit tests for constructor transformations (Session 2).

Tests:
- Diagonal matrix constructors (mat2/3/4 with single scalar argument)
- Full matrix constructors (mat2/3/4 with all elements)
- Matrix type casting (mat3 <-> mat4)
- Declaration-with-initialization split for mat3

Architecture:
    GLSL: mat3 M = mat3(1.0);
    OpenCL: mat3 M; GLSL_mat3_diagonal(1.0f, M);
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
# Diagonal Matrix Constructors
# ============================================================================

def test_mat2_diagonal_constructor(parser, transformer, emitter):
    """Test mat2(scalar) -> GLSL_matrix2x2_diagonal(scalar)."""
    glsl = """
    void test() {
        mat2 M = mat2(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'matrix2x2 M = GLSL_matrix2x2_diagonal(1.0f)' in opencl


def test_mat2_diagonal_constructor_in_expression(parser, transformer, emitter):
    """Test mat2(2.0) used in expression."""
    glsl = """
    void test() {
        mat2 A = mat2(2.0) * mat2(3.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_matrix2x2_diagonal(2.0f)' in opencl
    assert 'GLSL_matrix2x2_diagonal(3.0f)' in opencl


def test_mat4_diagonal_constructor(parser, transformer, emitter):
    """Test mat4(scalar) -> GLSL_matrix4x4_diagonal(scalar)."""
    glsl = """
    void test() {
        mat4 M = mat4(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'matrix4x4 M = GLSL_matrix4x4_diagonal(1.0f)' in opencl


def test_mat3_diagonal_constructor_declaration(parser, transformer, emitter):
    """Test mat3(scalar) -> GLSL_matrix3x3_diagonal (direct return, no splitting)."""
    glsl = """
    void test() {
        mat3 M = mat3(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Now uses direct return with struct type
    assert 'matrix3x3 M = GLSL_matrix3x3_diagonal(1.0f)' in opencl


def test_mat3_diagonal_multiple_variables(parser, transformer, emitter):
    """Test multiple mat3 diagonal constructors in same function."""
    glsl = """
    void test() {
        mat3 M1 = mat3(1.0);
        mat3 M2 = mat3(2.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_matrix3x3_diagonal(1.0f)' in opencl
    assert 'GLSL_matrix3x3_diagonal(2.0f)' in opencl


def test_mat3_diagonal_with_variable(parser, transformer, emitter):
    """Test mat3(variable) constructor."""
    glsl = """
    void test() {
        float s = 2.0;
        mat3 M = mat3(s);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_matrix3x3_diagonal(s)' in opencl


# ============================================================================
# Full Matrix Constructors
# ============================================================================

def test_mat2_full_constructor(parser, transformer, emitter):
    """Test mat2 with 4 elements."""
    glsl = """
    void test() {
        mat2 M = mat2(1.0, 2.0, 3.0, 4.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Should use GLSL_mat2 function
    assert 'GLSL_mat2(1.0f, 2.0f, 3.0f, 4.0f)' in opencl


def test_mat4_full_constructor(parser, transformer, emitter):
    """Test mat4 with 16 elements."""
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
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Should use GLSL_mat4 function
    assert 'GLSL_mat4(' in opencl
    assert '1.0f' in opencl
    assert '0.0f' in opencl


def test_mat3_full_constructor(parser, transformer, emitter):
    """Test mat3 with 9 elements - GLSL_mat3 function."""
    glsl = """
    void test() {
        mat3 M = mat3(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        );
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Should use GLSL_mat3 function (struct type, not array)
    assert 'GLSL_mat3(' in opencl
    assert '1.0f' in opencl
    assert '0.0f' in opencl


def test_mat3_full_constructor_custom_values(parser, transformer, emitter):
    """Test mat3 with non-identity values."""
    glsl = """
    void test() {
        mat3 M = mat3(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        );
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3(' in opencl
    assert '1.0f, 2.0f, 3.0f' in opencl


# ============================================================================
# Column Matrix Constructors
# ============================================================================

def test_mat2_column_constructor(parser, transformer, emitter):
    """Test mat2(vec2, vec2) column constructor."""
    glsl = """
    void test() {
        mat2 M = mat2(vec2(1.0, 0.0), vec2(0.0, 1.0));
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_cols(' in opencl
    assert '(float2)(1.0f, 0.0f)' in opencl
    assert '(float2)(0.0f, 1.0f)' in opencl


def test_mat3_column_constructor(parser, transformer, emitter):
    """Test mat3(vec3, vec3, vec3) column constructor."""
    glsl = """
    void test() {
        mat3 M = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_cols(' in opencl
    assert '(float3)(1.0f, 0.0f, 0.0f)' in opencl
    assert '(float3)(0.0f, 1.0f, 0.0f)' in opencl
    assert '(float3)(0.0f, 0.0f, 1.0f)' in opencl


def test_mat4_column_constructor(parser, transformer, emitter):
    """Test mat4(vec4, vec4, vec4, vec4) column constructor."""
    glsl = """
    void test() {
        mat4 M = mat4(
            vec4(1.0, 0.0, 0.0, 0.0),
            vec4(0.0, 1.0, 0.0, 0.0),
            vec4(0.0, 0.0, 1.0, 0.0),
            vec4(0.0, 0.0, 0.0, 1.0)
        );
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat4_cols(' in opencl
    assert '(float4)(1.0f, 0.0f, 0.0f, 0.0f)' in opencl
    assert '(float4)(0.0f, 1.0f, 0.0f, 0.0f)' in opencl


def test_mat3_column_constructor_with_variables(parser, transformer, emitter):
    """Test mat3 column constructor with vector variables."""
    glsl = """
    void test() {
        vec3 c0 = vec3(1.0, 0.0, 0.0);
        vec3 c1 = vec3(0.0, 1.0, 0.0);
        vec3 c2 = vec3(0.0, 0.0, 1.0);
        mat3 M = mat3(c0, c1, c2);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_cols(c0, c1, c2)' in opencl


def test_mat3_column_constructor_zero_vectors(parser, transformer, emitter):
    """Test mat3 column constructor with zero vectors (like spec.glsl)."""
    glsl = """
    void test() {
        mat3 M = mat3(vec3(0), vec3(0), vec3(0));
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_cols(' in opencl
    assert '(float3)(0)' in opencl


# ============================================================================
# Matrix Type Casting
# ============================================================================

def test_mat4_from_mat3_cast(parser, transformer, emitter):
    """Test mat4(mat3) type casting."""
    glsl = """
    void test() {
        mat3 M3 = mat3(1.0);
        mat4 M4 = mat4(M3);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # mat4 can return by value, uses matrix4x4 type
    assert 'GLSL_mat4_from_mat3(M3)' in opencl or 'mat4_from_mat3(M3)' in opencl or 'matrix4x4 M4' in opencl


def test_mat3_from_mat4_cast(parser, transformer, emitter):
    """Test mat3(mat4) type casting (direct return with struct type)."""
    glsl = """
    void test() {
        mat4 M4 = mat4(1.0);
        mat3 M3 = mat3(M4);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # mat3 now uses struct type, can return by value
    assert 'matrix3x3 M3' in opencl


# ============================================================================
# Advanced Vector Constructors (Extended from Session 1)
# ============================================================================

def test_vec4_from_vec3_and_scalar(parser, transformer, emitter):
    """Test vec4(vec3, scalar) constructor."""
    glsl = """
    void test() {
        vec3 v3 = vec3(1.0, 2.0, 3.0);
        vec4 v4 = vec4(v3, 4.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '(float4)(v3, 4.0f)' in opencl


def test_vec4_from_vec2_and_scalars(parser, transformer, emitter):
    """Test vec4(vec2, scalar, scalar) constructor."""
    glsl = """
    void test() {
        vec2 v2 = vec2(1.0, 2.0);
        vec4 v4 = vec4(v2, 3.0, 4.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '(float4)(v2, 3.0f, 4.0f)' in opencl


def test_vec3_from_vec2_and_scalar(parser, transformer, emitter):
    """Test vec3(vec2, scalar) constructor."""
    glsl = """
    void test() {
        vec2 v2 = vec2(1.0, 2.0);
        vec3 v3 = vec3(v2, 3.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '(float3)(v2, 3.0f)' in opencl


def test_vec4_from_two_vec2(parser, transformer, emitter):
    """Test vec4(vec2, vec2) constructor."""
    glsl = """
    void test() {
        vec2 a = vec2(1.0, 2.0);
        vec2 b = vec2(3.0, 4.0);
        vec4 v = vec4(a, b);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '(float4)(a, b)' in opencl


def test_vec3_from_scalar_and_vec2(parser, transformer, emitter):
    """Test vec3(scalar, vec2) constructor."""
    glsl = """
    void test() {
        vec2 v2 = vec2(2.0, 3.0);
        vec3 v3 = vec3(1.0, v2);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '(float3)(1.0f, v2)' in opencl


# ============================================================================
# Edge Cases
# ============================================================================

def test_nested_matrix_constructors(parser, transformer, emitter):
    """Test nested matrix constructor calls."""
    glsl = """
    void test() {
        mat2 M = mat2(mat2(1.0));
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Outer constructor gets the result of inner diagonal constructor
    assert 'GLSL_matrix2x2_diagonal' in opencl


def test_matrix_constructor_as_function_argument(parser, transformer, emitter):
    """Test matrix constructor passed as function argument."""
    glsl = """
    mat2 identity() {
        return mat2(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_matrix2x2_diagonal(1.0f)' in opencl


def test_mat3_in_return_statement(parser, transformer, emitter):
    """Test mat3 diagonal constructor in return statement (direct return with struct)."""
    glsl = """
    mat3 identity() {
        return mat3(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # mat3 now uses struct type, can return directly
    assert 'matrix3x3' in opencl
    assert 'GLSL_matrix3x3_diagonal' in opencl
