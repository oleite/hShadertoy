"""
Unit tests for parenthesized expression transformations.

Tests:
- Parentheses preserve order of operations
- Nested parentheses
- Parentheses in various contexts (declarations, assignments, returns)
- Unary operators with parentheses

Bug Context:
- The transformer was unwrapping parenthesized expressions
- This caused loss of explicit order-of-operations control
- Example: (2.0/iResolution.y) became 2.0/iResolution.y, losing parentheses
- Result: 1.0*(2.0/iResolution.y)*1.0 became 1.0*2.0/iResolution.y*1.0 (WRONG!)
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter


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
    """Create OpenCL code emitter."""
    return OpenCLEmitter()


def transform_and_emit(glsl_code, parser, transformer, emitter):
    """Helper: parse, transform, and emit code."""
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    opencl = emitter.emit(transformed)
    return opencl


# ============================================================================
# 1. Basic Parenthesized Expressions (Order of Operations)
# ============================================================================

def test_parentheses_in_multiplication_division(parser, transformer, emitter):
    """Test: 1.0*(2.0/iResolution.y)*(1.0/fov) preserves parentheses."""
    glsl = """
    void test() {
        float px = 1.0*(2.0/iResolution.y)*(1.0/fov);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Should preserve parentheses around division operations
    assert '(2.0f / iResolution.y)' in opencl
    assert '(1.0f / fov)' in opencl
    # Should NOT flatten to: 1.0f * 2.0f / iResolution.y * 1.0f / fov
    assert '1.0f * 2.0f / iResolution.y * 1.0f / fov' not in opencl


def test_parentheses_with_unary_minus(parser, transformer, emitter):
    """Test: -(1.0+E.y)/I.y preserves parentheses."""
    glsl = """
    void test() {
        float t = -(1.0+E.y)/I.y;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Should preserve parentheses around addition
    assert '-(1.0f + E.y)' in opencl
    # Should NOT become: -1.0f + E.y / I.y
    assert '-1.0f + E.y / I.y' not in opencl


def test_parentheses_in_addition_subtraction(parser, transformer, emitter):
    """Test: (a + b) * c preserves parentheses."""
    glsl = """
    void test() {
        float result = (a + b) * c;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '(a + b) * c' in opencl


def test_parentheses_override_precedence(parser, transformer, emitter):
    """Test: a * (b + c) preserves parentheses."""
    glsl = """
    void test() {
        float result = a * (b + c);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'a * (b + c)' in opencl


# ============================================================================
# 2. Nested Parentheses
# ============================================================================

def test_nested_parentheses(parser, transformer, emitter):
    """Test: ((a + b) * c) preserves both levels of parentheses."""
    glsl = """
    void test() {
        float result = ((a + b) * c);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '((a + b) * c)' in opencl


def test_multiple_parenthesized_groups(parser, transformer, emitter):
    """Test: (a + b) * (c + d) preserves both groups."""
    glsl = """
    void test() {
        float result = (a + b) * (c + d);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '(a + b) * (c + d)' in opencl


# ============================================================================
# 3. Parentheses in Various Contexts
# ============================================================================

def test_parentheses_in_assignment(parser, transformer, emitter):
    """Test parentheses in assignment expression."""
    glsl = """
    void test() {
        float x;
        x = (a + b) / c;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'x = (a + b) / c' in opencl


def test_parentheses_in_return(parser, transformer, emitter):
    """Test parentheses in return statement."""
    glsl = """
    float test() {
        return (a + b) * c;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'return (a + b) * c' in opencl


def test_parentheses_in_function_argument(parser, transformer, emitter):
    """Test parentheses in function call argument."""
    glsl = """
    void test() {
        float result = sin((a + b) * c);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_sin((a + b) * c)' in opencl


def test_parentheses_in_array_index(parser, transformer, emitter):
    """Test parentheses in array indexing."""
    glsl = """
    void test() {
        float result = arr[(i + j) * 2];
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'arr[(i + j) * 2]' in opencl


# ============================================================================
# 4. Complex Real-World Cases (from oporder.glsl)
# ============================================================================

def test_oporder_px_expression(parser, transformer, emitter):
    """Test the exact px expression from oporder.glsl bug report."""
    glsl = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        float fov = 1.8;
        float px = 1.0*(2.0/iResolution.y)*(1.0/fov);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Should have parentheses around both division operations
    assert '1.0f * (2.0f / iResolution.y) * (1.0f / fov)' in opencl


def test_oporder_t_expression(parser, transformer, emitter):
    """Test the exact t expression from oporder.glsl bug report."""
    glsl = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        vec3 E;
        vec3 I;
        float t = -(1.0+E.y)/I.y;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Should have parentheses around the addition
    assert '-(1.0f + E.y) / I.y' in opencl


def test_oporder_full_shader(parser, transformer, emitter):
    """Test the complete oporder.glsl shader."""
    glsl = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        float fov = 1.8;
        vec3 E;
        vec3 I;
        float px = 1.0*(2.0/iResolution.y)*(1.0/fov);
        float t = -(1.0+E.y)/I.y;
        vec3 col = vec3(px,t,0.0);
        fragColor = vec4(col,1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Verify both critical expressions are correct
    assert '1.0f * (2.0f / iResolution.y) * (1.0f / fov)' in opencl
    assert '-(1.0f + E.y) / I.y' in opencl


# ============================================================================
# 5. Edge Cases
# ============================================================================

def test_unnecessary_but_harmless_parentheses(parser, transformer, emitter):
    """Test that even unnecessary parentheses are preserved."""
    glsl = """
    void test() {
        float result = (a) + (b);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Even single-operand parentheses should be preserved
    assert '(a) + (b)' in opencl


def test_parentheses_with_vector_operations(parser, transformer, emitter):
    """Test parentheses with vector types."""
    glsl = """
    void test() {
        vec3 result = (a + b) * vec3(1.0, 2.0, 3.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '(a + b) *' in opencl
    assert '(float3)' in opencl


def test_parentheses_in_conditional(parser, transformer, emitter):
    """Test parentheses in conditional expression."""
    glsl = """
    void test() {
        float result = (x > 0.0) ? (a + b) : (c - d);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '(x > 0.0f)' in opencl
    assert '(a + b)' in opencl
    assert '(c - d)' in opencl
