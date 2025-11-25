"""
Unit tests for if/else/else-if statement transformations.

Tests:
- Simple if statements
- if-else statements
- if-else-if chains
- Nested if statements
- if statements with complex conditions

This test validates the fix for the bug where else and else-if blocks were being skipped.

Bug Context:
- The tree-sitter parser wraps else blocks in an 'else_clause' node
- The transformer was missing a handler for 'else_clause', causing it to return None
- This resulted in else and else-if blocks being completely skipped
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
# 1. Simple If Statements
# ============================================================================

def test_simple_if_statement(parser, transformer, emitter):
    """Test simple if statement without else."""
    glsl = """
    void test() {
        float x = 1.0;
        if (x > 0.0) {
            x = 2.0;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (x > 0.0f)' in opencl
    assert 'x = 2.0f' in opencl


def test_if_statement_with_complex_condition(parser, transformer, emitter):
    """Test if statement with complex boolean condition."""
    glsl = """
    void test() {
        float x = 1.0;
        float y = 2.0;
        if (x > 0.0 && y < 10.0) {
            x = y;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (x > 0.0f && y < 10.0f)' in opencl


# ============================================================================
# 2. If-Else Statements
# ============================================================================

def test_if_else_statement(parser, transformer, emitter):
    """Test if-else statement (basic else block)."""
    glsl = """
    void test() {
        float x = 1.0;
        if (x > 0.0) {
            x = 2.0;
        } else {
            x = -1.0;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (x > 0.0f)' in opencl
    assert 'x = 2.0f' in opencl
    assert 'else' in opencl
    assert 'x = -1.0f' in opencl


def test_if_else_with_vectors(parser, transformer, emitter):
    """Test if-else with vector operations."""
    glsl = """
    void test() {
        vec3 col;
        float t = 0.5;
        if (t > 0.5) {
            col = vec3(1.0, 0.0, 0.0);
        } else {
            col = vec3(0.0, 0.0, 1.0);
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (t > 0.5f)' in opencl
    assert '(float3)(1.0f, 0.0f, 0.0f)' in opencl
    assert 'else' in opencl
    assert '(float3)(0.0f, 0.0f, 1.0f)' in opencl


# ============================================================================
# 3. If-Else-If Chains (THE BUG FIX)
# ============================================================================

def test_if_else_if_chain(parser, transformer, emitter):
    """Test if-else-if chain (validates the bug fix)."""
    glsl = """
    void test() {
        float x = 1.0;
        if (x > 0.7) {
            x = 1.0;
        } else if (x > 0.4) {
            x = 0.5;
        } else {
            x = 0.0;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)

    # Verify all three blocks are present
    assert 'if (x > 0.7f)' in opencl
    assert 'x = 1.0f' in opencl
    assert 'else if (x > 0.4f)' in opencl
    assert 'x = 0.5f' in opencl
    assert 'x = 0.0f' in opencl


def test_multiple_else_if(parser, transformer, emitter):
    """Test multiple else-if blocks in a chain."""
    glsl = """
    void test() {
        int x = 1;
        if (x == 1) {
            x = 10;
        } else if (x == 2) {
            x = 20;
        } else if (x == 3) {
            x = 30;
        } else {
            x = 0;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)

    # Verify all blocks are present
    assert 'if (x == 1)' in opencl
    assert 'x = 10' in opencl
    assert 'else if (x == 2)' in opencl
    assert 'x = 20' in opencl
    assert 'else if (x == 3)' in opencl
    assert 'x = 30' in opencl
    assert 'x = 0' in opencl


def test_else_if_with_vector_comparison(parser, transformer, emitter):
    """Test else-if with vector comparisons (like the original bug report)."""
    glsl = """
    void test(vec2 uv) {
        vec3 col;
        if (uv.x > 0.7) {
            col = vec3(1.0, 0.0, 0.0);
        } else if (uv.x > 0.4) {
            col = vec3(0.0, 1.0, 0.0);
        } else {
            col = vec3(0.0, 0.0, 1.0);
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)

    # Verify all three color assignments are present
    assert '(float3)(1.0f, 0.0f, 0.0f)' in opencl
    assert '(float3)(0.0f, 1.0f, 0.0f)' in opencl
    assert '(float3)(0.0f, 0.0f, 1.0f)' in opencl


# ============================================================================
# 4. Nested If Statements
# ============================================================================

def test_nested_if_statements(parser, transformer, emitter):
    """Test nested if statements."""
    glsl = """
    void test() {
        float x = 1.0;
        float y = 2.0;
        if (x > 0.0) {
            if (y > 0.0) {
                x = y;
            }
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (x > 0.0f)' in opencl
    assert 'if (y > 0.0f)' in opencl


def test_nested_if_else_in_else_block(parser, transformer, emitter):
    """Test nested if-else inside an else block."""
    glsl = """
    void test() {
        float x = 1.0;
        float y = 2.0;
        if (x > 0.0) {
            x = 1.0;
        } else {
            if (y > 0.0) {
                y = 1.0;
            } else {
                y = -1.0;
            }
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (x > 0.0f)' in opencl
    assert 'else' in opencl
    assert 'if (y > 0.0f)' in opencl
    assert 'y = 1.0f' in opencl
    assert 'y = -1.0f' in opencl


# ============================================================================
# 5. If Statements with Function Calls
# ============================================================================

def test_if_with_function_call_condition(parser, transformer, emitter):
    """Test if statement with function call in condition."""
    glsl = """
    void test(vec3 v) {
        if (length(v) > 1.0) {
            v = normalize(v);
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (GLSL_length(v) > 1.0f)' in opencl
    assert 'GLSL_normalize' in opencl


def test_else_if_with_function_calls(parser, transformer, emitter):
    """Test else-if with function calls in conditions."""
    glsl = """
    void test(vec2 uv) {
        float x;
        if (length(uv) > 1.0) {
            x = 1.0;
        } else if (length(uv) > 0.5) {
            x = 0.5;
        } else {
            x = 0.0;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (GLSL_length(uv) > 1.0f)' in opencl
    assert 'else if (GLSL_length(uv) > 0.5f)' in opencl


# ============================================================================
# 6. Edge Cases
# ============================================================================

def test_if_without_braces(parser, transformer, emitter):
    """Test if statement without braces (single statement)."""
    glsl = """
    void test() {
        float x = 1.0;
        if (x > 0.0) x = 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (x > 0.0f)' in opencl
    assert 'x = 2.0f' in opencl


def test_if_else_without_braces(parser, transformer, emitter):
    """Test if-else without braces."""
    glsl = """
    void test() {
        float x = 1.0;
        if (x > 0.0) x = 2.0; else x = -1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (x > 0.0f)' in opencl
    assert 'x = 2.0f' in opencl
    assert 'else' in opencl
    assert 'x = -1.0f' in opencl


def test_ternary_operator_not_affected(parser, transformer, emitter):
    """Test that ternary operator (conditional expression) still works."""
    glsl = """
    void test() {
        float x = 1.0;
        float y = x > 0.0 ? 1.0 : -1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'x > 0.0f ? 1.0f : -1.0f' in opencl
