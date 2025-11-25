"""
Unit tests for matrix function transformations (Session 5).

Tests matrix functions: transpose(), inverse(), determinant()

Key transformations:
- transpose(mat2/mat4) -> GLSL_transpose(mat2/mat4) (return by value)
- transpose(mat3) -> mat3 result; GLSL_transpose(mat3, &result); (out-param)
- inverse(mat2/mat4) -> GLSL_inverse(mat2/mat4) (return by value)
- inverse(mat3) -> mat3 result; GLSL_inverse(mat3, &result); (out-param)
- determinant(matN) -> GLSL_determinant(matN) (returns float)

Test coverage:
- Each function for each matrix size (3 x 3 = 9 tests)
- mat3 out-parameter pattern (3 tests)
- Combined operations (transpose + multiply, etc.) (6 tests)
- Chained operations (transpose(inverse(M))) (3 tests)
- Expression contexts (return, ternary, function args) (6 tests)
- Edge cases (3 tests)

Total: 30 tests
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
def transformer():
    """Create transformer instance for testing."""
    parser = GLSLParser()
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    return ASTTransformer(type_checker), parser, CodeEmitter()


def transform_and_emit(glsl_code: str, transformer_fixture):
    """Helper to transform GLSL and emit OpenCL."""
    transformer, parser, emitter = transformer_fixture
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    opencl = emitter.emit(transformed)
    return opencl


# ============================================================================
# Basic Matrix Function Transformations
# ============================================================================

class TestTransposeBasic:
    """Test transpose() transformation for each matrix size."""

    def test_transpose_mat2(self, transformer):
        """Test transpose(mat2) returns by value."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            mat2 Mt = transpose(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # mat2 returns by value
        assert "GLSL_transpose(M)" in opencl
        assert "matrix2x2 Mt = GLSL_transpose(M)" in opencl

    def test_transpose_mat3(self, transformer):
        """Test transpose(mat3) returns by value (struct type)."""
        glsl = """
        void test() {
            mat3 M = mat3(1.0);
            mat3 Mt = transpose(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # mat3 now uses struct type, returns by value with _mat3 suffix
        assert "matrix3x3 Mt = GLSL_transpose_mat3(M)" in opencl

    def test_transpose_mat4(self, transformer):
        """Test transpose(mat4) returns by value."""
        glsl = """
        void test() {
            mat4 M = mat4(1.0);
            mat4 Mt = transpose(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # mat4 returns by value with _mat4 suffix
        assert "GLSL_transpose_mat4(M)" in opencl
        assert "matrix4x4 Mt = GLSL_transpose_mat4(M)" in opencl


class TestInverseBasic:
    """Test inverse() transformation for each matrix size."""

    def test_inverse_mat2(self, transformer):
        """Test inverse(mat2) returns by value."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            mat2 Minv = inverse(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # mat2 returns by value
        assert "GLSL_inverse(M)" in opencl
        assert "matrix2x2 Minv = GLSL_inverse(M)" in opencl

    def test_inverse_mat3(self, transformer):
        """Test inverse(mat3) returns by value (struct type)."""
        glsl = """
        void test() {
            mat3 M = mat3(1.0);
            mat3 Minv = inverse(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # mat3 now uses struct type, returns by value with _mat3 suffix
        assert "matrix3x3 Minv = GLSL_inverse_mat3(M)" in opencl

    def test_inverse_mat4(self, transformer):
        """Test inverse(mat4) returns by value."""
        glsl = """
        void test() {
            mat4 M = mat4(1.0);
            mat4 Minv = inverse(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # mat4 returns by value with _mat4 suffix
        assert "GLSL_inverse_mat4(M)" in opencl
        assert "matrix4x4 Minv = GLSL_inverse_mat4(M)" in opencl


class TestDeterminantBasic:
    """Test determinant() transformation for each matrix size."""

    def test_determinant_mat2(self, transformer):
        """Test determinant(mat2) returns scalar."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            float det = determinant(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # determinant returns float
        assert "GLSL_determinant(M)" in opencl
        assert "float det = GLSL_determinant(M)" in opencl

    def test_determinant_mat3(self, transformer):
        """Test determinant(mat3) returns scalar."""
        glsl = """
        void test() {
            mat3 M = mat3(1.0);
            float det = determinant(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # determinant returns float with _mat3 suffix
        assert "GLSL_determinant_mat3(M)" in opencl
        assert "float det = GLSL_determinant_mat3(M)" in opencl

    def test_determinant_mat4(self, transformer):
        """Test determinant(mat4) returns scalar."""
        glsl = """
        void test() {
            mat4 M = mat4(1.0);
            float det = determinant(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # determinant returns float with _mat4 suffix
        assert "GLSL_determinant_mat4(M)" in opencl
        assert "float det = GLSL_determinant_mat4(M)" in opencl


# ============================================================================
# mat3 Out-Parameter Pattern Tests
# ============================================================================

class TestMat3OutParameter:
    """Test mat3 declaration splitting for transpose/inverse."""

    def test_mat3_transpose_declaration_splitting(self, transformer):
        """Verify mat3 transpose uses direct return (no splitting with struct)."""
        glsl = """
        void test() {
            mat3 M = mat3(1.0);
            mat3 result = transpose(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # mat3 now uses struct type, returns by value with _mat3 suffix
        assert "matrix3x3 result = GLSL_transpose_mat3(M)" in opencl

    def test_mat3_inverse_declaration_splitting(self, transformer):
        """Verify mat3 inverse uses direct return (no splitting with struct)."""
        glsl = """
        void test() {
            mat3 M = mat3(1.0);
            mat3 result = inverse(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # mat3 now uses struct type, returns by value with _mat3 suffix
        assert "matrix3x3 result = GLSL_inverse_mat3(M)" in opencl

    def test_mat3_multiple_operations(self, transformer):
        """Test multiple mat3 operations in same scope."""
        glsl = """
        void test() {
            mat3 M = mat3(1.0);
            mat3 Mt = transpose(M);
            mat3 Minv = inverse(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # Both use direct returns with _mat3 suffix
        assert "matrix3x3 Mt = GLSL_transpose_mat3(M)" in opencl
        assert "matrix3x3 Minv = GLSL_inverse_mat3(M)" in opencl


# ============================================================================
# Combined Operations Tests
# ============================================================================

class TestCombinedOperations:
    """Test matrix functions combined with other operations."""

    def test_transpose_then_multiply_mat2(self, transformer):
        """Test transpose(M) * v for mat2."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            vec2 v = vec2(1.0, 0.0);
            vec2 result = transpose(M) * v;
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # transpose returns mat2, then multiply with typed function
        assert "GLSL_transpose(M)" in opencl
        assert "GLSL_mul_mat2_vec2" in opencl

    def test_inverse_then_multiply_mat2(self, transformer):
        """Test inverse(M) * v for mat2."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            vec2 v = vec2(1.0, 0.0);
            vec2 result = inverse(M) * v;
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # inverse returns mat2, then multiply with typed function
        assert "GLSL_inverse(M)" in opencl
        assert "GLSL_mul_mat2_vec2" in opencl

    def test_multiply_then_transpose_mat4(self, transformer):
        """Test transpose(M * M2) for mat4."""
        glsl = """
        void test() {
            mat4 M1 = mat4(1.0);
            mat4 M2 = mat4(2.0);
            mat4 result = transpose(M1 * M2);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # Nested: multiply first with typed function, then transpose with _mat4 suffix
        assert "GLSL_mul_mat4_mat4" in opencl
        assert "GLSL_transpose_mat4" in opencl

    def test_transpose_mat3_with_vector_multiply(self, transformer):
        """Test v * transpose(M3) for mat3."""
        glsl = """
        void test() {
            mat3 M = mat3(1.0);
            vec3 v = vec3(1.0, 0.0, 0.0);
            vec3 result = v * transpose(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # mat3 transpose in expression context (not declaration)
        # This is an edge case - transpose should be called inline
        assert "GLSL_transpose(M)" in opencl or "transpose" in opencl.lower()

    def test_determinant_in_condition(self, transformer):
        """Test determinant(M) in if condition."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            if (determinant(M) > 0.0) {
                float x = 1.0;
            }
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # determinant in condition
        assert "GLSL_determinant(M)" in opencl
        assert "if" in opencl

    def test_matrix_function_in_expression(self, transformer):
        """Test matrix function result used in arithmetic."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            float det = determinant(M) * 2.0;
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # determinant result in arithmetic
        assert "GLSL_determinant(M)" in opencl
        assert "* 2.0f" in opencl


# ============================================================================
# Chained Matrix Function Tests
# ============================================================================

class TestChainedFunctions:
    """Test chained matrix function calls."""

    def test_transpose_of_inverse_mat2(self, transformer):
        """Test transpose(inverse(M)) for mat2."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            mat2 result = transpose(inverse(M));
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # Nested function calls
        assert "GLSL_inverse(M)" in opencl
        assert "GLSL_transpose(GLSL_inverse(M))" in opencl

    def test_inverse_of_transpose_mat4(self, transformer):
        """Test inverse(transpose(M)) for mat4."""
        glsl = """
        void test() {
            mat4 M = mat4(1.0);
            mat4 result = inverse(transpose(M));
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # Nested function calls with _mat4 suffix
        assert "GLSL_transpose_mat4(M)" in opencl
        assert "GLSL_inverse_mat4(GLSL_transpose_mat4(M))" in opencl

    def test_determinant_of_transpose_mat3(self, transformer):
        """Test determinant(transpose(M)) for mat3."""
        glsl = """
        void test() {
            mat3 M = mat3(1.0);
            float det = determinant(transpose(M));
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # mat3 transpose in nested context
        # This is tricky - transpose can't split here
        assert "GLSL_transpose" in opencl
        assert "GLSL_determinant" in opencl


# ============================================================================
# Expression Context Tests
# ============================================================================

class TestExpressionContexts:
    """Test matrix functions in various expression contexts."""

    def test_transpose_in_return_statement(self, transformer):
        """Test transpose in return statement."""
        glsl = """
        mat2 test(mat2 M) {
            return transpose(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # transpose in return
        assert "return GLSL_transpose(M)" in opencl

    def test_inverse_in_function_argument(self, transformer):
        """Test inverse as function argument."""
        glsl = """
        float compute(mat2 M) {
            return 1.0;
        }
        void test() {
            mat2 M = mat2(1.0);
            float result = compute(inverse(M));
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # inverse as argument
        assert "compute(GLSL_inverse(M))" in opencl

    def test_determinant_in_ternary(self, transformer):
        """Test determinant in ternary operator."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            float result = determinant(M) > 0.0 ? 1.0 : -1.0;
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # determinant in ternary
        assert "GLSL_determinant(M)" in opencl
        assert "?" in opencl

    def test_transpose_mat4_in_assignment(self, transformer):
        """Test transpose in assignment (not declaration)."""
        glsl = """
        void test() {
            mat4 M = mat4(1.0);
            mat4 result;
            result = transpose(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # Assignment form with _mat4 suffix
        assert "result = GLSL_transpose_mat4(M)" in opencl

    def test_multiple_functions_same_line(self, transformer):
        """Test multiple matrix functions in same statement."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            float value = determinant(M) + determinant(transpose(M));
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # Multiple function calls
        assert opencl.count("GLSL_determinant") == 2
        assert "GLSL_transpose(M)" in opencl

    def test_matrix_function_array_access(self, transformer):
        """Test matrix function on array element."""
        glsl = """
        void test() {
            mat2 matrices[2];
            mat2 result = transpose(matrices[0]);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # Array access in function call
        assert "GLSL_transpose(matrices[0])" in opencl


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_all_three_functions_same_scope(self, transformer):
        """Test transpose, inverse, determinant in same scope."""
        glsl = """
        void test() {
            mat2 M = mat2(1.0);
            mat2 Mt = transpose(M);
            mat2 Minv = inverse(M);
            float det = determinant(M);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # All three functions
        assert "GLSL_transpose(M)" in opencl
        assert "GLSL_inverse(M)" in opencl
        assert "GLSL_determinant(M)" in opencl

    def test_matrix_functions_different_sizes(self, transformer):
        """Test matrix functions on different matrix sizes."""
        glsl = """
        void test() {
            mat2 M2 = mat2(1.0);
            mat3 M3 = mat3(1.0);
            mat4 M4 = mat4(1.0);
            mat2 M2t = transpose(M2);
            mat3 M3t = transpose(M3);
            mat4 M4t = transpose(M4);
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # All matrices return by value with struct types, mat3/mat4 use suffixes
        assert "matrix2x2 M2t = GLSL_transpose(M2)" in opencl
        assert "matrix3x3 M3t = GLSL_transpose_mat3(M3)" in opencl
        assert "matrix4x4 M4t = GLSL_transpose_mat4(M4)" in opencl

    def test_complex_shader_pattern(self, transformer):
        """Test realistic shader pattern with multiple matrix operations."""
        glsl = """
        void mainImage() {
            mat3 normalMatrix = mat3(1.0);
            mat3 invNormal = inverse(transpose(normalMatrix));
            vec3 normal = vec3(0.0, 1.0, 0.0);
            vec3 transformed = invNormal * normal;
        }
        """
        opencl = transform_and_emit(glsl, transformer)

        # Complex transformation chain with mat3 suffixes and typed mul
        assert "GLSL_transpose_mat3" in opencl
        assert "GLSL_inverse_mat3" in opencl
        assert "GLSL_mul_mat3_vec3(invNormal, normal)" in opencl
