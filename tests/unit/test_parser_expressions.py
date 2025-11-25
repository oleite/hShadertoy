"""
Expression parsing tests - Binary, unary, and complex expressions.

Tests parsing of GLSL expressions and operators.
"""

import pytest
from glsl_to_opencl.parser import GLSLParser, BinaryExpression, CallExpression


class TestBinaryExpressions:
    """Test parsing of binary expressions."""

    def test_parse_addition(self):
        """Test a + b."""
        source = """
        void main() {
            float z = x + y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_subtraction(self):
        """Test a - b."""
        source = """
        void main() {
            float z = x - y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_multiplication(self):
        """Test a * b."""
        source = """
        void main() {
            float z = x * y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_division(self):
        """Test a / b."""
        source = """
        void main() {
            float z = x / y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_modulo(self):
        """Test a % b (integers only)."""
        source = """
        void main() {
            int z = x % y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestComparisonExpressions:
    """Test parsing of comparison expressions."""

    def test_parse_less_than(self):
        """Test a < b."""
        source = """
        void main() {
            bool result = x < y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_greater_than(self):
        """Test a > b."""
        source = """
        void main() {
            bool result = x > y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_less_or_equal(self):
        """Test a <= b."""
        source = """
        void main() {
            bool result = x <= y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_greater_or_equal(self):
        """Test a >= b."""
        source = """
        void main() {
            bool result = x >= y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_equals(self):
        """Test a == b."""
        source = """
        void main() {
            bool result = x == y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_not_equals(self):
        """Test a != b."""
        source = """
        void main() {
            bool result = x != y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestLogicalExpressions:
    """Test parsing of logical expressions."""

    def test_parse_logical_and(self):
        """Test a && b."""
        source = """
        void main() {
            bool result = a && b;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_logical_or(self):
        """Test a || b."""
        source = """
        void main() {
            bool result = a || b;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_logical_not(self):
        """Test !a."""
        source = """
        void main() {
            bool result = !flag;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestComplexExpressions:
    """Test parsing of complex expressions."""

    def test_parse_parenthesized_expression(self):
        """Test (a + b) * c."""
        source = """
        void main() {
            float result = (a + b) * c;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_nested_parentheses(self):
        """Test ((a + b) * (c - d)) / e."""
        source = """
        void main() {
            float result = ((a + b) * (c - d)) / e;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_mixed_operators(self):
        """Test a + b * c - d / e."""
        source = """
        void main() {
            float result = a + b * c - d / e;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestVectorExpressions:
    """Test parsing of vector expressions."""

    def test_parse_vector_addition(self):
        """Test vec3 + vec3."""
        source = """
        void main() {
            vec3 result = v1 + v2;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_vector_scalar_multiply(self):
        """Test vec3 * float."""
        source = """
        void main() {
            vec3 result = v * 2.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_xy(self):
        """Test v.xy swizzling."""
        source = """
        void main() {
            vec2 result = v.xy;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_rgb(self):
        """Test v.rgb swizzling."""
        source = """
        void main() {
            vec3 result = v.rgb;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_complex(self):
        """Test v.xxyz complex swizzling."""
        source = """
        void main() {
            vec4 result = v.xxyz;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestMatrixExpressions:
    """Test parsing of matrix expressions."""

    def test_parse_matrix_multiply(self):
        """Test mat3 * mat3."""
        source = """
        void main() {
            mat3 result = M1 * M2;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_matrix_vector_multiply(self):
        """Test mat3 * vec3."""
        source = """
        void main() {
            vec3 result = M * v;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestArrayAccess:
    """Test parsing of array/subscript expressions."""

    def test_parse_array_access(self):
        """Test arr[i]."""
        source = """
        void main() {
            float value = arr[i];
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_matrix_element_access(self):
        """Test M[0][1]."""
        source = """
        void main() {
            float value = M[0][1];
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestTernaryExpression:
    """Test parsing of conditional ternary expressions."""

    def test_parse_ternary(self):
        """Test a ? b : c."""
        source = """
        void main() {
            float result = x > 0.0 ? 1.0 : -1.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_nested_ternary(self):
        """Test nested ternary."""
        source = """
        void main() {
            float result = a ? (b ? c : d) : e;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestAssignmentExpressions:
    """Test parsing of assignment expressions."""

    def test_parse_simple_assignment(self):
        """Test x = value."""
        source = """
        void main() {
            x = 1.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_add_assign(self):
        """Test x += value."""
        source = """
        void main() {
            x += 1.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_subtract_assign(self):
        """Test x -= value."""
        source = """
        void main() {
            x -= 1.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_multiply_assign(self):
        """Test x *= value."""
        source = """
        void main() {
            x *= 2.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_divide_assign(self):
        """Test x /= value."""
        source = """
        void main() {
            x /= 2.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestUnaryExpressions:
    """Test parsing of unary expressions."""

    def test_parse_negate(self):
        """Test -x."""
        source = """
        void main() {
            float result = -x;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_increment(self):
        """Test ++x."""
        source = """
        void main() {
            ++i;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_decrement(self):
        """Test --x."""
        source = """
        void main() {
            --i;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_postfix_increment(self):
        """Test x++."""
        source = """
        void main() {
            i++;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_postfix_decrement(self):
        """Test x--."""
        source = """
        void main() {
            i--;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None
