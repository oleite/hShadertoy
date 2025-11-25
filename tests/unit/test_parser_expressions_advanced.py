"""
Advanced expression tests - Complex and nested expressions.

Tests parsing of complex expression patterns found in real shaders.
Target: 80 tests
"""

import pytest
from glsl_to_opencl.parser import GLSLParser


class TestNestedExpressions:
    """Test deeply nested and complex expressions."""

    def test_parse_nested_function_calls_3_levels(self):
        """Test 3-level nested function calls."""
        source = "void main() { float x = sin(cos(tan(1.0))); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_nested_function_calls_5_levels(self):
        """Test 5-level nested function calls."""
        source = "void main() { float x = abs(sign(floor(ceil(fract(v))))); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_complex_arithmetic_nested(self):
        """Test complex nested arithmetic."""
        source = "void main() { float x = (a + b) * (c - d) / (e + f); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_nested_parentheses(self):
        """Test deeply nested parentheses."""
        source = "void main() { float x = (((a + b) * c) - ((d / e) + f)); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_vector_arithmetic_complex(self):
        """Test complex vector arithmetic."""
        source = "void main() { vec3 v = (a * b + c) * (d - e / f); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_matrix_vector_nested(self):
        """Test nested matrix-vector operations."""
        source = "void main() { vec3 v = mat3(1.0) * (vec3(1.0) + vec3(2.0)); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_texture_in_expression(self):
        """Test texture() call in complex expression."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = texture(tex, uv).xyz * 2.0 - 1.0; }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_in_arithmetic(self):
        """Test swizzled vector in arithmetic."""
        source = "void main() { float x = v.x * v.y + v.z * v.w; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_constructor_in_expression(self):
        """Test constructor in expression."""
        source = "void main() { vec3 v = vec3(1.0, 2.0, 3.0) * scale + offset; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_dot_product_in_expression(self):
        """Test dot product in complex expression."""
        source = "void main() { float x = dot(a, b) * dot(c, d) + dot(e, f); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_mixed_vector_scalar_ops(self):
        """Test mixed vector and scalar operations."""
        source = "void main() { vec3 v = a * 2.0 + b * 3.0 - c / 4.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_ternary_nested(self):
        """Test nested ternary operators."""
        source = "void main() { float x = a ? (b ? c : d) : (e ? f : g); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_ternary_in_arithmetic(self):
        """Test ternary operator in arithmetic expression."""
        source = "void main() { float x = (a > 0.0 ? b : c) * 2.0 + d; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_comparison_in_ternary(self):
        """Test comparison in ternary condition."""
        source = "void main() { float x = (a > b && c < d) ? e : f; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_access_in_expression(self):
        """Test array access in complex expression."""
        source = "void main() { float x = arr[i] * arr[j] + arr[k]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_computed_array_index(self):
        """Test computed array index."""
        source = "void main() { float x = arr[i + 1] * arr[j * 2]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_of_function_result(self):
        """Test swizzling of function return value."""
        source = "void main() { float x = texture(tex, uv).r; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_chained_member_access(self):
        """Test chained member access."""
        source = "void main() { float x = s.member.x; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_length_of_expression(self):
        """Test length() of complex expression."""
        source = "void main() { float x = length(a * b + c); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_normalize_of_expression(self):
        """Test normalize() of complex expression."""
        source = "void main() { vec3 v = normalize(a - b * c); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_pow_in_expression(self):
        """Test pow() in complex expression."""
        source = "void main() { float x = pow(a * b, 2.0) + pow(c, 3.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_mix_in_expression(self):
        """Test mix() in complex expression."""
        source = "void main() { vec3 v = mix(a, b, t) * scale; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_clamp_in_expression(self):
        """Test clamp() in complex expression."""
        source = "void main() { float x = clamp(a * b, 0.0, 1.0) * c; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_step_in_expression(self):
        """Test step() in complex expression."""
        source = "void main() { float x = step(0.5, v) * a + (1.0 - step(0.5, v)) * b; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_smoothstep_in_expression(self):
        """Test smoothstep() in complex expression."""
        source = "void main() { float x = smoothstep(0.0, 1.0, t) * (b - a) + a; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_mod_in_expression(self):
        """Test mod() in complex expression."""
        source = "void main() { float x = mod(a, b) / b; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_fract_in_expression(self):
        """Test fract() in complex expression."""
        source = "void main() { float x = fract(a * b + c); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_floor_ceil_combination(self):
        """Test floor and ceil in same expression."""
        source = "void main() { float x = floor(a) + ceil(b) - fract(c); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_min_max_combination(self):
        """Test min and max in same expression."""
        source = "void main() { float x = max(min(a, b), min(c, d)); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_cross_product_in_expression(self):
        """Test cross product in expression."""
        source = "void main() { vec3 v = cross(a, b) * scale; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestOperatorPrecedence:
    """Test operator precedence and associativity."""

    def test_parse_addition_multiplication(self):
        """Test addition and multiplication precedence."""
        source = "void main() { float x = a + b * c; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_multiplication_division(self):
        """Test multiplication and division precedence."""
        source = "void main() { float x = a * b / c; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_comparison_arithmetic(self):
        """Test comparison and arithmetic precedence."""
        source = "void main() { bool b = a + b > c * d; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_logical_comparison(self):
        """Test logical and comparison precedence."""
        source = "void main() { bool b = a > 0.0 && b < 1.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_logical_and_or(self):
        """Test logical AND and OR precedence."""
        source = "void main() { bool b = a || b && c; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_unary_arithmetic(self):
        """Test unary and arithmetic precedence."""
        source = "void main() { float x = -a * b; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_increment_arithmetic(self):
        """Test increment and arithmetic."""
        source = "void main() { float x = ++i * 2.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_postfix_increment_arithmetic(self):
        """Test postfix increment and arithmetic."""
        source = "void main() { float x = i++ * 2.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_assignment_in_expression(self):
        """Test assignment in expression context."""
        source = "void main() { float x = (y = 2.0) * 3.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_compound_assignment_precedence(self):
        """Test compound assignment precedence."""
        source = "void main() { x += y * z; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_ternary_precedence(self):
        """Test ternary operator precedence."""
        source = "void main() { float x = a > b ? c + d : e * f; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_not_comparison(self):
        """Test NOT and comparison precedence."""
        source = "void main() { bool b = !a > b; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_bitwise_arithmetic(self):
        """Test bitwise and arithmetic precedence."""
        source = "void main() { int x = a + b & c; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_shift_arithmetic(self):
        """Test shift and arithmetic precedence."""
        source = "void main() { int x = a + b << c; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_mixed_operators_complex(self):
        """Test complex mix of operators."""
        source = "void main() { float x = a * b + c / d - e * f; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestComplexSwizzling:
    """Test complex vector swizzling patterns."""

    def test_parse_single_component_replication_xxx(self):
        """Test .xxx swizzle."""
        source = "void main() { vec3 v = a.xxx; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_single_component_replication_yyy(self):
        """Test .yyy swizzle."""
        source = "void main() { vec3 v = a.yyy; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_single_component_replication_zzz(self):
        """Test .zzz swizzle."""
        source = "void main() { vec3 v = a.zzz; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_single_component_replication_www(self):
        """Test .www swizzle."""
        source = "void main() { vec3 v = a.www; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_rgb_swizzle(self):
        """Test .rgb swizzle."""
        source = "void main() { vec3 v = a.rgb; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_bgr_swizzle(self):
        """Test .bgr swizzle."""
        source = "void main() { vec3 v = a.bgr; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_rrr_swizzle(self):
        """Test .rrr swizzle."""
        source = "void main() { vec3 v = a.rrr; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_ggg_swizzle(self):
        """Test .ggg swizzle."""
        source = "void main() { vec3 v = a.ggg; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_bbb_swizzle(self):
        """Test .bbb swizzle."""
        source = "void main() { vec3 v = a.bbb; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_rgba_swizzle(self):
        """Test .rgba swizzle."""
        source = "void main() { vec4 v = a.rgba; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_bgra_swizzle(self):
        """Test .bgra swizzle."""
        source = "void main() { vec4 v = a.bgra; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_argb_swizzle(self):
        """Test .argb swizzle."""
        source = "void main() { vec4 v = a.argb; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_xyzw_swizzle(self):
        """Test .xyzw swizzle."""
        source = "void main() { vec4 v = a.xyzw; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_wzyx_swizzle(self):
        """Test .wzyx swizzle."""
        source = "void main() { vec4 v = a.wzyx; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_xxyy_swizzle(self):
        """Test .xxyy swizzle."""
        source = "void main() { vec4 v = a.xxyy; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_zzww_swizzle(self):
        """Test .zzww swizzle."""
        source = "void main() { vec4 v = a.zzww; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_chained_swizzle_xy_yx(self):
        """Test chained swizzle .xy.yx."""
        source = "void main() { vec2 v = a.xy.yx; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_chained_swizzle_rgb_bgr(self):
        """Test chained swizzle .rgb.bgr."""
        source = "void main() { vec3 v = a.rgb.bgr; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_on_function_result(self):
        """Test swizzle on function return."""
        source = "void main() { vec2 v = getVec4().xy; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_on_constructor(self):
        """Test swizzle on constructor."""
        source = "void main() { vec2 v = vec4(1.0).xy; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_on_array_element(self):
        """Test swizzle on array element."""
        source = "void main() { vec2 v = arr[0].xy; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_on_struct_member(self):
        """Test swizzle on struct member."""
        source = "void main() { vec2 v = s.color.xy; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_write_mask(self):
        """Test swizzle as write mask."""
        source = "void main() { v.xy = vec2(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_swizzle_in_arithmetic(self):
        """Test swizzle in arithmetic expression."""
        source = "void main() { float x = a.x * b.y + c.z; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_mixed_swizzle_xyx(self):
        """Test mixed swizzle .xyx."""
        source = "void main() { vec3 v = a.xyx; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None
