"""
Unit tests for matrix operations on swizzled vector components.

Tests transformation of GLSL matrix operations on isolated vector components:
- vec3.xy * mat2 -> GLSL_mul_vec2_mat2(vec3.xy, mat2)
- vec4.xy *= mat2 -> vec4.xy = GLSL_mul_vec2_mat2(vec4.xy, mat2)
- vec4.xyz * mat3 -> GLSL_mul_vec3_mat3(vec4.xyz, mat3)

This ensures swizzle type inference properly propagates to matrix operation detection.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter


def transform_glsl(glsl_code: str) -> str:
    """Helper to transform GLSL code to OpenCL."""
    parser = GLSLParser()
    ast = parser.parse(glsl_code)
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    transformer = ASTTransformer(type_checker)
    transformed = transformer.transform(ast)
    emitter = OpenCLEmitter()
    return emitter.emit(transformed)


class TestSwizzleMatrixMultiplication:
    """Test matrix multiplication on swizzled vector components."""

    def test_vec3_xy_times_mat2(self):
        """Test vec3.xy * mat2 transformation."""
        glsl = """
        void main() {
            vec3 foo = vec3(1.0, 0.0, 0.0);
            mat2 bar = mat2(1.0, 0.0, 0.0, 1.0);
            vec2 result = foo.xy * bar;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'GLSL_mul_vec2_mat2(foo.xy, bar)' in opencl

    def test_vec3_yz_times_mat2(self):
        """Test vec3.yz * mat2 transformation."""
        glsl = """
        void main() {
            vec3 foo = vec3(1.0, 0.0, 0.0);
            mat2 bar = mat2(1.0, 0.0, 0.0, 1.0);
            vec2 result = foo.yz * bar;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'GLSL_mul_vec2_mat2(foo.yz, bar)' in opencl

    def test_vec4_xy_times_mat2(self):
        """Test vec4.xy * mat2 transformation."""
        glsl = """
        void main() {
            vec4 foo = vec4(1.0, 0.0, 0.0, 0.0);
            mat2 bar = mat2(1.0, 0.0, 0.0, 1.0);
            vec2 result = foo.xy * bar;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'GLSL_mul_vec2_mat2(foo.xy, bar)' in opencl

    def test_vec4_xyz_times_mat3(self):
        """Test vec4.xyz * mat3 transformation."""
        glsl = """
        void main() {
            vec4 foo = vec4(1.0, 0.0, 0.0, 0.0);
            mat3 bar = mat3(1.0);
            vec3 result = foo.xyz * bar;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'GLSL_mul_vec3_mat3(foo.xyz, bar)' in opencl


class TestSwizzleMatrixCompoundAssignment:
    """Test compound assignment with swizzled vector components."""

    def test_vec3_xy_compound_assign_mat2(self):
        """Test foo.xy *= bar transformation."""
        glsl = """
        void main() {
            vec3 foo = vec3(1.0, 0.0, 0.0);
            mat2 bar = mat2(1.0, 0.0, 0.0, 1.0);
            foo.xy *= bar;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'foo.xy = GLSL_mul_vec2_mat2(foo.xy, bar)' in opencl

    def test_vec4_xy_compound_assign_mat2(self):
        """Test vec4.xy *= mat2 transformation."""
        glsl = """
        void main() {
            vec4 foo = vec4(1.0, 0.0, 0.0, 0.0);
            mat2 bar = mat2(1.0, 0.0, 0.0, 1.0);
            foo.xy *= bar;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'foo.xy = GLSL_mul_vec2_mat2(foo.xy, bar)' in opencl

    def test_vec4_xyz_compound_assign_mat3(self):
        """Test vec4.xyz *= mat3 transformation."""
        glsl = """
        void main() {
            vec4 foo = vec4(1.0, 0.0, 0.0, 0.0);
            mat3 bar = mat3(1.0);
            foo.xyz *= bar;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'foo.xyz = GLSL_mul_vec3_mat3(foo.xyz, bar)' in opencl


class TestSwizzleAssignment:
    """Test assignment to swizzled vector components."""

    def test_vec2_assign_vec3_xy_times_mat2(self):
        """Test V2 = V3.xy * M2 transformation."""
        glsl = """
        void main() {
            vec2 V2;
            vec3 V3 = vec3(1.0, 0.0, 0.0);
            mat2 M2 = mat2(1.0, 0.0, 0.0, 1.0);
            V2 = V3.xy * M2;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'V2 = GLSL_mul_vec2_mat2(V3.xy, M2)' in opencl


class TestIntegerVectorSwizzles:
    """Test matrix operations on integer vector swizzles."""

    def test_ivec3_xy_times_mat2(self):
        """Test ivec3.xy * mat2 (should work - swizzle is vec2)."""
        glsl = """
        void main() {
            ivec3 foo = ivec3(1, 0, 0);
            mat2 bar = mat2(1.0, 0.0, 0.0, 1.0);
            vec2 result = vec2(foo.xy) * bar;
        }
        """
        opencl = transform_glsl(glsl)
        # Type cast ivec2 -> vec2 happens first, then matrix mult
        assert 'GLSL_mul_vec2_mat2' in opencl


class TestColorSwizzles:
    """Test matrix operations with color swizzle patterns (rgba)."""

    def test_vec3_rg_times_mat2(self):
        """Test vec3.rg * mat2 transformation."""
        glsl = """
        void main() {
            vec3 color = vec3(1.0, 0.0, 0.0);
            mat2 bar = mat2(1.0, 0.0, 0.0, 1.0);
            vec2 result = color.rg * bar;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'GLSL_mul_vec2_mat2(color.rg, bar)' in opencl

    def test_vec4_rgb_times_mat3(self):
        """Test vec4.rgb * mat3 transformation."""
        glsl = """
        void main() {
            vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
            mat3 bar = mat3(1.0);
            vec3 result = color.rgb * bar;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'GLSL_mul_vec3_mat3(color.rgb, bar)' in opencl


class TestMatrixGLSLExample:
    """Test the specific examples from the problem statement."""

    def test_problem_statement_example(self):
        """Test the exact example from the problem statement."""
        glsl = """
        void main() {
            vec3 foo = vec3(1.0, 0.0, 0.0);
            mat2 bar = mat2(1.0, 0.0, 0.0, 1.0);
            foo.xy = foo.yz * bar;
            foo.xy *= bar;
        }
        """
        opencl = transform_glsl(glsl)
        # Check first assignment
        assert 'foo.xy = GLSL_mul_vec2_mat2(foo.yz, bar)' in opencl
        # Check compound assignment
        # Note: The second foo.xy *= bar should also become foo.xy = GLSL_mul_vec2_mat2(foo.xy, bar)
        # Count occurrences
        assert opencl.count('GLSL_mul_vec2_mat2') >= 2


class TestSwizzleTypeInference:
    """Test that swizzle type inference works correctly."""

    def test_vec3_xy_infers_vec2(self):
        """Verify vec3.xy is inferred as vec2."""
        glsl = """
        void main() {
            vec3 v = vec3(1.0, 2.0, 3.0);
            float x = v.xy.x;
        }
        """
        # Should not crash during transformation
        opencl = transform_glsl(glsl)
        assert 'v.xy.x' in opencl

    def test_vec4_xyz_infers_vec3(self):
        """Verify vec4.xyz is inferred as vec3."""
        glsl = """
        void main() {
            vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
            float x = v.xyz.x;
        }
        """
        # Should not crash during transformation
        opencl = transform_glsl(glsl)
        assert 'v.xyz.x' in opencl


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_chained_swizzles(self):
        """Test chained swizzle operations (if supported by GLSL)."""
        # Note: GLSL doesn't actually support v.xy.x, but we test robustness
        glsl = """
        void main() {
            vec3 v = vec3(1.0);
            vec2 result = v.xy;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'v.xy' in opencl

    def test_swizzle_in_complex_expression(self):
        """Test swizzle in complex expression with multiple operations."""
        glsl = """
        void main() {
            vec3 a = vec3(1.0, 0.0, 0.0);
            vec3 b = vec3(0.0, 1.0, 0.0);
            mat2 M = mat2(1.0, 0.0, 0.0, 1.0);
            vec2 result = (a.xy + b.xy) * M;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'GLSL_mul_vec2_mat2' in opencl

    def test_multiple_swizzles_different_vectors(self):
        """Test multiple swizzle operations on different vectors."""
        glsl = """
        void main() {
            vec3 a = vec3(1.0, 0.0, 0.0);
            vec4 b = vec4(0.0, 1.0, 0.0, 0.0);
            mat2 M2 = mat2(1.0, 0.0, 0.0, 1.0);
            mat3 M3 = mat3(1.0);
            vec2 r1 = a.xy * M2;
            vec3 r2 = b.xyz * M3;
        }
        """
        opencl = transform_glsl(glsl)
        assert 'GLSL_mul_vec2_mat2(a.xy, M2)' in opencl
        assert 'GLSL_mul_vec3_mat3(b.xyz, M3)' in opencl
