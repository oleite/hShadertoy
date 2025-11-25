"""
Built-in function tests - GLSL ES 3.0 built-in functions.

Tests parsing of all GLSL ES 3.0 built-in functions.
Target: 100 tests
"""

import pytest
from glsl_to_opencl.parser import GLSLParser


class TestMathFunctions:
    """Test trigonometric and mathematical functions."""

    def test_parse_sin(self):
        """Test sin() function."""
        source = "void main() { float x = sin(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_cos(self):
        """Test cos() function."""
        source = "void main() { float x = cos(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_tan(self):
        """Test tan() function."""
        source = "void main() { float x = tan(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_asin(self):
        """Test asin() function."""
        source = "void main() { float x = asin(0.5); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_acos(self):
        """Test acos() function."""
        source = "void main() { float x = acos(0.5); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_atan_one_arg(self):
        """Test atan() with one argument."""
        source = "void main() { float x = atan(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_atan_two_args(self):
        """Test atan() with two arguments (atan2)."""
        source = "void main() { float x = atan(y, x); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_pow(self):
        """Test pow() function."""
        source = "void main() { float x = pow(2.0, 3.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_exp(self):
        """Test exp() function."""
        source = "void main() { float x = exp(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_exp2(self):
        """Test exp2() function."""
        source = "void main() { float x = exp2(3.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_log(self):
        """Test log() function."""
        source = "void main() { float x = log(10.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_log2(self):
        """Test log2() function."""
        source = "void main() { float x = log2(8.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_sqrt(self):
        """Test sqrt() function."""
        source = "void main() { float x = sqrt(9.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_inversesqrt(self):
        """Test inversesqrt() function."""
        source = "void main() { float x = inversesqrt(4.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_abs(self):
        """Test abs() function."""
        source = "void main() { float x = abs(-5.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_sign(self):
        """Test sign() function."""
        source = "void main() { float x = sign(-3.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_floor(self):
        """Test floor() function."""
        source = "void main() { float x = floor(3.7); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_ceil(self):
        """Test ceil() function."""
        source = "void main() { float x = ceil(3.2); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_fract(self):
        """Test fract() function."""
        source = "void main() { float x = fract(3.7); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_mod(self):
        """Test mod() function."""
        source = "void main() { float x = mod(5.0, 2.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_min(self):
        """Test min() function."""
        source = "void main() { float x = min(3.0, 5.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_max(self):
        """Test max() function."""
        source = "void main() { float x = max(3.0, 5.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_clamp(self):
        """Test clamp() function."""
        source = "void main() { float x = clamp(v, 0.0, 1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_mix(self):
        """Test mix() function."""
        source = "void main() { float x = mix(a, b, 0.5); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_step(self):
        """Test step() function."""
        source = "void main() { float x = step(0.5, v); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_smoothstep(self):
        """Test smoothstep() function."""
        source = "void main() { float x = smoothstep(0.0, 1.0, v); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_radians(self):
        """Test radians() function."""
        source = "void main() { float x = radians(180.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_degrees(self):
        """Test degrees() function."""
        source = "void main() { float x = degrees(3.14159); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_sinh(self):
        """Test sinh() function."""
        source = "void main() { float x = sinh(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_cosh(self):
        """Test cosh() function."""
        source = "void main() { float x = cosh(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_tanh(self):
        """Test tanh() function."""
        source = "void main() { float x = tanh(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_asinh(self):
        """Test asinh() function."""
        source = "void main() { float x = asinh(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_acosh(self):
        """Test acosh() function."""
        source = "void main() { float x = acosh(2.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_atanh(self):
        """Test atanh() function."""
        source = "void main() { float x = atanh(0.5); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_trunc(self):
        """Test trunc() function."""
        source = "void main() { float x = trunc(3.7); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_round(self):
        """Test round() function."""
        source = "void main() { float x = round(3.5); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_roundEven(self):
        """Test roundEven() function."""
        source = "void main() { float x = roundEven(3.5); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_modf(self):
        """Test modf() function."""
        source = "void main() { float i; float x = modf(3.7, i); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_isnan(self):
        """Test isnan() function."""
        source = "void main() { bool x = isnan(v); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_isinf(self):
        """Test isinf() function."""
        source = "void main() { bool x = isinf(v); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestGeometricFunctions:
    """Test geometric functions."""

    def test_parse_length(self):
        """Test length() function."""
        source = "void main() { float x = length(vec3(1.0, 2.0, 3.0)); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_distance(self):
        """Test distance() function."""
        source = "void main() { float x = distance(a, b); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_dot(self):
        """Test dot() function."""
        source = "void main() { float x = dot(vec3(1.0), vec3(2.0)); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_cross(self):
        """Test cross() function."""
        source = "void main() { vec3 x = cross(a, b); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_normalize(self):
        """Test normalize() function."""
        source = "void main() { vec3 x = normalize(vec3(1.0, 2.0, 3.0)); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_faceforward(self):
        """Test faceforward() function."""
        source = "void main() { vec3 x = faceforward(N, I, Nref); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_reflect(self):
        """Test reflect() function."""
        source = "void main() { vec3 x = reflect(I, N); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_refract(self):
        """Test refract() function."""
        source = "void main() { vec3 x = refract(I, N, 1.5); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestMatrixFunctions:
    """Test matrix functions."""

    def test_parse_matrixCompMult(self):
        """Test matrixCompMult() function."""
        source = "void main() { mat4 x = matrixCompMult(a, b); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_outerProduct(self):
        """Test outerProduct() function."""
        source = "void main() { mat4 x = outerProduct(vec4(1.0), vec4(2.0)); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_transpose(self):
        """Test transpose() function."""
        source = "void main() { mat4 x = transpose(m); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_determinant(self):
        """Test determinant() function."""
        source = "void main() { float x = determinant(m); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_inverse(self):
        """Test inverse() function."""
        source = "void main() { mat4 x = inverse(m); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestVectorRelationalFunctions:
    """Test vector relational functions."""

    def test_parse_lessThan(self):
        """Test lessThan() function."""
        source = "void main() { bvec3 x = lessThan(a, b); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_lessThanEqual(self):
        """Test lessThanEqual() function."""
        source = "void main() { bvec3 x = lessThanEqual(a, b); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_greaterThan(self):
        """Test greaterThan() function."""
        source = "void main() { bvec3 x = greaterThan(a, b); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_greaterThanEqual(self):
        """Test greaterThanEqual() function."""
        source = "void main() { bvec3 x = greaterThanEqual(a, b); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_equal(self):
        """Test equal() function."""
        source = "void main() { bvec3 x = equal(a, b); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_notEqual(self):
        """Test notEqual() function."""
        source = "void main() { bvec3 x = notEqual(a, b); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_any(self):
        """Test any() function."""
        source = "void main() { bool x = any(bvec3(true, false, true)); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_all(self):
        """Test all() function."""
        source = "void main() { bool x = all(bvec3(true, true, true)); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_not(self):
        """Test not() function."""
        source = "void main() { bvec3 x = not(bvec3(true, false, true)); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestTextureFunctions:
    """Test texture sampling functions."""

    def test_parse_texture_2d(self):
        """Test texture() with sampler2D."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = texture(tex, vec2(0.5)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_texture_cube(self):
        """Test texture() with samplerCube."""
        source = """
        uniform samplerCube tex;
        void main() { vec4 c = texture(tex, vec3(1.0)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_texture_3d(self):
        """Test texture() with sampler3D."""
        source = """
        uniform sampler3D tex;
        void main() { vec4 c = texture(tex, vec3(0.5)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureLod(self):
        """Test textureLod() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureLod(tex, vec2(0.5), 0.0); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureGrad(self):
        """Test textureGrad() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureGrad(tex, uv, dPdx, dPdy); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureProj(self):
        """Test textureProj() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureProj(tex, vec3(0.5)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureOffset(self):
        """Test textureOffset() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureOffset(tex, uv, ivec2(1, 1)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_texelFetch(self):
        """Test texelFetch() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = texelFetch(tex, ivec2(10, 10), 0); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_texelFetchOffset(self):
        """Test texelFetchOffset() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = texelFetchOffset(tex, ivec2(10, 10), 0, ivec2(1, 1)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureProjLod(self):
        """Test textureProjLod() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureProjLod(tex, vec3(0.5), 0.0); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureProjGrad(self):
        """Test textureProjGrad() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureProjGrad(tex, vec3(0.5), dPdx, dPdy); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureLodOffset(self):
        """Test textureLodOffset() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureLodOffset(tex, uv, 0.0, ivec2(1, 1)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureGradOffset(self):
        """Test textureGradOffset() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureGradOffset(tex, uv, dPdx, dPdy, ivec2(1, 1)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureProjOffset(self):
        """Test textureProjOffset() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureProjOffset(tex, vec3(0.5), ivec2(1, 1)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureProjLodOffset(self):
        """Test textureProjLodOffset() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureProjLodOffset(tex, vec3(0.5), 0.0, ivec2(1, 1)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureProjGradOffset(self):
        """Test textureProjGradOffset() function."""
        source = """
        uniform sampler2D tex;
        void main() { vec4 c = textureProjGradOffset(tex, vec3(0.5), dPdx, dPdy, ivec2(1, 1)); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_textureSize(self):
        """Test textureSize() function."""
        source = """
        uniform sampler2D tex;
        void main() { ivec2 size = textureSize(tex, 0); }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestIntegerFunctions:
    """Test integer functions."""

    def test_parse_uaddCarry(self):
        """Test uaddCarry() function."""
        source = "void main() { uint carry; uint result = uaddCarry(a, b, carry); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_usubBorrow(self):
        """Test usubBorrow() function."""
        source = "void main() { uint borrow; uint result = usubBorrow(a, b, borrow); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_umulExtended(self):
        """Test umulExtended() function."""
        source = "void main() { uint msb, lsb; umulExtended(a, b, msb, lsb); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_imulExtended(self):
        """Test imulExtended() function."""
        source = "void main() { int msb, lsb; imulExtended(a, b, msb, lsb); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_bitfieldExtract(self):
        """Test bitfieldExtract() function."""
        source = "void main() { int result = bitfieldExtract(value, 0, 8); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_bitfieldInsert(self):
        """Test bitfieldInsert() function."""
        source = "void main() { int result = bitfieldInsert(base, insert, 0, 8); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_bitfieldReverse(self):
        """Test bitfieldReverse() function."""
        source = "void main() { int result = bitfieldReverse(value); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_bitCount(self):
        """Test bitCount() function."""
        source = "void main() { int result = bitCount(value); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_findLSB(self):
        """Test findLSB() function."""
        source = "void main() { int result = findLSB(value); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_findMSB(self):
        """Test findMSB() function."""
        source = "void main() { int result = findMSB(value); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None
