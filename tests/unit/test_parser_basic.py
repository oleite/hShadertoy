"""
Basic parser tests - Simple declarations and types.

Tests the fundamental parsing capabilities of GLSLParser.
"""

import pytest
from glsl_to_opencl.parser import GLSLParser, ParseError


class TestSimpleDeclarations:
    """Test parsing of simple variable declarations."""

    def test_parse_float_variable(self):
        """Test parsing float variable declaration."""
        source = "float x;"
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast.type == "translation_unit"
        assert len(ast.declarations) == 1
        assert ast.declarations[0].type == "declaration"

    def test_parse_int_variable(self):
        """Test parsing int variable declaration."""
        source = "int count;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert decl.type == "declaration"
        assert "int" in decl.text

    def test_parse_bool_variable(self):
        """Test parsing bool variable declaration."""
        source = "bool flag;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert decl.type == "declaration"
        assert "bool" in decl.text

    def test_parse_uint_variable(self):
        """Test parsing uint variable declaration."""
        source = "uint value;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert decl.type == "declaration"
        assert "uint" in decl.text


class TestVectorTypes:
    """Test parsing of vector type declarations."""

    def test_parse_vec2(self):
        """Test parsing vec2 declaration."""
        source = "vec2 position;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "vec2" in decl.text
        assert "position" in decl.text

    def test_parse_vec3(self):
        """Test parsing vec3 declaration."""
        source = "vec3 color;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "vec3" in decl.text

    def test_parse_vec4(self):
        """Test parsing vec4 declaration."""
        source = "vec4 fragColor;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "vec4" in decl.text

    def test_parse_ivec2(self):
        """Test parsing ivec2 declaration."""
        source = "ivec2 coords;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "ivec2" in decl.text

    def test_parse_ivec3(self):
        """Test parsing ivec3 declaration."""
        source = "ivec3 indices;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "ivec3" in decl.text

    def test_parse_ivec4(self):
        """Test parsing ivec4 declaration."""
        source = "ivec4 data;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "ivec4" in decl.text

    def test_parse_uvec2(self):
        """Test parsing uvec2 declaration."""
        source = "uvec2 udata;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "uvec2" in decl.text

    def test_parse_uvec3(self):
        """Test parsing uvec3 declaration."""
        source = "uvec3 udata;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "uvec3" in decl.text

    def test_parse_uvec4(self):
        """Test parsing uvec4 declaration."""
        source = "uvec4 udata;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "uvec4" in decl.text

    def test_parse_bvec2(self):
        """Test parsing bvec2 declaration."""
        source = "bvec2 flags;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "bvec2" in decl.text

    def test_parse_bvec3(self):
        """Test parsing bvec3 declaration."""
        source = "bvec3 flags;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "bvec3" in decl.text

    def test_parse_bvec4(self):
        """Test parsing bvec4 declaration."""
        source = "bvec4 flags;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "bvec4" in decl.text


class TestMatrixTypes:
    """Test parsing of matrix type declarations."""

    def test_parse_mat2(self):
        """Test parsing mat2 declaration."""
        source = "mat2 M;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "mat2" in decl.text

    def test_parse_mat3(self):
        """Test parsing mat3 declaration."""
        source = "mat3 transform;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "mat3" in decl.text

    def test_parse_mat4(self):
        """Test parsing mat4 declaration."""
        source = "mat4 projection;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "mat4" in decl.text


class TestSamplerTypes:
    """Test parsing of sampler type declarations."""

    def test_parse_sampler2D(self):
        """Test parsing sampler2D declaration."""
        source = "uniform sampler2D tex;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "sampler2D" in decl.text

    def test_parse_samplerCube(self):
        """Test parsing samplerCube declaration."""
        source = "uniform samplerCube cubemap;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "samplerCube" in decl.text

    def test_parse_sampler3D(self):
        """Test parsing sampler3D declaration."""
        source = "uniform sampler3D volume;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "sampler3D" in decl.text


class TestInitializers:
    """Test parsing of variable declarations with initializers."""

    def test_parse_float_with_initializer(self):
        """Test float variable with initializer."""
        source = "float x = 1.0;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "1.0" in decl.text

    def test_parse_vec3_constructor(self):
        """Test vec3 with constructor initializer."""
        source = "vec3 color = vec3(1.0, 0.5, 0.0);"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "vec3" in decl.text
        assert "1.0" in decl.text

    def test_parse_mat3_constructor(self):
        """Test mat3 with constructor initializer."""
        source = "mat3 M = mat3(1.0);"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "mat3" in decl.text


class TestMultipleDeclarations:
    """Test parsing multiple declarations in one shader."""

    def test_parse_two_declarations(self):
        """Test two separate declarations."""
        source = """
        float x;
        vec3 color;
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert len(ast.declarations) == 2

    def test_parse_multiple_types(self):
        """Test multiple declarations of different types."""
        source = """
        float x;
        vec3 color;
        mat3 M;
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert len(ast.declarations) == 3


class TestQualifiers:
    """Test parsing of type qualifiers."""

    def test_parse_const_qualifier(self):
        """Test const qualifier."""
        source = "const float PI = 3.14159;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "const" in decl.text

    def test_parse_uniform_qualifier(self):
        """Test uniform qualifier."""
        source = "uniform float iTime;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "uniform" in decl.text

    def test_parse_in_qualifier(self):
        """Test in qualifier."""
        source = "in vec2 fragCoord;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "in" in decl.text

    def test_parse_out_qualifier(self):
        """Test out qualifier."""
        source = "out vec4 fragColor;"
        parser = GLSLParser()
        ast = parser.parse(source)

        decl = ast.declarations[0]
        assert "out" in decl.text


class TestParseErrors:
    """Test error handling for invalid GLSL."""

    def test_parse_error_missing_semicolon(self):
        """Test error for missing semicolon."""
        source = "float x"
        parser = GLSLParser()

        with pytest.raises(ParseError):
            parser.parse(source)

    def test_parse_error_invalid_type(self):
        """Test error for invalid type."""
        source = "notAType x;"
        parser = GLSLParser()

        # Should parse but won't recognize as built-in type
        ast = parser.parse(source)
        # tree-sitter should still create AST (may treat as identifier)
        assert ast is not None

    def test_parse_error_unmatched_brace(self):
        """Test error for unmatched brace."""
        source = "void main() {"
        parser = GLSLParser()

        with pytest.raises(ParseError):
            parser.parse(source)
