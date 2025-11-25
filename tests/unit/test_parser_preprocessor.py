"""
Preprocessor tests - #define macro expansion.

Tests basic GLSL preprocessor support.
"""

import pytest
from glsl_to_opencl.parser import GLSLPreprocessor


class TestDefineDirective:
    """Test #define directive processing."""

    def test_parse_simple_define(self):
        """Test simple #define."""
        source = "#define PI 3.14159"
        preprocessor = GLSLPreprocessor()
        result = preprocessor.process(source)

        assert "PI" in result
        # Define should be commented out
        assert "//" in result

    def test_expand_simple_macro(self):
        """Test macro expansion."""
        source = """
        #define MAX_ITER 100
        int x = MAX_ITER;
        """
        preprocessor = GLSLPreprocessor()
        result = preprocessor.process(source)

        assert "100" in result

    def test_expand_multiple_macros(self):
        """Test multiple macro expansions."""
        source = """
        #define WIDTH 800
        #define HEIGHT 600
        vec2 resolution = vec2(WIDTH, HEIGHT);
        """
        preprocessor = GLSLPreprocessor()
        result = preprocessor.process(source)

        assert "800" in result
        assert "600" in result

    def test_macro_not_expanded_in_define(self):
        """Test macro not expanded in #define line itself."""
        source = """
        #define FOO 123
        #define BAR FOO
        """
        preprocessor = GLSLPreprocessor()
        result = preprocessor.process(source)

        # BAR should be defined as "FOO" (not expanded)
        lines = result.split('\n')
        assert any("FOO" in line and "#define BAR" in line for line in lines)


class TestMacroExpansion:
    """Test macro expansion in code."""

    def test_expand_in_expression(self):
        """Test macro expansion in expression."""
        source = """
        #define SIZE 10
        int arr[SIZE];
        """
        preprocessor = GLSLPreprocessor()
        result = preprocessor.process(source)

        assert "arr[10]" in result or "arr[ 10 ]" in result

    def test_expand_float_constant(self):
        """Test expanding float constant."""
        source = """
        #define PI 3.14159
        float x = PI;
        """
        preprocessor = GLSLPreprocessor()
        result = preprocessor.process(source)

        assert "3.14159" in result

    def test_word_boundary_respected(self):
        """Test macro only expands at word boundaries."""
        source = """
        #define X 5
        int X = X;
        int XX = 10;
        """
        preprocessor = GLSLPreprocessor()
        result = preprocessor.process(source)

        # XX should not be expanded
        assert "XX" in result


class TestPreprocessorEdgeCases:
    """Test edge cases for preprocessor."""

    def test_empty_source(self):
        """Test empty source."""
        source = ""
        preprocessor = GLSLPreprocessor()
        result = preprocessor.process(source)

        assert result == ""

    def test_no_defines(self):
        """Test source with no #define directives."""
        source = "float x = 1.0;"
        preprocessor = GLSLPreprocessor()
        result = preprocessor.process(source)

        assert "float x = 1.0;" in result

    def test_define_with_spaces(self):
        """Test #define with various spacing."""
        source = "#define   SPACING    42"
        preprocessor = GLSLPreprocessor()
        preprocessor.process(source)

        assert "SPACING" in preprocessor.defines
        assert preprocessor.defines["SPACING"] == "42"
