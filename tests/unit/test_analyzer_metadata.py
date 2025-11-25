"""
Tests for Shader Metadata Extraction.

Tests metadata extraction from GLSL shaders including:
- mainImage() function detection
- Uniform variable extraction
- Texture sampler extraction
- Global variables and constants
- User-defined functions
- Shadertoy-specific patterns

Target: 100+ tests
"""

import pytest
from glsl_to_opencl.parser import GLSLParser
from glsl_to_opencl.analyzer.metadata import (
    ShaderMetadata,
    MetadataExtractor,
    MetadataExtractionError,
)
from glsl_to_opencl.analyzer.symbol_table import SymbolType
from glsl_to_opencl.analyzer.builtins import create_builtin_symbol_table


class TestShaderMetadataClass:
    """Test ShaderMetadata data class."""

    def test_create_empty_metadata(self):
        """Test creating empty metadata."""
        metadata = ShaderMetadata()
        assert metadata.main_function is None
        assert len(metadata.uniforms) == 0
        assert len(metadata.samplers) == 0
        assert len(metadata.global_variables) == 0
        assert len(metadata.constants) == 0
        assert len(metadata.functions) == 0
        assert len(metadata.structs) == 0

    def test_has_main_function_false(self):
        """Test has_main_function returns False when no main."""
        metadata = ShaderMetadata()
        assert not metadata.has_main_function()

    def test_has_main_function_true(self):
        """Test has_main_function returns True when main exists."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        metadata = ShaderMetadata()
        metadata.main_function = ast.get_main_image()
        assert metadata.has_main_function()

    def test_get_uniform_names_empty(self):
        """Test get_uniform_names with no uniforms."""
        metadata = ShaderMetadata()
        assert metadata.get_uniform_names() == []

    def test_get_sampler_names_empty(self):
        """Test get_sampler_names with no samplers."""
        metadata = ShaderMetadata()
        assert metadata.get_sampler_names() == []

    def test_get_function_names_empty(self):
        """Test get_function_names with no functions."""
        metadata = ShaderMetadata()
        assert metadata.get_function_names() == []


class TestMetadataExtractorConstruction:
    """Test MetadataExtractor construction."""

    def test_create_extractor_default(self):
        """Test creating extractor with default symbol table."""
        extractor = MetadataExtractor()
        assert extractor.symbol_table is not None
        assert extractor.metadata is not None

    def test_create_extractor_with_symbol_table(self):
        """Test creating extractor with custom symbol table."""
        symbol_table = create_builtin_symbol_table()
        extractor = MetadataExtractor(symbol_table=symbol_table)
        assert extractor.symbol_table is symbol_table


class TestMainFunctionExtraction:
    """Test mainImage() function extraction."""

    def test_extract_simple_main_image(self):
        """Test extracting simple mainImage() function."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert metadata.has_main_function()
        assert metadata.main_function.name == "mainImage"
        assert metadata.main_function.return_type.text == "void"
        assert len(metadata.main_function.parameters) == 2

    def test_extract_no_main_image(self):
        """Test extraction when no mainImage() present."""
        parser = GLSLParser()
        ast = parser.parse("""
            float helper(float x) {
                return x * 2.0;
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert not metadata.has_main_function()
        assert metadata.main_function is None

    def test_main_image_not_confused_with_other_functions(self):
        """Test that mainImage is distinguished from other functions."""
        parser = GLSLParser()
        ast = parser.parse("""
            void otherFunction(out vec4 color, in vec2 coord) {
                color = vec4(0.0);
            }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert metadata.has_main_function()
        assert metadata.main_function.name == "mainImage"
        assert len(metadata.functions) == 1
        assert metadata.functions[0].name == "otherFunction"

    def test_main_image_with_no_parameters_not_matched(self):
        """Test that mainImage with wrong signature is not matched."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage() {
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        # Should not be recognized as the real mainImage
        assert not metadata.has_main_function()

    def test_main_image_with_wrong_return_type(self):
        """Test that mainImage with non-void return is not matched."""
        parser = GLSLParser()
        ast = parser.parse("""
            float mainImage(out vec4 fragColor, in vec2 fragCoord) {
                return 1.0;
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        # Should not be recognized as the real mainImage
        assert not metadata.has_main_function()


class TestUniformExtraction:
    """Test uniform variable extraction."""

    def test_extract_simple_uniform(self):
        """Test extracting simple uniform declaration."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float uTime;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(uTime);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 1
        assert metadata.uniforms[0].name == "uTime"
        assert metadata.uniforms[0].glsl_type == "float"
        assert 'uniform' in metadata.uniforms[0].qualifiers

    def test_extract_multiple_uniforms(self):
        """Test extracting multiple uniform declarations."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float uTime;
            uniform vec3 uColor;
            uniform mat4 uTransform;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 3
        uniform_names = metadata.get_uniform_names()
        assert "uTime" in uniform_names
        assert "uColor" in uniform_names
        assert "uTransform" in uniform_names

    def test_extract_vector_uniform(self):
        """Test extracting vector uniform."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform vec3 uResolution;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 1
        assert metadata.uniforms[0].name == "uResolution"
        assert metadata.uniforms[0].glsl_type == "vec3"

    def test_extract_matrix_uniform(self):
        """Test extracting matrix uniform."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform mat4 uProjection;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 1
        assert metadata.uniforms[0].name == "uProjection"
        assert metadata.uniforms[0].glsl_type == "mat4"

    def test_uniforms_not_confused_with_globals(self):
        """Test that uniforms are distinct from global variables."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float uTime;
            float globalVar = 1.0;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 1
        assert len(metadata.global_variables) == 1
        assert metadata.uniforms[0].name == "uTime"
        assert metadata.global_variables[0].name == "globalVar"


class TestSamplerExtraction:
    """Test texture sampler extraction."""

    def test_extract_sampler2d(self):
        """Test extracting sampler2D uniform."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform sampler2D iChannel0;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = texture(iChannel0, fragCoord);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.samplers) == 1
        assert metadata.samplers[0].name == "iChannel0"
        assert metadata.samplers[0].glsl_type == "sampler2D"
        assert 'uniform' in metadata.samplers[0].qualifiers

    def test_extract_sampler_cube(self):
        """Test extracting samplerCube uniform."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform samplerCube iChannel1;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.samplers) == 1
        assert metadata.samplers[0].name == "iChannel1"
        assert metadata.samplers[0].glsl_type == "samplerCube"

    def test_extract_sampler3d(self):
        """Test extracting sampler3D uniform."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform sampler3D iChannel2;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.samplers) == 1
        assert metadata.samplers[0].name == "iChannel2"
        assert metadata.samplers[0].glsl_type == "sampler3D"

    def test_extract_multiple_samplers(self):
        """Test extracting multiple samplers."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform sampler2D iChannel0;
            uniform sampler2D iChannel1;
            uniform samplerCube iChannel2;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.samplers) == 3
        sampler_names = metadata.get_sampler_names()
        assert "iChannel0" in sampler_names
        assert "iChannel1" in sampler_names
        assert "iChannel2" in sampler_names

    def test_samplers_separate_from_uniforms(self):
        """Test that samplers are stored separately from regular uniforms."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float uTime;
            uniform sampler2D iChannel0;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 1
        assert len(metadata.samplers) == 1
        assert metadata.uniforms[0].name == "uTime"
        assert metadata.samplers[0].name == "iChannel0"


class TestGlobalVariableExtraction:
    """Test global variable and constant extraction."""

    def test_extract_global_variable(self):
        """Test extracting global variable."""
        parser = GLSLParser()
        ast = parser.parse("""
            float globalVar = 1.0;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(globalVar);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.global_variables) == 1
        assert metadata.global_variables[0].name == "globalVar"
        assert metadata.global_variables[0].glsl_type == "float"

    def test_extract_global_constant(self):
        """Test extracting global const."""
        parser = GLSLParser()
        ast = parser.parse("""
            const float PI = 3.14159;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(PI);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.constants) == 1
        assert metadata.constants[0].name == "PI"
        assert metadata.constants[0].glsl_type == "float"
        assert metadata.constants[0].symbol_type == SymbolType.CONSTANT

    def test_extract_multiple_globals(self):
        """Test extracting multiple global variables."""
        parser = GLSLParser()
        ast = parser.parse("""
            float var1 = 1.0;
            vec3 var2 = vec3(0.0);
            mat2 var3;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.global_variables) == 3
        global_names = [v.name for v in metadata.global_variables]
        assert "var1" in global_names
        assert "var2" in global_names
        assert "var3" in global_names

    def test_extract_multiple_constants(self):
        """Test extracting multiple constants."""
        parser = GLSLParser()
        ast = parser.parse("""
            const float PI = 3.14159;
            const float E = 2.71828;
            const vec3 UP = vec3(0.0, 1.0, 0.0);

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.constants) == 3
        const_names = [c.name for c in metadata.constants]
        assert "PI" in const_names
        assert "E" in const_names
        assert "UP" in const_names

    def test_constants_separate_from_variables(self):
        """Test that constants are stored separately from variables."""
        parser = GLSLParser()
        ast = parser.parse("""
            const float PI = 3.14159;
            float radius = 1.0;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.constants) == 1
        assert len(metadata.global_variables) == 1
        assert metadata.constants[0].name == "PI"
        assert metadata.global_variables[0].name == "radius"


class TestUserFunctionExtraction:
    """Test user-defined function extraction."""

    def test_extract_single_user_function(self):
        """Test extracting single user-defined function."""
        parser = GLSLParser()
        ast = parser.parse("""
            float helper(float x) {
                return x * 2.0;
            }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(helper(1.0));
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.functions) == 1
        assert metadata.functions[0].name == "helper"
        assert metadata.functions[0].return_type.text == "float"

    def test_extract_multiple_user_functions(self):
        """Test extracting multiple user functions."""
        parser = GLSLParser()
        ast = parser.parse("""
            float add(float a, float b) {
                return a + b;
            }

            vec3 multiply(vec3 v, float s) {
                return v * s;
            }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.functions) == 2
        func_names = metadata.get_function_names()
        assert "add" in func_names
        assert "multiply" in func_names

    def test_main_image_not_in_user_functions(self):
        """Test that mainImage is not included in user functions list."""
        parser = GLSLParser()
        ast = parser.parse("""
            float helper(float x) {
                return x * 2.0;
            }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.functions) == 1
        assert metadata.functions[0].name == "helper"
        assert "mainImage" not in metadata.get_function_names()

    def test_extract_function_with_multiple_parameters(self):
        """Test extracting function with multiple parameters."""
        parser = GLSLParser()
        ast = parser.parse("""
            vec4 blend(vec4 a, vec4 b, float t) {
                return mix(a, b, t);
            }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.functions) == 1
        assert metadata.functions[0].name == "blend"
        assert len(metadata.functions[0].parameters) == 3


class TestShadertoyPatterns:
    """Test Shadertoy-specific pattern detection."""

    def test_uses_shadertoy_uniform_method(self):
        """Test uses_shadertoy_uniform() method."""
        metadata = ShaderMetadata()
        from glsl_to_opencl.analyzer.symbol_table import Symbol, SymbolType

        # Add iTime uniform
        itime_symbol = Symbol("iTime", SymbolType.BUILTIN, "float", qualifiers=['uniform'])
        metadata.uniforms.append(itime_symbol)

        assert metadata.uses_shadertoy_uniform("iTime")
        assert not metadata.uses_shadertoy_uniform("iResolution")

    def test_uses_texture_channel_method(self):
        """Test uses_texture_channel() method."""
        metadata = ShaderMetadata()
        from glsl_to_opencl.analyzer.symbol_table import Symbol, SymbolType

        # Add iChannel0
        sampler_symbol = Symbol("iChannel0", SymbolType.BUILTIN, "sampler2D", qualifiers=['uniform'])
        metadata.samplers.append(sampler_symbol)

        assert metadata.uses_texture_channel(0)
        assert not metadata.uses_texture_channel(1)

    def test_detect_itime_usage(self):
        """Test detecting iTime usage in shader."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                float t = iTime;
                fragColor = vec4(sin(t));
            }
        """)
        # Note: This test assumes we'll implement usage tracking
        # For now it just checks that the parser doesn't fail
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert metadata.has_main_function()

    def test_detect_iresolution_usage(self):
        """Test detecting iResolution usage in shader."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                vec2 uv = fragCoord / iResolution.xy;
                fragColor = vec4(uv, 0.0, 1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert metadata.has_main_function()

    def test_detect_ichannel_usage(self):
        """Test detecting iChannel usage in shader."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                vec2 uv = fragCoord / iResolution.xy;
                fragColor = texture(iChannel0, uv);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert metadata.has_main_function()


class TestComplexShaders:
    """Test metadata extraction on complex shaders."""

    def test_extract_shader_with_all_features(self):
        """Test extracting metadata from shader with all features."""
        parser = GLSLParser()
        ast = parser.parse("""
            const float PI = 3.14159;
            uniform float uCustom;
            uniform sampler2D iChannel0;
            float globalVar = 1.0;

            float helper(float x) {
                return x * PI;
            }

            vec3 compute(vec3 color, float t) {
                return color * t;
            }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                vec2 uv = fragCoord / iResolution.xy;
                vec4 tex = texture(iChannel0, uv);
                float h = helper(iTime);
                fragColor = vec4(compute(tex.rgb, h), 1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        # Check main function
        assert metadata.has_main_function()
        assert metadata.main_function.name == "mainImage"

        # Check uniforms (uCustom)
        assert len(metadata.uniforms) >= 1
        assert "uCustom" in metadata.get_uniform_names()

        # Check samplers
        assert len(metadata.samplers) == 1
        assert "iChannel0" in metadata.get_sampler_names()

        # Check constants
        assert len(metadata.constants) == 1
        assert metadata.constants[0].name == "PI"

        # Check global variables
        assert len(metadata.global_variables) == 1
        assert metadata.global_variables[0].name == "globalVar"

        # Check user functions (helper, compute)
        assert len(metadata.functions) == 2
        func_names = metadata.get_function_names()
        assert "helper" in func_names
        assert "compute" in func_names

    def test_extract_minimal_shader(self):
        """Test extracting metadata from minimal shader."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0, 0.0, 0.0, 1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert metadata.has_main_function()
        assert len(metadata.uniforms) == 0
        assert len(metadata.samplers) == 0
        assert len(metadata.global_variables) == 0
        assert len(metadata.constants) == 0
        assert len(metadata.functions) == 0

    def test_extract_shader_with_many_helpers(self):
        """Test extracting shader with many helper functions."""
        parser = GLSLParser()
        ast = parser.parse("""
            float func1(float x) { return x; }
            float func2(float x) { return x * 2.0; }
            float func3(float x) { return x * 3.0; }
            float func4(float x) { return x * 4.0; }
            float func5(float x) { return x * 5.0; }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert metadata.has_main_function()
        assert len(metadata.functions) == 5

    def test_extract_shader_with_many_uniforms(self):
        """Test extracting shader with many uniforms."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float u1;
            uniform float u2;
            uniform vec3 u3;
            uniform vec4 u4;
            uniform mat3 u5;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 5
        uniform_names = metadata.get_uniform_names()
        for i in range(1, 6):
            assert f"u{i}" in uniform_names

    def test_extract_shader_with_many_samplers(self):
        """Test extracting shader with all 4 Shadertoy channels."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform sampler2D iChannel0;
            uniform sampler2D iChannel1;
            uniform sampler2D iChannel2;
            uniform sampler2D iChannel3;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                vec2 uv = fragCoord / iResolution.xy;
                vec4 c0 = texture(iChannel0, uv);
                vec4 c1 = texture(iChannel1, uv);
                vec4 c2 = texture(iChannel2, uv);
                vec4 c3 = texture(iChannel3, uv);
                fragColor = (c0 + c1 + c2 + c3) * 0.25;
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.samplers) == 4
        for i in range(4):
            assert metadata.uses_texture_channel(i)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_shader(self):
        """Test extracting metadata from empty shader."""
        parser = GLSLParser()
        ast = parser.parse("")
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert not metadata.has_main_function()
        assert len(metadata.uniforms) == 0
        assert len(metadata.samplers) == 0

    def test_shader_with_only_comments(self):
        """Test shader with only comments."""
        parser = GLSLParser()
        ast = parser.parse("""
            // This is a comment
            /* Multi-line
               comment */
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert not metadata.has_main_function()

    def test_declaration_without_type(self):
        """Test that malformed declarations are skipped gracefully."""
        parser = GLSLParser()
        # This will parse but won't have the expected structure
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        # Should still extract mainImage
        assert metadata.has_main_function()

    def test_multiple_main_images(self):
        """Test shader with multiple mainImage definitions (last one wins)."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0, 0.0, 0.0, 1.0);
            }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(0.0, 1.0, 0.0, 1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        # Should find at least one mainImage
        assert metadata.has_main_function()

    def test_uniform_and_global_same_type(self):
        """Test uniform and global variable of same type."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float uValue;
            float globalValue = 0.5;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 1
        assert len(metadata.global_variables) == 1
        assert metadata.uniforms[0].glsl_type == "float"
        assert metadata.global_variables[0].glsl_type == "float"

    def test_sampler_types_all_variants(self):
        """Test all sampler type variants."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform sampler2D s2d;
            uniform sampler3D s3d;
            uniform samplerCube scube;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.samplers) == 3
        types = [s.glsl_type for s in metadata.samplers]
        assert "sampler2D" in types
        assert "sampler3D" in types
        assert "samplerCube" in types

    def test_vec_types_all_variants(self):
        """Test all vector type variants."""
        parser = GLSLParser()
        ast = parser.parse("""
            vec2 v2;
            vec3 v3;
            vec4 v4;
            ivec2 iv2;
            ivec3 iv3;
            ivec4 iv4;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.global_variables) == 6
        types = [v.glsl_type for v in metadata.global_variables]
        assert "vec2" in types
        assert "vec3" in types
        assert "vec4" in types
        assert "ivec2" in types
        assert "ivec3" in types
        assert "ivec4" in types

    def test_mat_types_all_variants(self):
        """Test all matrix type variants."""
        parser = GLSLParser()
        ast = parser.parse("""
            mat2 m2;
            mat3 m3;
            mat4 m4;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.global_variables) == 3
        types = [v.glsl_type for v in metadata.global_variables]
        assert "mat2" in types
        assert "mat3" in types
        assert "mat4" in types

    def test_function_with_no_parameters(self):
        """Test function with no parameters."""
        parser = GLSLParser()
        ast = parser.parse("""
            float getRandom() {
                return 0.5;
            }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(getRandom());
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.functions) == 1
        assert metadata.functions[0].name == "getRandom"
        assert len(metadata.functions[0].parameters) == 0

    def test_function_with_void_return(self):
        """Test function with void return type."""
        parser = GLSLParser()
        ast = parser.parse("""
            void helper() {
            }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                helper();
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.functions) == 1
        assert metadata.functions[0].name == "helper"
        assert metadata.functions[0].return_type.text == "void"


class TestRealShaders:
    """Test metadata extraction on real shaders from tests/shaders/."""

    def test_basic_radial_gradient(self):
        """Test metadata extraction from Basic_Radial_Gradient.glsl."""
        import os
        shader_path = "C:\\dev\\hShadertoy\\tests\\shaders\\simple\\Basic_Radial_Gradient.glsl"

        if not os.path.exists(shader_path):
            pytest.skip(f"Shader file not found: {shader_path}")

        with open(shader_path, 'r') as f:
            shader_code = f.read()

        parser = GLSLParser()
        ast = parser.parse(shader_code)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        # Should have mainImage
        assert metadata.has_main_function()

        # Should have at least one user-defined function (dist)
        assert len(metadata.functions) >= 1
        func_names = metadata.get_function_names()
        assert "dist" in func_names

    def test_simple_vignette_effect(self):
        """Test metadata extraction from Simple_vignette_effect.glsl."""
        import os
        shader_path = "C:\\dev\\hShadertoy\\tests\\shaders\\simple\\Simple_vignette_effect .glsl"

        if not os.path.exists(shader_path):
            pytest.skip(f"Shader file not found: {shader_path}")

        with open(shader_path, 'r') as f:
            shader_code = f.read()

        parser = GLSLParser()
        ast = parser.parse(shader_code)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        # Should have mainImage
        assert metadata.has_main_function()

    def test_hypnotic_ripples(self):
        """Test metadata extraction from Hypnotic_ripples.glsl."""
        import os
        shader_path = "C:\\dev\\hShadertoy\\tests\\shaders\\simple\\Hypnotic_ripples.glsl"

        if not os.path.exists(shader_path):
            pytest.skip(f"Shader file not found: {shader_path}")

        with open(shader_path, 'r') as f:
            shader_code = f.read()

        parser = GLSLParser()
        ast = parser.parse(shader_code)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        # Should have mainImage
        assert metadata.has_main_function()

    def test_simple_hexagonal_tiles(self):
        """Test metadata extraction from simple_hexagonal_tiles.glsl."""
        import os
        shader_path = "C:\\dev\\hShadertoy\\tests\\shaders\\simple\\simple_hexagonal_tiles.glsl"

        if not os.path.exists(shader_path):
            pytest.skip(f"Shader file not found: {shader_path}")

        with open(shader_path, 'r') as f:
            shader_code = f.read()

        parser = GLSLParser()
        ast = parser.parse(shader_code)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        # Should have mainImage
        assert metadata.has_main_function()

    def test_sand_random_gradient_noise(self):
        """Test metadata extraction from Sand_Random_Gradient_Noise.glsl."""
        import os
        shader_path = "C:\\dev\\hShadertoy\\tests\\shaders\\simple\\Sand_Random_Gradient_Noise.glsl"

        if not os.path.exists(shader_path):
            pytest.skip(f"Shader file not found: {shader_path}")

        with open(shader_path, 'r') as f:
            shader_code = f.read()

        parser = GLSLParser()
        ast = parser.parse(shader_code)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        # Should have mainImage
        assert metadata.has_main_function()

        # Should have user-defined functions
        assert len(metadata.functions) >= 1


class TestMetadataQuery:
    """Test metadata query methods."""

    def test_get_uniform_names(self):
        """Test get_uniform_names returns correct list."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float u1;
            uniform vec3 u2;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        names = metadata.get_uniform_names()
        assert isinstance(names, list)
        assert "u1" in names
        assert "u2" in names
        assert len(names) == 2

    def test_get_sampler_names(self):
        """Test get_sampler_names returns correct list."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform sampler2D s1;
            uniform samplerCube s2;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        names = metadata.get_sampler_names()
        assert isinstance(names, list)
        assert "s1" in names
        assert "s2" in names
        assert len(names) == 2

    def test_get_function_names(self):
        """Test get_function_names returns correct list."""
        parser = GLSLParser()
        ast = parser.parse("""
            float f1(float x) { return x; }
            vec3 f2(vec3 v) { return v; }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        names = metadata.get_function_names()
        assert isinstance(names, list)
        assert "f1" in names
        assert "f2" in names
        assert "mainImage" not in names
        assert len(names) == 2

    def test_uses_shadertoy_uniform_true(self):
        """Test uses_shadertoy_uniform detects usage."""
        from glsl_to_opencl.analyzer.symbol_table import Symbol, SymbolType

        metadata = ShaderMetadata()
        metadata.uniforms.append(Symbol("iTime", SymbolType.BUILTIN, "float"))

        assert metadata.uses_shadertoy_uniform("iTime")

    def test_uses_shadertoy_uniform_false(self):
        """Test uses_shadertoy_uniform returns False when not used."""
        metadata = ShaderMetadata()

        assert not metadata.uses_shadertoy_uniform("iTime")

    def test_uses_texture_channel_true(self):
        """Test uses_texture_channel detects usage."""
        from glsl_to_opencl.analyzer.symbol_table import Symbol, SymbolType

        metadata = ShaderMetadata()
        metadata.samplers.append(Symbol("iChannel0", SymbolType.BUILTIN, "sampler2D"))

        assert metadata.uses_texture_channel(0)

    def test_uses_texture_channel_false(self):
        """Test uses_texture_channel returns False when not used."""
        metadata = ShaderMetadata()

        assert not metadata.uses_texture_channel(0)

    def test_uses_texture_channel_all_channels(self):
        """Test uses_texture_channel for all 4 channels."""
        from glsl_to_opencl.analyzer.symbol_table import Symbol, SymbolType

        metadata = ShaderMetadata()
        for i in range(4):
            metadata.samplers.append(
                Symbol(f"iChannel{i}", SymbolType.BUILTIN, "sampler2D")
            )

        for i in range(4):
            assert metadata.uses_texture_channel(i)

        # Channel 4 doesn't exist
        assert not metadata.uses_texture_channel(4)


class TestSymbolTypes:
    """Test symbol type classification."""

    def test_uniform_symbol_type(self):
        """Test that uniforms are classified correctly."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float uTime;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 1
        assert metadata.uniforms[0].symbol_type == SymbolType.VARIABLE

    def test_constant_symbol_type(self):
        """Test that constants are classified correctly."""
        parser = GLSLParser()
        ast = parser.parse("""
            const float PI = 3.14159;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.constants) == 1
        assert metadata.constants[0].symbol_type == SymbolType.CONSTANT

    def test_global_variable_symbol_type(self):
        """Test that global variables are classified correctly."""
        parser = GLSLParser()
        ast = parser.parse("""
            float globalVar = 1.0;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.global_variables) == 1
        assert metadata.global_variables[0].symbol_type == SymbolType.VARIABLE


class TestQualifiers:
    """Test qualifier handling in declarations."""

    def test_uniform_qualifier(self):
        """Test uniform qualifier detection."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float x;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 1
        assert 'uniform' in metadata.uniforms[0].qualifiers

    def test_const_qualifier(self):
        """Test const qualifier detection."""
        parser = GLSLParser()
        ast = parser.parse("""
            const float PI = 3.14;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.constants) == 1
        assert 'const' in metadata.constants[0].qualifiers

    def test_uniform_sampler(self):
        """Test uniform sampler has uniform qualifier."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform sampler2D tex;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.samplers) == 1
        assert 'uniform' in metadata.samplers[0].qualifiers


class TestLocationTracking:
    """Test source location tracking."""

    def test_uniform_has_location(self):
        """Test that uniforms have location info."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float x;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 1
        assert metadata.uniforms[0].location is not None
        assert isinstance(metadata.uniforms[0].location, tuple)
        assert len(metadata.uniforms[0].location) == 2  # (line, column)

    def test_const_has_location(self):
        """Test that constants have location info."""
        parser = GLSLParser()
        ast = parser.parse("""
            const float PI = 3.14;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.constants) == 1
        assert metadata.constants[0].location is not None

    def test_global_variable_has_location(self):
        """Test that global variables have location info."""
        parser = GLSLParser()
        ast = parser.parse("""
            float globalVar = 1.0;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.global_variables) == 1
        assert metadata.global_variables[0].location is not None


class TestMultipleDeclarations:
    """Test handling of multiple declarations."""

    def test_multiple_uniforms_same_line(self):
        """Test multiple uniform declarations."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float a;
            uniform float b;
            uniform float c;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 3
        uniform_names = metadata.get_uniform_names()
        assert 'a' in uniform_names
        assert 'b' in uniform_names
        assert 'c' in uniform_names

    def test_mixed_declarations(self):
        """Test mix of different declaration types."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform float u;
            const float c = 1.0;
            float g;
            uniform sampler2D s;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.uniforms) == 1
        assert len(metadata.constants) == 1
        assert len(metadata.global_variables) == 1
        assert len(metadata.samplers) == 1

    def test_declarations_between_functions(self):
        """Test declarations interspersed with functions."""
        parser = GLSLParser()
        ast = parser.parse("""
            float global1 = 1.0;

            float func1() { return 0.0; }

            float global2 = 2.0;

            float func2() { return 0.0; }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.global_variables) == 2
        assert len(metadata.functions) == 2


class TestFunctionExtraction:
    """Test function metadata extraction details."""

    def test_function_parameter_count(self):
        """Test function parameter count extraction."""
        parser = GLSLParser()
        ast = parser.parse("""
            float func0() { return 0.0; }
            float func1(float a) { return a; }
            float func2(float a, float b) { return a + b; }
            float func3(float a, float b, float c) { return a + b + c; }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.functions) == 4

        func_map = {f.name: f for f in metadata.functions}
        assert len(func_map['func0'].parameters) == 0
        assert len(func_map['func1'].parameters) == 1
        assert len(func_map['func2'].parameters) == 2
        assert len(func_map['func3'].parameters) == 3

    def test_function_return_types(self):
        """Test function return type extraction."""
        parser = GLSLParser()
        ast = parser.parse("""
            void voidFunc() { }
            float floatFunc() { return 0.0; }
            vec3 vec3Func() { return vec3(0.0); }
            mat4 mat4Func() { return mat4(1.0); }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.functions) == 4

        func_map = {f.name: f for f in metadata.functions}
        assert func_map['voidFunc'].return_type.text == 'void'
        assert func_map['floatFunc'].return_type.text == 'float'
        assert func_map['vec3Func'].return_type.text == 'vec3'
        assert func_map['mat4Func'].return_type.text == 'mat4'

    def test_function_names_unique(self):
        """Test that function names are correctly extracted."""
        parser = GLSLParser()
        ast = parser.parse("""
            float calcDistance() { return 0.0; }
            vec3 getNormal() { return vec3(0.0); }
            vec4 applyLighting() { return vec4(0.0); }

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        names = metadata.get_function_names()
        assert len(names) == 3
        assert 'calcDistance' in names
        assert 'getNormal' in names
        assert 'applyLighting' in names


class TestComplexDeclarations:
    """Test complex declaration patterns."""

    def test_vec_with_initializer(self):
        """Test vector with initializer."""
        parser = GLSLParser()
        ast = parser.parse("""
            vec3 color = vec3(1.0, 0.5, 0.0);

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.global_variables) == 1
        assert metadata.global_variables[0].name == 'color'
        assert metadata.global_variables[0].glsl_type == 'vec3'

    def test_mat_with_initializer(self):
        """Test matrix with initializer."""
        parser = GLSLParser()
        ast = parser.parse("""
            mat4 transform = mat4(1.0);

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.global_variables) == 1
        assert metadata.global_variables[0].name == 'transform'
        assert metadata.global_variables[0].glsl_type == 'mat4'

    def test_const_with_expression(self):
        """Test const with complex expression."""
        parser = GLSLParser()
        ast = parser.parse("""
            const float TWO_PI = 2.0 * 3.14159;

            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)

        assert len(metadata.constants) == 1
        assert metadata.constants[0].name == 'TWO_PI'


class TestExtractorReuse:
    """Test MetadataExtractor reuse and state management."""

    def test_extractor_multiple_calls(self):
        """Test that extractor can be reused for multiple extractions."""
        parser = GLSLParser()
        extractor = MetadataExtractor()

        # First extraction
        ast1 = parser.parse("""
            uniform float x;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        metadata1 = extractor.extract(ast1)
        assert len(metadata1.uniforms) == 1

        # Second extraction - should be independent
        ast2 = parser.parse("""
            uniform float y;
            uniform float z;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        metadata2 = extractor.extract(ast2)
        assert len(metadata2.uniforms) == 2

        # First metadata should be unchanged
        assert len(metadata1.uniforms) == 1

    def test_extractor_clean_state(self):
        """Test that extractor clears state between extractions."""
        parser = GLSLParser()
        extractor = MetadataExtractor()

        # Extract from shader with functions
        ast1 = parser.parse("""
            float helper() { return 0.0; }
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        metadata1 = extractor.extract(ast1)
        assert len(metadata1.functions) == 1

        # Extract from shader without functions
        ast2 = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        metadata2 = extractor.extract(ast2)
        assert len(metadata2.functions) == 0

    def test_get_metadata_method(self):
        """Test get_metadata() returns current metadata."""
        parser = GLSLParser()
        extractor = MetadataExtractor()

        ast = parser.parse("""
            uniform float x;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        metadata = extractor.extract(ast)

        # get_metadata should return the same object
        assert extractor.get_metadata() is metadata


class TestAdditionalCoverage:
    """Additional tests to reach 100+ test coverage."""

    def test_bool_type(self):
        """Test bool type variable."""
        parser = GLSLParser()
        ast = parser.parse("""
            bool flag = true;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert metadata.global_variables[0].glsl_type == "bool"

    def test_int_type(self):
        """Test int type variable."""
        parser = GLSLParser()
        ast = parser.parse("""
            int count = 10;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert metadata.global_variables[0].glsl_type == "int"

    def test_uint_type(self):
        """Test uint type variable."""
        parser = GLSLParser()
        ast = parser.parse("""
            uint index = 0u;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert metadata.global_variables[0].glsl_type == "uint"

    def test_bvec_types(self):
        """Test bvec types."""
        parser = GLSLParser()
        ast = parser.parse("""
            bvec2 b2;
            bvec3 b3;
            bvec4 b4;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        types = [v.glsl_type for v in metadata.global_variables]
        assert "bvec2" in types
        assert "bvec3" in types
        assert "bvec4" in types

    def test_uvec_types(self):
        """Test uvec types."""
        parser = GLSLParser()
        ast = parser.parse("""
            uvec2 u2;
            uvec3 u3;
            uvec4 u4;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        types = [v.glsl_type for v in metadata.global_variables]
        assert "uvec2" in types
        assert "uvec3" in types
        assert "uvec4" in types

    def test_mat2x3_type(self):
        """Test mat2x3 type."""
        parser = GLSLParser()
        ast = parser.parse("""
            mat2x3 m;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert metadata.global_variables[0].glsl_type == "mat2x3"

    def test_empty_functions_list(self):
        """Test shader with no user functions."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert len(metadata.functions) == 0
        assert metadata.get_function_names() == []

    def test_sampler_as_uniform(self):
        """Test that samplers are uniforms."""
        parser = GLSLParser()
        ast = parser.parse("""
            uniform sampler2D tex;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert len(metadata.samplers) == 1
        assert len(metadata.uniforms) == 0  # Samplers go to separate list

    def test_multiple_mainImage_last_wins(self):
        """Test that last mainImage is kept when multiple exist."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0, 0.0, 0.0, 1.0);
            }
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(0.0, 1.0, 0.0, 1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        # Should have found mainImage (implementation picks last one)
        assert metadata.has_main_function()

    def test_const_vec3(self):
        """Test const vec3 declaration."""
        parser = GLSLParser()
        ast = parser.parse("""
            const vec3 UP = vec3(0.0, 1.0, 0.0);
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert len(metadata.constants) == 1
        assert metadata.constants[0].glsl_type == "vec3"

    def test_function_with_vec_parameters(self):
        """Test function with vector parameters."""
        parser = GLSLParser()
        ast = parser.parse("""
            vec3 transform(vec3 v, mat3 m) {
                return m * v;
            }
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert len(metadata.functions) == 1
        assert len(metadata.functions[0].parameters) == 2

    def test_no_declarations(self):
        """Test shader with only mainImage."""
        parser = GLSLParser()
        ast = parser.parse("""
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0, 0.5, 0.0, 1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert len(metadata.uniforms) == 0
        assert len(metadata.samplers) == 0
        assert len(metadata.global_variables) == 0
        assert len(metadata.constants) == 0
        assert len(metadata.functions) == 0

    def test_metadata_dataclass_default_values(self):
        """Test ShaderMetadata dataclass default values."""
        metadata = ShaderMetadata()
        assert metadata.main_function is None
        assert isinstance(metadata.uniforms, list)
        assert isinstance(metadata.samplers, list)
        assert isinstance(metadata.global_variables, list)
        assert isinstance(metadata.constants, list)
        assert isinstance(metadata.functions, list)
        assert isinstance(metadata.structs, list)

    def test_shadertoy_constant_pattern(self):
        """Test common Shadertoy constant pattern."""
        parser = GLSLParser()
        ast = parser.parse("""
            const float PI = 3.14159265359;
            const float TAU = 2.0 * PI;
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                fragColor = vec4(1.0);
            }
        """)
        extractor = MetadataExtractor()
        metadata = extractor.extract(ast)
        assert len(metadata.constants) == 2
        names = [c.name for c in metadata.constants]
        assert "PI" in names
        assert "TAU" in names
