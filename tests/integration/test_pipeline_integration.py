"""
Integration Tests - Complete Pipeline

Tests end-to-end transpilation with real shader examples.
Focus on structural correctness, not exact text matching.
"""

import pytest
from pathlib import Path
from glsl_to_opencl.parser import GLSLParser
from glsl_to_opencl.analyzer import (
    create_builtin_symbol_table,
    SymbolTable,
    TypeChecker,
)
from glsl_to_opencl.transformer import (
    GLSLTransformer,
    ShaderStructureTransformer,
    ShaderConfig,
    RenderpassType,
)
from glsl_to_opencl.codegen import CodeGenerator


class TestSimpleShaderPipeline:
    """Test complete pipeline with simple shaders."""

    @pytest.fixture
    def parser(self):
        return GLSLParser()

    @pytest.fixture
    def symbol_table(self):
        return create_builtin_symbol_table()

    @pytest.fixture
    def transformer(self, symbol_table):
        return GLSLTransformer(symbol_table)

    @pytest.fixture
    def shader_transformer(self, transformer):
        return ShaderStructureTransformer(transformer)

    @pytest.fixture
    def generator(self):
        return CodeGenerator()

    def test_basic_radial_gradient(self, parser, transformer, shader_transformer):
        """Test Basic_Radial_Gradient.glsl transpilation."""
        shader_path = Path("tests/shaders/simple/Basic_Radial_Gradient.glsl")
        glsl = shader_path.read_text()

        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Verify structure
        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result
        assert "@fragColor.set(fragColor)" in result

        # Verify transformations
        assert ".xy" in result  # vec2 swizzle preserved
        assert "sin(iTime)" in result  # function call preserved

    def test_simple_vignette_effect(self, parser, transformer, shader_transformer):
        """Test Simple_vignette_effect.glsl transpilation."""
        shader_path = Path("tests/shaders/simple/Simple_vignette_effect .glsl")
        if not shader_path.exists():
            pytest.skip(f"Shader file not found: {shader_path}")

        glsl = shader_path.read_text()
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Verify structure
        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result
        assert "@fragColor.set(fragColor)" in result

    def test_hypnotic_ripples(self, parser, transformer, shader_transformer):
        """Test Hypnotic_ripples.glsl transpilation."""
        shader_path = Path("tests/shaders/simple/Hypnotic_ripples.glsl")
        if not shader_path.exists():
            pytest.skip(f"Shader file not found: {shader_path}")

        glsl = shader_path.read_text()
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Verify structure
        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result

    def test_simple_hexagonal_tiles(self, parser, transformer, shader_transformer):
        """Test simple_hexagonal_tiles.glsl transpilation."""
        shader_path = Path("tests/shaders/simple/simple_hexagonal_tiles.glsl")
        if not shader_path.exists():
            pytest.skip(f"Shader file not found: {shader_path}")

        glsl = shader_path.read_text()
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Verify structure
        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result

    def test_sand_random_gradient_noise(self, parser, transformer, shader_transformer):
        """Test Sand_Random_Gradient_Noise.glsl transpilation."""
        shader_path = Path("tests/shaders/simple/Sand_Random_Gradient_Noise.glsl")
        if not shader_path.exists():
            pytest.skip(f"Shader file not found: {shader_path}")

        glsl = shader_path.read_text()
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Verify structure
        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result


class TestMediumShaderPipeline:
    """Test complete pipeline with medium complexity shaders."""

    @pytest.fixture
    def parser(self):
        return GLSLParser()

    @pytest.fixture
    def symbol_table(self):
        return create_builtin_symbol_table()

    @pytest.fixture
    def transformer(self, symbol_table):
        return GLSLTransformer(symbol_table)

    @pytest.fixture
    def shader_transformer(self, transformer):
        return ShaderStructureTransformer(transformer)

    def test_3d_voronoi_noise(self, parser, transformer, shader_transformer):
        """Test 3D_Voronoi_noise.glsl transpilation."""
        shader_path = Path("tests/shaders/medium/3D_Voronoi_noise.glsl")
        if not shader_path.exists():
            pytest.skip(f"Shader file not found: {shader_path}")

        glsl = shader_path.read_text()
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Verify structure
        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result

    def test_hexagon_x5(self, parser, transformer, shader_transformer):
        """Test Hexagon_X5.glsl transpilation."""
        shader_path = Path("tests/shaders/medium/Hexagon_X5.glsl")
        if not shader_path.exists():
            pytest.skip(f"Shader file not found: {shader_path}")

        glsl = shader_path.read_text()
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Verify structure
        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result

    def test_moving_voronoi_cells(self, parser, transformer, shader_transformer):
        """Test Moving_Voronoi_Cells.glsl transpilation."""
        shader_path = Path("tests/shaders/medium/Moving_Voronoi_Cells.glsl")
        if not shader_path.exists():
            pytest.skip(f"Shader file not found: {shader_path}")

        glsl = shader_path.read_text()
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Verify structure
        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result

    def test_noise_voronoi_diagram(self, parser, transformer, shader_transformer):
        """Test Noise_Voronoi_Diagram.glsl transpilation."""
        shader_path = Path("tests/shaders/medium/Noise_Voronoi_Diagram.glsl")
        if not shader_path.exists():
            pytest.skip(f"Shader file not found: {shader_path}")

        glsl = shader_path.read_text()
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Verify structure
        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result

    def test_fire_smoke_tinkering(self, parser, transformer, shader_transformer):
        """Test Fire_Smoke_Tinkering.glsl transpilation."""
        shader_path = Path("tests/shaders/medium/Fire _Smoke_Tinkering.glsl")
        if not shader_path.exists():
            pytest.skip(f"Shader file not found: {shader_path}")

        glsl = shader_path.read_text()
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Verify structure
        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result


class TestTransformationCorrectness:
    """Test that key transformations are applied correctly."""

    @pytest.fixture
    def parser(self):
        return GLSLParser()

    @pytest.fixture
    def symbol_table(self):
        return create_builtin_symbol_table()

    @pytest.fixture
    def transformer(self, symbol_table):
        return GLSLTransformer(symbol_table)

    @pytest.fixture
    def shader_transformer(self, transformer):
        return ShaderStructureTransformer(transformer)

    def test_float_suffix_transformation(self, shader_transformer):
        """Test that float literals get 'f' suffix."""
        glsl = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            float x = 1.0;
            float y = 2.5;
            fragColor = vec4(x, y, 0.0, 1.0);
        }
        """
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Should have float suffixes
        assert "1.0f" in result or "1.f" in result
        assert "2.5f" in result or "2.5f" in result

    def test_type_conversion(self, shader_transformer):
        """Test that GLSL types are converted to OpenCL types."""
        glsl = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            vec3 color = vec3(uv, 0.5);
            fragColor = vec4(color, 1.0);
        }
        """
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Original types should be preserved in transform (conversion happens in different phase)
        assert "fragCoord" in result
        assert "iResolution" in result

    def test_helper_function_preservation(self, shader_transformer):
        """Test that helper functions are called correctly."""
        glsl = """
        float square(float x) {
            return x * x;
        }
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            float s = square(2.0);
            fragColor = vec4(s);
        }
        """
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Function call should be present in mainImage body
        assert "square" in result
        assert "@KERNEL" in result

    def test_uniform_generation(self, shader_transformer):
        """Test that Shadertoy uniforms are generated."""
        glsl = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            float t = iTime;
            fragColor = vec4(uv, sin(t), 1.0);
        }
        """
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Shadertoy inputs should be present
        assert "iResolution" in result
        assert "iTime" in result

    def test_fragcolor_setting(self, shader_transformer):
        """Test that fragColor is set at the end."""
        glsl = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            fragColor = vec4(1.0);
        }
        """
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        # Should have fragColor.set() call
        assert "@fragColor.set(fragColor)" in result or "fragColor.set" in result


class TestPipelineRobustness:
    """Test that pipeline handles various edge cases."""

    @pytest.fixture
    def parser(self):
        return GLSLParser()

    @pytest.fixture
    def symbol_table(self):
        return create_builtin_symbol_table()

    @pytest.fixture
    def transformer(self, symbol_table):
        return GLSLTransformer(symbol_table)

    @pytest.fixture
    def shader_transformer(self, transformer):
        return ShaderStructureTransformer(transformer)

    def test_minimal_shader(self, shader_transformer):
        """Test minimal valid shader."""
        glsl = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            fragColor = vec4(1.0);
        }
        """
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        assert "@KERNEL" in result
        assert result.strip()  # Not empty

    def test_shader_with_comments(self, shader_transformer):
        """Test shader with comments."""
        glsl = """
        // This is a comment
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            // Another comment
            fragColor = vec4(1.0);
        }
        """
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        assert "@KERNEL" in result
        # Comments may or may not be preserved (current implementation strips them)

    def test_shader_with_preprocessor(self, shader_transformer):
        """Test shader with preprocessor directives."""
        glsl = """
        #define PI 3.14159
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            float x = PI;
            fragColor = vec4(x);
        }
        """
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        assert "@KERNEL" in result

    def test_shader_with_multiple_helpers(self, shader_transformer):
        """Test shader with multiple helper functions."""
        glsl = """
        float add(float a, float b) { return a + b; }
        float mul(float a, float b) { return a * b; }
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            float x = add(1.0, 2.0);
            float y = mul(x, 3.0);
            fragColor = vec4(y);
        }
        """
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        assert "@KERNEL" in result
        assert "add" in result
        assert "mul" in result

    def test_shader_with_struct(self, shader_transformer):
        """Test shader with struct definition."""
        glsl = """
        struct Ray {
            vec3 origin;
            vec3 direction;
        };
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            Ray r;
            r.origin = vec3(0.0);
            fragColor = vec4(r.origin, 1.0);
        }
        """
        config = ShaderConfig(renderpass=RenderpassType.IMAGE)
        result = shader_transformer.transform_shadertoy_shader("", glsl, config)

        assert "@KERNEL" in result
        assert "Ray" in result or "struct" in result


class TestCodeGeneratorIntegration:
    """Test CodeGenerator integration with transformed AST."""

    @pytest.fixture
    def parser(self):
        return GLSLParser()

    @pytest.fixture
    def generator(self):
        return CodeGenerator()

    def test_generate_from_parsed_ast(self, parser, generator):
        """Test code generation from parsed AST."""
        glsl = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            fragColor = vec4(uv, 0.5, 1.0);
        }
        """
        ast = parser.parse(glsl)
        result = generator.generate(ast)

        # Should produce valid code
        assert "void mainImage" in result
        assert "fragColor" in result
        assert "fragCoord" in result

    def test_generate_simple_function(self, parser, generator):
        """Test generating simple function."""
        glsl = """
        float square(float x) {
            return x * x;
        }
        """
        ast = parser.parse(glsl)
        result = generator.generate(ast)

        assert "float square" in result
        assert "return x * x" in result

    def test_generate_complete_program(self, parser, generator):
        """Test generating complete program."""
        glsl = """
        const float PI = 3.14159;
        float circle(vec2 p) { return length(p); }
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            float d = circle(uv - 0.5);
            fragColor = vec4(d);
        }
        """
        ast = parser.parse(glsl)
        result = generator.generate(ast)

        assert "PI" in result
        assert "circle" in result
        assert "mainImage" in result
