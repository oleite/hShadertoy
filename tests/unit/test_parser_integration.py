"""
Integration tests - Complete shader parsing.

Tests parsing of complete, realistic GLSL shaders.
"""

import pytest
from glsl_to_opencl.parser import GLSLParser


class TestCompleteShaders:
    """Test parsing of complete shaders."""

    def test_parse_solid_color_shader(self):
        """Test simplest possible shader - solid color."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            fragColor = vec4(1.0, 0.5, 0.0, 1.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None
        main = ast.get_main_image()
        assert main is not None
        assert main.name == "mainImage"

    def test_parse_uv_gradient_shader(self):
        """Test UV-based gradient shader."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            vec3 col = vec3(uv, 0.5);
            fragColor = vec4(col, 1.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None
        main = ast.get_main_image()
        assert main is not None

    def test_parse_shader_with_helper_function(self):
        """Test shader with helper function."""
        source = """
        float hash(float n) {
            return fract(sin(n) * 43758.5453);
        }

        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            float n = hash(fragCoord.x);
            fragColor = vec4(n);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None
        functions = ast.get_functions()
        assert len(functions) == 2

    def test_parse_shader_with_defines(self):
        """Test shader with #define macros."""
        source = """
        #define PI 3.14159
        #define MAX_STEPS 100

        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            float angle = PI * 2.0;
            fragColor = vec4(angle / PI);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None

    def test_parse_shader_with_texture_sampling(self):
        """Test shader with texture sampling."""
        source = """
        uniform sampler2D iChannel0;

        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            vec4 tex = texture(iChannel0, uv);
            fragColor = tex;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None


class TestShadertoyExamples:
    """Test parsing of real Shadertoy-style shaders."""

    def test_parse_basic_example(self):
        """Test basic Shadertoy example."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
            fragColor = vec4(col, 1.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None

    def test_parse_with_multiple_uniforms(self):
        """Test shader using multiple Shadertoy uniforms."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            float t = iTime;
            vec4 mouse = iMouse;

            vec3 col = vec3(uv, sin(t));
            fragColor = vec4(col, 1.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None

    def test_parse_noise_function(self):
        """Test shader with noise function."""
        source = """
        float noise(vec2 p) {
            return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
        }

        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            float n = noise(uv * 10.0 + iTime);
            fragColor = vec4(vec3(n), 1.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None
        functions = ast.get_functions()
        assert len(functions) == 2

    def test_parse_raymarching_template(self):
        """Test simple raymarching structure."""
        source = """
        float sdSphere(vec3 p, float r) {
            return length(p) - r;
        }

        float map(vec3 p) {
            return sdSphere(p, 1.0);
        }

        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
            vec3 ro = vec3(0.0, 0.0, -3.0);
            vec3 rd = normalize(vec3(uv, 1.0));

            float t = 0.0;
            for (int i = 0; i < 100; i++) {
                vec3 p = ro + rd * t;
                float d = map(p);
                if (d < 0.001) break;
                t += d;
            }

            vec3 col = vec3(t * 0.1);
            fragColor = vec4(col, 1.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None
        functions = ast.get_functions()
        assert len(functions) >= 3


class TestControlFlow:
    """Test parsing of control flow statements."""

    def test_parse_if_statement(self):
        """Test if statement."""
        source = """
        void main() {
            if (x > 0.0) {
                y = 1.0;
            }
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_if_else_statement(self):
        """Test if-else statement."""
        source = """
        void main() {
            if (x > 0.0) {
                y = 1.0;
            } else {
                y = -1.0;
            }
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_for_loop(self):
        """Test for loop."""
        source = """
        void main() {
            for (int i = 0; i < 10; i++) {
                sum += i;
            }
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_while_loop(self):
        """Test while loop."""
        source = """
        void main() {
            while (x < 10.0) {
                x += 1.0;
            }
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_do_while_loop(self):
        """Test do-while loop."""
        source = """
        void main() {
            do {
                x += 1.0;
            } while (x < 10.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_break_statement(self):
        """Test break statement."""
        source = """
        void main() {
            for (int i = 0; i < 100; i++) {
                if (found) break;
            }
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_continue_statement(self):
        """Test continue statement."""
        source = """
        void main() {
            for (int i = 0; i < 100; i++) {
                if (skip) continue;
                process(i);
            }
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestComments:
    """Test parsing with comments."""

    def test_parse_with_single_line_comment(self):
        """Test single-line comment."""
        source = """
        void main() {
            // This is a comment
            float x = 1.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_with_multi_line_comment(self):
        """Test multi-line comment."""
        source = """
        void main() {
            /*
             * Multi-line comment
             * spanning multiple lines
             */
            float x = 1.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_comment_before_function(self):
        """Test comment before function."""
        source = """
        // Helper function
        float getValue() { return 1.0; }

        // Main entry point
        void main() { }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None
