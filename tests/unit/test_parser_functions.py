"""
Function parsing tests - Function definitions and calls.

Tests parsing of GLSL functions, especially Shadertoy's mainImage().
"""

import pytest
from glsl_to_opencl.parser import GLSLParser, FunctionDefinition


class TestSimpleFunctions:
    """Test parsing of simple function definitions."""

    def test_parse_void_main(self):
        """Test parsing void main() function."""
        source = "void main() { }"
        parser = GLSLParser()
        ast = parser.parse(source)

        functions = ast.get_functions()
        assert len(functions) == 1
        assert functions[0].name == "main"

    def test_parse_empty_function(self):
        """Test parsing empty function."""
        source = "void foo() { }"
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert func.name == "foo"
        assert func.return_type.text == "void"

    def test_parse_function_return_float(self):
        """Test function returning float."""
        source = "float getValue() { return 1.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert func.name == "getValue"
        assert "float" in func.return_type.text

    def test_parse_function_return_vec3(self):
        """Test function returning vec3."""
        source = "vec3 getColor() { return vec3(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert func.name == "getColor"
        assert "vec3" in func.return_type.text


class TestFunctionParameters:
    """Test parsing of function parameters."""

    def test_parse_one_parameter(self):
        """Test function with one parameter."""
        source = "float square(float x) { return x * x; }"
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert func.name == "square"
        assert len(func.parameters) == 1

    def test_parse_two_parameters(self):
        """Test function with two parameters."""
        source = "float add(float a, float b) { return a + b; }"
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert len(func.parameters) == 2

    def test_parse_vec_parameter(self):
        """Test function with vector parameter."""
        source = "float length2(vec3 v) { return dot(v, v); }"
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert len(func.parameters) == 1
        assert "vec3" in func.parameters[0].text

    def test_parse_in_parameter(self):
        """Test function with 'in' parameter qualifier."""
        source = "void process(in float x) { }"
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert "in" in func.parameters[0].text

    def test_parse_out_parameter(self):
        """Test function with 'out' parameter qualifier."""
        source = "void getValues(out float x) { x = 1.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert "out" in func.parameters[0].text

    def test_parse_inout_parameter(self):
        """Test function with 'inout' parameter qualifier."""
        source = "void modify(inout float x) { x += 1.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert "inout" in func.parameters[0].text


class TestShadertoyMainImage:
    """Test parsing of Shadertoy mainImage() function."""

    def test_parse_mainimage_signature(self):
        """Test mainImage with correct signature."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            fragColor = vec4(1.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        main = ast.get_main_image()
        assert main is not None
        assert main.name == "mainImage"

    def test_mainimage_return_type(self):
        """Test mainImage return type is void."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) { }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        main = ast.get_main_image()
        assert "void" in main.return_type.text

    def test_mainimage_parameters(self):
        """Test mainImage has 2 parameters."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) { }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        main = ast.get_main_image()
        assert len(main.parameters) == 2

    def test_mainimage_first_param_out_vec4(self):
        """Test first parameter is out vec4."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) { }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        main = ast.get_main_image()
        param0 = main.parameters[0]
        assert "out" in param0.text
        assert "vec4" in param0.text

    def test_mainimage_second_param_in_vec2(self):
        """Test second parameter is in vec2."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) { }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        main = ast.get_main_image()
        param1 = main.parameters[1]
        assert "in" in param1.text
        assert "vec2" in param1.text

    def test_mainimage_with_body(self):
        """Test mainImage with function body."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            fragColor = vec4(1.0, 0.5, 0.0, 1.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        main = ast.get_main_image()
        assert main.body is not None
        assert main.body.type == "compound_statement"


class TestMultipleFunctions:
    """Test parsing shaders with multiple functions."""

    def test_parse_two_functions(self):
        """Test parsing two functions."""
        source = """
        float helper() { return 1.0; }
        void main() { }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        functions = ast.get_functions()
        assert len(functions) == 2

    def test_parse_helper_and_mainimage(self):
        """Test helper function before mainImage."""
        source = """
        float noise(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            float n = noise(fragCoord);
            fragColor = vec4(n);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        functions = ast.get_functions()
        assert len(functions) == 2

        main = ast.get_main_image()
        assert main is not None

    def test_multiple_helpers(self):
        """Test multiple helper functions."""
        source = """
        float hash(float n) { return fract(sin(n) * 43758.5453); }
        vec2 hash2(vec2 p) { return vec2(hash(p.x), hash(p.y)); }
        vec3 hash3(vec3 p) { return vec3(hash(p.x), hash(p.y), hash(p.z)); }

        void mainImage(out vec4 fragColor, in vec2 fragCoord) { }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        functions = ast.get_functions()
        assert len(functions) == 4


class TestFunctionCalls:
    """Test parsing of function call expressions."""

    def test_parse_simple_call(self):
        """Test simple function call."""
        source = """
        void main() {
            float x = sin(1.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        # Should parse successfully
        assert ast is not None

    def test_parse_vec3_constructor_call(self):
        """Test vec3 constructor call."""
        source = """
        void main() {
            vec3 color = vec3(1.0, 0.5, 0.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None

    def test_parse_texture_call(self):
        """Test texture() function call."""
        source = """
        uniform sampler2D tex;
        void main() {
            vec4 color = texture(tex, vec2(0.5));
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None

    def test_parse_nested_calls(self):
        """Test nested function calls."""
        source = """
        void main() {
            float result = sin(cos(1.0));
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        assert ast is not None


class TestFunctionBodies:
    """Test parsing of function bodies."""

    def test_empty_body(self):
        """Test empty function body."""
        source = "void main() { }"
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert func.body is not None

    def test_single_statement_body(self):
        """Test body with one statement."""
        source = """
        void main() {
            float x = 1.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert func.body is not None

    def test_multiple_statements_body(self):
        """Test body with multiple statements."""
        source = """
        void main() {
            float x = 1.0;
            float y = 2.0;
            float z = x + y;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert func.body is not None

    def test_return_statement(self):
        """Test function with return statement."""
        source = """
        float getValue() {
            return 42.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        func = ast.get_functions()[0]
        assert "return" in func.body.text
