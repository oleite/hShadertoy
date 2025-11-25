"""
Unit tests for Houdini transpiler module
Tests the transpile_glsl.py module functionality
"""
import sys
sys.path.insert(0, 'C:/dev/hShadertoy/houdini/scripts/python')
sys.path.insert(0, 'C:/dev/hShadertoy')

import pytest
from hshadertoy.transpiler.transpile_glsl import (
    transpile,
    TranspilationError,
    _detect_renderpass_type,
    _split_header_and_body,
    _format_for_houdini
)


class TestRenderpassDetection:
    """Test renderpass type detection"""

    def test_detect_mainImage(self):
        glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { }"
        assert _detect_renderpass_type(glsl) == "mainImage"

    def test_detect_mainCubemap(self):
        glsl = "void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir) { }"
        assert _detect_renderpass_type(glsl) == "mainCubemap"

    def test_detect_mainSound(self):
        glsl = "vec2 mainSound(in int sampleRate, in float time) { return vec2(0.0); }"
        assert _detect_renderpass_type(glsl) == "mainSound"

    def test_detect_common(self):
        glsl = "#define PI 3.14\nfloat helper() { return 1.0; }"
        assert _detect_renderpass_type(glsl) == "Common"

    def test_detect_common_empty(self):
        glsl = ""
        assert _detect_renderpass_type(glsl) == "Common"


class TestHeaderBodySplit:
    """Test splitting OpenCL code into header and body"""

    def test_split_simple_shader(self):
        opencl = """#define PI 3.14f

void mainImage(__private float4* fragColor, float2 fragCoord) {
    *fragColor = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
}"""
        header, body = _split_header_and_body(opencl, "mainImage")

        assert "#define PI 3.14f" in header
        assert "void mainImage" not in header
        assert "*fragColor = (float4)(1.0f, 0.0f, 0.0f, 1.0f);" in body

    def test_split_with_functions(self):
        opencl = """float4 helper(float x) {
    return (float4)(x, x, 0.0f, 1.0f);
}

void mainImage(__private float4* fragColor, float2 fragCoord) {
    *fragColor = helper(0.5f);
}"""
        header, body = _split_header_and_body(opencl, "mainImage")

        assert "float4 helper" in header
        assert "*fragColor = helper(0.5f);" in body

    def test_split_common_renderpass(self):
        opencl = """#define PI 3.14f
float helper() { return 1.0f; }"""
        header, body = _split_header_and_body(opencl, "Common")

        assert header == opencl
        assert body == ""


class TestHoudiniFormatting:
    """Test Houdini output formatting"""

    def test_format_simple_shader(self):
        header = "#define PI 3.14f"
        body = "*fragColor = (float4)(1.0f, 0.0f, 0.0f, 1.0f);"
        result = _format_for_houdini(header, body, "mainImage")

        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result
        assert "@fragColor.set(fragColor);" in result
        assert "#define PI 3.14f" in result
        # Pointer syntax should be removed by _fix_houdini_pointers
        assert "fragColor = (float4)(1.0f, 0.0f, 0.0f, 1.0f);" in result
        assert "*fragColor" not in result  # Verify pointer syntax was removed

    def test_format_common_renderpass(self):
        header = "#define PI 3.14f\nfloat helper() { return 1.0f; }"
        body = ""
        result = _format_for_houdini(header, body, "Common")

        assert "@KERNEL" not in result
        assert result == header

    def test_format_with_empty_header(self):
        header = ""
        body = "*fragColor = (float4)(1.0f, 0.0f, 0.0f, 1.0f);"
        result = _format_for_houdini(header, body, "mainImage")

        assert "@KERNEL" in result
        assert "// ---- HEADER:" not in result  # No header comment if empty


class TestTranspileFunction:
    """Test main transpile function"""

    def test_transpile_simple_mainImage(self):
        glsl = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = vec4(uv, 0.5, 1.0);
}
"""
        result = transpile(glsl)

        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result
        assert "@fragColor.set(fragColor);" in result
        assert "float2" in result
        assert "float4" in result

    def test_transpile_with_helper_function(self):
        glsl = """
vec4 helper(float x) {
    return vec4(x, x, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = helper(0.5);
}
"""
        result = transpile(glsl)

        assert "float4 helper" in result
        assert "@KERNEL" in result

    def test_transpile_with_define(self):
        glsl = """
#define PI 3.14

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(PI, 0.0, 0.0, 1.0);
}
"""
        result = transpile(glsl)

        assert "#define PI 3.14f" in result
        assert "@KERNEL" in result

    def test_transpile_common_renderpass(self):
        glsl = """
#define PI 3.14
float helper() { return 1.0; }
"""
        result = transpile(glsl)

        assert "@KERNEL" not in result
        assert "#define PI 3.14f" in result

    def test_transpile_auto_detect_mode(self):
        glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }"
        result = transpile(glsl, mode=None)  # Auto-detect

        assert "@KERNEL" in result

    def test_transpile_explicit_mode(self):
        glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }"
        result = transpile(glsl, mode="mainImage")  # Explicit

        assert "@KERNEL" in result

    def test_transpile_invalid_syntax_raises_error(self):
        glsl = "this is not valid GLSL code @#$%"
        with pytest.raises(TranspilationError):
            transpile(glsl)


class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_empty_body(self):
        glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { }"
        result = transpile(glsl)

        assert "@KERNEL" in result
        assert "@fragColor.set(fragColor);" in result

    def test_multiline_function_signature(self):
        glsl = """
void mainImage(
    out vec4 fragColor,
    in vec2 fragCoord
) {
    fragColor = vec4(1.0);
}
"""
        result = transpile(glsl)

        assert "@KERNEL" in result

    def test_nested_braces(self):
        glsl = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    if (fragCoord.x > 0.5) {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        fragColor = vec4(0.0, 1.0, 0.0, 1.0);
    }
}
"""
        result = transpile(glsl)

        assert "@KERNEL" in result
        assert "if" in result


if __name__ == "__main__":
    # Allow running tests directly with: hython tests/unit/test_houdini_transpiler.py
    pytest.main([__file__, "-v"])
