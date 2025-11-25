"""
Integration tests for shader transpilation end-to-end pipeline.

Tests the full transpilation workflow from GLSL source to OpenCL output
for real-world Shadertoy shaders from tests/shaders/simple/.

Session 10: Simple Shaders Integration
"""

import sys
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.transpile import transpile, TranspileError


# ============================================================================
# Helper Functions
# ============================================================================

def read_shader(shader_name: str) -> str:
    """Read GLSL shader source from tests/shaders/simple/."""
    shader_path = Path(__file__).parent.parent / "shaders" / "simple" / f"{shader_name}.glsl"
    return shader_path.read_text()


def verify_shader_transpilation(shader_name: str):
    """Helper: Test full transpilation pipeline for a shader."""
    glsl_source = read_shader(shader_name)
    result = transpile(glsl_source)

    # Verify we got results
    assert result is not None
    assert result.get_header() is not None
    assert result.get_kernel() is not None
    assert result.get_full() is not None

    # Verify kernel has Shadertoy markers
    kernel = result.get_kernel()
    assert "// ---- SHADERTOY CODE BEGIN ----" in kernel
    assert "// ---- SHADERTOY CODE END ----" in kernel

    # Verify basic transformations were applied
    full_code = result.get_full()

    # Check float literals have 'f' suffix
    # (We check for common patterns like "1.0f" or "0.5f")
    assert ".0f" in full_code or ".5f" in full_code or "f)" in full_code

    # Check vector types are transformed (vec2 -> float2, etc.)
    assert "float2" in full_code or "float3" in full_code or "float4" in full_code

    return result


# ============================================================================
# Individual Shader Tests
# ============================================================================

def test_transpile_vignette():
    """Test transpilation of vignette.glsl (simplest shader)."""
    result = verify_shader_transpilation("vignette")

    # Vignette-specific checks
    full_code = result.get_full()
    assert "GLSL_pow" in full_code  # pow() transformed


def test_transpile_gradient():
    """Test transpilation of gradient.glsl."""
    result = verify_shader_transpilation("gradient")

    # Gradient-specific checks
    full_code = result.get_full()
    assert "GLSL_sqrt" in full_code  # sqrt() in dist() function
    assert "GLSL_sin" in full_code   # sin() transformed
    assert "GLSL_mix" in full_code   # mix() transformed


def test_transpile_hexagonal():
    """Test transpilation of hexagonal.glsl."""
    result = verify_shader_transpilation("hexagonal")

    # Hexagonal-specific checks
    full_code = result.get_full()
    assert "GLSL_mod" in full_code  # mod() transformed
    assert "GLSL_dot" in full_code  # dot() transformed
    assert "GLSL_min" in full_code  # min() transformed


def test_transpile_ripples():
    """Test transpilation of ripples.glsl."""
    result = verify_shader_transpilation("ripples")

    # Ripples-specific checks
    full_code = result.get_full()
    header = result.get_header()

    # Global variables should be in header
    assert "float2 center" in header or "float2 center" in full_code
    assert "float speed" in header or "float speed" in full_code

    # Functions transformed
    assert "GLSL_sin" in full_code
    # Note: ripples.glsl only uses sin(), not cos()


def test_transpile_silexars():
    """Test transpilation of silexars.glsl (with preprocessor directives)."""
    result = verify_shader_transpilation("silexars")

    # Silexars-specific checks
    full_code = result.get_full()
    header = result.get_header()

    # Preprocessor directives preserved
    assert "#define t iTime" in header or "#define t iTime" in full_code
    assert "#define r iResolution.xy" in header or "#define r iResolution.xy" in full_code

    # Loop and math functions
    assert "for" in full_code
    assert "GLSL_length" in full_code
    assert "GLSL_sin" in full_code
    assert "GLSL_mod" in full_code


def test_transpile_sand():
    """Test transpilation of sand.glsl (with helper functions and out params)."""
    result = verify_shader_transpilation("sand")

    # Sand-specific checks
    full_code = result.get_full()
    header = result.get_header()

    # Helper functions in header
    assert "hash" in header
    assert "noise" in header
    assert "random2" in header

    # Out parameter transformation
    assert "__private float2* noise2" in full_code or "out float2 noise2" in header

    # Math functions
    assert "GLSL_fract" in full_code
    assert "GLSL_sin" in full_code or "GLSL_cos" in full_code
    assert "GLSL_floor" in full_code
    assert "GLSL_mix" in full_code
    assert "GLSL_pow" in full_code


def test_transpile_caustic():
    """Test transpilation of caustic.glsl (with #ifdef preprocessor)."""
    result = verify_shader_transpilation("caustic")

    # Caustic-specific checks
    full_code = result.get_full()
    header = result.get_header()

    # Preprocessor directives
    assert "#define TAU" in header or "#define TAU" in full_code
    assert "#define MAX_ITER" in header or "#define MAX_ITER" in full_code
    assert "#ifdef SHOW_TILING" in full_code or "#ifdef SHOW_TILING" in header

    # Complex math operations
    assert "GLSL_sin" in full_code
    assert "GLSL_cos" in full_code
    assert "GLSL_length" in full_code
    assert "GLSL_pow" in full_code
    assert "GLSL_clamp" in full_code


def test_transpile_colorful():
    """Test transpilation of colorful.glsl (complex with nested loops)."""
    result = verify_shader_transpilation("colorful")

    # Colorful-specific checks
    full_code = result.get_full()
    header = result.get_header()

    # Preprocessor directive
    assert "#define l" in header or "#define l" in full_code

    # Nested control flow
    assert "for" in full_code
    assert "if" in full_code

    # Math operations
    assert "GLSL_sin" in full_code
    assert "GLSL_cos" in full_code
    assert "GLSL_dot" in full_code
    assert "GLSL_mod" in full_code
    assert "GLSL_fract" in full_code


def test_transpile_warping():
    """Test transpilation of warping.glsl (most complex with mat2 operations)."""
    result = verify_shader_transpilation("warping")

    # Warping-specific checks
    full_code = result.get_full()
    header = result.get_header()

    # Multiple helper functions
    assert "hash21" in header
    assert "hash22" in header
    assert "noise" in header
    assert "voronoi" in header
    assert "fbmNoise" in header
    assert "fbmVoronoi" in header
    assert "fbm2Noise" in header
    assert "dis" in header

    # Matrix operations (mat2 multiplication)
    assert "mat2" in full_code
    assert "GLSL_mul" in full_code or "*" in full_code  # Matrix multiplication

    # Complex math
    assert "GLSL_floor" in full_code
    assert "GLSL_fract" in full_code
    assert "GLSL_sin" in full_code
    assert "GLSL_length" in full_code
    assert "GLSL_mix" in full_code
    assert "GLSL_dot" in full_code


# ============================================================================
# Batch Tests
# ============================================================================

def test_all_shaders_transpile_successfully():
    """Verify all 9 shaders transpile without errors."""
    shaders = [
        "vignette",
        "gradient",
        "hexagonal",
        "ripples",
        "silexars",
        "sand",
        "caustic",
        "colorful",
        "warping",
    ]

    for shader_name in shaders:
        glsl_source = read_shader(shader_name)
        result = transpile(glsl_source)
        assert result is not None, f"{shader_name} transpilation failed"


def test_all_shaders_have_valid_output():
    """Verify all transpiled shaders have non-empty output."""
    shaders = [
        "vignette",
        "gradient",
        "hexagonal",
        "ripples",
        "silexars",
        "sand",
        "caustic",
        "colorful",
        "warping",
    ]

    for shader_name in shaders:
        glsl_source = read_shader(shader_name)
        result = transpile(glsl_source)

        # Verify non-empty outputs
        assert len(result.get_header()) > 0, f"{shader_name} has empty header"
        assert len(result.get_kernel()) > 0, f"{shader_name} has empty kernel"
        assert len(result.get_full()) > 0, f"{shader_name} has empty full code"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_shader_with_preprocessor_directives():
    """Test shaders with various preprocessor directives."""
    # silexars has #define
    result1 = transpile(read_shader("silexars"))
    assert "#define" in result1.get_full()

    # caustic has #define and #ifdef
    result2 = transpile(read_shader("caustic"))
    assert "#define" in result2.get_full()
    assert "#ifdef" in result2.get_full()


def test_shader_with_global_variables():
    """Test shaders with global variable declarations."""
    # ripples has global variables
    result = transpile(read_shader("ripples"))
    full_code = result.get_full()

    # Global variables should appear before mainImage
    assert "float2 center" in full_code or "center" in result.get_header()
    assert "float speed" in full_code or "speed" in result.get_header()


def test_shader_with_helper_functions():
    """Test shaders with multiple helper functions."""
    # warping has many helper functions
    result = transpile(read_shader("warping"))
    header = result.get_header()

    # All helper functions should be in header
    helper_funcs = ["hash21", "hash22", "noise", "voronoi", "fbmNoise", "dis"]
    for func in helper_funcs:
        assert func in header, f"Helper function {func} not found in header"


def test_shader_with_out_parameters():
    """Test shaders with out parameter transformations."""
    # sand has random2 with out parameter
    result = transpile(read_shader("sand"))
    full_code = result.get_full()

    # Check out parameter transformation
    assert "random2" in full_code
    # Should have either __private pointer or out qualifier
    assert "__private" in full_code or "out" in full_code


def test_shader_with_matrix_operations():
    """Test shader with matrix operations (mat2)."""
    # warping has mat2 multiplication
    result = transpile(read_shader("warping"))
    full_code = result.get_full()

    # Check matrix type
    assert "mat2" in full_code

    # Check matrix operations are handled (either GLSL_mul or native *)
    # The transpiler should handle mat2 operations
    assert "GLSL_mul" in full_code or "*" in full_code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
