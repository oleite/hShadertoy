"""
Integration tests for shader compilation with PyOpenCL.

Tests that all transpiled shaders compile successfully with OpenCL.
Validates the complete pipeline: GLSL -> Transpile -> OpenCL -> Compile.

Session 10: Simple Shaders Integration
"""

import sys
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pyopencl as cl
    HAS_PYOPENCL = True
except ImportError:
    HAS_PYOPENCL = False

from tests.transpile import transpile


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def opencl_context():
    """Create OpenCL context for tests."""
    if not HAS_PYOPENCL:
        pytest.skip("PyOpenCL not available")

    platforms = cl.get_platforms()
    if not platforms:
        pytest.skip("No OpenCL platforms available")

    devices = platforms[0].get_devices()
    if not devices:
        pytest.skip("No OpenCL devices available")

    return cl.Context([devices[0]])


@pytest.fixture(scope="module")
def main_header():
    """Load main header file."""
    header_path = Path(__file__).parent.parent / "ocl" / "main_header.cl"
    return header_path.read_text()


@pytest.fixture(scope="module")
def main_kernel():
    """Load main kernel file."""
    kernel_path = Path(__file__).parent.parent / "ocl" / "main_kernel.cl"
    return kernel_path.read_text()


@pytest.fixture(scope="module")
def build_options():
    """Get OpenCL build options."""
    # These are the standard build options for hShadertoy
    houdini_include = "C:/PROGRA~1/SIDEEF~1/HOUDIN~1.440/houdini/ocl/include"
    project_include = str(Path(__file__).parent.parent.parent / "houdini" / "ocl" / "include")
    test_include = str(Path(__file__).parent.parent / "ocl")

    return (
        f"-I {project_include} "
        f"-I {houdini_include} "
        f"-I {test_include} "
        "-D __H_CPU__ "
        "-D __H_INTEL__ "
        "-DFILTER_BOX "
        "-D_RUNOVER_LAYER=_bound_fragColor_layer "
        "-DHAS_size_ref "
        "-DALIGNED_size_ref "
        "-DCONSTANT_size_ref "
        "-D_bound_size_ref_border=IMX_WRAP "
        "-D_bound_size_ref_storage=FLOAT32 "
        "-D_bound_size_ref_channels=1 "
        "-D_bound_fragColor_storage=FLOAT32 "
        "-D_bound_fragColor_channels=4"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def read_shader(shader_name: str) -> str:
    """Read GLSL shader source from tests/shaders/simple/."""
    shader_path = Path(__file__).parent.parent / "shaders" / "simple" / f"{shader_name}.glsl"
    return shader_path.read_text()


def compile_shader(
    shader_name: str,
    opencl_context,
    main_header: str,
    main_kernel: str,
    build_options: str
):
    """
    Transpile and compile a shader.

    Returns:
        tuple: (program, kernel_source_length, transpile_result)
    """
    # Transpile GLSL to OpenCL
    glsl_source = read_shader(shader_name)
    result = transpile(glsl_source)

    # Construct full kernel source
    kernel_source = (
        main_header +
        "\n" +
        result.get_header() +
        "\n" +
        main_kernel +
        "\n" +
        result.get_kernel() +
        "\nAT_fragColor_set(fragColor);}"  # Close main kernel
    )

    # Compile with OpenCL
    program = cl.Program(opencl_context, kernel_source).build(options=build_options)

    return program, len(kernel_source), result


# ============================================================================
# Individual Shader Compilation Tests
# ============================================================================

@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_vignette(opencl_context, main_header, main_kernel, build_options):
    """Test vignette.glsl compiles successfully."""
    program, source_len, result = compile_shader(
        "vignette", opencl_context, main_header, main_kernel, build_options
    )

    assert program is not None
    assert source_len > 0
    assert "generickernel" in [k.function_name for k in program.all_kernels()]


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_gradient(opencl_context, main_header, main_kernel, build_options):
    """Test gradient.glsl compiles successfully."""
    program, source_len, result = compile_shader(
        "gradient", opencl_context, main_header, main_kernel, build_options
    )

    assert program is not None
    assert source_len > 0
    assert "generickernel" in [k.function_name for k in program.all_kernels()]


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_hexagonal(opencl_context, main_header, main_kernel, build_options):
    """Test hexagonal.glsl compiles successfully."""
    program, source_len, result = compile_shader(
        "hexagonal", opencl_context, main_header, main_kernel, build_options
    )

    assert program is not None
    assert source_len > 0
    assert "generickernel" in [k.function_name for k in program.all_kernels()]


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_ripples(opencl_context, main_header, main_kernel, build_options):
    """Test ripples.glsl compiles successfully."""
    program, source_len, result = compile_shader(
        "ripples", opencl_context, main_header, main_kernel, build_options
    )

    assert program is not None
    assert source_len > 0
    assert "generickernel" in [k.function_name for k in program.all_kernels()]


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_silexars(opencl_context, main_header, main_kernel, build_options):
    """Test silexars.glsl compiles successfully (with preprocessor)."""
    program, source_len, result = compile_shader(
        "silexars", opencl_context, main_header, main_kernel, build_options
    )

    assert program is not None
    assert source_len > 0
    assert "generickernel" in [k.function_name for k in program.all_kernels()]


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_sand(opencl_context, main_header, main_kernel, build_options):
    """Test sand.glsl compiles successfully (with helper functions)."""
    program, source_len, result = compile_shader(
        "sand", opencl_context, main_header, main_kernel, build_options
    )

    assert program is not None
    assert source_len > 0
    assert "generickernel" in [k.function_name for k in program.all_kernels()]


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_caustic(opencl_context, main_header, main_kernel, build_options):
    """Test caustic.glsl compiles successfully (with #ifdef)."""
    program, source_len, result = compile_shader(
        "caustic", opencl_context, main_header, main_kernel, build_options
    )

    assert program is not None
    assert source_len > 0
    assert "generickernel" in [k.function_name for k in program.all_kernels()]


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_colorful(opencl_context, main_header, main_kernel, build_options):
    """Test colorful.glsl compiles successfully (complex with nested loops)."""
    program, source_len, result = compile_shader(
        "colorful", opencl_context, main_header, main_kernel, build_options
    )

    assert program is not None
    assert source_len > 0
    assert "generickernel" in [k.function_name for k in program.all_kernels()]


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_warping(opencl_context, main_header, main_kernel, build_options):
    """Test warping.glsl compiles successfully (most complex with mat2)."""
    program, source_len, result = compile_shader(
        "warping", opencl_context, main_header, main_kernel, build_options
    )

    assert program is not None
    assert source_len > 0
    assert "generickernel" in [k.function_name for k in program.all_kernels()]


# ============================================================================
# Batch Compilation Tests
# ============================================================================

@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_all_shaders_compile(opencl_context, main_header, main_kernel, build_options):
    """Test that all 9 shaders compile successfully."""
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

    compiled_count = 0
    failed_shaders = []

    for shader_name in shaders:
        try:
            program, source_len, result = compile_shader(
                shader_name, opencl_context, main_header, main_kernel, build_options
            )
            assert program is not None
            compiled_count += 1
        except Exception as e:
            failed_shaders.append((shader_name, str(e)))

    # All shaders should compile
    assert compiled_count == len(shaders), \
        f"Only {compiled_count}/{len(shaders)} shaders compiled. Failed: {failed_shaders}"


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_shader_compilation_metrics(opencl_context, main_header, main_kernel, build_options):
    """Test compilation metrics for all shaders."""
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

    metrics = {}

    for shader_name in shaders:
        program, source_len, result = compile_shader(
            shader_name, opencl_context, main_header, main_kernel, build_options
        )

        metrics[shader_name] = {
            "compiled": program is not None,
            "source_length": source_len,
            "header_length": len(result.get_header()),
            "kernel_length": len(result.get_kernel()),
        }

    # Verify all compiled
    for shader_name, metric in metrics.items():
        assert metric["compiled"], f"{shader_name} failed to compile"
        assert metric["source_length"] > 0, f"{shader_name} has empty source"


# ============================================================================
# Feature-Specific Compilation Tests
# ============================================================================

@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_with_preprocessor_directives(
    opencl_context, main_header, main_kernel, build_options
):
    """Test compilation of shaders with preprocessor directives."""
    # silexars has #define
    program1, _, _ = compile_shader(
        "silexars", opencl_context, main_header, main_kernel, build_options
    )
    assert program1 is not None

    # caustic has #define and #ifdef
    program2, _, _ = compile_shader(
        "caustic", opencl_context, main_header, main_kernel, build_options
    )
    assert program2 is not None


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_with_helper_functions(
    opencl_context, main_header, main_kernel, build_options
):
    """Test compilation of shaders with multiple helper functions."""
    # sand has helper functions
    program1, _, _ = compile_shader(
        "sand", opencl_context, main_header, main_kernel, build_options
    )
    assert program1 is not None

    # warping has many helper functions
    program2, _, _ = compile_shader(
        "warping", opencl_context, main_header, main_kernel, build_options
    )
    assert program2 is not None


@pytest.mark.skipif(not HAS_PYOPENCL, reason="PyOpenCL not available")
def test_compile_with_matrix_operations(
    opencl_context, main_header, main_kernel, build_options
):
    """Test compilation of shader with matrix operations (mat2)."""
    # warping has mat2 multiplication
    program, _, _ = compile_shader(
        "warping", opencl_context, main_header, main_kernel, build_options
    )
    assert program is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
