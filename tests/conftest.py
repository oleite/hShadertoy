"""
Pytest configuration and fixtures for hShadertoy transpiler tests.

This file provides reusable fixtures for:
- OpenCL context and queue management
- Transpiler instance creation
- Test data loading
- Kernel execution helpers
"""

import pytest
import pyopencl as cl
import numpy as np
from pathlib import Path


# ============================================================================
# OpenCL Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def opencl_context():
    """
    Create OpenCL context once per test session.

    Returns:
        pyopencl.Context: OpenCL context for kernel compilation and execution

    Note:
        Session scope means this is created once and shared across all tests.
        This improves performance but tests must not modify context state.
    """
    try:
        platforms = cl.get_platforms()
        if not platforms:
            pytest.skip("No OpenCL platforms found")

        # Use first available device (usually CPU or integrated GPU)
        platform = platforms[0]
        devices = platform.get_devices()

        if not devices:
            pytest.skip("No OpenCL devices found")

        context = cl.Context([devices[0]])
        return context

    except Exception as e:
        pytest.skip(f"OpenCL initialization failed: {e}")


@pytest.fixture
def opencl_queue(opencl_context):
    """
    Create OpenCL command queue for each test.

    Args:
        opencl_context: Session-scoped OpenCL context

    Returns:
        pyopencl.CommandQueue: Command queue for kernel execution

    Note:
        Function scope means each test gets a fresh queue.
        This ensures test isolation.
    """
    return cl.CommandQueue(opencl_context)


@pytest.fixture
def opencl_device_info(opencl_context):
    """
    Get information about the OpenCL device being used for testing.

    Returns:
        dict: Device information (name, type, vendor, etc.)
    """
    device = opencl_context.devices[0]
    return {
        "name": device.name,
        "type": cl.device_type.to_string(device.type),
        "vendor": device.vendor,
        "version": device.version,
        "driver_version": device.driver_version,
        "max_work_group_size": device.max_work_group_size,
        "max_compute_units": device.max_compute_units,
    }


# ============================================================================
# Transpiler Fixtures (will be implemented in future)
# ============================================================================

@pytest.fixture
def transpiler():
    """
    Create transpiler instance for testing.

    Returns:
        GLSLToOpenCLTranspiler: Fresh transpiler instance

    Note:
        This fixture will be implemented once the transpiler class exists.
        For now, it's a placeholder to document the intended API.
    """
    pytest.skip("Transpiler not yet implemented")
    # Future implementation:
    # from glsl_to_opencl import GLSLToOpenCLTranspiler
    # return GLSLToOpenCLTranspiler()


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def test_fixtures_dir():
    """
    Get path to test fixtures directory.

    Returns:
        Path: Absolute path to tests/fixtures/
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_shaders_dir(test_fixtures_dir):
    """
    Get path to simple test shaders.

    Returns:
        Path: Absolute path to tests/fixtures/simple_shaders/
    """
    return test_fixtures_dir / "simple_shaders"


@pytest.fixture
def complex_shaders_dir(test_fixtures_dir):
    """
    Get path to complex test shaders.

    Returns:
        Path: Absolute path to tests/fixtures/complex_shaders/
    """
    return test_fixtures_dir / "complex_shaders"


@pytest.fixture
def reference_images_dir(test_fixtures_dir):
    """
    Get path to reference images for visual comparison.

    Returns:
        Path: Absolute path to tests/fixtures/reference_images/
    """
    return test_fixtures_dir / "reference_images"


# ============================================================================
# Example Shader Fixtures
# ============================================================================

@pytest.fixture
def basic_glsl_shader():
    """
    Minimal valid GLSL fragment shader for basic testing.

    Returns:
        str: GLSL source code
    """
    return """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Solid color shader - simplest possible
    fragColor = vec4(1.0, 0.5, 0.0, 1.0);
}
"""


@pytest.fixture
def vec_operations_shader():
    """
    GLSL shader with vector operations for type conversion testing.

    Returns:
        str: GLSL source code
    """
    return """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = vec3(uv, 0.5);
    fragColor = vec4(col, 1.0);
}
"""


# ============================================================================
# Helper Functions
# ============================================================================

def compile_opencl_kernel(context, source, kernel_name="test_kernel"):
    """
    Compile OpenCL kernel source and return program.

    Args:
        context: OpenCL context
        source: OpenCL C source code
        kernel_name: Name of kernel function (for error messages)

    Returns:
        pyopencl.Program: Compiled program

    Raises:
        RuntimeError: If compilation fails
    """
    try:
        program = cl.Program(context, source).build()
        return program
    except cl.RuntimeError as e:
        raise RuntimeError(f"OpenCL compilation failed for {kernel_name}: {e}")


def execute_simple_kernel(context, queue, program, kernel_name, output_size):
    """
    Execute a simple kernel that writes to an output buffer.

    Args:
        context: OpenCL context
        queue: OpenCL command queue
        program: Compiled OpenCL program
        kernel_name: Name of kernel to execute
        output_size: Size of output buffer (number of float4 elements)

    Returns:
        numpy.ndarray: Output data from kernel
    """
    # Create output buffer
    output = np.zeros(output_size * 4, dtype=np.float32)
    output_buf = cl.Buffer(
        context,
        cl.mem_flags.WRITE_ONLY,
        output.nbytes
    )

    # Execute kernel
    kernel = getattr(program, kernel_name)
    kernel(queue, (output_size,), None, output_buf)

    # Read results
    cl.enqueue_copy(queue, output, output_buf).wait()

    return output.reshape(-1, 4)


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_configure(config):
    """
    Pytest configuration hook - runs once at startup.

    Can be used to:
    - Set up test database
    - Initialize logging
    - Register custom markers
    """
    # Print test session info
    print("\n" + "=" * 70)
    print("hShadertoy GLSL to OpenCL Transpiler - Test Suite")
    print("=" * 70)


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection - runs after test discovery.

    Can be used to:
    - Add markers automatically
    - Skip tests based on conditions
    - Reorder tests
    """
    # Auto-mark tests based on file location
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark OpenCL tests
        if "opencl" in item.name.lower():
            item.add_marker(pytest.mark.opencl)
