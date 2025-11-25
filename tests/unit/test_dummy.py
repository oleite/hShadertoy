"""
Dummy test to verify pytest infrastructure is working correctly.

This test should be the ONLY test passing in Week 1.
It will be removed once real tests are added in Week 2.
"""

import pytest


def test_pytest_works():
    """
    Verify pytest can discover and run tests.

    This is a trivial test that always passes.
    Purpose: Confirm pytest infrastructure is operational.
    """
    assert True


def test_python_version():
    """
    Verify Python version is 3.11+ as required.

    Note: Specification requires Python 3.11.7, but 3.11+ is acceptable.
    """
    import sys
    version = sys.version_info
    assert version.major == 3
    assert version.minor >= 11, f"Python 3.11+ required, got {version.major}.{version.minor}"


@pytest.mark.unit
def test_imports_work():
    """
    Verify all required dependencies can be imported.
    """
    # Parser dependency
    import tree_sitter_glsl
    from tree_sitter import Language, Parser

    # Testing dependencies
    import pyopencl as cl
    import numpy as np

    # All imports successful
    assert True


@pytest.mark.opencl
def test_opencl_context_fixture(opencl_context):
    """
    Verify OpenCL context fixture works.

    Args:
        opencl_context: Session-scoped OpenCL context from conftest.py
    """
    assert opencl_context is not None
    assert len(opencl_context.devices) > 0


@pytest.mark.opencl
def test_opencl_queue_fixture(opencl_queue):
    """
    Verify OpenCL queue fixture works.

    Args:
        opencl_queue: Function-scoped OpenCL queue from conftest.py
    """
    assert opencl_queue is not None


@pytest.mark.opencl
def test_opencl_device_info(opencl_device_info):
    """
    Verify OpenCL device info fixture and print device details.

    This helps identify which OpenCL device is being used for testing.

    Args:
        opencl_device_info: Device information dict from conftest.py
    """
    print("\n" + "=" * 60)
    print("OpenCL Test Device Information:")
    print("=" * 60)
    for key, value in opencl_device_info.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    assert "name" in opencl_device_info
    assert "type" in opencl_device_info


def test_basic_glsl_shader_fixture(basic_glsl_shader):
    """
    Verify basic GLSL shader fixture provides valid shader source.

    Args:
        basic_glsl_shader: GLSL source from conftest.py
    """
    assert isinstance(basic_glsl_shader, str)
    assert len(basic_glsl_shader) > 0
    assert "mainImage" in basic_glsl_shader
    assert "fragColor" in basic_glsl_shader


def test_test_directories_exist(test_fixtures_dir, simple_shaders_dir,
                                 complex_shaders_dir, reference_images_dir):
    """
    Verify test fixture directories were created correctly.

    Args:
        test_fixtures_dir: Path to tests/fixtures/
        simple_shaders_dir: Path to tests/fixtures/simple_shaders/
        complex_shaders_dir: Path to tests/fixtures/complex_shaders/
        reference_images_dir: Path to tests/fixtures/reference_images/
    """
    assert test_fixtures_dir.exists()
    assert simple_shaders_dir.exists()
    assert complex_shaders_dir.exists()
    assert reference_images_dir.exists()


@pytest.mark.skip(reason="Transpiler not yet implemented (Phase 1, Week 1)")
def test_transpiler_fixture_placeholder(transpiler):
    """
    Placeholder test for transpiler fixture.

    This test is skipped because the transpiler doesn't exist yet.
    It will be enabled in Week 2+ when parser development begins.

    Args:
        transpiler: Transpiler instance from conftest.py (not yet implemented)
    """
    # This test should skip automatically due to fixture skip
    assert transpiler is not None
