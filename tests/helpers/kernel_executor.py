"""
OpenCL Kernel Execution Helpers

Utilities for executing OpenCL kernels during testing.
These helpers will be used extensively in Phase 4+ for validating
generated OpenCL code.
"""

import pyopencl as cl
import numpy as np
from typing import Tuple, Optional


class KernelExecutor:
    """
    Helper class for executing OpenCL kernels in tests.

    Usage:
        executor = KernelExecutor(context, queue)
        result = executor.execute_simple(kernel_source, output_size=1024)
    """

    def __init__(self, context: cl.Context, queue: cl.CommandQueue):
        """
        Initialize kernel executor.

        Args:
            context: OpenCL context
            queue: OpenCL command queue
        """
        self.context = context
        self.queue = queue

    def compile(self, source: str, build_options: str = "") -> cl.Program:
        """
        Compile OpenCL kernel source.

        Args:
            source: OpenCL C source code
            build_options: Compiler flags (e.g., "-I /path/to/includes")

        Returns:
            Compiled OpenCL program

        Raises:
            RuntimeError: If compilation fails
        """
        try:
            program = cl.Program(self.context, source).build(options=build_options)
            return program
        except cl.RuntimeError as e:
            raise RuntimeError(f"OpenCL compilation failed: {e}")

    def execute_simple(
        self,
        kernel_source: str,
        kernel_name: str = "test_kernel",
        output_size: int = 256,
        build_options: str = ""
    ) -> np.ndarray:
        """
        Execute a simple kernel that writes to an output buffer.

        Assumes kernel signature: kernel void test_kernel(global float4* output)

        Args:
            kernel_source: Complete OpenCL source including kernel
            kernel_name: Name of kernel function to execute
            output_size: Number of work items (output buffer size)
            build_options: Compiler flags

        Returns:
            Output data as numpy array of shape (output_size, 4)
        """
        # Compile
        program = self.compile(kernel_source, build_options)

        # Create output buffer
        output = np.zeros(output_size * 4, dtype=np.float32)
        output_buf = cl.Buffer(
            self.context,
            cl.mem_flags.WRITE_ONLY,
            output.nbytes
        )

        # Execute kernel
        kernel = getattr(program, kernel_name)
        kernel(self.queue, (output_size,), None, output_buf)

        # Read results
        cl.enqueue_copy(self.queue, output, output_buf).wait()

        return output.reshape(output_size, 4)


def execute_opencl_kernel(
    context: cl.Context,
    queue: cl.CommandQueue,
    kernel_source: str,
    kernel_name: str = "test_kernel",
    global_size: Tuple[int, ...] = (256,),
    build_options: str = ""
) -> None:
    """
    Compile and execute OpenCL kernel (no output verification).

    Useful for testing that generated code compiles and runs without error.

    Args:
        context: OpenCL context
        queue: OpenCL command queue
        kernel_source: Complete OpenCL source
        kernel_name: Name of kernel to execute
        global_size: Global work size tuple (e.g., (256,) or (16, 16))
        build_options: Compiler flags

    Raises:
        RuntimeError: If compilation or execution fails
    """
    try:
        # Compile
        program = cl.Program(context, kernel_source).build(options=build_options)

        # Execute (assuming no arguments for simple test)
        kernel = getattr(program, kernel_name)
        kernel(queue, global_size, None)
        queue.finish()

    except cl.RuntimeError as e:
        raise RuntimeError(f"OpenCL kernel execution failed: {e}")


def validate_compilation_only(
    context: cl.Context,
    kernel_source: str,
    build_options: str = ""
) -> bool:
    """
    Validate that OpenCL source compiles successfully.

    Does not execute the kernel. Useful for quick compilation tests.

    Args:
        context: OpenCL context
        kernel_source: OpenCL C source code
        build_options: Compiler flags

    Returns:
        True if compilation succeeds, False otherwise
    """
    try:
        cl.Program(context, kernel_source).build(options=build_options)
        return True
    except cl.RuntimeError:
        return False


# Will be expanded in future weeks with more sophisticated execution helpers
