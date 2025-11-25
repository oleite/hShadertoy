"""
hShadertoy GLSL to OpenCL Transpiler

A production-ready transpiler that converts GLSL ES 3.0 fragment shaders from
Shadertoy.com into OpenCL 1.2 kernels for Houdini 21 Copernicus compositor.

Architecture:
    1. Parser - GLSL tokenization and AST generation (tree-sitter-glsl)
    2. Analyzer - Semantic analysis and type checking
    3. Transformer - GLSL to OpenCL conversion (CRITICAL PATH)
    4. Code Generator - OpenCL source emission
    5. Validator - Compilation checking

Usage:
    from glsl_to_opencl import transpile

    glsl_source = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { ... }"
    opencl_source = transpile(glsl_source)
"""

__version__ = "0.1.0-dev"
__author__ = "hShadertoy Development Team"

# Will be populated as modules are implemented
__all__ = []
