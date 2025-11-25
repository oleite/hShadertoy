"""
Code generation module for hShadertoy GLSL-to-OpenCL transpiler.

Session 6: Production code generator for transformed AST.
"""

from .opencl_emitter import OpenCLEmitter

__all__ = ['OpenCLEmitter']
