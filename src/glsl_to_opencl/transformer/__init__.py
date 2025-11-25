"""
Transformer module for hShadertoy GLSL-to-OpenCL transpiler.

Session 6: Now uses production code generator from codegen module.
"""

from .transformed_ast import *
from .ast_transformer import ASTTransformer

# Session 6: For backward compatibility, keep temporary code_emitter available
# Tests can import from transformer.code_emitter or use codegen.OpenCLEmitter
# Note: The production emitter is in codegen.opencl_emitter

__all__ = [
    'ASTTransformer',
]
