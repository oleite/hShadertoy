"""
Analyzer Module

Performs semantic analysis and type checking on parsed GLSL AST.

Components:
    - symbol_table.py: Hierarchical scoping and symbol management (COMPLETE)
    - builtins.py: GLSL ES 3.0 built-in symbols (COMPLETE)
    - type_checker.py: Type system and operator checking (COMPLETE)
    - metadata.py: Shader interface extraction (COMPLETE)

Development Phase: Phase 3 (Weeks 7-9)
Week 7 Status: COMPLETE (204 tests passing)
Week 8 Status: COMPLETE (304 tests passing)
Week 9 Status: COMPLETE (100 tests passing)
Risk Level: MEDIUM
Test Target: 700+ tests (608 tests achieved - 87%)
"""

from .symbol_table import (
    Symbol,
    SymbolType,
    SymbolTable,
    SymbolTableError,
    DuplicateSymbolError,
    UndeclaredSymbolError,
    BuiltinRedeclarationError,
)

from .builtins import (
    register_builtins,
    create_builtin_symbol_table,
    GLSL_MATH_FUNCTIONS,
    GLSL_GEOMETRIC_FUNCTIONS,
    GLSL_MATRIX_FUNCTIONS,
    GLSL_TEXTURE_FUNCTIONS,
    SHADERTOY_UNIFORMS,
)

from .type_checker import (
    GLSLType,
    TypeCategory,
    ScalarType,
    TypeChecker,
    TypeCheckError,
    TypeMismatchError,
    InvalidOperationError,
    UndefinedTypeError,
    parse_type_string,
    # Type constants
    FLOAT, INT, UINT, BOOL, VOID,
    VEC2, VEC3, VEC4,
    IVEC2, IVEC3, IVEC4,
    UVEC2, UVEC3, UVEC4,
    BVEC2, BVEC3, BVEC4,
    MAT2, MAT3, MAT4,
    SAMPLER2D, SAMPLER3D, SAMPLERCUBE,
)

from .metadata import (
    ShaderMetadata,
    MetadataExtractor,
    MetadataExtractionError,
)

__all__ = [
    # Symbol table classes
    "Symbol",
    "SymbolType",
    "SymbolTable",
    "SymbolTableError",
    "DuplicateSymbolError",
    "UndeclaredSymbolError",
    "BuiltinRedeclarationError",
    # Built-in functions
    "register_builtins",
    "create_builtin_symbol_table",
    # Built-in lists
    "GLSL_MATH_FUNCTIONS",
    "GLSL_GEOMETRIC_FUNCTIONS",
    "GLSL_MATRIX_FUNCTIONS",
    "GLSL_TEXTURE_FUNCTIONS",
    "SHADERTOY_UNIFORMS",
    # Type checker classes
    "GLSLType",
    "TypeCategory",
    "ScalarType",
    "TypeChecker",
    "TypeCheckError",
    "TypeMismatchError",
    "InvalidOperationError",
    "UndefinedTypeError",
    "parse_type_string",
    # Type constants
    "FLOAT", "INT", "UINT", "BOOL", "VOID",
    "VEC2", "VEC3", "VEC4",
    "IVEC2", "IVEC3", "IVEC4",
    "UVEC2", "UVEC3", "UVEC4",
    "BVEC2", "BVEC3", "BVEC4",
    "MAT2", "MAT3", "MAT4",
    "SAMPLER2D", "SAMPLER3D", "SAMPLERCUBE",
    # Metadata extraction
    "ShaderMetadata",
    "MetadataExtractor",
    "MetadataExtractionError",
]
