"""
GLSL ES 3.0 Built-in Symbols

Registers all built-in functions, variables, and constants defined in GLSL ES 3.0.
Used by SymbolTable to provide built-in symbol information for semantic analysis.

Phase 3, Week 7
"""

from typing import List
from .symbol_table import Symbol, SymbolType, SymbolTable


def register_builtins(symbol_table: SymbolTable) -> None:
    """
    Register all GLSL ES 3.0 built-in symbols into the symbol table.

    Args:
        symbol_table: SymbolTable to populate with built-ins
    """
    _register_math_functions(symbol_table)
    _register_geometric_functions(symbol_table)
    _register_matrix_functions(symbol_table)
    _register_vector_relational_functions(symbol_table)
    _register_texture_functions(symbol_table)
    _register_integer_functions(symbol_table)
    _register_interpolation_functions(symbol_table)
    _register_derivative_functions(symbol_table)
    _register_builtin_variables(symbol_table)
    _register_shadertoy_uniforms(symbol_table)
    _register_builtin_constants(symbol_table)


def _register_math_functions(table: SymbolTable) -> None:
    """Register trigonometric, exponential, and common math functions."""

    # Trigonometric functions (11)
    trig_funcs = [
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"
    ]
    for func in trig_funcs:
        table.insert(func, Symbol(func, SymbolType.BUILTIN, "genType"))

    # Exponential functions (7)
    exp_funcs = ["pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt"]
    for func in exp_funcs:
        table.insert(func, Symbol(func, SymbolType.BUILTIN, "genType"))

    # Common math functions (20)
    common_funcs = [
        "abs", "sign", "floor", "ceil", "round", "roundEven",
        "trunc", "fract", "mod", "modf", "min", "max",
        "clamp", "mix", "step", "smoothstep", "isnan", "isinf",
        "floatBitsToInt", "intBitsToFloat"
    ]
    for func in common_funcs:
        table.insert(func, Symbol(func, SymbolType.BUILTIN, "genType"))


def _register_geometric_functions(table: SymbolTable) -> None:
    """Register geometric functions."""

    geo_funcs = [
        "length", "distance", "dot", "cross",
        "normalize", "faceforward", "reflect", "refract"
    ]
    for func in geo_funcs:
        table.insert(func, Symbol(func, SymbolType.BUILTIN, "genType"))


def _register_matrix_functions(table: SymbolTable) -> None:
    """Register matrix operation functions."""

    matrix_funcs = [
        "matrixCompMult", "outerProduct", "transpose",
        "determinant", "inverse"
    ]
    for func in matrix_funcs:
        table.insert(func, Symbol(func, SymbolType.BUILTIN, "mat"))


def _register_vector_relational_functions(table: SymbolTable) -> None:
    """Register vector relational and component-wise comparison functions."""

    vec_rel_funcs = [
        "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual",
        "equal", "notEqual", "any", "all", "not"
    ]
    for func in vec_rel_funcs:
        table.insert(func, Symbol(func, SymbolType.BUILTIN, "bvec"))


def _register_texture_functions(table: SymbolTable) -> None:
    """Register texture sampling functions."""

    texture_funcs = [
        "texture", "textureProj", "textureLod", "textureOffset",
        "texelFetch", "texelFetchOffset", "textureProjOffset",
        "textureLodOffset", "textureProjLod", "textureProjLodOffset",
        "textureGrad", "textureGradOffset", "textureProjGrad",
        "textureProjGradOffset", "textureSize", "textureQueryLod",
        "textureQueryLevels", "textureSamples", "textureGather",
        "textureGatherOffset", "textureGatherOffsets"
    ]
    for func in texture_funcs:
        table.insert(func, Symbol(func, SymbolType.BUILTIN, "vec4"))


def _register_integer_functions(table: SymbolTable) -> None:
    """Register integer/bitwise operation functions."""

    int_funcs = [
        "uaddCarry", "usubBorrow", "umulExtended", "imulExtended",
        "bitfieldExtract", "bitfieldInsert", "bitfieldReverse",
        "bitCount", "findLSB", "findMSB"
    ]
    for func in int_funcs:
        table.insert(func, Symbol(func, SymbolType.BUILTIN, "genIType"))


def _register_interpolation_functions(table: SymbolTable) -> None:
    """Register fragment interpolation functions."""

    interp_funcs = [
        "interpolateAtCentroid", "interpolateAtSample", "interpolateAtOffset"
    ]
    for func in interp_funcs:
        table.insert(func, Symbol(func, SymbolType.BUILTIN, "genType"))


def _register_derivative_functions(table: SymbolTable) -> None:
    """Register derivative functions (fragment shader only)."""

    deriv_funcs = ["dFdx", "dFdy", "fwidth"]
    for func in deriv_funcs:
        symbol = Symbol(func, SymbolType.BUILTIN, "genType")
        symbol.metadata["fragment_only"] = True
        table.insert(func, symbol)


def _register_builtin_variables(table: SymbolTable) -> None:
    """Register built-in input/output variables."""

    # Vertex shader inputs
    table.insert("gl_VertexID", Symbol(
        "gl_VertexID", SymbolType.BUILTIN, "int",
        qualifiers=["in"], metadata={"shader_stage": "vertex"}
    ))
    table.insert("gl_InstanceID", Symbol(
        "gl_InstanceID", SymbolType.BUILTIN, "int",
        qualifiers=["in"], metadata={"shader_stage": "vertex"}
    ))

    # Fragment shader inputs
    table.insert("gl_FragCoord", Symbol(
        "gl_FragCoord", SymbolType.BUILTIN, "vec4",
        qualifiers=["in"], metadata={"shader_stage": "fragment"}
    ))
    table.insert("gl_FrontFacing", Symbol(
        "gl_FrontFacing", SymbolType.BUILTIN, "bool",
        qualifiers=["in"], metadata={"shader_stage": "fragment"}
    ))
    table.insert("gl_PointCoord", Symbol(
        "gl_PointCoord", SymbolType.BUILTIN, "vec2",
        qualifiers=["in"], metadata={"shader_stage": "fragment"}
    ))

    # Fragment shader outputs (GLSL ES 1.0 compatibility)
    table.insert("gl_FragColor", Symbol(
        "gl_FragColor", SymbolType.BUILTIN, "vec4",
        qualifiers=["out"], metadata={"shader_stage": "fragment", "deprecated": True}
    ))
    table.insert("gl_FragDepth", Symbol(
        "gl_FragDepth", SymbolType.BUILTIN, "float",
        qualifiers=["out"], metadata={"shader_stage": "fragment"}
    ))

    # Vertex shader outputs
    table.insert("gl_Position", Symbol(
        "gl_Position", SymbolType.BUILTIN, "vec4",
        qualifiers=["out"], metadata={"shader_stage": "vertex"}
    ))
    table.insert("gl_PointSize", Symbol(
        "gl_PointSize", SymbolType.BUILTIN, "float",
        qualifiers=["out"], metadata={"shader_stage": "vertex"}
    ))


def _register_shadertoy_uniforms(table: SymbolTable) -> None:
    """Register Shadertoy-specific uniform variables."""

    # Time uniforms
    table.insert("iTime", Symbol(
        "iTime", SymbolType.BUILTIN, "float",
        qualifiers=["uniform"], metadata={"shadertoy": True}
    ))
    table.insert("iTimeDelta", Symbol(
        "iTimeDelta", SymbolType.BUILTIN, "float",
        qualifiers=["uniform"], metadata={"shadertoy": True}
    ))
    table.insert("iFrame", Symbol(
        "iFrame", SymbolType.BUILTIN, "int",
        qualifiers=["uniform"], metadata={"shadertoy": True}
    ))
    table.insert("iFrameRate", Symbol(
        "iFrameRate", SymbolType.BUILTIN, "float",
        qualifiers=["uniform"], metadata={"shadertoy": True}
    ))

    # Resolution and mouse
    table.insert("iResolution", Symbol(
        "iResolution", SymbolType.BUILTIN, "vec3",
        qualifiers=["uniform"], metadata={"shadertoy": True}
    ))
    table.insert("iMouse", Symbol(
        "iMouse", SymbolType.BUILTIN, "vec4",
        qualifiers=["uniform"], metadata={"shadertoy": True}
    ))

    # Date and audio
    table.insert("iDate", Symbol(
        "iDate", SymbolType.BUILTIN, "vec4",
        qualifiers=["uniform"], metadata={"shadertoy": True}
    ))
    table.insert("iSampleRate", Symbol(
        "iSampleRate", SymbolType.BUILTIN, "float",
        qualifiers=["uniform"], metadata={"shadertoy": True}
    ))

    # Input channels (textures)
    for i in range(4):
        table.insert(f"iChannel{i}", Symbol(
            f"iChannel{i}", SymbolType.BUILTIN, "sampler2D",
            qualifiers=["uniform"], metadata={"shadertoy": True}
        ))

    # Channel time and resolution arrays
    table.insert("iChannelTime", Symbol(
        "iChannelTime", SymbolType.BUILTIN, "float[4]",
        qualifiers=["uniform"], metadata={"shadertoy": True}
    ))
    table.insert("iChannelResolution", Symbol(
        "iChannelResolution", SymbolType.BUILTIN, "vec3[4]",
        qualifiers=["uniform"], metadata={"shadertoy": True}
    ))


def _register_builtin_constants(table: SymbolTable) -> None:
    """Register GLSL ES 3.0 implementation-defined constants."""

    constants = [
        ("gl_MaxVertexAttribs", "int"),
        ("gl_MaxVertexUniformVectors", "int"),
        ("gl_MaxVaryingVectors", "int"),
        ("gl_MaxVertexTextureImageUnits", "int"),
        ("gl_MaxCombinedTextureImageUnits", "int"),
        ("gl_MaxTextureImageUnits", "int"),
        ("gl_MaxFragmentUniformVectors", "int"),
        ("gl_MaxDrawBuffers", "int"),
        ("gl_MinProgramTexelOffset", "int"),
        ("gl_MaxProgramTexelOffset", "int"),
    ]

    for name, glsl_type in constants:
        table.insert(name, Symbol(
            name, SymbolType.BUILTIN, glsl_type,
            qualifiers=["const"], metadata={"constant": True}
        ))


# Convenience function to create a pre-populated symbol table
def create_builtin_symbol_table() -> SymbolTable:
    """
    Create a symbol table pre-populated with all GLSL ES 3.0 built-ins.

    Returns:
        SymbolTable with all built-in symbols registered
    """
    table = SymbolTable()
    register_builtins(table)
    return table


# Lists for easy access
GLSL_MATH_FUNCTIONS: List[str] = [
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
    "abs", "sign", "floor", "ceil", "round", "roundEven",
    "trunc", "fract", "mod", "modf", "min", "max",
    "clamp", "mix", "step", "smoothstep", "isnan", "isinf",
    "floatBitsToInt", "intBitsToFloat"
]

GLSL_GEOMETRIC_FUNCTIONS: List[str] = [
    "length", "distance", "dot", "cross",
    "normalize", "faceforward", "reflect", "refract"
]

GLSL_MATRIX_FUNCTIONS: List[str] = [
    "matrixCompMult", "outerProduct", "transpose",
    "determinant", "inverse"
]

GLSL_TEXTURE_FUNCTIONS: List[str] = [
    "texture", "textureProj", "textureLod", "textureOffset",
    "texelFetch", "texelFetchOffset", "textureProjOffset",
    "textureLodOffset", "textureProjLod", "textureProjLodOffset",
    "textureGrad", "textureGradOffset", "textureProjGrad",
    "textureProjGradOffset", "textureSize", "textureQueryLod",
    "textureQueryLevels", "textureSamples", "textureGather",
    "textureGatherOffset", "textureGatherOffsets"
]

SHADERTOY_UNIFORMS: List[str] = [
    "iTime", "iTimeDelta", "iFrame", "iFrameRate",
    "iResolution", "iMouse", "iDate", "iSampleRate",
    "iChannel0", "iChannel1", "iChannel2", "iChannel3",
    "iChannelTime", "iChannelResolution"
]
