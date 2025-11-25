"""
Tests for GLSL ES 3.0 built-in symbols.

Tests that all built-in functions, variables, and constants are
correctly registered in the symbol table.
Target: 60 tests
"""

import pytest
from glsl_to_opencl.analyzer.symbol_table import SymbolTable, SymbolType
from glsl_to_opencl.analyzer.builtins import (
    register_builtins,
    create_builtin_symbol_table,
    GLSL_MATH_FUNCTIONS,
    GLSL_GEOMETRIC_FUNCTIONS,
    GLSL_MATRIX_FUNCTIONS,
    GLSL_TEXTURE_FUNCTIONS,
    SHADERTOY_UNIFORMS,
)


class TestMathFunctions:
    """Test math function registration."""

    def test_sin_registered(self):
        """Test sin() function is registered."""
        table = create_builtin_symbol_table()
        sin_func = table.lookup("sin")
        assert sin_func is not None
        assert sin_func.is_builtin()

    def test_cos_registered(self):
        """Test cos() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("cos")

    def test_tan_registered(self):
        """Test tan() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("tan")

    def test_sqrt_registered(self):
        """Test sqrt() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("sqrt")

    def test_pow_registered(self):
        """Test pow() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("pow")

    def test_abs_registered(self):
        """Test abs() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("abs")

    def test_floor_registered(self):
        """Test floor() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("floor")

    def test_ceil_registered(self):
        """Test ceil() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("ceil")

    def test_min_registered(self):
        """Test min() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("min")

    def test_max_registered(self):
        """Test max() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("max")

    def test_clamp_registered(self):
        """Test clamp() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("clamp")

    def test_mix_registered(self):
        """Test mix() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("mix")

    def test_step_registered(self):
        """Test step() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("step")

    def test_smoothstep_registered(self):
        """Test smoothstep() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("smoothstep")

    def test_all_math_functions_registered(self):
        """Test all math functions from list are registered."""
        table = create_builtin_symbol_table()
        for func_name in GLSL_MATH_FUNCTIONS:
            assert table.has_symbol(func_name), f"{func_name} not registered"


class TestGeometricFunctions:
    """Test geometric function registration."""

    def test_length_registered(self):
        """Test length() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("length")

    def test_distance_registered(self):
        """Test distance() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("distance")

    def test_dot_registered(self):
        """Test dot() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("dot")

    def test_cross_registered(self):
        """Test cross() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("cross")

    def test_normalize_registered(self):
        """Test normalize() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("normalize")

    def test_reflect_registered(self):
        """Test reflect() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("reflect")

    def test_refract_registered(self):
        """Test refract() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("refract")

    def test_all_geometric_functions_registered(self):
        """Test all geometric functions from list are registered."""
        table = create_builtin_symbol_table()
        for func_name in GLSL_GEOMETRIC_FUNCTIONS:
            assert table.has_symbol(func_name), f"{func_name} not registered"


class TestMatrixFunctions:
    """Test matrix function registration."""

    def test_transpose_registered(self):
        """Test transpose() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("transpose")

    def test_determinant_registered(self):
        """Test determinant() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("determinant")

    def test_inverse_registered(self):
        """Test inverse() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("inverse")

    def test_all_matrix_functions_registered(self):
        """Test all matrix functions from list are registered."""
        table = create_builtin_symbol_table()
        for func_name in GLSL_MATRIX_FUNCTIONS:
            assert table.has_symbol(func_name), f"{func_name} not registered"


class TestTextureFunctions:
    """Test texture function registration."""

    def test_texture_registered(self):
        """Test texture() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("texture")

    def test_textureLod_registered(self):
        """Test textureLod() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("textureLod")

    def test_textureGrad_registered(self):
        """Test textureGrad() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("textureGrad")

    def test_texelFetch_registered(self):
        """Test texelFetch() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("texelFetch")

    def test_textureSize_registered(self):
        """Test textureSize() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("textureSize")

    def test_all_texture_functions_registered(self):
        """Test all texture functions from list are registered."""
        table = create_builtin_symbol_table()
        for func_name in GLSL_TEXTURE_FUNCTIONS:
            assert table.has_symbol(func_name), f"{func_name} not registered"


class TestDerivativeFunctions:
    """Test derivative function registration."""

    def test_dFdx_registered(self):
        """Test dFdx() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("dFdx")

    def test_dFdy_registered(self):
        """Test dFdy() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("dFdy")

    def test_fwidth_registered(self):
        """Test fwidth() function is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("fwidth")

    def test_derivative_functions_marked_fragment_only(self):
        """Test derivative functions have fragment_only metadata."""
        table = create_builtin_symbol_table()
        dFdx = table.lookup("dFdx")
        assert dFdx.metadata.get("fragment_only") is True


class TestBuiltinVariables:
    """Test built-in variable registration."""

    def test_gl_FragCoord_registered(self):
        """Test gl_FragCoord is registered."""
        table = create_builtin_symbol_table()
        frag_coord = table.lookup("gl_FragCoord")
        assert frag_coord is not None
        assert frag_coord.glsl_type == "vec4"

    def test_gl_FrontFacing_registered(self):
        """Test gl_FrontFacing is registered."""
        table = create_builtin_symbol_table()
        front_facing = table.lookup("gl_FrontFacing")
        assert front_facing is not None
        assert front_facing.glsl_type == "bool"

    def test_gl_Position_registered(self):
        """Test gl_Position is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("gl_Position")

    def test_gl_FragColor_registered(self):
        """Test gl_FragColor is registered."""
        table = create_builtin_symbol_table()
        frag_color = table.lookup("gl_FragColor")
        assert frag_color is not None
        assert frag_color.glsl_type == "vec4"

    def test_gl_FragDepth_registered(self):
        """Test gl_FragDepth is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("gl_FragDepth")


class TestShadertoyUniforms:
    """Test Shadertoy uniform registration."""

    def test_iTime_registered(self):
        """Test iTime uniform is registered."""
        table = create_builtin_symbol_table()
        itime = table.lookup("iTime")
        assert itime is not None
        assert itime.glsl_type == "float"
        assert itime.is_uniform()

    def test_iResolution_registered(self):
        """Test iResolution uniform is registered."""
        table = create_builtin_symbol_table()
        ires = table.lookup("iResolution")
        assert ires is not None
        assert ires.glsl_type == "vec3"

    def test_iMouse_registered(self):
        """Test iMouse uniform is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("iMouse")

    def test_iFrame_registered(self):
        """Test iFrame uniform is registered."""
        table = create_builtin_symbol_table()
        iframe = table.lookup("iFrame")
        assert iframe is not None
        assert iframe.glsl_type == "int"

    def test_iChannel0_registered(self):
        """Test iChannel0 uniform is registered."""
        table = create_builtin_symbol_table()
        ch0 = table.lookup("iChannel0")
        assert ch0 is not None
        assert ch0.glsl_type == "sampler2D"

    def test_all_iChannels_registered(self):
        """Test all 4 iChannel uniforms are registered."""
        table = create_builtin_symbol_table()
        for i in range(4):
            assert table.has_symbol(f"iChannel{i}")

    def test_iChannelTime_registered(self):
        """Test iChannelTime array is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("iChannelTime")

    def test_iChannelResolution_registered(self):
        """Test iChannelResolution array is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("iChannelResolution")

    def test_all_shadertoy_uniforms_registered(self):
        """Test all Shadertoy uniforms from list are registered."""
        table = create_builtin_symbol_table()
        for uniform_name in SHADERTOY_UNIFORMS:
            assert table.has_symbol(uniform_name), f"{uniform_name} not registered"


class TestBuiltinConstants:
    """Test built-in constant registration."""

    def test_gl_MaxDrawBuffers_registered(self):
        """Test gl_MaxDrawBuffers constant is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("gl_MaxDrawBuffers")

    def test_gl_MaxTextureImageUnits_registered(self):
        """Test gl_MaxTextureImageUnits constant is registered."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("gl_MaxTextureImageUnits")


class TestRegisterBuiltins:
    """Test register_builtins function."""

    def test_register_builtins_populates_table(self):
        """Test register_builtins adds symbols to table."""
        table = SymbolTable()
        assert len(table.symbols) == 0

        register_builtins(table)
        assert len(table.symbols) > 100  # Should have 100+ built-in symbols

    def test_create_builtin_symbol_table_returns_populated(self):
        """Test create_builtin_symbol_table returns pre-populated table."""
        table = create_builtin_symbol_table()
        assert len(table.symbols) > 100

    def test_builtin_symbols_marked_as_builtins(self):
        """Test all registered symbols are marked as built-ins."""
        table = create_builtin_symbol_table()

        # Check a few examples
        assert table.lookup("sin").symbol_type == SymbolType.BUILTIN
        assert table.lookup("cos").symbol_type == SymbolType.BUILTIN
        assert table.lookup("texture").symbol_type == SymbolType.BUILTIN
        assert table.lookup("gl_FragCoord").symbol_type == SymbolType.BUILTIN
        assert table.lookup("iTime").symbol_type == SymbolType.BUILTIN
