"""
Tests for Symbol class and SymbolType enum.

Tests basic symbol creation, properties, and type checking.
Target: 20 tests
"""

import pytest
from glsl_to_opencl.analyzer.symbol_table import (
    Symbol,
    SymbolType,
)


class TestSymbolType:
    """Test SymbolType enum."""

    def test_symbol_type_variable(self):
        """Test VARIABLE symbol type."""
        assert SymbolType.VARIABLE.value == "variable"

    def test_symbol_type_function(self):
        """Test FUNCTION symbol type."""
        assert SymbolType.FUNCTION.value == "function"

    def test_symbol_type_struct(self):
        """Test STRUCT symbol type."""
        assert SymbolType.STRUCT.value == "struct"

    def test_symbol_type_parameter(self):
        """Test PARAMETER symbol type."""
        assert SymbolType.PARAMETER.value == "parameter"

    def test_symbol_type_builtin(self):
        """Test BUILTIN symbol type."""
        assert SymbolType.BUILTIN.value == "builtin"

    def test_symbol_type_constant(self):
        """Test CONSTANT symbol type."""
        assert SymbolType.CONSTANT.value == "constant"


class TestSymbolConstruction:
    """Test Symbol class construction."""

    def test_create_simple_symbol(self):
        """Test creating a simple variable symbol."""
        symbol = Symbol("x", SymbolType.VARIABLE, "float")
        assert symbol.name == "x"
        assert symbol.symbol_type == SymbolType.VARIABLE
        assert symbol.glsl_type == "float"
        assert symbol.qualifiers == []
        assert symbol.location is None
        assert symbol.metadata == {}

    def test_create_symbol_with_qualifiers(self):
        """Test creating symbol with qualifiers."""
        symbol = Symbol("x", SymbolType.VARIABLE, "float", qualifiers=["const"])
        assert symbol.name == "x"
        assert symbol.qualifiers == ["const"]

    def test_create_symbol_with_location(self):
        """Test creating symbol with source location."""
        symbol = Symbol("x", SymbolType.VARIABLE, "float", location=(10, 5))
        assert symbol.location == (10, 5)

    def test_create_symbol_with_metadata(self):
        """Test creating symbol with metadata."""
        symbol = Symbol(
            "myFunc", SymbolType.FUNCTION, "void", metadata={"params": ["float", "int"]}
        )
        assert symbol.metadata["params"] == ["float", "int"]

    def test_create_vector_symbol(self):
        """Test creating vec3 symbol."""
        symbol = Symbol("pos", SymbolType.VARIABLE, "vec3")
        assert symbol.glsl_type == "vec3"

    def test_create_matrix_symbol(self):
        """Test creating mat4 symbol."""
        symbol = Symbol("transform", SymbolType.VARIABLE, "mat4")
        assert symbol.glsl_type == "mat4"

    def test_create_uniform_symbol(self):
        """Test creating uniform symbol."""
        symbol = Symbol("iTime", SymbolType.VARIABLE, "float", qualifiers=["uniform"])
        assert "uniform" in symbol.qualifiers


class TestSymbolProperties:
    """Test Symbol property methods."""

    def test_is_const_true(self):
        """Test is_const() returns True for const symbol."""
        symbol = Symbol("x", SymbolType.VARIABLE, "float", qualifiers=["const"])
        assert symbol.is_const() is True

    def test_is_const_false(self):
        """Test is_const() returns False for non-const symbol."""
        symbol = Symbol("x", SymbolType.VARIABLE, "float")
        assert symbol.is_const() is False

    def test_is_uniform_true(self):
        """Test is_uniform() returns True for uniform symbol."""
        symbol = Symbol("iTime", SymbolType.VARIABLE, "float", qualifiers=["uniform"])
        assert symbol.is_uniform() is True

    def test_is_uniform_false(self):
        """Test is_uniform() returns False for non-uniform symbol."""
        symbol = Symbol("x", SymbolType.VARIABLE, "float")
        assert symbol.is_uniform() is False

    def test_is_builtin_true(self):
        """Test is_builtin() returns True for built-in symbol."""
        symbol = Symbol("sin", SymbolType.BUILTIN, "float(float)")
        assert symbol.is_builtin() is True

    def test_is_builtin_false(self):
        """Test is_builtin() returns False for non-built-in symbol."""
        symbol = Symbol("myFunc", SymbolType.FUNCTION, "void")
        assert symbol.is_builtin() is False

    def test_multiple_qualifiers(self):
        """Test symbol with multiple qualifiers."""
        symbol = Symbol("x", SymbolType.PARAMETER, "float", qualifiers=["in", "const"])
        assert "in" in symbol.qualifiers
        assert "const" in symbol.qualifiers
        assert symbol.is_const() is True
