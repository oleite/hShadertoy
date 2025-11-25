"""
Tests for function symbol handling.

Tests function definitions, parameters, overloading, and return types.
Target: 30 tests
"""

import pytest
from glsl_to_opencl.analyzer.symbol_table import Symbol, SymbolType, SymbolTable


class TestFunctionSymbols:
    """Test function symbol creation and properties."""

    def test_create_function_symbol(self):
        """Test creating a function symbol."""
        func = Symbol("myFunc", SymbolType.FUNCTION, "void")
        assert func.name == "myFunc"
        assert func.symbol_type == SymbolType.FUNCTION
        assert func.glsl_type == "void"

    def test_function_with_return_type(self):
        """Test function with return type."""
        func = Symbol("calculate", SymbolType.FUNCTION, "float")
        assert func.glsl_type == "float"

    def test_function_with_parameters_metadata(self):
        """Test function with parameter metadata."""
        func = Symbol("add", SymbolType.FUNCTION, "float")
        func.metadata["params"] = ["float", "float"]
        func.metadata["param_names"] = ["a", "b"]

        assert len(func.metadata["params"]) == 2
        assert func.metadata["params"] == ["float", "float"]

    def test_function_with_no_parameters(self):
        """Test function with no parameters."""
        func = Symbol("getTime", SymbolType.FUNCTION, "float")
        func.metadata["params"] = []
        assert func.metadata["params"] == []

    def test_function_with_vector_return(self):
        """Test function returning vector."""
        func = Symbol("getColor", SymbolType.FUNCTION, "vec3")
        assert func.glsl_type == "vec3"


class TestFunctionRegistration:
    """Test registering functions in symbol table."""

    def test_register_simple_function(self):
        """Test registering a simple function."""
        table = SymbolTable()
        func = Symbol("helper", SymbolType.FUNCTION, "float")
        table.insert("helper", func)

        found = table.lookup("helper")
        assert found is not None
        assert found.symbol_type == SymbolType.FUNCTION

    def test_register_function_with_parameters(self):
        """Test registering function with parameters."""
        table = SymbolTable()
        func = Symbol("multiply", SymbolType.FUNCTION, "float")
        func.metadata["params"] = ["float", "float"]

        table.insert("multiply", func)

        found = table.lookup("multiply")
        assert found.metadata["params"] == ["float", "float"]

    def test_register_multiple_functions(self):
        """Test registering multiple functions."""
        table = SymbolTable()

        func1 = Symbol("func1", SymbolType.FUNCTION, "void")
        func2 = Symbol("func2", SymbolType.FUNCTION, "float")

        table.insert("func1", func1)
        table.insert("func2", func2)

        assert table.has_symbol("func1")
        assert table.has_symbol("func2")

    def test_get_all_functions(self):
        """Test getting all function symbols."""
        table = SymbolTable()

        table.insert("func1", Symbol("func1", SymbolType.FUNCTION, "void"))
        table.insert("func2", Symbol("func2", SymbolType.FUNCTION, "float"))
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "int"))

        functions = table.get_symbols_by_type(SymbolType.FUNCTION)
        assert len(functions) == 2


class TestFunctionParameters:
    """Test function parameter handling."""

    def test_parameter_symbol(self):
        """Test creating parameter symbol."""
        param = Symbol("x", SymbolType.PARAMETER, "float", qualifiers=["in"])
        assert param.symbol_type == SymbolType.PARAMETER
        assert "in" in param.qualifiers

    def test_out_parameter(self):
        """Test out parameter."""
        param = Symbol("result", SymbolType.PARAMETER, "float", qualifiers=["out"])
        assert "out" in param.qualifiers

    def test_inout_parameter(self):
        """Test inout parameter."""
        param = Symbol("value", SymbolType.PARAMETER, "vec3", qualifiers=["inout"])
        assert "inout" in param.qualifiers

    def test_function_scope_with_parameters(self):
        """Test function scope containing parameters."""
        global_scope = SymbolTable()

        # Function scope
        func_scope = global_scope.push_scope()
        func_scope.insert("param1", Symbol("param1", SymbolType.PARAMETER, "float", qualifiers=["in"]))
        func_scope.insert("param2", Symbol("param2", SymbolType.PARAMETER, "int", qualifiers=["in"]))

        # Can find parameters in function scope
        assert func_scope.has_symbol("param1")
        assert func_scope.has_symbol("param2")

        # Parameters not visible in global scope
        assert not global_scope.has_symbol("param1")


class TestFunctionOverloading:
    """Test function overloading scenarios."""

    def test_store_overload_signatures(self):
        """Test storing overloaded function signatures in metadata."""
        table = SymbolTable()

        # Store first signature
        func1 = Symbol("mix", SymbolType.FUNCTION, "float")
        func1.metadata["signature"] = "float(float,float,float)"
        func1.metadata["params"] = ["float", "float", "float"]

        # In real implementation, we'd need special handling for overloading
        # For now, we can only store one function per name
        # This test documents the limitation
        table.insert("mix", func1)

        found = table.lookup("mix")
        assert found.metadata["signature"] == "float(float,float,float)"

    def test_function_signature_metadata(self):
        """Test function with complete signature metadata."""
        func = Symbol("texture", SymbolType.FUNCTION, "vec4")
        func.metadata["signature"] = "vec4(sampler2D,vec2)"
        func.metadata["params"] = ["sampler2D", "vec2"]
        func.metadata["param_names"] = ["sampler", "uv"]

        assert func.metadata["signature"] == "vec4(sampler2D,vec2)"
        assert len(func.metadata["params"]) == 2


class TestFunctionScopes:
    """Test function-related scope handling."""

    def test_function_with_local_variables(self):
        """Test function scope with local variables."""
        global_scope = SymbolTable()

        func_scope = global_scope.push_scope()
        func_scope.insert("localVar", Symbol("localVar", SymbolType.VARIABLE, "float"))

        # Local variable exists in function scope
        assert func_scope.has_symbol("localVar")

        # Local variable not visible in global scope
        assert not global_scope.has_symbol("localVar")

    def test_nested_function_calls(self):
        """Test nested scopes for nested function calls."""
        global_scope = SymbolTable()
        global_scope.insert("helper", Symbol("helper", SymbolType.FUNCTION, "float"))

        outer_func = global_scope.push_scope()
        outer_func.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        # Can see global function from outer function
        assert outer_func.has_symbol("helper")

    def test_function_cannot_see_other_function_locals(self):
        """Test function scopes are isolated."""
        global_scope = SymbolTable()

        func1_scope = global_scope.push_scope()
        func1_scope.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        # Return to global and create another function
        func2_scope = global_scope.push_scope()

        # func2 cannot see func1's local variables
        assert not func2_scope.has_symbol("x")

    def test_recursive_function(self):
        """Test recursive function can see itself."""
        global_scope = SymbolTable()
        global_scope.insert("factorial", Symbol("factorial", SymbolType.FUNCTION, "int"))

        func_scope = global_scope.push_scope()

        # Function can see itself (recursive call)
        assert func_scope.has_symbol("factorial")


class TestComplexFunctionScenarios:
    """Test complex function-related scenarios."""

    def test_mainImage_function(self):
        """Test Shadertoy mainImage function."""
        global_scope = SymbolTable()

        mainImage = Symbol("mainImage", SymbolType.FUNCTION, "void")
        mainImage.metadata["params"] = ["vec4", "vec2"]
        mainImage.metadata["param_names"] = ["fragColor", "fragCoord"]
        mainImage.metadata["param_qualifiers"] = [["out"], ["in"]]

        global_scope.insert("mainImage", mainImage)

        found = global_scope.lookup("mainImage")
        assert found.metadata["param_names"] == ["fragColor", "fragCoord"]

    def test_helper_functions_before_main(self):
        """Test multiple helper functions defined before main."""
        global_scope = SymbolTable()

        helper1 = Symbol("noise", SymbolType.FUNCTION, "float")
        helper2 = Symbol("rotate", SymbolType.FUNCTION, "vec2")
        mainFunc = Symbol("mainImage", SymbolType.FUNCTION, "void")

        global_scope.insert("noise", helper1)
        global_scope.insert("rotate", helper2)
        global_scope.insert("mainImage", mainFunc)

        # All functions visible at global scope
        assert global_scope.has_symbol("noise")
        assert global_scope.has_symbol("rotate")
        assert global_scope.has_symbol("mainImage")

    def test_function_with_struct_parameter(self):
        """Test function with struct parameter."""
        global_scope = SymbolTable()

        # Define struct
        global_scope.insert("Light", Symbol("Light", SymbolType.STRUCT, "struct"))

        # Function taking struct parameter
        func = Symbol("applyLight", SymbolType.FUNCTION, "vec3")
        func.metadata["params"] = ["Light", "vec3"]

        global_scope.insert("applyLight", func)

        found = global_scope.lookup("applyLight")
        assert "Light" in found.metadata["params"]

    def test_function_returning_struct(self):
        """Test function returning struct."""
        global_scope = SymbolTable()

        global_scope.insert("Material", Symbol("Material", SymbolType.STRUCT, "struct"))

        func = Symbol("getMaterial", SymbolType.FUNCTION, "Material")
        global_scope.insert("getMaterial", func)

        found = global_scope.lookup("getMaterial")
        assert found.glsl_type == "Material"
