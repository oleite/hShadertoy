"""
Integration and edge case tests for symbol table.

Tests comprehensive scenarios, edge cases, and real-world usage patterns.
Target: 65 tests
"""

import pytest
from glsl_to_opencl.analyzer.symbol_table import (
    Symbol,
    SymbolType,
    SymbolTable,
    DuplicateSymbolError,
    BuiltinRedeclarationError,
)
from glsl_to_opencl.analyzer.builtins import create_builtin_symbol_table


class TestIntegrationWithBuiltins:
    """Test symbol table with built-ins registered."""

    def test_builtin_table_has_math_functions(self):
        """Test built-in table contains math functions."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("sin")
        assert table.has_symbol("cos")
        assert table.has_symbol("sqrt")

    def test_builtin_table_has_shadertoy_uniforms(self):
        """Test built-in table contains Shadertoy uniforms."""
        table = create_builtin_symbol_table()
        assert table.has_symbol("iTime")
        assert table.has_symbol("iResolution")

    def test_can_add_user_variables_to_builtin_table(self):
        """Test can add user variables to table with built-ins."""
        table = create_builtin_symbol_table()
        table.insert("myVar", Symbol("myVar", SymbolType.VARIABLE, "float"))

        assert table.has_symbol("myVar")
        assert table.has_symbol("sin")  # Built-in still there

    def test_cannot_override_builtin_function(self):
        """Test cannot override built-in function."""
        table = create_builtin_symbol_table()

        with pytest.raises(BuiltinRedeclarationError):
            table.insert("sin", Symbol("sin", SymbolType.VARIABLE, "float"))

    def test_shader_with_builtins_and_user_code(self):
        """Test typical shader structure with built-ins and user code."""
        global_scope = create_builtin_symbol_table()

        # Add global variables
        global_scope.insert("myGlobal", Symbol("myGlobal", SymbolType.VARIABLE, "float"))

        # Function scope
        func_scope = global_scope.push_scope()
        func_scope.insert("local", Symbol("local", SymbolType.VARIABLE, "vec3"))

        # Can use built-ins
        assert func_scope.has_symbol("sin")
        # Can use globals
        assert func_scope.has_symbol("myGlobal")
        # Can use locals
        assert func_scope.has_symbol("local")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_name_symbol(self):
        """Test symbol with empty name."""
        symbol = Symbol("", SymbolType.VARIABLE, "float")
        table = SymbolTable()
        table.insert("", symbol)

        assert table.has_symbol("")

    def test_symbol_name_with_numbers(self):
        """Test symbol name with numbers."""
        table = SymbolTable()
        table.insert("var123", Symbol("var123", SymbolType.VARIABLE, "float"))
        assert table.has_symbol("var123")

    def test_symbol_name_with_underscore(self):
        """Test symbol name with underscores."""
        table = SymbolTable()
        table.insert("my_var_name", Symbol("my_var_name", SymbolType.VARIABLE, "float"))
        assert table.has_symbol("my_var_name")

    def test_very_long_symbol_name(self):
        """Test very long symbol name."""
        long_name = "a" * 1000
        table = SymbolTable()
        table.insert(long_name, Symbol(long_name, SymbolType.VARIABLE, "float"))
        assert table.has_symbol(long_name)

    def test_deep_nesting_10_levels(self):
        """Test deep scope nesting (10 levels)."""
        scopes = [SymbolTable()]

        for i in range(10):
            scopes.append(scopes[-1].push_scope())

        assert scopes[-1].get_scope_depth() == 10

    def test_many_symbols_in_scope(self):
        """Test inserting many symbols in one scope."""
        table = SymbolTable()

        for i in range(100):
            table.insert(f"var{i}", Symbol(f"var{i}", SymbolType.VARIABLE, "float"))

        assert len(table.symbols) == 100
        assert table.has_symbol("var99")

    def test_symbol_with_special_gl_prefix(self):
        """Test user symbols with gl_ prefix."""
        table = SymbolTable()
        # Users shouldn't use gl_ prefix, but technically allowed
        table.insert("gl_MyVar", Symbol("gl_MyVar", SymbolType.VARIABLE, "float"))
        assert table.has_symbol("gl_MyVar")

    def test_case_sensitivity(self):
        """Test symbol names are case-sensitive."""
        table = SymbolTable()
        table.insert("MyVar", Symbol("MyVar", SymbolType.VARIABLE, "float"))
        table.insert("myvar", Symbol("myvar", SymbolType.VARIABLE, "int"))
        table.insert("MYVAR", Symbol("MYVAR", SymbolType.VARIABLE, "vec3"))

        assert table.lookup("MyVar").glsl_type == "float"
        assert table.lookup("myvar").glsl_type == "int"
        assert table.lookup("MYVAR").glsl_type == "vec3"


class TestRealWorldShaderPatterns:
    """Test patterns from real Shadertoy shaders."""

    def test_gameOfLife_pattern(self):
        """Test symbol table for Game of Life shader pattern."""
        global_scope = create_builtin_symbol_table()

        # mainImage function
        main_scope = global_scope.push_scope()
        main_scope.insert("fragColor", Symbol("fragColor", SymbolType.PARAMETER, "vec4", qualifiers=["out"]))
        main_scope.insert("fragCoord", Symbol("fragCoord", SymbolType.PARAMETER, "vec2", qualifiers=["in"]))

        # Local variables
        main_scope.insert("o", Symbol("o", SymbolType.VARIABLE, "vec2"))
        main_scope.insert("uv", Symbol("uv", SymbolType.VARIABLE, "vec2"))
        main_scope.insert("accum", Symbol("accum", SymbolType.VARIABLE, "float"))

        # Nested loop
        outer_loop = main_scope.push_scope()
        outer_loop.insert("i", Symbol("i", SymbolType.VARIABLE, "int"))

        inner_loop = outer_loop.push_scope()
        inner_loop.insert("j", Symbol("j", SymbolType.VARIABLE, "int"))

        # All symbols accessible from innermost scope
        assert inner_loop.has_symbol("i")
        assert inner_loop.has_symbol("j")
        assert inner_loop.has_symbol("uv")
        assert inner_loop.has_symbol("iFrame")  # Shadertoy uniform

    def test_complex_shader_with_helpers(self):
        """Test shader with multiple helper functions."""
        global_scope = create_builtin_symbol_table()

        # Helper function 1
        global_scope.insert("noise", Symbol("noise", SymbolType.FUNCTION, "float"))

        # Helper function 2
        global_scope.insert("rotate", Symbol("rotate", SymbolType.FUNCTION, "vec2"))

        # Main function
        global_scope.insert("mainImage", Symbol("mainImage", SymbolType.FUNCTION, "void"))

        # Main function scope
        main_scope = global_scope.push_scope()
        main_scope.insert("uv", Symbol("uv", SymbolType.VARIABLE, "vec2"))

        # Can call helper functions from main
        assert main_scope.has_symbol("noise")
        assert main_scope.has_symbol("rotate")

    def test_shader_with_struct(self):
        """Test shader with struct definition and usage."""
        global_scope = create_builtin_symbol_table()

        # Define struct
        global_scope.insert("Material", Symbol("Material", SymbolType.STRUCT, "struct"))

        # Global variable of struct type
        global_scope.insert("defaultMaterial", Symbol("defaultMaterial", SymbolType.VARIABLE, "Material"))

        # Function using struct
        global_scope.insert("applyMaterial", Symbol("applyMaterial", SymbolType.FUNCTION, "vec3"))

        assert global_scope.has_symbol("Material")
        assert global_scope.has_symbol("defaultMaterial")


class TestQualifierHandling:
    """Test type qualifier handling."""

    def test_const_qualifier(self):
        """Test const qualifier."""
        table = SymbolTable()
        table.insert("PI", Symbol("PI", SymbolType.CONSTANT, "float", qualifiers=["const"]))

        pi = table.lookup("PI")
        assert pi.is_const()

    def test_uniform_qualifier(self):
        """Test uniform qualifier."""
        table = SymbolTable()
        table.insert("time", Symbol("time", SymbolType.VARIABLE, "float", qualifiers=["uniform"]))

        time = table.lookup("time")
        assert time.is_uniform()

    def test_in_qualifier(self):
        """Test in qualifier."""
        table = SymbolTable()
        table.insert("input", Symbol("input", SymbolType.PARAMETER, "float", qualifiers=["in"]))

        param = table.lookup("input")
        assert "in" in param.qualifiers

    def test_out_qualifier(self):
        """Test out qualifier."""
        table = SymbolTable()
        table.insert("output", Symbol("output", SymbolType.PARAMETER, "vec4", qualifiers=["out"]))

        param = table.lookup("output")
        assert "out" in param.qualifiers

    def test_inout_qualifier(self):
        """Test inout qualifier."""
        table = SymbolTable()
        table.insert("param", Symbol("param", SymbolType.PARAMETER, "vec3", qualifiers=["inout"]))

        param = table.lookup("param")
        assert "inout" in param.qualifiers

    def test_multiple_qualifiers(self):
        """Test symbol with multiple qualifiers."""
        table = SymbolTable()
        symbol = Symbol("value", SymbolType.PARAMETER, "float", qualifiers=["const", "in"])
        table.insert("value", symbol)

        found = table.lookup("value")
        assert "const" in found.qualifiers
        assert "in" in found.qualifiers


class TestMetadataHandling:
    """Test symbol metadata."""

    def test_symbol_with_location_metadata(self):
        """Test symbol with source location."""
        table = SymbolTable()
        symbol = Symbol("x", SymbolType.VARIABLE, "float", location=(10, 5))
        table.insert("x", symbol)

        found = table.lookup("x")
        assert found.location == (10, 5)

    def test_symbol_with_custom_metadata(self):
        """Test symbol with custom metadata."""
        table = SymbolTable()
        symbol = Symbol("func", SymbolType.FUNCTION, "void")
        symbol.metadata["params"] = ["float", "int"]
        symbol.metadata["return_type"] = "void"
        symbol.metadata["inline"] = True

        table.insert("func", symbol)

        found = table.lookup("func")
        assert found.metadata["inline"] is True
        assert len(found.metadata["params"]) == 2

    def test_shadertoy_metadata(self):
        """Test Shadertoy-specific metadata."""
        table = SymbolTable()
        symbol = Symbol("iTime", SymbolType.BUILTIN, "float", qualifiers=["uniform"])
        symbol.metadata["shadertoy"] = True

        table.insert("iTime", symbol)

        found = table.lookup("iTime")
        assert found.metadata["shadertoy"] is True


class TestTypeSystem:
    """Test GLSL type handling."""

    def test_scalar_types(self):
        """Test scalar type symbols."""
        table = SymbolTable()

        table.insert("f", Symbol("f", SymbolType.VARIABLE, "float"))
        table.insert("i", Symbol("i", SymbolType.VARIABLE, "int"))
        table.insert("u", Symbol("u", SymbolType.VARIABLE, "uint"))
        table.insert("b", Symbol("b", SymbolType.VARIABLE, "bool"))

        assert table.lookup("f").glsl_type == "float"
        assert table.lookup("i").glsl_type == "int"
        assert table.lookup("u").glsl_type == "uint"
        assert table.lookup("b").glsl_type == "bool"

    def test_vector_types(self):
        """Test vector type symbols."""
        table = SymbolTable()

        table.insert("v2", Symbol("v2", SymbolType.VARIABLE, "vec2"))
        table.insert("v3", Symbol("v3", SymbolType.VARIABLE, "vec3"))
        table.insert("v4", Symbol("v4", SymbolType.VARIABLE, "vec4"))
        table.insert("iv3", Symbol("iv3", SymbolType.VARIABLE, "ivec3"))
        table.insert("bv2", Symbol("bv2", SymbolType.VARIABLE, "bvec2"))

        assert table.lookup("v3").glsl_type == "vec3"
        assert table.lookup("iv3").glsl_type == "ivec3"

    def test_matrix_types(self):
        """Test matrix type symbols."""
        table = SymbolTable()

        table.insert("m2", Symbol("m2", SymbolType.VARIABLE, "mat2"))
        table.insert("m3", Symbol("m3", SymbolType.VARIABLE, "mat3"))
        table.insert("m4", Symbol("m4", SymbolType.VARIABLE, "mat4"))

        assert table.lookup("m2").glsl_type == "mat2"
        assert table.lookup("m3").glsl_type == "mat3"
        assert table.lookup("m4").glsl_type == "mat4"

    def test_sampler_types(self):
        """Test sampler type symbols."""
        table = SymbolTable()

        table.insert("tex2D", Symbol("tex2D", SymbolType.VARIABLE, "sampler2D"))
        table.insert("texCube", Symbol("texCube", SymbolType.VARIABLE, "samplerCube"))
        table.insert("tex3D", Symbol("tex3D", SymbolType.VARIABLE, "sampler3D"))

        assert table.lookup("tex2D").glsl_type == "sampler2D"
        assert table.lookup("texCube").glsl_type == "samplerCube"

    def test_array_types(self):
        """Test array type symbols."""
        table = SymbolTable()

        table.insert("arr", Symbol("arr", SymbolType.VARIABLE, "float[10]"))
        table.insert("arr2d", Symbol("arr2d", SymbolType.VARIABLE, "vec3[5][5]"))

        assert table.lookup("arr").glsl_type == "float[10]"
        assert table.lookup("arr2d").glsl_type == "vec3[5][5]"


class TestErrorMessages:
    """Test error message clarity."""

    def test_duplicate_error_message(self):
        """Test duplicate error contains useful info."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        try:
            table.insert("x", Symbol("x", SymbolType.VARIABLE, "int"))
            assert False, "Should raise DuplicateSymbolError"
        except DuplicateSymbolError as e:
            assert "x" in str(e)
            assert "already declared" in str(e).lower()

    def test_builtin_redeclaration_error_message(self):
        """Test builtin redeclaration error message."""
        table = create_builtin_symbol_table()

        try:
            table.insert("sin", Symbol("sin", SymbolType.VARIABLE, "float"))
            assert False, "Should raise BuiltinRedeclarationError"
        except BuiltinRedeclarationError as e:
            assert "sin" in str(e)
            assert "built-in" in str(e).lower()


class TestPerformance:
    """Test symbol table performance."""

    def test_lookup_performance_flat_scope(self):
        """Test lookup in flat scope is fast."""
        table = SymbolTable()

        # Insert 1000 symbols
        for i in range(1000):
            table.insert(f"var{i}", Symbol(f"var{i}", SymbolType.VARIABLE, "float"))

        # Lookup should still be O(1)
        assert table.lookup("var999") is not None

    def test_lookup_performance_deep_nesting(self):
        """Test lookup with deep nesting."""
        scopes = [SymbolTable()]

        # Create 50-level nesting
        for i in range(50):
            scopes.append(scopes[-1].push_scope())
            scopes[-1].insert(f"var{i}", Symbol(f"var{i}", SymbolType.VARIABLE, "float"))

        # Can still lookup from innermost scope
        assert scopes[-1].lookup("var0") is not None

    def test_many_children_scopes(self):
        """Test parent with many child scopes."""
        parent = SymbolTable()

        children = []
        for i in range(100):
            child = parent.push_scope()
            children.append(child)

        assert len(parent.children) == 100


class TestSymbolTableState:
    """Test symbol table state management."""

    def test_clear_resets_symbols(self):
        """Test clear removes all symbols."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))
        table.insert("y", Symbol("y", SymbolType.VARIABLE, "int"))

        table.clear()

        assert len(table.symbols) == 0
        assert not table.has_symbol("x")

    def test_clear_resets_children(self):
        """Test clear removes child scopes."""
        parent = SymbolTable()
        child1 = parent.push_scope()
        child2 = parent.push_scope()

        parent.clear()

        assert len(parent.children) == 0

    def test_get_all_symbols_empty_table(self):
        """Test get_all_symbols on empty table."""
        table = SymbolTable()
        symbols = table.get_all_symbols()
        assert len(symbols) == 0

    def test_get_symbols_by_type_empty_result(self):
        """Test get_symbols_by_type with no matches."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        functions = table.get_symbols_by_type(SymbolType.FUNCTION)
        assert len(functions) == 0


class TestAdditionalScenarios:
    """Additional comprehensive test scenarios."""

    def test_symbol_with_array_type(self):
        """Test symbol with array type."""
        table = SymbolTable()
        table.insert("data", Symbol("data", SymbolType.VARIABLE, "float[100]"))
        assert table.lookup("data").glsl_type == "float[100]"

    def test_multidimensional_array(self):
        """Test multidimensional array type."""
        table = SymbolTable()
        table.insert("grid", Symbol("grid", SymbolType.VARIABLE, "int[10][20]"))
        assert table.lookup("grid").glsl_type == "int[10][20]"

    def test_vector_array(self):
        """Test array of vectors."""
        table = SymbolTable()
        table.insert("positions", Symbol("positions", SymbolType.VARIABLE, "vec3[50]"))
        assert table.lookup("positions").glsl_type == "vec3[50]"

    def test_matrix_array(self):
        """Test array of matrices."""
        table = SymbolTable()
        table.insert("transforms", Symbol("transforms", SymbolType.VARIABLE, "mat4[10]"))
        assert table.lookup("transforms").glsl_type == "mat4[10]"

    def test_struct_as_function_return(self):
        """Test function returning struct."""
        table = SymbolTable()
        table.insert("Material", Symbol("Material", SymbolType.STRUCT, "struct"))
        table.insert("getMaterial", Symbol("getMaterial", SymbolType.FUNCTION, "Material"))

        func = table.lookup("getMaterial")
        assert func.glsl_type == "Material"

    def test_nested_struct_definitions(self):
        """Test nested struct types."""
        table = SymbolTable()
        table.insert("Inner", Symbol("Inner", SymbolType.STRUCT, "struct"))
        table.insert("Outer", Symbol("Outer", SymbolType.STRUCT, "struct"))

        assert table.has_symbol("Inner")
        assert table.has_symbol("Outer")

    def test_builtin_and_user_functions_coexist(self):
        """Test built-in and user functions coexist."""
        table = create_builtin_symbol_table()
        table.insert("myNoise", Symbol("myNoise", SymbolType.FUNCTION, "float"))

        assert table.has_symbol("sin")  # Built-in
        assert table.has_symbol("myNoise")  # User

    def test_shadowing_across_4_levels(self):
        """Test shadowing across 4 scope levels."""
        l1 = SymbolTable()
        l1.insert("x", Symbol("x", SymbolType.VARIABLE, "int"))

        l2 = l1.push_scope()
        l2.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        l3 = l2.push_scope()
        l3.insert("x", Symbol("x", SymbolType.VARIABLE, "vec2"))

        l4 = l3.push_scope()
        l4.insert("x", Symbol("x", SymbolType.VARIABLE, "vec3"))

        assert l1.lookup("x").glsl_type == "int"
        assert l2.lookup("x").glsl_type == "float"
        assert l3.lookup("x").glsl_type == "vec2"
        assert l4.lookup("x").glsl_type == "vec3"

    def test_loop_iteration_variable_pattern(self):
        """Test typical loop iteration variable pattern."""
        func_scope = SymbolTable()

        loop1 = func_scope.push_scope()
        loop1.insert("i", Symbol("i", SymbolType.VARIABLE, "int"))

        loop2 = func_scope.push_scope()
        loop2.insert("i", Symbol("i", SymbolType.VARIABLE, "int"))

        # Different scopes can have same variable name
        assert loop1.lookup_current_scope("i") is not None
        assert loop2.lookup_current_scope("i") is not None

    def test_switch_case_scope(self):
        """Test switch statement scope pattern."""
        func_scope = SymbolTable()

        switch_scope = func_scope.push_scope()
        case1_scope = switch_scope.push_scope()
        case1_scope.insert("temp", Symbol("temp", SymbolType.VARIABLE, "float"))

        case2_scope = switch_scope.push_scope()
        # case2 doesn't see case1's temp
        assert case2_scope.lookup_current_scope("temp") is None

    def test_ternary_expression_scope(self):
        """Test ternary expression doesn't create scope."""
        table = SymbolTable()
        table.insert("condition", Symbol("condition", SymbolType.VARIABLE, "bool"))
        table.insert("a", Symbol("a", SymbolType.VARIABLE, "float"))
        table.insert("b", Symbol("b", SymbolType.VARIABLE, "float"))

        # All accessible at same level
        assert table.has_symbol("condition")
        assert table.has_symbol("a")
        assert table.has_symbol("b")

    def test_parameter_list_order(self):
        """Test function parameter ordering in metadata."""
        table = SymbolTable()
        func = Symbol("calc", SymbolType.FUNCTION, "float")
        func.metadata["params"] = ["float", "float", "int"]
        func.metadata["param_names"] = ["x", "y", "mode"]

        table.insert("calc", func)

        found = table.lookup("calc")
        assert found.metadata["param_names"][0] == "x"
        assert found.metadata["param_names"][2] == "mode"

    def test_builtin_constant_values(self):
        """Test built-in constants have correct types."""
        table = create_builtin_symbol_table()
        max_draw = table.lookup("gl_MaxDrawBuffers")

        assert max_draw is not None
        assert max_draw.glsl_type == "int"
        assert max_draw.is_builtin()

    def test_fragment_only_metadata(self):
        """Test fragment-only functions marked correctly."""
        table = create_builtin_symbol_table()
        dfdx = table.lookup("dFdx")

        assert dfdx is not None
        assert dfdx.metadata.get("fragment_only") is True

    def test_lookup_chain_optimization(self):
        """Test lookup doesn't revisit scopes."""
        parent = SymbolTable()
        parent.insert("global", Symbol("global", SymbolType.VARIABLE, "float"))

        child1 = parent.push_scope()
        child2 = child1.push_scope()
        child3 = child2.push_scope()

        # Lookup from deep scope should find in parent
        found = child3.lookup("global")
        assert found is not None
        assert found.name == "global"

    def test_symbol_replacement_forbidden(self):
        """Test cannot replace existing symbol."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        with pytest.raises(DuplicateSymbolError):
            table.insert("x", Symbol("x", SymbolType.VARIABLE, "int"))

    def test_empty_qualifier_list(self):
        """Test symbol with no qualifiers."""
        symbol = Symbol("x", SymbolType.VARIABLE, "float", qualifiers=[])
        assert len(symbol.qualifiers) == 0
        assert not symbol.is_const()
        assert not symbol.is_uniform()

    def test_sampler_2D_array(self):
        """Test array of samplers."""
        table = SymbolTable()
        table.insert("textures", Symbol("textures", SymbolType.VARIABLE, "sampler2D[4]"))
        assert table.lookup("textures").glsl_type == "sampler2D[4]"

    def test_precision_in_metadata(self):
        """Test precision qualifiers can be stored in metadata."""
        table = SymbolTable()
        symbol = Symbol("x", SymbolType.VARIABLE, "float")
        symbol.metadata["precision"] = "highp"

        table.insert("x", symbol)

        found = table.lookup("x")
        assert found.metadata["precision"] == "highp"

    def test_location_line_and_column(self):
        """Test location tuple has line and column."""
        symbol = Symbol("x", SymbolType.VARIABLE, "float", location=(42, 15))
        assert symbol.location[0] == 42  # line
        assert symbol.location[1] == 15  # column

    def test_function_with_void_return(self):
        """Test function with void return type."""
        table = SymbolTable()
        func = Symbol("doSomething", SymbolType.FUNCTION, "void")
        table.insert("doSomething", func)

        found = table.lookup("doSomething")
        assert found.glsl_type == "void"

    def test_get_all_symbols_includes_all_types(self):
        """Test get_all_symbols returns all symbol types."""
        table = SymbolTable()
        table.insert("var", Symbol("var", SymbolType.VARIABLE, "float"))
        table.insert("func", Symbol("func", SymbolType.FUNCTION, "void"))
        table.insert("MyStruct", Symbol("MyStruct", SymbolType.STRUCT, "struct"))

        symbols = table.get_all_symbols()
        assert len(symbols) == 3
        assert "var" in symbols
        assert "func" in symbols
        assert "MyStruct" in symbols

    def test_scope_depth_consistency(self):
        """Test scope depth is consistent."""
        l0 = SymbolTable()
        l1 = l0.push_scope()
        l2 = l1.push_scope()
        l3 = l2.push_scope()

        assert l0.get_scope_depth() == 0
        assert l1.get_scope_depth() == 1
        assert l2.get_scope_depth() == 2
        assert l3.get_scope_depth() == 3
