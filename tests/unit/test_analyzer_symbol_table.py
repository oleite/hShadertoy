"""
Tests for SymbolTable class and operations.

Tests symbol table insert/lookup, hierarchical scoping, shadowing,
and error handling.
Target: 90 tests
"""

import pytest
from glsl_to_opencl.analyzer.symbol_table import (
    Symbol,
    SymbolType,
    SymbolTable,
    DuplicateSymbolError,
    UndeclaredSymbolError,
    BuiltinRedeclarationError,
)


class TestSymbolTableConstruction:
    """Test SymbolTable construction."""

    def test_create_empty_symbol_table(self):
        """Test creating empty symbol table."""
        table = SymbolTable()
        assert table.parent is None
        assert len(table.symbols) == 0
        assert table.get_scope_depth() == 0

    def test_create_symbol_table_with_parent(self):
        """Test creating symbol table with parent."""
        parent = SymbolTable()
        child = SymbolTable(parent=parent)
        assert child.parent is parent
        assert child.get_scope_depth() == 1


class TestBasicInsertLookup:
    """Test basic insert and lookup operations."""

    def test_insert_simple_variable(self):
        """Test inserting a simple variable."""
        table = SymbolTable()
        symbol = Symbol("x", SymbolType.VARIABLE, "float")
        table.insert("x", symbol)
        assert table.has_symbol("x")

    def test_lookup_existing_symbol(self):
        """Test looking up existing symbol."""
        table = SymbolTable()
        symbol = Symbol("x", SymbolType.VARIABLE, "float")
        table.insert("x", symbol)

        found = table.lookup("x")
        assert found is not None
        assert found.name == "x"
        assert found.glsl_type == "float"

    def test_lookup_nonexistent_symbol(self):
        """Test looking up non-existent symbol returns None."""
        table = SymbolTable()
        found = table.lookup("x")
        assert found is None

    def test_insert_multiple_variables(self):
        """Test inserting multiple variables."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))
        table.insert("y", Symbol("y", SymbolType.VARIABLE, "int"))
        table.insert("z", Symbol("z", SymbolType.VARIABLE, "vec3"))

        assert table.has_symbol("x")
        assert table.has_symbol("y")
        assert table.has_symbol("z")

    def test_lookup_current_scope_only(self):
        """Test lookup_current_scope doesn't check parents."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        child = table.push_scope()
        found = child.lookup_current_scope("x")
        assert found is None  # Not in current scope

    def test_lookup_with_recursive_false(self):
        """Test lookup with recursive=False."""
        parent = SymbolTable()
        parent.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        child = parent.push_scope()
        found = child.lookup("x", recursive=False)
        assert found is None  # Not in current scope

    def test_has_symbol_true(self):
        """Test has_symbol returns True for existing symbol."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))
        assert table.has_symbol("x") is True

    def test_has_symbol_false(self):
        """Test has_symbol returns False for non-existent symbol."""
        table = SymbolTable()
        assert table.has_symbol("x") is False


class TestDuplicateDetection:
    """Test duplicate symbol detection."""

    def test_insert_duplicate_raises_error(self):
        """Test inserting duplicate symbol raises error."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        with pytest.raises(DuplicateSymbolError) as excinfo:
            table.insert("x", Symbol("x", SymbolType.VARIABLE, "int"))

        assert "x" in str(excinfo.value)

    def test_duplicate_error_contains_name(self):
        """Test DuplicateSymbolError contains symbol name."""
        table = SymbolTable()
        table.insert("myVar", Symbol("myVar", SymbolType.VARIABLE, "float"))

        try:
            table.insert("myVar", Symbol("myVar", SymbolType.VARIABLE, "int"))
            assert False, "Should have raised DuplicateSymbolError"
        except DuplicateSymbolError as e:
            assert e.name == "myVar"

    def test_duplicate_in_different_scopes_allowed(self):
        """Test duplicate names in different scopes is allowed."""
        parent = SymbolTable()
        parent.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        child = parent.push_scope()
        child.insert("x", Symbol("x", SymbolType.VARIABLE, "int"))

        # Both should exist
        assert parent.lookup_current_scope("x").glsl_type == "float"
        assert child.lookup_current_scope("x").glsl_type == "int"


class TestHierarchicalScoping:
    """Test hierarchical scope operations."""

    def test_push_scope_creates_child(self):
        """Test push_scope creates child scope."""
        parent = SymbolTable()
        child = parent.push_scope()

        assert child.parent is parent
        assert child in parent.children
        assert child.get_scope_depth() == 1

    def test_pop_scope_returns_parent(self):
        """Test pop_scope returns parent scope."""
        parent = SymbolTable()
        child = parent.push_scope()

        returned = child.pop_scope()
        assert returned is parent

    def test_pop_scope_at_global_returns_none(self):
        """Test pop_scope at global scope returns None."""
        table = SymbolTable()
        assert table.pop_scope() is None

    def test_lookup_in_parent_scope(self):
        """Test lookup finds symbol in parent scope."""
        parent = SymbolTable()
        parent.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        child = parent.push_scope()
        found = child.lookup("x")

        assert found is not None
        assert found.glsl_type == "float"

    def test_lookup_in_grandparent_scope(self):
        """Test lookup finds symbol in grandparent scope."""
        grandparent = SymbolTable()
        grandparent.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        parent = grandparent.push_scope()
        child = parent.push_scope()

        found = child.lookup("x")
        assert found is not None
        assert found.glsl_type == "float"

    def test_three_level_nesting(self):
        """Test three levels of nested scopes."""
        level1 = SymbolTable()
        level2 = level1.push_scope()
        level3 = level2.push_scope()

        assert level1.get_scope_depth() == 0
        assert level2.get_scope_depth() == 1
        assert level3.get_scope_depth() == 2

    def test_multiple_children(self):
        """Test parent can have multiple children."""
        parent = SymbolTable()
        child1 = parent.push_scope()
        child2 = parent.push_scope()

        assert len(parent.children) == 2
        assert child1 in parent.children
        assert child2 in parent.children


class TestSymbolShadowing:
    """Test symbol shadowing behavior."""

    def test_local_shadows_global(self):
        """Test local variable shadows global variable."""
        global_scope = SymbolTable()
        global_scope.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        function_scope = global_scope.push_scope()
        function_scope.insert("x", Symbol("x", SymbolType.VARIABLE, "int"))

        # Lookup in function scope finds int version
        found = function_scope.lookup("x")
        assert found.glsl_type == "int"

        # Lookup in global scope finds float version
        found_global = global_scope.lookup("x")
        assert found_global.glsl_type == "float"

    def test_block_shadows_function(self):
        """Test block variable shadows function variable."""
        function_scope = SymbolTable()
        function_scope.insert("i", Symbol("i", SymbolType.VARIABLE, "int"))

        block_scope = function_scope.push_scope()
        block_scope.insert("i", Symbol("i", SymbolType.VARIABLE, "float"))

        found_block = block_scope.lookup("i")
        assert found_block.glsl_type == "float"

        found_function = function_scope.lookup("i")
        assert found_function.glsl_type == "int"

    def test_shadowing_chain(self):
        """Test shadowing across multiple levels."""
        level1 = SymbolTable()
        level1.insert("x", Symbol("x", SymbolType.VARIABLE, "int"))

        level2 = level1.push_scope()
        level2.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        level3 = level2.push_scope()
        level3.insert("x", Symbol("x", SymbolType.VARIABLE, "vec3"))

        # Each level sees its own version
        assert level1.lookup("x").glsl_type == "int"
        assert level2.lookup("x").glsl_type == "float"
        assert level3.lookup("x").glsl_type == "vec3"


class TestBuiltinProtection:
    """Test built-in symbol protection."""

    def test_cannot_redeclare_builtin(self):
        """Test cannot redeclare built-in symbol."""
        table = SymbolTable()
        table.insert("sin", Symbol("sin", SymbolType.BUILTIN, "float(float)"))

        with pytest.raises(BuiltinRedeclarationError):
            table.insert("sin", Symbol("sin", SymbolType.VARIABLE, "float"))

    def test_cannot_shadow_builtin_in_child_scope(self):
        """Test cannot shadow built-in in child scope."""
        parent = SymbolTable()
        parent.insert("cos", Symbol("cos", SymbolType.BUILTIN, "float(float)"))

        child = parent.push_scope()
        with pytest.raises(BuiltinRedeclarationError):
            child.insert("cos", Symbol("cos", SymbolType.VARIABLE, "float"))

    def test_builtin_error_contains_name(self):
        """Test BuiltinRedeclarationError contains symbol name."""
        table = SymbolTable()
        table.insert("sqrt", Symbol("sqrt", SymbolType.BUILTIN, "float(float)"))

        try:
            table.insert("sqrt", Symbol("sqrt", SymbolType.VARIABLE, "float"))
            assert False, "Should have raised BuiltinRedeclarationError"
        except BuiltinRedeclarationError as e:
            assert e.name == "sqrt"


class TestGetAllSymbols:
    """Test get_all_symbols method."""

    def test_get_all_symbols_current_scope(self):
        """Test get_all_symbols for current scope only."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))
        table.insert("y", Symbol("y", SymbolType.VARIABLE, "int"))

        symbols = table.get_all_symbols(include_parents=False)
        assert len(symbols) == 2
        assert "x" in symbols
        assert "y" in symbols

    def test_get_all_symbols_with_parents(self):
        """Test get_all_symbols includes parent symbols."""
        parent = SymbolTable()
        parent.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        child = parent.push_scope()
        child.insert("y", Symbol("y", SymbolType.VARIABLE, "int"))

        symbols = child.get_all_symbols(include_parents=True)
        assert len(symbols) == 2
        assert "x" in symbols
        assert "y" in symbols

    def test_get_all_symbols_shadowing(self):
        """Test get_all_symbols with shadowing."""
        parent = SymbolTable()
        parent.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        child = parent.push_scope()
        child.insert("x", Symbol("x", SymbolType.VARIABLE, "int"))

        symbols = child.get_all_symbols(include_parents=True)
        # Should get child's version (int)
        assert symbols["x"].glsl_type == "int"


class TestGetSymbolsByType:
    """Test get_symbols_by_type method."""

    def test_get_variables_only(self):
        """Test getting only variable symbols."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))
        table.insert("y", Symbol("y", SymbolType.VARIABLE, "int"))
        table.insert("func", Symbol("func", SymbolType.FUNCTION, "void"))

        variables = table.get_symbols_by_type(SymbolType.VARIABLE)
        assert len(variables) == 2

    def test_get_functions_only(self):
        """Test getting only function symbols."""
        table = SymbolTable()
        table.insert("func1", Symbol("func1", SymbolType.FUNCTION, "void"))
        table.insert("func2", Symbol("func2", SymbolType.FUNCTION, "float"))
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        functions = table.get_symbols_by_type(SymbolType.FUNCTION)
        assert len(functions) == 2

    def test_get_builtins_only(self):
        """Test getting only built-in symbols."""
        table = SymbolTable()
        table.insert("sin", Symbol("sin", SymbolType.BUILTIN, "float(float)"))
        table.insert("cos", Symbol("cos", SymbolType.BUILTIN, "float(float)"))
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        builtins = table.get_symbols_by_type(SymbolType.BUILTIN)
        assert len(builtins) == 2


class TestClearMethod:
    """Test clear() method."""

    def test_clear_removes_all_symbols(self):
        """Test clear removes all symbols."""
        table = SymbolTable()
        table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))
        table.insert("y", Symbol("y", SymbolType.VARIABLE, "int"))

        table.clear()
        assert len(table.symbols) == 0
        assert table.has_symbol("x") is False

    def test_clear_removes_children(self):
        """Test clear removes child scopes."""
        parent = SymbolTable()
        child1 = parent.push_scope()
        child2 = parent.push_scope()

        parent.clear()
        assert len(parent.children) == 0


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_function_with_local_variables(self):
        """Test function scope with local variables."""
        global_scope = SymbolTable()
        global_scope.insert("globalVar", Symbol("globalVar", SymbolType.VARIABLE, "float"))

        function_scope = global_scope.push_scope()
        function_scope.insert("param", Symbol("param", SymbolType.PARAMETER, "int"))
        function_scope.insert("localVar", Symbol("localVar", SymbolType.VARIABLE, "float"))

        # Function can see global variable
        assert function_scope.lookup("globalVar") is not None

        # Function can see its own symbols
        assert function_scope.lookup("param") is not None
        assert function_scope.lookup("localVar") is not None

        # Global cannot see function symbols
        assert global_scope.lookup("param") is None
        assert global_scope.lookup("localVar") is None

    def test_nested_for_loops(self):
        """Test nested for loop scopes."""
        function_scope = SymbolTable()

        # Outer loop
        outer_loop = function_scope.push_scope()
        outer_loop.insert("i", Symbol("i", SymbolType.VARIABLE, "int"))

        # Inner loop
        inner_loop = outer_loop.push_scope()
        inner_loop.insert("j", Symbol("j", SymbolType.VARIABLE, "int"))

        # Inner can see outer's i
        assert inner_loop.lookup("i") is not None
        # Outer cannot see inner's j
        assert outer_loop.lookup("j") is None

    def test_if_block_scope(self):
        """Test if-block scope."""
        function_scope = SymbolTable()
        function_scope.insert("condition", Symbol("condition", SymbolType.VARIABLE, "bool"))

        if_block = function_scope.push_scope()
        if_block.insert("temp", Symbol("temp", SymbolType.VARIABLE, "float"))

        # if-block can see function variable
        assert if_block.lookup("condition") is not None

        # Function cannot see if-block variable
        assert function_scope.lookup("temp") is None

    def test_struct_and_variables(self):
        """Test struct definition and variables of that type."""
        global_scope = SymbolTable()

        # Define struct
        global_scope.insert("Light", Symbol("Light", SymbolType.STRUCT, "struct"))

        # Create variable of struct type
        global_scope.insert("mainLight", Symbol("mainLight", SymbolType.VARIABLE, "Light"))

        assert global_scope.lookup("Light") is not None
        assert global_scope.lookup("mainLight") is not None

    def test_function_overloading(self):
        """Test multiple functions with same name (overloading)."""
        table = SymbolTable()

        # In real implementation, we'd handle overloading with metadata
        # For now, just test that we can store function signature info
        func1 = Symbol("mix", SymbolType.BUILTIN, "float(float,float,float)")
        func1.metadata["params"] = ["float", "float", "float"]

        func2 = Symbol("mix", SymbolType.BUILTIN, "vec3(vec3,vec3,vec3)")
        func2.metadata["params"] = ["vec3", "vec3", "vec3"]

        # Would need special handling for overloading
        # For now just test that metadata can be stored
        assert func1.metadata["params"] == ["float", "float", "float"]
        assert func2.metadata["params"] == ["vec3", "vec3", "vec3"]

    def test_loop_variable_shadowing_outer(self):
        """Test loop variable shadowing outer variable."""
        outer_scope = SymbolTable()
        outer_scope.insert("i", Symbol("i", SymbolType.VARIABLE, "float"))

        loop_scope = outer_scope.push_scope()
        loop_scope.insert("i", Symbol("i", SymbolType.VARIABLE, "int"))

        # Loop sees int version
        found = loop_scope.lookup("i")
        assert found.glsl_type == "int"

        # Outer sees float version
        found_outer = outer_scope.lookup("i")
        assert found_outer.glsl_type == "float"

    def test_multiple_functions_same_level(self):
        """Test multiple functions at same scope level."""
        global_scope = SymbolTable()

        func1_scope = global_scope.push_scope()
        func1_scope.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        # Pop back to global
        # func2_scope = func1_scope.pop_scope().push_scope()
        func2_scope = global_scope.push_scope()
        func2_scope.insert("y", Symbol("y", SymbolType.VARIABLE, "int"))

        # func1 symbols not visible in func2
        assert func2_scope.lookup("x") is None

        # Each function has its own symbols
        assert func1_scope.lookup_current_scope("x") is not None
        assert func2_scope.lookup_current_scope("y") is not None

    def test_const_and_regular_variable(self):
        """Test const and regular variables."""
        table = SymbolTable()

        const_var = Symbol("PI", SymbolType.CONSTANT, "float", qualifiers=["const"])
        regular_var = Symbol("angle", SymbolType.VARIABLE, "float")

        table.insert("PI", const_var)
        table.insert("angle", regular_var)

        found_const = table.lookup("PI")
        assert found_const.is_const()

        found_regular = table.lookup("angle")
        assert not found_regular.is_const()

    def test_uniform_variables(self):
        """Test uniform variables."""
        global_scope = SymbolTable()

        global_scope.insert(
            "iTime", Symbol("iTime", SymbolType.VARIABLE, "float", qualifiers=["uniform"])
        )
        global_scope.insert(
            "iResolution",
            Symbol("iResolution", SymbolType.VARIABLE, "vec3", qualifiers=["uniform"]),
        )

        itime = global_scope.lookup("iTime")
        assert itime.is_uniform()

        iresolution = global_scope.lookup("iResolution")
        assert iresolution.is_uniform()
