"""
Test Type Checker

Tests for GLSL type inference and type checking.
Validates type system, operator compatibility, function signatures, and conversions.
"""

import pytest
from glsl_to_opencl.analyzer.type_checker import (
    GLSLType, TypeCategory, ScalarType,
    FLOAT, INT, UINT, BOOL, VOID,
    VEC2, VEC3, VEC4, IVEC2, IVEC3, IVEC4, UVEC2, UVEC3, UVEC4, BVEC2, BVEC3, BVEC4,
    MAT2, MAT3, MAT4, SAMPLER2D, SAMPLER3D, SAMPLERCUBE,
    TYPE_NAME_MAP, parse_type_string,
    TypeChecker, TypeCheckError, TypeMismatchError, InvalidOperationError, UndefinedTypeError,
)
from glsl_to_opencl.analyzer.symbol_table import SymbolTable, Symbol, SymbolType


# ============================================================================
# 1. GLSLType Tests (40 tests)
# ============================================================================

class TestGLSLType:
    """Tests for GLSLType class and type system."""

    def test_scalar_types(self):
        """Test scalar type creation."""
        assert FLOAT.category == TypeCategory.SCALAR
        assert FLOAT.base_type == ScalarType.FLOAT
        assert str(FLOAT) == "float"

        assert INT.category == TypeCategory.SCALAR
        assert INT.base_type == ScalarType.INT
        assert str(INT) == "int"

        assert UINT.category == TypeCategory.SCALAR
        assert UINT.base_type == ScalarType.UINT
        assert str(UINT) == "uint"

        assert BOOL.category == TypeCategory.SCALAR
        assert BOOL.base_type == ScalarType.BOOL
        assert str(BOOL) == "bool"

    def test_void_type(self):
        """Test void type."""
        assert VOID.category == TypeCategory.VOID
        assert str(VOID) == "void"

    def test_vector_types(self):
        """Test vector type creation."""
        assert VEC2.category == TypeCategory.VECTOR
        assert VEC2.base_type == ScalarType.FLOAT
        assert VEC2.dimensions == (2,)
        assert VEC2.vector_size() == 2
        assert str(VEC2) == "vec2"

        assert VEC3.vector_size() == 3
        assert str(VEC3) == "vec3"

        assert VEC4.vector_size() == 4
        assert str(VEC4) == "vec4"

    def test_integer_vector_types(self):
        """Test integer vector types."""
        assert IVEC2.base_type == ScalarType.INT
        assert str(IVEC2) == "ivec2"

        assert IVEC3.base_type == ScalarType.INT
        assert str(IVEC3) == "ivec3"

        assert IVEC4.base_type == ScalarType.INT
        assert str(IVEC4) == "ivec4"

    def test_unsigned_vector_types(self):
        """Test unsigned vector types."""
        assert UVEC2.base_type == ScalarType.UINT
        assert str(UVEC2) == "uvec2"

        assert UVEC3.base_type == ScalarType.UINT
        assert str(UVEC3) == "uvec3"

        assert UVEC4.base_type == ScalarType.UINT
        assert str(UVEC4) == "uvec4"

    def test_bool_vector_types(self):
        """Test bool vector types."""
        assert BVEC2.base_type == ScalarType.BOOL
        assert str(BVEC2) == "bvec2"

        assert BVEC3.base_type == ScalarType.BOOL
        assert str(BVEC3) == "bvec3"

        assert BVEC4.base_type == ScalarType.BOOL
        assert str(BVEC4) == "bvec4"

    def test_matrix_types(self):
        """Test matrix type creation."""
        assert MAT2.category == TypeCategory.MATRIX
        assert MAT2.base_type == ScalarType.FLOAT
        assert MAT2.dimensions == (2, 2)
        assert MAT2.matrix_dimensions() == (2, 2)
        assert str(MAT2) == "mat2"

        assert MAT3.matrix_dimensions() == (3, 3)
        assert str(MAT3) == "mat3"

        assert MAT4.matrix_dimensions() == (4, 4)
        assert str(MAT4) == "mat4"

    def test_non_square_matrix_types(self):
        """Test non-square matrix types."""
        mat2x3 = GLSLType(TypeCategory.MATRIX, ScalarType.FLOAT, (2, 3))
        assert mat2x3.matrix_dimensions() == (2, 3)
        assert str(mat2x3) == "mat2x3"

        mat3x4 = GLSLType(TypeCategory.MATRIX, ScalarType.FLOAT, (3, 4))
        assert str(mat3x4) == "mat3x4"

    def test_sampler_types(self):
        """Test sampler type creation."""
        assert SAMPLER2D.category == TypeCategory.SAMPLER
        assert SAMPLER2D.name == "sampler2D"
        assert str(SAMPLER2D) == "sampler2D"

        assert SAMPLER3D.name == "sampler3D"
        assert str(SAMPLER3D) == "sampler3D"

        assert SAMPLERCUBE.name == "samplerCube"
        assert str(SAMPLERCUBE) == "samplerCube"

    def test_type_predicates_scalar(self):
        """Test type predicate methods for scalars."""
        assert FLOAT.is_numeric()
        assert FLOAT.is_float()
        assert not FLOAT.is_integer()
        assert not FLOAT.is_bool()
        assert not FLOAT.is_vector()
        assert not FLOAT.is_matrix()

        assert INT.is_numeric()
        assert INT.is_integer()
        assert not INT.is_float()
        assert not INT.is_bool()

        assert BOOL.is_bool()
        assert not BOOL.is_numeric()

    def test_type_predicates_vector(self):
        """Test type predicate methods for vectors."""
        assert VEC3.is_numeric()
        assert VEC3.is_float()
        assert VEC3.is_vector()
        assert not VEC3.is_matrix()
        assert not VEC3.is_integer()

        assert IVEC2.is_numeric()
        assert IVEC2.is_integer()
        assert IVEC2.is_vector()
        assert not IVEC2.is_float()

        assert BVEC4.is_bool()
        assert BVEC4.is_vector()
        assert not BVEC4.is_numeric()

    def test_type_predicates_matrix(self):
        """Test type predicate methods for matrices."""
        assert MAT3.is_numeric()
        assert MAT3.is_float()
        assert MAT3.is_matrix()
        assert not MAT3.is_vector()
        assert not MAT3.is_integer()

    def test_type_predicates_sampler(self):
        """Test type predicate methods for samplers."""
        assert not SAMPLER2D.is_numeric()
        assert not SAMPLER2D.is_float()
        assert not SAMPLER2D.is_vector()
        assert not SAMPLER2D.is_matrix()

    def test_vector_size(self):
        """Test vector_size method."""
        assert VEC2.vector_size() == 2
        assert VEC3.vector_size() == 3
        assert VEC4.vector_size() == 4
        assert IVEC3.vector_size() == 3
        assert FLOAT.vector_size() is None
        assert MAT3.vector_size() is None

    def test_matrix_dimensions(self):
        """Test matrix_dimensions method."""
        assert MAT2.matrix_dimensions() == (2, 2)
        assert MAT3.matrix_dimensions() == (3, 3)
        assert MAT4.matrix_dimensions() == (4, 4)
        assert VEC3.matrix_dimensions() is None
        assert FLOAT.matrix_dimensions() is None

    def test_type_equality(self):
        """Test type equality."""
        assert FLOAT == FLOAT
        assert VEC3 == VEC3
        assert MAT4 == MAT4
        assert FLOAT != INT
        assert VEC2 != VEC3
        assert IVEC3 != VEC3

    def test_type_const_qualifier(self):
        """Test const qualifier."""
        const_float = GLSLType(TypeCategory.SCALAR, ScalarType.FLOAT, is_const=True)
        assert const_float.is_const
        assert not FLOAT.is_const

    def test_type_name_map(self):
        """Test TYPE_NAME_MAP completeness."""
        assert TYPE_NAME_MAP["float"] == FLOAT
        assert TYPE_NAME_MAP["int"] == INT
        assert TYPE_NAME_MAP["vec2"] == VEC2
        assert TYPE_NAME_MAP["vec3"] == VEC3
        assert TYPE_NAME_MAP["vec4"] == VEC4
        assert TYPE_NAME_MAP["ivec2"] == IVEC2
        assert TYPE_NAME_MAP["mat2"] == MAT2
        assert TYPE_NAME_MAP["mat3"] == MAT3
        assert TYPE_NAME_MAP["mat4"] == MAT4
        assert TYPE_NAME_MAP["sampler2D"] == SAMPLER2D

    def test_parse_type_string_scalars(self):
        """Test parsing scalar type strings."""
        assert parse_type_string("float") == FLOAT
        assert parse_type_string("int") == INT
        assert parse_type_string("uint") == UINT
        assert parse_type_string("bool") == BOOL

    def test_parse_type_string_vectors(self):
        """Test parsing vector type strings."""
        assert parse_type_string("vec2") == VEC2
        assert parse_type_string("vec3") == VEC3
        assert parse_type_string("vec4") == VEC4
        assert parse_type_string("ivec2") == IVEC2
        assert parse_type_string("uvec3") == UVEC3
        assert parse_type_string("bvec4") == BVEC4

    def test_parse_type_string_matrices(self):
        """Test parsing matrix type strings."""
        assert parse_type_string("mat2") == MAT2
        assert parse_type_string("mat3") == MAT3
        assert parse_type_string("mat4") == MAT4

    def test_parse_type_string_samplers(self):
        """Test parsing sampler type strings."""
        assert parse_type_string("sampler2D") == SAMPLER2D
        assert parse_type_string("sampler3D") == SAMPLER3D
        assert parse_type_string("samplerCube") == SAMPLERCUBE

    def test_parse_type_string_whitespace(self):
        """Test parsing with whitespace."""
        assert parse_type_string(" float ") == FLOAT
        assert parse_type_string("  vec3  ") == VEC3

    def test_parse_type_string_invalid(self):
        """Test parsing invalid type strings."""
        with pytest.raises(UndefinedTypeError):
            parse_type_string("invalid")

        with pytest.raises(UndefinedTypeError):
            parse_type_string("vec5")

    def test_type_string_representation(self):
        """Test string representation of types."""
        assert str(FLOAT) == "float"
        assert str(VEC3) == "vec3"
        assert str(IVEC2) == "ivec2"
        assert str(UVEC4) == "uvec4"
        assert str(BVEC3) == "bvec3"
        assert str(MAT2) == "mat2"
        assert str(MAT4) == "mat4"
        assert str(SAMPLER2D) == "sampler2D"
        assert str(VOID) == "void"


# ============================================================================
# 2. Type Conversion Tests (20 tests)
# ============================================================================

class TestTypeConversions:
    """Tests for implicit type conversions."""

    def setup_method(self):
        """Create type checker for each test."""
        self.symbol_table = SymbolTable()
        self.checker = TypeChecker(self.symbol_table)

    def test_exact_match_scalar(self):
        """Test exact type match for scalars."""
        assert self.checker.can_convert(FLOAT, FLOAT)
        assert self.checker.can_convert(INT, INT)
        assert self.checker.can_convert(UINT, UINT)
        assert self.checker.can_convert(BOOL, BOOL)

    def test_exact_match_vector(self):
        """Test exact type match for vectors."""
        assert self.checker.can_convert(VEC2, VEC2)
        assert self.checker.can_convert(VEC3, VEC3)
        assert self.checker.can_convert(VEC4, VEC4)
        assert self.checker.can_convert(IVEC3, IVEC3)

    def test_exact_match_matrix(self):
        """Test exact type match for matrices."""
        assert self.checker.can_convert(MAT2, MAT2)
        assert self.checker.can_convert(MAT3, MAT3)
        assert self.checker.can_convert(MAT4, MAT4)

    def test_int_to_float(self):
        """Test int to float conversion (widening)."""
        assert self.checker.can_convert(INT, FLOAT)

    def test_uint_to_float(self):
        """Test uint to float conversion (widening)."""
        assert self.checker.can_convert(UINT, FLOAT)

    def test_no_float_to_int(self):
        """Test float to int conversion not allowed (narrowing)."""
        assert not self.checker.can_convert(FLOAT, INT)

    def test_no_float_to_uint(self):
        """Test float to uint conversion not allowed (narrowing)."""
        assert not self.checker.can_convert(FLOAT, UINT)

    def test_no_int_to_uint(self):
        """Test int to uint conversion not allowed."""
        assert not self.checker.can_convert(INT, UINT)

    def test_no_uint_to_int(self):
        """Test uint to int conversion not allowed."""
        assert not self.checker.can_convert(UINT, INT)

    def test_scalar_to_vector_splatting_float(self):
        """Test float to vec splatting."""
        assert self.checker.can_convert(FLOAT, VEC2)
        assert self.checker.can_convert(FLOAT, VEC3)
        assert self.checker.can_convert(FLOAT, VEC4)

    def test_scalar_to_vector_splatting_int(self):
        """Test int to ivec splatting."""
        assert self.checker.can_convert(INT, IVEC2)
        assert self.checker.can_convert(INT, IVEC3)
        assert self.checker.can_convert(INT, IVEC4)

    def test_scalar_to_vector_splatting_bool(self):
        """Test bool to bvec splatting."""
        assert self.checker.can_convert(BOOL, BVEC2)
        assert self.checker.can_convert(BOOL, BVEC3)
        assert self.checker.can_convert(BOOL, BVEC4)

    def test_no_scalar_to_wrong_vector(self):
        """Test scalar to wrong vector type not allowed."""
        assert not self.checker.can_convert(FLOAT, IVEC3)
        assert not self.checker.can_convert(INT, VEC3)
        assert not self.checker.can_convert(BOOL, VEC3)

    def test_no_vector_to_scalar(self):
        """Test vector to scalar conversion not allowed."""
        assert not self.checker.can_convert(VEC3, FLOAT)
        assert not self.checker.can_convert(IVEC2, INT)

    def test_no_vector_size_conversion(self):
        """Test vector size conversion not allowed."""
        assert not self.checker.can_convert(VEC2, VEC3)
        assert not self.checker.can_convert(VEC4, VEC2)

    def test_no_matrix_conversions(self):
        """Test matrix conversions not allowed."""
        assert not self.checker.can_convert(MAT2, MAT3)
        assert not self.checker.can_convert(MAT3, MAT4)
        assert not self.checker.can_convert(FLOAT, MAT2)
        assert not self.checker.can_convert(VEC3, MAT3)

    def test_no_sampler_conversions(self):
        """Test sampler conversions not allowed."""
        assert not self.checker.can_convert(SAMPLER2D, SAMPLER3D)
        assert not self.checker.can_convert(INT, SAMPLER2D)

    def test_no_bool_conversions(self):
        """Test bool conversions not allowed (except exact match)."""
        assert not self.checker.can_convert(BOOL, INT)
        assert not self.checker.can_convert(BOOL, FLOAT)
        assert not self.checker.can_convert(INT, BOOL)
        assert not self.checker.can_convert(FLOAT, BOOL)


# ============================================================================
# 3. TypeChecker Basic Tests (20 tests)
# ============================================================================

class TestTypeCheckerBasics:
    """Tests for TypeChecker initialization and basic functionality."""

    def test_init_with_symbol_table(self):
        """Test TypeChecker initialization."""
        symbol_table = SymbolTable()
        checker = TypeChecker(symbol_table)
        assert checker.symbol_table == symbol_table
        assert isinstance(checker.type_map, dict)
        assert len(checker.type_map) == 0

    def test_check_returns_type_map(self):
        """Test check method returns type map."""
        from glsl_to_opencl.parser import GLSLParser
        parser = GLSLParser()
        ast = parser.parse("void main() {}")

        symbol_table = SymbolTable()
        checker = TypeChecker(symbol_table)
        result = checker.check(ast)

        assert isinstance(result, dict)
        assert result is checker.type_map

    def test_type_map_empty_initially(self):
        """Test type map is empty before checking."""
        symbol_table = SymbolTable()
        checker = TypeChecker(symbol_table)
        assert len(checker.type_map) == 0

    def test_infer_type_not_implemented(self):
        """Test infer_type raises NotImplementedError initially."""
        symbol_table = SymbolTable()
        checker = TypeChecker(symbol_table)
        with pytest.raises(NotImplementedError):
            checker.infer_type(None)

    def test_check_binary_op_works(self):
        """Test check_binary_op is implemented."""
        symbol_table = SymbolTable()
        checker = TypeChecker(symbol_table)
        result = checker.check_binary_op("+", FLOAT, FLOAT)
        assert result == FLOAT

    def test_check_unary_op_implemented(self):
        """Test check_unary_op is implemented."""
        symbol_table = SymbolTable()
        checker = TypeChecker(symbol_table)
        result = checker.check_unary_op("-", FLOAT)
        assert result == FLOAT

    def test_check_function_call_implemented(self):
        """Test check_function_call is implemented for constructors."""
        symbol_table = SymbolTable()
        checker = TypeChecker(symbol_table)
        # Test with constructor (doesn't need symbol table)
        result = checker.check_function_call("vec3", [FLOAT])
        assert result == VEC3

    def test_check_member_access_implemented(self):
        """Test check_member_access is implemented."""
        symbol_table = SymbolTable()
        checker = TypeChecker(symbol_table)
        result = checker.check_member_access(VEC3, "xy")
        assert result == VEC2

    def test_type_check_error_with_location(self):
        """Test TypeCheckError with location."""
        error = TypeCheckError("test error", (10, 5))
        assert error.message == "test error"
        assert error.location == (10, 5)
        assert "test error at (10, 5)" in str(error)

    def test_type_check_error_without_location(self):
        """Test TypeCheckError without location."""
        error = TypeCheckError("test error")
        assert error.message == "test error"
        assert error.location is None
        assert str(error) == "test error"

    def test_type_mismatch_error(self):
        """Test TypeMismatchError."""
        error = TypeMismatchError("type mismatch", (5, 10))
        assert isinstance(error, TypeCheckError)
        assert error.message == "type mismatch"

    def test_invalid_operation_error(self):
        """Test InvalidOperationError."""
        error = InvalidOperationError("invalid operation", (1, 2))
        assert isinstance(error, TypeCheckError)
        assert error.message == "invalid operation"

    def test_undefined_type_error(self):
        """Test UndefinedTypeError."""
        error = UndefinedTypeError("undefined type")
        assert isinstance(error, TypeCheckError)
        assert error.message == "undefined type"

    def test_checker_with_populated_symbol_table(self):
        """Test TypeChecker with symbol table containing symbols."""
        symbol_table = SymbolTable()
        symbol_table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))
        symbol_table.insert("y", Symbol("y", SymbolType.VARIABLE, "vec3"))

        checker = TypeChecker(symbol_table)
        assert checker.symbol_table.lookup("x").glsl_type == "float"
        assert checker.symbol_table.lookup("y").glsl_type == "vec3"

    def test_multiple_checkers_independent(self):
        """Test multiple TypeChecker instances are independent."""
        table1 = SymbolTable()
        table2 = SymbolTable()
        checker1 = TypeChecker(table1)
        checker2 = TypeChecker(table2)

        assert checker1.symbol_table is not checker2.symbol_table
        assert checker1.type_map is not checker2.type_map

    def test_checker_preserves_symbol_table(self):
        """Test TypeChecker doesn't modify symbol table during init."""
        symbol_table = SymbolTable()
        symbol_table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))

        checker = TypeChecker(symbol_table)
        assert symbol_table.lookup("x") is not None
        assert len(symbol_table.get_all_symbols()) == 1

    def test_error_message_content(self):
        """Test error messages contain useful information."""
        error = TypeMismatchError("Expected 'float', got 'int'", (10, 20))
        assert "float" in error.message
        assert "int" in error.message
        assert "(10, 20)" in str(error)

    def test_type_checker_str_representation(self):
        """Test TypeChecker has useful string representation."""
        symbol_table = SymbolTable()
        checker = TypeChecker(symbol_table)
        # Just ensure it doesn't crash
        repr_str = repr(checker)
        assert "TypeChecker" in str(type(checker).__name__)

    def test_empty_ast_check(self):
        """Test checking empty AST doesn't crash."""
        from glsl_to_opencl.parser import GLSLParser
        parser = GLSLParser()
        ast = parser.parse("")  # Empty shader

        symbol_table = SymbolTable()
        checker = TypeChecker(symbol_table)
        result = checker.check(ast)
        assert isinstance(result, dict)


# ============================================================================
# 4. Binary Operator Tests (120 tests)
# ============================================================================

class TestBinaryOperators:
    """Tests for binary operator type checking."""

    def setup_method(self):
        """Create type checker for each test."""
        self.symbol_table = SymbolTable()
        self.checker = TypeChecker(self.symbol_table)

    # Arithmetic Operators: + - * / (40 tests)

    def test_arithmetic_float_float(self):
        """Test float + float -> float."""
        result = self.checker.check_binary_op("+", FLOAT, FLOAT)
        assert result == FLOAT

        result = self.checker.check_binary_op("-", FLOAT, FLOAT)
        assert result == FLOAT

        result = self.checker.check_binary_op("*", FLOAT, FLOAT)
        assert result == FLOAT

        result = self.checker.check_binary_op("/", FLOAT, FLOAT)
        assert result == FLOAT

    def test_arithmetic_int_int(self):
        """Test int + int -> int."""
        result = self.checker.check_binary_op("+", INT, INT)
        assert result == INT

        result = self.checker.check_binary_op("-", INT, INT)
        assert result == INT

        result = self.checker.check_binary_op("*", INT, INT)
        assert result == INT

        result = self.checker.check_binary_op("/", INT, INT)
        assert result == INT

    def test_arithmetic_uint_uint(self):
        """Test uint + uint -> uint."""
        result = self.checker.check_binary_op("+", UINT, UINT)
        assert result == UINT

        result = self.checker.check_binary_op("-", UINT, UINT)
        assert result == UINT

        result = self.checker.check_binary_op("*", UINT, UINT)
        assert result == UINT

        result = self.checker.check_binary_op("/", UINT, UINT)
        assert result == UINT

    def test_arithmetic_vec_vec(self):
        """Test vec3 + vec3 -> vec3 (component-wise)."""
        result = self.checker.check_binary_op("+", VEC3, VEC3)
        assert result == VEC3

        result = self.checker.check_binary_op("-", VEC3, VEC3)
        assert result == VEC3

        result = self.checker.check_binary_op("*", VEC3, VEC3)
        assert result == VEC3

        result = self.checker.check_binary_op("/", VEC3, VEC3)
        assert result == VEC3

    def test_arithmetic_vec_scalar(self):
        """Test vec3 + float -> vec3 (scalar promotion)."""
        result = self.checker.check_binary_op("+", VEC3, FLOAT)
        assert result == VEC3

        result = self.checker.check_binary_op("-", VEC3, FLOAT)
        assert result == VEC3

        result = self.checker.check_binary_op("*", VEC3, FLOAT)
        assert result == VEC3

        result = self.checker.check_binary_op("/", VEC3, FLOAT)
        assert result == VEC3

    def test_arithmetic_scalar_vec(self):
        """Test float + vec3 -> vec3 (scalar promotion, commutative)."""
        result = self.checker.check_binary_op("+", FLOAT, VEC3)
        assert result == VEC3

        result = self.checker.check_binary_op("-", FLOAT, VEC3)
        assert result == VEC3

        result = self.checker.check_binary_op("*", FLOAT, VEC3)
        assert result == VEC3

        result = self.checker.check_binary_op("/", FLOAT, VEC3)
        assert result == VEC3

    def test_arithmetic_ivec_ivec(self):
        """Test ivec3 + ivec3 -> ivec3."""
        result = self.checker.check_binary_op("+", IVEC3, IVEC3)
        assert result == IVEC3

        result = self.checker.check_binary_op("*", IVEC3, IVEC3)
        assert result == IVEC3

    def test_arithmetic_ivec_int(self):
        """Test ivec3 + int -> ivec3."""
        result = self.checker.check_binary_op("+", IVEC3, INT)
        assert result == IVEC3

        result = self.checker.check_binary_op("*", INT, IVEC3)
        assert result == IVEC3

    def test_arithmetic_different_vector_sizes_error(self):
        """Test vec2 + vec3 is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("+", VEC2, VEC3)

        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("*", VEC4, VEC2)

    def test_arithmetic_mixed_types_error(self):
        """Test float + int requires explicit conversion."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("+", FLOAT, INT)

        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("*", VEC3, IVEC3)

    def test_arithmetic_bool_error(self):
        """Test bool + bool is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("+", BOOL, BOOL)

        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("*", BOOL, FLOAT)

    # Matrix-Vector Multiplication (10 tests)

    def test_matrix_vector_multiply_mat2_vec2(self):
        """Test mat2 * vec2 -> vec2."""
        result = self.checker.check_binary_op("*", MAT2, VEC2)
        assert result == VEC2

    def test_matrix_vector_multiply_mat3_vec3(self):
        """Test mat3 * vec3 -> vec3."""
        result = self.checker.check_binary_op("*", MAT3, VEC3)
        assert result == VEC3

    def test_matrix_vector_multiply_mat4_vec4(self):
        """Test mat4 * vec4 -> vec4."""
        result = self.checker.check_binary_op("*", MAT4, VEC4)
        assert result == VEC4

    def test_vector_matrix_multiply_vec2_mat2(self):
        """Test vec2 * mat2 -> vec2."""
        result = self.checker.check_binary_op("*", VEC2, MAT2)
        assert result == VEC2

    def test_matrix_matrix_multiply_mat3_mat3(self):
        """Test mat3 * mat3 -> mat3."""
        result = self.checker.check_binary_op("*", MAT3, MAT3)
        assert result == MAT3

    def test_matrix_scalar_multiply(self):
        """Test mat3 * float -> mat3."""
        result = self.checker.check_binary_op("*", MAT3, FLOAT)
        assert result == MAT3

        result = self.checker.check_binary_op("*", FLOAT, MAT3)
        assert result == MAT3

    def test_matrix_size_mismatch_error(self):
        """Test mat2 * vec3 is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("*", MAT2, VEC3)

        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("*", MAT3, MAT2)

    def test_matrix_addition_mat3_mat3(self):
        """Test mat3 + mat3 -> mat3."""
        result = self.checker.check_binary_op("+", MAT3, MAT3)
        assert result == MAT3

    def test_matrix_vector_addition_error(self):
        """Test mat3 + vec3 is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("+", MAT3, VEC3)

    def test_matrix_different_sizes_error(self):
        """Test mat2 + mat3 is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("+", MAT2, MAT3)

    # Comparison Operators: < <= > >= == != (30 tests)

    def test_comparison_float_float(self):
        """Test float < float -> bool."""
        result = self.checker.check_binary_op("<", FLOAT, FLOAT)
        assert result == BOOL

        result = self.checker.check_binary_op("<=", FLOAT, FLOAT)
        assert result == BOOL

        result = self.checker.check_binary_op(">", FLOAT, FLOAT)
        assert result == BOOL

        result = self.checker.check_binary_op(">=", FLOAT, FLOAT)
        assert result == BOOL

    def test_comparison_int_int(self):
        """Test int < int -> bool."""
        result = self.checker.check_binary_op("<", INT, INT)
        assert result == BOOL

        result = self.checker.check_binary_op(">=", INT, INT)
        assert result == BOOL

    def test_comparison_uint_uint(self):
        """Test uint < uint -> bool."""
        result = self.checker.check_binary_op("<", UINT, UINT)
        assert result == BOOL

    def test_equality_float_float(self):
        """Test float == float -> bool."""
        result = self.checker.check_binary_op("==", FLOAT, FLOAT)
        assert result == BOOL

        result = self.checker.check_binary_op("!=", FLOAT, FLOAT)
        assert result == BOOL

    def test_equality_int_int(self):
        """Test int == int -> bool."""
        result = self.checker.check_binary_op("==", INT, INT)
        assert result == BOOL

        result = self.checker.check_binary_op("!=", INT, INT)
        assert result == BOOL

    def test_equality_bool_bool(self):
        """Test bool == bool -> bool."""
        result = self.checker.check_binary_op("==", BOOL, BOOL)
        assert result == BOOL

        result = self.checker.check_binary_op("!=", BOOL, BOOL)
        assert result == BOOL

    def test_equality_vec_vec(self):
        """Test vec3 == vec3 -> bvec3 (component-wise)."""
        result = self.checker.check_binary_op("==", VEC3, VEC3)
        assert result == BVEC3

        result = self.checker.check_binary_op("!=", VEC3, VEC3)
        assert result == BVEC3

    def test_comparison_vec_vec(self):
        """Test vec3 < vec3 -> bvec3 (component-wise)."""
        result = self.checker.check_binary_op("<", VEC3, VEC3)
        assert result == BVEC3

        result = self.checker.check_binary_op("<=", VEC3, VEC3)
        assert result == BVEC3

        result = self.checker.check_binary_op(">", VEC3, VEC3)
        assert result == BVEC3

        result = self.checker.check_binary_op(">=", VEC3, VEC3)
        assert result == BVEC3

    def test_comparison_ivec_ivec(self):
        """Test ivec2 < ivec2 -> bvec2."""
        result = self.checker.check_binary_op("<", IVEC2, IVEC2)
        assert result == BVEC2

    def test_comparison_mixed_types_error(self):
        """Test float < int is invalid (requires explicit conversion)."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("<", FLOAT, INT)

        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("==", VEC3, IVEC3)

    def test_comparison_vector_sizes_error(self):
        """Test vec2 < vec3 is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("<", VEC2, VEC3)

    def test_comparison_vec_scalar_error(self):
        """Test vec3 < float is invalid (not component-wise in GLSL)."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("<", VEC3, FLOAT)

    def test_equality_matrix_error(self):
        """Test mat3 == mat3 is invalid in GLSL (no direct comparison)."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("==", MAT3, MAT3)

    # Logical Operators: && || (15 tests)

    def test_logical_and_bool_bool(self):
        """Test bool && bool -> bool."""
        result = self.checker.check_binary_op("&&", BOOL, BOOL)
        assert result == BOOL

    def test_logical_or_bool_bool(self):
        """Test bool || bool -> bool."""
        result = self.checker.check_binary_op("||", BOOL, BOOL)
        assert result == BOOL

    def test_logical_and_requires_bool(self):
        """Test float && float is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("&&", FLOAT, FLOAT)

        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("&&", INT, INT)

    def test_logical_or_requires_bool(self):
        """Test int || int is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("||", INT, INT)

    def test_logical_vector_error(self):
        """Test bvec3 && bvec3 is invalid (use all/any)."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("&&", BVEC3, BVEC3)

    # Bitwise Operators: & | ^ << >> (15 tests)

    def test_bitwise_and_int_int(self):
        """Test int & int -> int."""
        result = self.checker.check_binary_op("&", INT, INT)
        assert result == INT

    def test_bitwise_or_int_int(self):
        """Test int | int -> int."""
        result = self.checker.check_binary_op("|", INT, INT)
        assert result == INT

    def test_bitwise_xor_int_int(self):
        """Test int ^ int -> int."""
        result = self.checker.check_binary_op("^", INT, INT)
        assert result == INT

    def test_bitwise_shift_left_int_int(self):
        """Test int << int -> int."""
        result = self.checker.check_binary_op("<<", INT, INT)
        assert result == INT

    def test_bitwise_shift_right_int_int(self):
        """Test int >> int -> int."""
        result = self.checker.check_binary_op(">>", INT, INT)
        assert result == INT

    def test_bitwise_uint_uint(self):
        """Test uint & uint -> uint."""
        result = self.checker.check_binary_op("&", UINT, UINT)
        assert result == UINT

        result = self.checker.check_binary_op("|", UINT, UINT)
        assert result == UINT

    def test_bitwise_ivec_ivec(self):
        """Test ivec3 & ivec3 -> ivec3 (component-wise)."""
        result = self.checker.check_binary_op("&", IVEC3, IVEC3)
        assert result == IVEC3

        result = self.checker.check_binary_op("|", IVEC3, IVEC3)
        assert result == IVEC3

    def test_bitwise_float_error(self):
        """Test float & float is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("&", FLOAT, FLOAT)

    def test_bitwise_mixed_int_uint_error(self):
        """Test int & uint is invalid (different signedness)."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("&", INT, UINT)

    def test_bitwise_vector_scalar(self):
        """Test ivec3 & int -> ivec3."""
        result = self.checker.check_binary_op("&", IVEC3, INT)
        assert result == IVEC3

    # Modulo Operator: % (5 tests)

    def test_modulo_int_int(self):
        """Test int % int -> int."""
        result = self.checker.check_binary_op("%", INT, INT)
        assert result == INT

    def test_modulo_uint_uint(self):
        """Test uint % uint -> uint."""
        result = self.checker.check_binary_op("%", UINT, UINT)
        assert result == UINT

    def test_modulo_float_error(self):
        """Test float % float is invalid (use mod() function)."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("%", FLOAT, FLOAT)

    def test_modulo_ivec_ivec(self):
        """Test ivec3 % ivec3 -> ivec3."""
        result = self.checker.check_binary_op("%", IVEC3, IVEC3)
        assert result == IVEC3

    def test_modulo_vec_error(self):
        """Test vec3 % vec3 is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("%", VEC3, VEC3)

    # Edge Cases (5 tests)

    def test_operator_with_location(self):
        """Test binary op with location tracking."""
        result = self.checker.check_binary_op("+", FLOAT, FLOAT, location=(10, 5))
        assert result == FLOAT

    def test_invalid_operator_string(self):
        """Test unknown operator."""
        with pytest.raises((InvalidOperationError, ValueError)):
            self.checker.check_binary_op("??", FLOAT, FLOAT)

    def test_sampler_arithmetic_error(self):
        """Test sampler2D + sampler2D is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("+", SAMPLER2D, SAMPLER2D)

    def test_void_arithmetic_error(self):
        """Test void + void is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_binary_op("+", VOID, VOID)

    def test_all_vector_sizes(self):
        """Test operators work with all vector sizes."""
        for vec_type in [VEC2, VEC3, VEC4]:
            result = self.checker.check_binary_op("+", vec_type, vec_type)
            assert result == vec_type


# ============================================================================
# 5. Unary Operator Tests (30 tests)
# ============================================================================

class TestUnaryOperators:
    """Tests for unary operator type checking."""

    def setup_method(self):
        """Create type checker for each test."""
        self.symbol_table = SymbolTable()
        self.checker = TypeChecker(self.symbol_table)

    # Unary minus: - (10 tests)

    def test_unary_minus_float(self):
        """Test -float -> float."""
        result = self.checker.check_unary_op("-", FLOAT)
        assert result == FLOAT

    def test_unary_minus_int(self):
        """Test -int -> int."""
        result = self.checker.check_unary_op("-", INT)
        assert result == INT

    def test_unary_minus_uint(self):
        """Test -uint -> uint."""
        result = self.checker.check_unary_op("-", UINT)
        assert result == UINT

    def test_unary_minus_vec(self):
        """Test -vec3 -> vec3 (component-wise)."""
        result = self.checker.check_unary_op("-", VEC3)
        assert result == VEC3

        result = self.checker.check_unary_op("-", VEC2)
        assert result == VEC2

    def test_unary_minus_ivec(self):
        """Test -ivec3 -> ivec3."""
        result = self.checker.check_unary_op("-", IVEC3)
        assert result == IVEC3

    def test_unary_minus_mat(self):
        """Test -mat3 -> mat3 (component-wise)."""
        result = self.checker.check_unary_op("-", MAT3)
        assert result == MAT3

        result = self.checker.check_unary_op("-", MAT4)
        assert result == MAT4

    def test_unary_minus_bool_error(self):
        """Test -bool is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("-", BOOL)

    def test_unary_minus_sampler_error(self):
        """Test -sampler2D is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("-", SAMPLER2D)

    def test_unary_minus_void_error(self):
        """Test -void is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("-", VOID)

    # Unary plus: + (3 tests)

    def test_unary_plus_float(self):
        """Test +float -> float (identity)."""
        result = self.checker.check_unary_op("+", FLOAT)
        assert result == FLOAT

    def test_unary_plus_vec(self):
        """Test +vec3 -> vec3."""
        result = self.checker.check_unary_op("+", VEC3)
        assert result == VEC3

    def test_unary_plus_bool_error(self):
        """Test +bool is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("+", BOOL)

    # Logical not: ! (7 tests)

    def test_logical_not_bool(self):
        """Test !bool -> bool."""
        result = self.checker.check_unary_op("!", BOOL)
        assert result == BOOL

    def test_logical_not_float_error(self):
        """Test !float is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("!", FLOAT)

    def test_logical_not_int_error(self):
        """Test !int is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("!", INT)

    def test_logical_not_vec_error(self):
        """Test !vec3 is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("!", VEC3)

    def test_logical_not_bvec_error(self):
        """Test !bvec3 is invalid (use not() function for component-wise)."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("!", BVEC3)

    def test_logical_not_mat_error(self):
        """Test !mat3 is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("!", MAT3)

    def test_logical_not_sampler_error(self):
        """Test !sampler2D is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("!", SAMPLER2D)

    # Bitwise not: ~ (10 tests)

    def test_bitwise_not_int(self):
        """Test ~int -> int."""
        result = self.checker.check_unary_op("~", INT)
        assert result == INT

    def test_bitwise_not_uint(self):
        """Test ~uint -> uint."""
        result = self.checker.check_unary_op("~", UINT)
        assert result == UINT

    def test_bitwise_not_ivec(self):
        """Test ~ivec3 -> ivec3 (component-wise)."""
        result = self.checker.check_unary_op("~", IVEC3)
        assert result == IVEC3

        result = self.checker.check_unary_op("~", IVEC2)
        assert result == IVEC2

    def test_bitwise_not_uvec(self):
        """Test ~uvec4 -> uvec4."""
        result = self.checker.check_unary_op("~", UVEC4)
        assert result == UVEC4

    def test_bitwise_not_float_error(self):
        """Test ~float is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("~", FLOAT)

    def test_bitwise_not_vec_error(self):
        """Test ~vec3 is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("~", VEC3)

    def test_bitwise_not_bool_error(self):
        """Test ~bool is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("~", BOOL)

    def test_bitwise_not_mat_error(self):
        """Test ~mat3 is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("~", MAT3)

    def test_bitwise_not_sampler_error(self):
        """Test ~sampler2D is invalid."""
        with pytest.raises(InvalidOperationError):
            self.checker.check_unary_op("~", SAMPLER2D)


# ============================================================================
# 6. Function Calls and Constructor Tests (130 tests)
# ============================================================================

class TestConstructors:
    """Tests for type constructor calls (vec3(), mat4(), etc.)."""

    def setup_method(self):
        """Create type checker for each test."""
        self.symbol_table = SymbolTable()
        self.checker = TypeChecker(self.symbol_table)

    # Scalar constructors (5 tests)

    def test_float_constructor_from_int(self):
        """Test float(int) -> float."""
        result = self.checker.check_function_call("float", [INT])
        assert result == FLOAT

    def test_int_constructor_from_float(self):
        """Test int(float) -> int."""
        result = self.checker.check_function_call("int", [FLOAT])
        assert result == INT

    def test_bool_constructor_from_int(self):
        """Test bool(int) -> bool."""
        result = self.checker.check_function_call("bool", [INT])
        assert result == BOOL

    def test_uint_constructor_from_float(self):
        """Test uint(float) -> uint."""
        result = self.checker.check_function_call("uint", [FLOAT])
        assert result == UINT

    def test_scalar_constructor_identity(self):
        """Test float(float) -> float (identity)."""
        result = self.checker.check_function_call("float", [FLOAT])
        assert result == FLOAT

    # Vector constructors from scalar (8 tests)

    def test_vec2_from_scalar(self):
        """Test vec2(float) -> vec2 (splat)."""
        result = self.checker.check_function_call("vec2", [FLOAT])
        assert result == VEC2

    def test_vec3_from_scalar(self):
        """Test vec3(float) -> vec3 (splat)."""
        result = self.checker.check_function_call("vec3", [FLOAT])
        assert result == VEC3

    def test_vec4_from_scalar(self):
        """Test vec4(float) -> vec4 (splat)."""
        result = self.checker.check_function_call("vec4", [FLOAT])
        assert result == VEC4

    def test_ivec2_from_scalar(self):
        """Test ivec2(int) -> ivec2."""
        result = self.checker.check_function_call("ivec2", [INT])
        assert result == IVEC2

    def test_ivec3_from_scalar(self):
        """Test ivec3(int) -> ivec3."""
        result = self.checker.check_function_call("ivec3", [INT])
        assert result == IVEC3

    def test_uvec4_from_scalar(self):
        """Test uvec4(uint) -> uvec4."""
        result = self.checker.check_function_call("uvec4", [UINT])
        assert result == UVEC4

    def test_bvec2_from_scalar(self):
        """Test bvec2(bool) -> bvec2."""
        result = self.checker.check_function_call("bvec2", [BOOL])
        assert result == BVEC2

    def test_bvec3_from_scalar(self):
        """Test bvec3(bool) -> bvec3."""
        result = self.checker.check_function_call("bvec3", [BOOL])
        assert result == BVEC3

    # Vector constructors from components (12 tests)

    def test_vec2_from_two_floats(self):
        """Test vec2(float, float) -> vec2."""
        result = self.checker.check_function_call("vec2", [FLOAT, FLOAT])
        assert result == VEC2

    def test_vec3_from_three_floats(self):
        """Test vec3(float, float, float) -> vec3."""
        result = self.checker.check_function_call("vec3", [FLOAT, FLOAT, FLOAT])
        assert result == VEC3

    def test_vec4_from_four_floats(self):
        """Test vec4(float, float, float, float) -> vec4."""
        result = self.checker.check_function_call("vec4", [FLOAT, FLOAT, FLOAT, FLOAT])
        assert result == VEC4

    def test_vec3_from_vec2_and_float(self):
        """Test vec3(vec2, float) -> vec3."""
        result = self.checker.check_function_call("vec3", [VEC2, FLOAT])
        assert result == VEC3

    def test_vec3_from_float_and_vec2(self):
        """Test vec3(float, vec2) -> vec3."""
        result = self.checker.check_function_call("vec3", [FLOAT, VEC2])
        assert result == VEC3

    def test_vec4_from_vec3_and_float(self):
        """Test vec4(vec3, float) -> vec4."""
        result = self.checker.check_function_call("vec4", [VEC3, FLOAT])
        assert result == VEC4

    def test_vec4_from_vec2_and_vec2(self):
        """Test vec4(vec2, vec2) -> vec4."""
        result = self.checker.check_function_call("vec4", [VEC2, VEC2])
        assert result == VEC4

    def test_vec4_from_float_vec2_float(self):
        """Test vec4(float, vec2, float) -> vec4."""
        result = self.checker.check_function_call("vec4", [FLOAT, VEC2, FLOAT])
        assert result == VEC4

    def test_ivec3_from_three_ints(self):
        """Test ivec3(int, int, int) -> ivec3."""
        result = self.checker.check_function_call("ivec3", [INT, INT, INT])
        assert result == IVEC3

    def test_ivec4_from_ivec2_and_ints(self):
        """Test ivec4(ivec2, int, int) -> ivec4."""
        result = self.checker.check_function_call("ivec4", [IVEC2, INT, INT])
        assert result == IVEC4

    def test_bvec4_from_bools(self):
        """Test bvec4(bool, bool, bool, bool) -> bvec4."""
        result = self.checker.check_function_call("bvec4", [BOOL, BOOL, BOOL, BOOL])
        assert result == BVEC4

    def test_uvec3_from_uints(self):
        """Test uvec3(uint, uint, uint) -> uvec3."""
        result = self.checker.check_function_call("uvec3", [UINT, UINT, UINT])
        assert result == UVEC3

    # Matrix constructors (10 tests)

    def test_mat2_from_scalar(self):
        """Test mat2(float) -> mat2 (diagonal)."""
        result = self.checker.check_function_call("mat2", [FLOAT])
        assert result == MAT2

    def test_mat3_from_scalar(self):
        """Test mat3(float) -> mat3 (diagonal)."""
        result = self.checker.check_function_call("mat3", [FLOAT])
        assert result == MAT3

    def test_mat4_from_scalar(self):
        """Test mat4(float) -> mat4 (diagonal)."""
        result = self.checker.check_function_call("mat4", [FLOAT])
        assert result == MAT4

    def test_mat2_from_four_floats(self):
        """Test mat2(float, float, float, float) -> mat2 (column-major)."""
        result = self.checker.check_function_call("mat2", [FLOAT, FLOAT, FLOAT, FLOAT])
        assert result == MAT2

    def test_mat2_from_two_vec2(self):
        """Test mat2(vec2, vec2) -> mat2 (column vectors)."""
        result = self.checker.check_function_call("mat2", [VEC2, VEC2])
        assert result == MAT2

    def test_mat3_from_three_vec3(self):
        """Test mat3(vec3, vec3, vec3) -> mat3 (column vectors)."""
        result = self.checker.check_function_call("mat3", [VEC3, VEC3, VEC3])
        assert result == MAT3

    def test_mat4_from_four_vec4(self):
        """Test mat4(vec4, vec4, vec4, vec4) -> mat4 (column vectors)."""
        result = self.checker.check_function_call("mat4", [VEC4, VEC4, VEC4, VEC4])
        assert result == MAT4

    def test_mat3_from_nine_floats(self):
        """Test mat3(float x 9) -> mat3."""
        args = [FLOAT] * 9
        result = self.checker.check_function_call("mat3", args)
        assert result == MAT3

    def test_mat4_from_sixteen_floats(self):
        """Test mat4(float x 16) -> mat4."""
        args = [FLOAT] * 16
        result = self.checker.check_function_call("mat4", args)
        assert result == MAT4

    def test_mat4_from_mat3(self):
        """Test mat4(mat3) -> mat4 (upcasting with identity)."""
        result = self.checker.check_function_call("mat4", [MAT3])
        assert result == MAT4

    # Constructor error cases (5 tests)

    def test_vec3_wrong_component_count_error(self):
        """Test vec3(float, float) is invalid (wrong count)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("vec3", [FLOAT, FLOAT])

    def test_vec2_wrong_type_error(self):
        """Test vec2(int) is invalid (wrong type)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("vec2", [INT])

    def test_mat2_wrong_vector_size_error(self):
        """Test mat2(vec3, vec3) is invalid."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("mat2", [VEC3, VEC3])

    def test_vec4_too_many_components_error(self):
        """Test vec4(vec3, vec2) is invalid (too many components)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("vec4", [VEC3, VEC2])

    def test_constructor_no_args_error(self):
        """Test vec3() is invalid (no arguments)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("vec3", [])


class TestBuiltinFunctions:
    """Tests for built-in function calls."""

    def setup_method(self):
        """Create type checker with built-in symbols."""
        from glsl_to_opencl.analyzer import register_builtins
        self.symbol_table = SymbolTable()
        register_builtins(self.symbol_table)
        self.checker = TypeChecker(self.symbol_table)

    # Math functions (genType) (20 tests)

    def test_sin_float(self):
        """Test sin(float) -> float."""
        result = self.checker.check_function_call("sin", [FLOAT])
        assert result == FLOAT

    def test_sin_vec3(self):
        """Test sin(vec3) -> vec3 (component-wise)."""
        result = self.checker.check_function_call("sin", [VEC3])
        assert result == VEC3

    def test_cos_vec2(self):
        """Test cos(vec2) -> vec2."""
        result = self.checker.check_function_call("cos", [VEC2])
        assert result == VEC2

    def test_tan_vec4(self):
        """Test tan(vec4) -> vec4."""
        result = self.checker.check_function_call("tan", [VEC4])
        assert result == VEC4

    def test_sqrt_float(self):
        """Test sqrt(float) -> float."""
        result = self.checker.check_function_call("sqrt", [FLOAT])
        assert result == FLOAT

    def test_exp_vec3(self):
        """Test exp(vec3) -> vec3."""
        result = self.checker.check_function_call("exp", [VEC3])
        assert result == VEC3

    def test_pow_float_float(self):
        """Test pow(float, float) -> float."""
        result = self.checker.check_function_call("pow", [FLOAT, FLOAT])
        assert result == FLOAT

    def test_pow_vec3_vec3(self):
        """Test pow(vec3, vec3) -> vec3."""
        result = self.checker.check_function_call("pow", [VEC3, VEC3])
        assert result == VEC3

    def test_min_float_float(self):
        """Test min(float, float) -> float."""
        result = self.checker.check_function_call("min", [FLOAT, FLOAT])
        assert result == FLOAT

    def test_max_vec3_vec3(self):
        """Test max(vec3, vec3) -> vec3."""
        result = self.checker.check_function_call("max", [VEC3, VEC3])
        assert result == VEC3

    def test_clamp_float_float_float(self):
        """Test clamp(float, float, float) -> float."""
        result = self.checker.check_function_call("clamp", [FLOAT, FLOAT, FLOAT])
        assert result == FLOAT

    def test_clamp_vec3_float_float(self):
        """Test clamp(vec3, float, float) -> vec3 (scalar min/max)."""
        result = self.checker.check_function_call("clamp", [VEC3, FLOAT, FLOAT])
        assert result == VEC3

    def test_mix_float_float_float(self):
        """Test mix(float, float, float) -> float (linear interpolation)."""
        result = self.checker.check_function_call("mix", [FLOAT, FLOAT, FLOAT])
        assert result == FLOAT

    def test_mix_vec3_vec3_vec3(self):
        """Test mix(vec3, vec3, vec3) -> vec3."""
        result = self.checker.check_function_call("mix", [VEC3, VEC3, VEC3])
        assert result == VEC3

    def test_mix_vec3_vec3_float(self):
        """Test mix(vec3, vec3, float) -> vec3 (scalar blend factor)."""
        result = self.checker.check_function_call("mix", [VEC3, VEC3, FLOAT])
        assert result == VEC3

    def test_step_float_float(self):
        """Test step(float, float) -> float."""
        result = self.checker.check_function_call("step", [FLOAT, FLOAT])
        assert result == FLOAT

    def test_smoothstep_float_float_float(self):
        """Test smoothstep(float, float, float) -> float."""
        result = self.checker.check_function_call("smoothstep", [FLOAT, FLOAT, FLOAT])
        assert result == FLOAT

    def test_abs_float(self):
        """Test abs(float) -> float."""
        result = self.checker.check_function_call("abs", [FLOAT])
        assert result == FLOAT

    def test_floor_vec2(self):
        """Test floor(vec2) -> vec2."""
        result = self.checker.check_function_call("floor", [VEC2])
        assert result == VEC2

    def test_mod_float_float(self):
        """Test mod(float, float) -> float."""
        result = self.checker.check_function_call("mod", [FLOAT, FLOAT])
        assert result == FLOAT

    # Geometric functions (15 tests)

    def test_length_vec3(self):
        """Test length(vec3) -> float."""
        result = self.checker.check_function_call("length", [VEC3])
        assert result == FLOAT

    def test_length_vec2(self):
        """Test length(vec2) -> float."""
        result = self.checker.check_function_call("length", [VEC2])
        assert result == FLOAT

    def test_distance_vec3_vec3(self):
        """Test distance(vec3, vec3) -> float."""
        result = self.checker.check_function_call("distance", [VEC3, VEC3])
        assert result == FLOAT

    def test_dot_vec3_vec3(self):
        """Test dot(vec3, vec3) -> float."""
        result = self.checker.check_function_call("dot", [VEC3, VEC3])
        assert result == FLOAT

    def test_dot_vec2_vec2(self):
        """Test dot(vec2, vec2) -> float."""
        result = self.checker.check_function_call("dot", [VEC2, VEC2])
        assert result == FLOAT

    def test_cross_vec3_vec3(self):
        """Test cross(vec3, vec3) -> vec3."""
        result = self.checker.check_function_call("cross", [VEC3, VEC3])
        assert result == VEC3

    def test_normalize_vec3(self):
        """Test normalize(vec3) -> vec3."""
        result = self.checker.check_function_call("normalize", [VEC3])
        assert result == VEC3

    def test_normalize_vec2(self):
        """Test normalize(vec2) -> vec2."""
        result = self.checker.check_function_call("normalize", [VEC2])
        assert result == VEC2

    def test_faceforward_vec3_vec3_vec3(self):
        """Test faceforward(vec3, vec3, vec3) -> vec3."""
        result = self.checker.check_function_call("faceforward", [VEC3, VEC3, VEC3])
        assert result == VEC3

    def test_reflect_vec3_vec3(self):
        """Test reflect(vec3, vec3) -> vec3."""
        result = self.checker.check_function_call("reflect", [VEC3, VEC3])
        assert result == VEC3

    def test_refract_vec3_vec3_float(self):
        """Test refract(vec3, vec3, float) -> vec3."""
        result = self.checker.check_function_call("refract", [VEC3, VEC3, FLOAT])
        assert result == VEC3

    def test_cross_wrong_type_error(self):
        """Test cross(vec2, vec2) is invalid (only vec3 allowed)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("cross", [VEC2, VEC2])

    def test_dot_wrong_size_error(self):
        """Test dot(vec2, vec3) is invalid (size mismatch)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("dot", [VEC2, VEC3])

    def test_length_int_error(self):
        """Test length(int) is invalid (not a vector)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("length", [INT])

    def test_normalize_float_error(self):
        """Test normalize(float) is invalid (not a vector)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("normalize", [FLOAT])

    # Texture functions (10 tests)

    def test_texture_sampler2d_vec2(self):
        """Test texture(sampler2D, vec2) -> vec4."""
        result = self.checker.check_function_call("texture", [SAMPLER2D, VEC2])
        assert result == VEC4

    def test_texture_samplercube_vec3(self):
        """Test texture(samplerCube, vec3) -> vec4."""
        result = self.checker.check_function_call("texture", [SAMPLERCUBE, VEC3])
        assert result == VEC4

    def test_texture_sampler3d_vec3(self):
        """Test texture(sampler3D, vec3) -> vec4."""
        result = self.checker.check_function_call("texture", [SAMPLER3D, VEC3])
        assert result == VEC4

    def test_texturelod_sampler2d_vec2_float(self):
        """Test textureLod(sampler2D, vec2, float) -> vec4."""
        result = self.checker.check_function_call("textureLod", [SAMPLER2D, VEC2, FLOAT])
        assert result == VEC4

    def test_texturesize_sampler2d_int(self):
        """Test textureSize(sampler2D, int) -> ivec2."""
        result = self.checker.check_function_call("textureSize", [SAMPLER2D, INT])
        # Note: actual return type depends on sampler type, but simplified here
        assert result.category == TypeCategory.VECTOR or result == VEC4

    def test_texelfetch_sampler2d_ivec2_int(self):
        """Test texelFetch(sampler2D, ivec2, int) -> vec4."""
        result = self.checker.check_function_call("texelFetch", [SAMPLER2D, IVEC2, INT])
        assert result == VEC4

    def test_texture_wrong_coord_type_error(self):
        """Test texture(sampler2D, vec3) is invalid (wrong coord size)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("texture", [SAMPLER2D, VEC3])

    def test_texture_no_sampler_error(self):
        """Test texture(vec2, vec2) is invalid (no sampler)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("texture", [VEC2, VEC2])

    def test_texelfetch_wrong_coord_type_error(self):
        """Test texelFetch(sampler2D, vec2, int) is invalid (needs ivec2)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("texelFetch", [SAMPLER2D, VEC2, INT])

    def test_texturelod_missing_arg_error(self):
        """Test textureLod(sampler2D, vec2) is invalid (missing lod)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("textureLod", [SAMPLER2D, VEC2])

    # Vector relational functions (10 tests)

    def test_lessThan_vec2_vec2(self):
        """Test lessThan(vec2, vec2) -> bvec2."""
        result = self.checker.check_function_call("lessThan", [VEC2, VEC2])
        assert result == BVEC2

    def test_equal_ivec3_ivec3(self):
        """Test equal(ivec3, ivec3) -> bvec3."""
        result = self.checker.check_function_call("equal", [IVEC3, IVEC3])
        assert result == BVEC3

    def test_all_bvec3(self):
        """Test all(bvec3) -> bool."""
        result = self.checker.check_function_call("all", [BVEC3])
        assert result == BOOL

    def test_any_bvec4(self):
        """Test any(bvec4) -> bool."""
        result = self.checker.check_function_call("any", [BVEC4])
        assert result == BOOL

    def test_not_bvec2(self):
        """Test not(bvec2) -> bvec2 (component-wise logical not)."""
        result = self.checker.check_function_call("not", [BVEC2])
        assert result == BVEC2

    def test_greaterThan_vec3_vec3(self):
        """Test greaterThan(vec3, vec3) -> bvec3."""
        result = self.checker.check_function_call("greaterThan", [VEC3, VEC3])
        assert result == BVEC3

    def test_notEqual_vec4_vec4(self):
        """Test notEqual(vec4, vec4) -> bvec4."""
        result = self.checker.check_function_call("notEqual", [VEC4, VEC4])
        assert result == BVEC4

    def test_lessThan_wrong_size_error(self):
        """Test lessThan(vec2, vec3) is invalid."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("lessThan", [VEC2, VEC3])

    def test_all_bool_error(self):
        """Test all(bool) is invalid (requires bvec)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("all", [BOOL])

    def test_not_float_error(self):
        """Test not(float) is invalid."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("not", [FLOAT])

    # Function lookup and error cases (10 tests)

    def test_undefined_function_error(self):
        """Test calling undefined function."""
        with pytest.raises((TypeCheckError, ValueError, KeyError)):
            self.checker.check_function_call("undefinedFunc", [FLOAT])

    def test_function_wrong_arg_count_error(self):
        """Test sin(float, float) is invalid (wrong arg count)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("sin", [FLOAT, FLOAT])

    def test_function_no_args_error(self):
        """Test sin() is invalid (no arguments)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("sin", [])

    def test_math_function_wrong_type_error(self):
        """Test sin(int) is invalid (genType is float-based)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("sin", [INT])

    def test_math_function_bool_error(self):
        """Test sqrt(bool) is invalid."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("sqrt", [BOOL])

    def test_pow_type_mismatch_error(self):
        """Test pow(float, int) is invalid (types must match)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("pow", [FLOAT, INT])

    def test_mix_wrong_blend_type_error(self):
        """Test mix(vec3, vec3, int) is invalid."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("mix", [VEC3, VEC3, INT])

    def test_clamp_size_mismatch_error(self):
        """Test clamp(vec2, vec3, vec3) is invalid."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_function_call("clamp", [VEC2, VEC3, VEC3])

    def test_function_with_location(self):
        """Test function call with location tracking."""
        result = self.checker.check_function_call("sin", [FLOAT], location=(10, 5))
        assert result == FLOAT

    def test_builtin_variable_not_callable_error(self):
        """Test calling a builtin variable as function."""
        # iTime is a variable, not a function
        with pytest.raises((TypeCheckError, InvalidOperationError, ValueError, KeyError)):
            self.checker.check_function_call("iTime", [FLOAT])


# ============================================================================
# 7. Member Access and Swizzling Tests (50 tests)
# ============================================================================

class TestMemberAccess:
    """Tests for member access and vector swizzling."""

    def setup_method(self):
        """Create type checker for each test."""
        self.symbol_table = SymbolTable()
        self.checker = TypeChecker(self.symbol_table)

    # Single component access (12 tests)

    def test_vec2_x(self):
        """Test vec2.x -> float."""
        result = self.checker.check_member_access(VEC2, "x")
        assert result == FLOAT

    def test_vec3_y(self):
        """Test vec3.y -> float."""
        result = self.checker.check_member_access(VEC3, "y")
        assert result == FLOAT

    def test_vec4_z(self):
        """Test vec4.z -> float."""
        result = self.checker.check_member_access(VEC4, "z")
        assert result == FLOAT

    def test_vec4_w(self):
        """Test vec4.w -> float."""
        result = self.checker.check_member_access(VEC4, "w")
        assert result == FLOAT

    def test_vec3_r(self):
        """Test vec3.r -> float (color swizzle)."""
        result = self.checker.check_member_access(VEC3, "r")
        assert result == FLOAT

    def test_vec4_g(self):
        """Test vec4.g -> float."""
        result = self.checker.check_member_access(VEC4, "g")
        assert result == FLOAT

    def test_vec4_b(self):
        """Test vec4.b -> float."""
        result = self.checker.check_member_access(VEC4, "b")
        assert result == FLOAT

    def test_vec4_a(self):
        """Test vec4.a -> float."""
        result = self.checker.check_member_access(VEC4, "a")
        assert result == FLOAT

    def test_vec2_s(self):
        """Test vec2.s -> float (texture coordinate)."""
        result = self.checker.check_member_access(VEC2, "s")
        assert result == FLOAT

    def test_vec3_t(self):
        """Test vec3.t -> float."""
        result = self.checker.check_member_access(VEC3, "t")
        assert result == FLOAT

    def test_vec3_p(self):
        """Test vec3.p -> float."""
        result = self.checker.check_member_access(VEC3, "p")
        assert result == FLOAT

    def test_vec4_q(self):
        """Test vec4.q -> float."""
        result = self.checker.check_member_access(VEC4, "q")
        assert result == FLOAT

    # Two component swizzles (8 tests)

    def test_vec3_xy(self):
        """Test vec3.xy -> vec2."""
        result = self.checker.check_member_access(VEC3, "xy")
        assert result == VEC2

    def test_vec4_zw(self):
        """Test vec4.zw -> vec2."""
        result = self.checker.check_member_access(VEC4, "zw")
        assert result == VEC2

    def test_vec4_rg(self):
        """Test vec4.rg -> vec2 (color swizzle)."""
        result = self.checker.check_member_access(VEC4, "rg")
        assert result == VEC2

    def test_vec4_ba(self):
        """Test vec4.ba -> vec2."""
        result = self.checker.check_member_access(VEC4, "ba")
        assert result == VEC2

    def test_vec3_st(self):
        """Test vec3.st -> vec2."""
        result = self.checker.check_member_access(VEC3, "st")
        assert result == VEC2

    def test_vec3_xx(self):
        """Test vec3.xx -> vec2 (repeated component)."""
        result = self.checker.check_member_access(VEC3, "xx")
        assert result == VEC2

    def test_vec4_yx(self):
        """Test vec4.yx -> vec2 (reversed order)."""
        result = self.checker.check_member_access(VEC4, "yx")
        assert result == VEC2

    def test_vec2_yy(self):
        """Test vec2.yy -> vec2."""
        result = self.checker.check_member_access(VEC2, "yy")
        assert result == VEC2

    # Three component swizzles (8 tests)

    def test_vec4_xyz(self):
        """Test vec4.xyz -> vec3."""
        result = self.checker.check_member_access(VEC4, "xyz")
        assert result == VEC3

    def test_vec4_rgb(self):
        """Test vec4.rgb -> vec3."""
        result = self.checker.check_member_access(VEC4, "rgb")
        assert result == VEC3

    def test_vec4_stp(self):
        """Test vec4.stp -> vec3."""
        result = self.checker.check_member_access(VEC4, "stp")
        assert result == VEC3

    def test_vec3_xxx(self):
        """Test vec3.xxx -> vec3 (repeated)."""
        result = self.checker.check_member_access(VEC3, "xxx")
        assert result == VEC3

    def test_vec4_zyx(self):
        """Test vec4.zyx -> vec3 (reversed)."""
        result = self.checker.check_member_access(VEC4, "zyx")
        assert result == VEC3

    def test_vec4_xxy(self):
        """Test vec4.xxy -> vec3 (mixed repeat)."""
        result = self.checker.check_member_access(VEC4, "xxy")
        assert result == VEC3

    def test_vec3_yzx(self):
        """Test vec3.yzx -> vec3 (rotated)."""
        result = self.checker.check_member_access(VEC3, "yzx")
        assert result == VEC3

    def test_vec4_wzy(self):
        """Test vec4.wzy -> vec3."""
        result = self.checker.check_member_access(VEC4, "wzy")
        assert result == VEC3

    # Four component swizzles (6 tests)

    def test_vec4_xyzw(self):
        """Test vec4.xyzw -> vec4."""
        result = self.checker.check_member_access(VEC4, "xyzw")
        assert result == VEC4

    def test_vec4_rgba(self):
        """Test vec4.rgba -> vec4."""
        result = self.checker.check_member_access(VEC4, "rgba")
        assert result == VEC4

    def test_vec4_stpq(self):
        """Test vec4.stpq -> vec4."""
        result = self.checker.check_member_access(VEC4, "stpq")
        assert result == VEC4

    def test_vec4_wzyx(self):
        """Test vec4.wzyx -> vec4 (reversed)."""
        result = self.checker.check_member_access(VEC4, "wzyx")
        assert result == VEC4

    def test_vec4_xxxx(self):
        """Test vec4.xxxx -> vec4 (all same)."""
        result = self.checker.check_member_access(VEC4, "xxxx")
        assert result == VEC4

    def test_vec3_xyzx(self):
        """Test vec3.xyzx -> vec4 (extended from vec3)."""
        result = self.checker.check_member_access(VEC3, "xyzx")
        assert result == VEC4

    # Integer vector swizzles (4 tests)

    def test_ivec3_xy(self):
        """Test ivec3.xy -> ivec2."""
        result = self.checker.check_member_access(IVEC3, "xy")
        assert result == IVEC2

    def test_ivec4_xyz(self):
        """Test ivec4.xyz -> ivec3."""
        result = self.checker.check_member_access(IVEC4, "xyz")
        assert result == IVEC3

    def test_uvec2_x(self):
        """Test uvec2.x -> uint."""
        result = self.checker.check_member_access(UVEC2, "x")
        assert result == UINT

    def test_uvec4_zw(self):
        """Test uvec4.zw -> uvec2."""
        result = self.checker.check_member_access(UVEC4, "zw")
        assert result == UVEC2

    # Error cases (12 tests)

    def test_scalar_member_error(self):
        """Test float.x is invalid."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(FLOAT, "x")

    def test_matrix_swizzle_error(self):
        """Test mat3.xy is invalid (matrices don't support swizzling)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(MAT3, "xy")

    def test_vec2_z_error(self):
        """Test vec2.z is invalid (out of range)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(VEC2, "z")

    def test_vec3_w_error(self):
        """Test vec3.w is invalid (out of range)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(VEC3, "w")

    def test_vec2_a_error(self):
        """Test vec2.a is invalid (out of range for color swizzle)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(VEC2, "a")

    def test_vec3_q_error(self):
        """Test vec3.q is invalid (out of range for texture swizzle)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(VEC3, "q")

    def test_invalid_swizzle_char_error(self):
        """Test vec3.k is invalid (invalid character)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(VEC3, "k")

    def test_mixed_swizzle_sets_error(self):
        """Test vec3.xr is invalid (can't mix xyzw and rgba)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(VEC3, "xr")

    def test_too_many_components_error(self):
        """Test vec3.xyzww is invalid (too many components)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(VEC3, "xyzww")

    def test_empty_swizzle_error(self):
        """Test vec3. is invalid (empty swizzle)."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(VEC3, "")

    def test_sampler_member_error(self):
        """Test sampler2D.x is invalid."""
        with pytest.raises((TypeCheckError, InvalidOperationError)):
            self.checker.check_member_access(SAMPLER2D, "x")

    def test_bool_vector_swizzle(self):
        """Test bvec3.xy -> bvec2 (bool vectors support swizzling)."""
        result = self.checker.check_member_access(BVEC3, "xy")
        assert result == BVEC2


# More test classes to be added:
# - TestIntegration (50+ tests)
