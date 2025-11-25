"""
Type Checker for GLSL AST

Performs type inference and validation on parsed GLSL code.
Validates operator compatibility, function signatures, and implicit conversions.

Components:
    - GLSLType: Represents GLSL types (scalars, vectors, matrices, samplers, structs, arrays)
    - TypeChecker: Main type checking class
    - Type compatibility and inference rules
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from .symbol_table import Symbol, SymbolTable, SymbolType


class TypeCategory(Enum):
    """Categories of GLSL types."""
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    SAMPLER = "sampler"
    STRUCT = "struct"
    ARRAY = "array"
    VOID = "void"


class ScalarType(Enum):
    """GLSL scalar types."""
    FLOAT = "float"
    INT = "int"
    UINT = "uint"
    BOOL = "bool"


@dataclass(frozen=True)
class GLSLType:
    """
    Represents a GLSL type.

    Attributes:
        category: Type category (scalar, vector, matrix, etc.)
        base_type: For scalars/vectors/matrices: the base scalar type
        dimensions: For vectors: (n,), for matrices: (rows, cols), for arrays: (size1, size2, ...)
        name: For structs and samplers: the type name
        is_const: Whether the type is const-qualified

    Examples:
        float: GLSLType(SCALAR, FLOAT)
        vec3: GLSLType(VECTOR, FLOAT, (3,))
        mat4: GLSLType(MATRIX, FLOAT, (4, 4))
        ivec2: GLSLType(VECTOR, INT, (2,))
        sampler2D: GLSLType(SAMPLER, name="sampler2D")
    """
    category: TypeCategory
    base_type: Optional[ScalarType] = None
    dimensions: Tuple[int, ...] = field(default_factory=tuple)
    name: Optional[str] = None
    is_const: bool = False

    def __str__(self) -> str:
        """String representation matching GLSL syntax."""
        if self.category == TypeCategory.SCALAR:
            return self.base_type.value
        elif self.category == TypeCategory.VECTOR:
            size = self.dimensions[0]
            prefix = {
                ScalarType.FLOAT: "vec",
                ScalarType.INT: "ivec",
                ScalarType.UINT: "uvec",
                ScalarType.BOOL: "bvec",
            }[self.base_type]
            return f"{prefix}{size}"
        elif self.category == TypeCategory.MATRIX:
            rows, cols = self.dimensions
            if rows == cols:
                return f"mat{rows}"
            else:
                return f"mat{rows}x{cols}"
        elif self.category == TypeCategory.SAMPLER:
            return self.name
        elif self.category == TypeCategory.STRUCT:
            return self.name
        elif self.category == TypeCategory.ARRAY:
            return f"{self.name}[{']['.join(str(d) for d in self.dimensions)}]"
        elif self.category == TypeCategory.VOID:
            return "void"
        return "unknown"

    def is_numeric(self) -> bool:
        """Check if type is numeric (float, int, uint, or vectors/matrices of these)."""
        return self.category in (TypeCategory.SCALAR, TypeCategory.VECTOR, TypeCategory.MATRIX) and \
               self.base_type in (ScalarType.FLOAT, ScalarType.INT, ScalarType.UINT)

    def is_integer(self) -> bool:
        """Check if type is integer-based."""
        return self.category in (TypeCategory.SCALAR, TypeCategory.VECTOR) and \
               self.base_type in (ScalarType.INT, ScalarType.UINT)

    def is_float(self) -> bool:
        """Check if type is float-based."""
        return self.category in (TypeCategory.SCALAR, TypeCategory.VECTOR, TypeCategory.MATRIX) and \
               self.base_type == ScalarType.FLOAT

    def is_bool(self) -> bool:
        """Check if type is bool-based."""
        return self.category in (TypeCategory.SCALAR, TypeCategory.VECTOR) and \
               self.base_type == ScalarType.BOOL

    def is_vector(self) -> bool:
        """Check if type is a vector."""
        return self.category == TypeCategory.VECTOR

    def is_matrix(self) -> bool:
        """Check if type is a matrix."""
        return self.category == TypeCategory.MATRIX

    def vector_size(self) -> Optional[int]:
        """Get vector size (2, 3, or 4), or None if not a vector."""
        return self.dimensions[0] if self.category == TypeCategory.VECTOR else None

    def matrix_dimensions(self) -> Optional[Tuple[int, int]]:
        """Get matrix dimensions (rows, cols), or None if not a matrix."""
        return tuple(self.dimensions) if self.category == TypeCategory.MATRIX else None

    def matrix_size(self) -> Optional[int]:
        """Get matrix size for square matrices (2, 3, or 4), or None if not a square matrix."""
        if self.category == TypeCategory.MATRIX:
            rows, cols = self.dimensions
            if rows == cols:
                return rows
        return None

    def to_glsl_name(self) -> str:
        """Get the GLSL type name as a string."""
        return str(self)


# Common GLSL types as constants
FLOAT = GLSLType(TypeCategory.SCALAR, ScalarType.FLOAT)
INT = GLSLType(TypeCategory.SCALAR, ScalarType.INT)
UINT = GLSLType(TypeCategory.SCALAR, ScalarType.UINT)
BOOL = GLSLType(TypeCategory.SCALAR, ScalarType.BOOL)
VOID = GLSLType(TypeCategory.VOID)

VEC2 = GLSLType(TypeCategory.VECTOR, ScalarType.FLOAT, (2,))
VEC3 = GLSLType(TypeCategory.VECTOR, ScalarType.FLOAT, (3,))
VEC4 = GLSLType(TypeCategory.VECTOR, ScalarType.FLOAT, (4,))

IVEC2 = GLSLType(TypeCategory.VECTOR, ScalarType.INT, (2,))
IVEC3 = GLSLType(TypeCategory.VECTOR, ScalarType.INT, (3,))
IVEC4 = GLSLType(TypeCategory.VECTOR, ScalarType.INT, (4,))

UVEC2 = GLSLType(TypeCategory.VECTOR, ScalarType.UINT, (2,))
UVEC3 = GLSLType(TypeCategory.VECTOR, ScalarType.UINT, (3,))
UVEC4 = GLSLType(TypeCategory.VECTOR, ScalarType.UINT, (4,))

BVEC2 = GLSLType(TypeCategory.VECTOR, ScalarType.BOOL, (2,))
BVEC3 = GLSLType(TypeCategory.VECTOR, ScalarType.BOOL, (3,))
BVEC4 = GLSLType(TypeCategory.VECTOR, ScalarType.BOOL, (4,))

MAT2 = GLSLType(TypeCategory.MATRIX, ScalarType.FLOAT, (2, 2))
MAT3 = GLSLType(TypeCategory.MATRIX, ScalarType.FLOAT, (3, 3))
MAT4 = GLSLType(TypeCategory.MATRIX, ScalarType.FLOAT, (4, 4))

SAMPLER2D = GLSLType(TypeCategory.SAMPLER, name="sampler2D")
SAMPLER3D = GLSLType(TypeCategory.SAMPLER, name="sampler3D")
SAMPLERCUBE = GLSLType(TypeCategory.SAMPLER, name="samplerCube")

# Type name to GLSLType mapping
TYPE_NAME_MAP: Dict[str, GLSLType] = {
    "float": FLOAT,
    "int": INT,
    "uint": UINT,
    "bool": BOOL,
    "void": VOID,
    "vec2": VEC2,
    "vec3": VEC3,
    "vec4": VEC4,
    "ivec2": IVEC2,
    "ivec3": IVEC3,
    "ivec4": IVEC4,
    "uvec2": UVEC2,
    "uvec3": UVEC3,
    "uvec4": UVEC4,
    "bvec2": BVEC2,
    "bvec3": BVEC3,
    "bvec4": BVEC4,
    "mat2": MAT2,
    "mat3": MAT3,
    "mat4": MAT4,
    "sampler2D": SAMPLER2D,
    "sampler3D": SAMPLER3D,
    "samplerCube": SAMPLERCUBE,
}


class TypeCheckError(Exception):
    """Base exception for type checking errors."""
    def __init__(self, message: str, location: Optional[Tuple[int, int]] = None):
        self.message = message
        self.location = location
        super().__init__(f"{message} at {location}" if location else message)


class TypeMismatchError(TypeCheckError):
    """Type mismatch in operation or assignment."""
    pass


class InvalidOperationError(TypeCheckError):
    """Invalid operation for given types."""
    pass


class UndefinedTypeError(TypeCheckError):
    """Reference to undefined type."""
    pass


class TypeChecker:
    """
    Type checker for GLSL AST.

    Performs type inference and validation:
    - Infer types for expressions
    - Validate operator compatibility
    - Check function signature matching
    - Handle implicit type conversions

    Usage:
        checker = TypeChecker(symbol_table)
        type_map = checker.check(ast)
    """

    def __init__(self, symbol_table: SymbolTable):
        """
        Initialize type checker.

        Args:
            symbol_table: Symbol table with declarations
        """
        self.symbol_table = symbol_table
        self.type_map: Dict[Any, GLSLType] = {}  # Maps AST nodes to their types

    def check(self, ast) -> Dict[Any, GLSLType]:
        """
        Perform type checking on AST.

        Args:
            ast: Root AST node (TranslationUnit)

        Returns:
            Dictionary mapping AST nodes to their inferred types

        Raises:
            TypeCheckError: If type checking fails
        """
        # Type checking implementation will be added incrementally via TDD
        return self.type_map

    def infer_type(self, node) -> GLSLType:
        """
        Infer the type of an expression node.

        Args:
            node: AST node to infer type for

        Returns:
            Inferred GLSLType

        Raises:
            TypeCheckError: If type cannot be inferred
        """
        # Will be implemented via TDD
        raise NotImplementedError("Type inference not yet implemented")

    def check_binary_op(self, op: str, left_type: GLSLType, right_type: GLSLType,
                        location: Optional[Tuple[int, int]] = None) -> GLSLType:
        """
        Check binary operation and return result type.

        Args:
            op: Operator string ("+", "-", "*", "/", etc.)
            left_type: Type of left operand
            right_type: Type of right operand
            location: Source location for error reporting

        Returns:
            Result type of operation

        Raises:
            InvalidOperationError: If operation is invalid for given types
        """
        # Categorize operators
        arithmetic_ops = {"+", "-", "*", "/"}
        comparison_ops = {"<", "<=", ">", ">="}
        equality_ops = {"==", "!="}
        logical_ops = {"&&", "||"}
        bitwise_ops = {"&", "|", "^", "<<", ">>"}
        modulo_op = {"%"}

        # Arithmetic operators: +, -, *, /
        if op in arithmetic_ops:
            return self._check_arithmetic_op(op, left_type, right_type, location)

        # Comparison operators: <, <=, >, >=
        elif op in comparison_ops:
            return self._check_comparison_op(op, left_type, right_type, location)

        # Equality operators: ==, !=
        elif op in equality_ops:
            return self._check_equality_op(op, left_type, right_type, location)

        # Logical operators: &&, ||
        elif op in logical_ops:
            return self._check_logical_op(op, left_type, right_type, location)

        # Bitwise operators: &, |, ^, <<, >>
        elif op in bitwise_ops:
            return self._check_bitwise_op(op, left_type, right_type, location)

        # Modulo operator: %
        elif op in modulo_op:
            return self._check_modulo_op(left_type, right_type, location)

        else:
            raise InvalidOperationError(f"Unknown operator: {op}", location)

    def _check_arithmetic_op(self, op: str, left: GLSLType, right: GLSLType,
                            location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check arithmetic operators: +, -, *, /"""

        # Reject invalid types
        if left.category == TypeCategory.VOID or right.category == TypeCategory.VOID:
            raise InvalidOperationError(f"Cannot use arithmetic operator {op} with void type", location)
        if left.category == TypeCategory.SAMPLER or right.category == TypeCategory.SAMPLER:
            raise InvalidOperationError(f"Cannot use arithmetic operator {op} with sampler type", location)
        if left.is_bool() or right.is_bool():
            raise InvalidOperationError(f"Cannot use arithmetic operator {op} with bool type", location)

        # Scalar-scalar
        if left.category == TypeCategory.SCALAR and right.category == TypeCategory.SCALAR:
            if left.base_type != right.base_type:
                raise InvalidOperationError(
                    f"Cannot {op} {left} and {right}: types must match exactly", location)
            return left

        # Matrix multiplication: mat * mat, mat * vec, vec * mat
        if op == "*":
            # mat * mat
            if left.is_matrix() and right.is_matrix():
                if left.dimensions != right.dimensions:
                    raise InvalidOperationError(
                        f"Cannot multiply matrices of different sizes: {left} * {right}", location)
                return left

            # mat * vec
            if left.is_matrix() and right.is_vector():
                rows, cols = left.dimensions
                vec_size = right.vector_size()
                if cols != vec_size:
                    raise InvalidOperationError(
                        f"Matrix-vector size mismatch: {left} * {right}", location)
                return right  # Result is vector

            # vec * mat
            if left.is_vector() and right.is_matrix():
                vec_size = left.vector_size()
                rows, cols = right.dimensions
                if vec_size != rows:
                    raise InvalidOperationError(
                        f"Vector-matrix size mismatch: {left} * {right}", location)
                return left  # Result is vector

            # mat * scalar or scalar * mat
            if left.is_matrix() and right.category == TypeCategory.SCALAR:
                if right.base_type != left.base_type:
                    raise InvalidOperationError(
                        f"Matrix-scalar type mismatch: {left} * {right}", location)
                return left
            if left.category == TypeCategory.SCALAR and right.is_matrix():
                if left.base_type != right.base_type:
                    raise InvalidOperationError(
                        f"Scalar-matrix type mismatch: {left} * {right}", location)
                return right

        # Matrix + matrix (component-wise, must be same size)
        if left.is_matrix() and right.is_matrix():
            if left.dimensions != right.dimensions:
                raise InvalidOperationError(
                    f"Cannot {op} matrices of different sizes: {left} and {right}", location)
            return left

        # Matrix + vector is invalid
        if (left.is_matrix() and right.is_vector()) or (left.is_vector() and right.is_matrix()):
            raise InvalidOperationError(
                f"Cannot {op} matrix and vector (incompatible types)", location)

        # Vector-vector (component-wise)
        if left.is_vector() and right.is_vector():
            if left.vector_size() != right.vector_size():
                raise InvalidOperationError(
                    f"Vector size mismatch: {left} {op} {right}", location)
            if left.base_type != right.base_type:
                raise InvalidOperationError(
                    f"Vector type mismatch: {left} {op} {right}", location)
            return left

        # Vector-scalar (scalar promotion)
        if left.is_vector() and right.category == TypeCategory.SCALAR:
            if left.base_type != right.base_type:
                raise InvalidOperationError(
                    f"Vector-scalar type mismatch: {left} {op} {right}", location)
            return left

        # Scalar-vector (scalar promotion)
        if left.category == TypeCategory.SCALAR and right.is_vector():
            if left.base_type != right.base_type:
                raise InvalidOperationError(
                    f"Scalar-vector type mismatch: {left} {op} {right}", location)
            return right

        raise InvalidOperationError(f"Invalid types for {op}: {left} and {right}", location)

    def _check_comparison_op(self, op: str, left: GLSLType, right: GLSLType,
                            location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check comparison operators: <, <=, >, >="""

        # Only numeric types can be compared
        if not left.is_numeric() or not right.is_numeric():
            raise InvalidOperationError(
                f"Comparison operator {op} requires numeric types", location)

        # Scalar-scalar -> bool
        if left.category == TypeCategory.SCALAR and right.category == TypeCategory.SCALAR:
            if left.base_type != right.base_type:
                raise InvalidOperationError(
                    f"Cannot compare {left} and {right}: types must match", location)
            return BOOL

        # Vector-vector -> bvec (component-wise)
        if left.is_vector() and right.is_vector():
            if left.vector_size() != right.vector_size():
                raise InvalidOperationError(
                    f"Vector size mismatch in comparison: {left} {op} {right}", location)
            if left.base_type != right.base_type:
                raise InvalidOperationError(
                    f"Vector type mismatch in comparison: {left} {op} {right}", location)
            # Return boolean vector of same size
            return GLSLType(TypeCategory.VECTOR, ScalarType.BOOL, (left.vector_size(),))

        # Vector-scalar comparison not allowed
        raise InvalidOperationError(
            f"Cannot compare vector and scalar directly: {left} {op} {right}", location)

    def _check_equality_op(self, op: str, left: GLSLType, right: GLSLType,
                          location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check equality operators: ==, !="""

        # Matrices cannot be compared directly
        if left.is_matrix() or right.is_matrix():
            raise InvalidOperationError(
                f"Cannot compare matrices directly: {left} {op} {right}", location)

        # Scalar-scalar -> bool
        if left.category == TypeCategory.SCALAR and right.category == TypeCategory.SCALAR:
            if left.base_type != right.base_type:
                raise InvalidOperationError(
                    f"Cannot compare {left} and {right}: types must match", location)
            return BOOL

        # Vector-vector -> bvec (component-wise)
        if left.is_vector() and right.is_vector():
            if left.vector_size() != right.vector_size():
                raise InvalidOperationError(
                    f"Vector size mismatch in equality: {left} {op} {right}", location)
            if left.base_type != right.base_type:
                raise InvalidOperationError(
                    f"Vector type mismatch in equality: {left} {op} {right}", location)
            # Return boolean vector of same size
            return GLSLType(TypeCategory.VECTOR, ScalarType.BOOL, (left.vector_size(),))

        # Other types cannot be compared
        raise InvalidOperationError(
            f"Invalid types for {op}: {left} and {right}", location)

    def _check_logical_op(self, op: str, left: GLSLType, right: GLSLType,
                         location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check logical operators: &&, ||"""

        # Only scalar bool allowed
        if left != BOOL or right != BOOL:
            raise InvalidOperationError(
                f"Logical operator {op} requires bool operands, got {left} and {right}", location)
        return BOOL

    def _check_bitwise_op(self, op: str, left: GLSLType, right: GLSLType,
                         location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check bitwise operators: &, |, ^, <<, >>"""

        # Only integer types allowed
        if not left.is_integer() or not right.is_integer():
            raise InvalidOperationError(
                f"Bitwise operator {op} requires integer types", location)

        # Must be same signedness
        if left.base_type != right.base_type:
            raise InvalidOperationError(
                f"Bitwise operator {op} requires matching types: {left} vs {right}", location)

        # Scalar-scalar
        if left.category == TypeCategory.SCALAR and right.category == TypeCategory.SCALAR:
            return left

        # Vector-vector
        if left.is_vector() and right.is_vector():
            if left.vector_size() != right.vector_size():
                raise InvalidOperationError(
                    f"Vector size mismatch in bitwise op: {left} {op} {right}", location)
            return left

        # Vector-scalar (scalar promotion)
        if left.is_vector() and right.category == TypeCategory.SCALAR:
            return left

        # Scalar-vector (scalar promotion)
        if left.category == TypeCategory.SCALAR and right.is_vector():
            return right

        raise InvalidOperationError(f"Invalid types for bitwise {op}: {left} and {right}", location)

    def _check_modulo_op(self, left: GLSLType, right: GLSLType,
                        location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check modulo operator: %"""

        # Only integer types allowed (float uses mod() function)
        if not left.is_integer() or not right.is_integer():
            raise InvalidOperationError(
                "Modulo operator % requires integer types (use mod() function for floats)", location)

        # Must be same type
        if left.base_type != right.base_type:
            raise InvalidOperationError(
                f"Modulo operator % requires matching types: {left} vs {right}", location)

        # Scalar-scalar
        if left.category == TypeCategory.SCALAR and right.category == TypeCategory.SCALAR:
            return left

        # Vector-vector
        if left.is_vector() and right.is_vector():
            if left.vector_size() != right.vector_size():
                raise InvalidOperationError(
                    f"Vector size mismatch in modulo: {left} % {right}", location)
            return left

        raise InvalidOperationError(f"Invalid types for modulo: {left} % {right}", location)

    def check_unary_op(self, op: str, operand_type: GLSLType,
                       location: Optional[Tuple[int, int]] = None) -> GLSLType:
        """
        Check unary operation and return result type.

        Args:
            op: Operator string ("-", "+", "!", "~")
            operand_type: Type of operand
            location: Source location for error reporting

        Returns:
            Result type of operation

        Raises:
            InvalidOperationError: If operation is invalid for given type
        """
        # Unary minus and plus: numeric types only
        if op in ("-", "+"):
            if not operand_type.is_numeric():
                raise InvalidOperationError(
                    f"Unary {op} requires numeric type, got {operand_type}", location)
            return operand_type

        # Logical not: bool only
        elif op == "!":
            if operand_type != BOOL:
                raise InvalidOperationError(
                    f"Logical not (!) requires bool, got {operand_type}", location)
            return BOOL

        # Bitwise not: integer types only
        elif op == "~":
            if not operand_type.is_integer():
                raise InvalidOperationError(
                    f"Bitwise not (~) requires integer type, got {operand_type}", location)
            return operand_type

        else:
            raise InvalidOperationError(f"Unknown unary operator: {op}", location)

    def check_function_call(self, func_name: str, arg_types: List[GLSLType],
                           location: Optional[Tuple[int, int]] = None) -> GLSLType:
        """
        Check function call and return result type.

        Handles:
        - Type constructors (vec3(), mat4(), float(), etc.)
        - Built-in functions (sin(), normalize(), texture(), etc.)
        - Overload resolution for generic types (genType, genIType, bvec, etc.)

        Args:
            func_name: Function name
            arg_types: List of argument types
            location: Source location for error reporting

        Returns:
            Return type of function

        Raises:
            TypeCheckError: If function not found or arguments don't match
        """
        # Check if it's a type constructor first
        constructor_type = self._check_constructor(func_name, arg_types, location)
        if constructor_type is not None:
            return constructor_type

        # Look up function in symbol table
        symbol = self.symbol_table.lookup(func_name)
        if symbol is None:
            raise TypeCheckError(f"Undefined function: {func_name}", location)

        # Check if it's actually a function (not a variable)
        if symbol.symbol_type not in (SymbolType.BUILTIN, SymbolType.FUNCTION):
            raise InvalidOperationError(
                f"{func_name} is not a function (it's a {symbol.symbol_type.value})", location)

        # Resolve generic function signature
        return self._check_builtin_function(func_name, symbol, arg_types, location)

    def _check_constructor(self, type_name: str, arg_types: List[GLSLType],
                          location: Optional[Tuple[int, int]]) -> Optional[GLSLType]:
        """
        Check type constructor calls (vec3(), mat4(), etc.).

        Returns the constructed type if valid, None if not a constructor.
        """
        # Scalar constructors: float(), int(), uint(), bool()
        if type_name in ("float", "int", "uint", "bool"):
            if len(arg_types) != 1:
                raise InvalidOperationError(
                    f"{type_name}() constructor requires exactly 1 argument", location)
            # Allow conversion from any scalar type
            if arg_types[0].category != TypeCategory.SCALAR:
                raise InvalidOperationError(
                    f"{type_name}() constructor requires scalar argument, got {arg_types[0]}", location)
            return TYPE_NAME_MAP[type_name]

        # Vector constructors
        vec_types = {
            "vec2": (VEC2, ScalarType.FLOAT, 2),
            "vec3": (VEC3, ScalarType.FLOAT, 3),
            "vec4": (VEC4, ScalarType.FLOAT, 4),
            "ivec2": (IVEC2, ScalarType.INT, 2),
            "ivec3": (IVEC3, ScalarType.INT, 3),
            "ivec4": (IVEC4, ScalarType.INT, 4),
            "uvec2": (UVEC2, ScalarType.UINT, 2),
            "uvec3": (UVEC3, ScalarType.UINT, 3),
            "uvec4": (UVEC4, ScalarType.UINT, 4),
            "bvec2": (BVEC2, ScalarType.BOOL, 2),
            "bvec3": (BVEC3, ScalarType.BOOL, 3),
            "bvec4": (BVEC4, ScalarType.BOOL, 4),
        }

        if type_name in vec_types:
            result_type, base_type, size = vec_types[type_name]
            return self._check_vector_constructor(result_type, base_type, size, arg_types, location)

        # Matrix constructors
        mat_types = {
            "mat2": (MAT2, 2),
            "mat3": (MAT3, 3),
            "mat4": (MAT4, 4),
        }

        if type_name in mat_types:
            result_type, size = mat_types[type_name]
            return self._check_matrix_constructor(result_type, size, arg_types, location)

        return None  # Not a constructor

    def _check_vector_constructor(self, result_type: GLSLType, base_type: ScalarType,
                                  size: int, arg_types: List[GLSLType],
                                  location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check vector constructor arguments."""
        if len(arg_types) == 0:
            raise InvalidOperationError(f"{result_type}() requires at least 1 argument", location)

        # Single argument: splat or identity
        if len(arg_types) == 1:
            arg = arg_types[0]
            # Scalar splat: vec3(float)
            if arg.category == TypeCategory.SCALAR and arg.base_type == base_type:
                return result_type
            # Vector identity: vec3(vec3)
            if arg == result_type:
                return result_type
            raise InvalidOperationError(
                f"{result_type}({arg}) is invalid (expected {base_type.value})", location)

        # Multiple arguments: component-wise construction
        # Count total components
        total_components = 0
        for arg in arg_types:
            if arg.category == TypeCategory.SCALAR:
                if arg.base_type != base_type:
                    raise InvalidOperationError(
                        f"{result_type}() requires {base_type.value} components, got {arg}", location)
                total_components += 1
            elif arg.category == TypeCategory.VECTOR:
                if arg.base_type != base_type:
                    raise InvalidOperationError(
                        f"{result_type}() requires {base_type.value} components, got {arg}", location)
                total_components += arg.vector_size()
            else:
                raise InvalidOperationError(
                    f"{result_type}() cannot construct from {arg}", location)

        if total_components != size:
            raise InvalidOperationError(
                f"{result_type}() requires exactly {size} components, got {total_components}", location)

        return result_type

    def _check_matrix_constructor(self, result_type: GLSLType, size: int,
                                  arg_types: List[GLSLType],
                                  location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check matrix constructor arguments."""
        if len(arg_types) == 0:
            raise InvalidOperationError(f"{result_type}() requires at least 1 argument", location)

        # Single float: diagonal matrix
        if len(arg_types) == 1:
            arg = arg_types[0]
            if arg == FLOAT:
                return result_type
            # Matrix upcasting: mat4(mat3)
            if arg.is_matrix():
                return result_type
            raise InvalidOperationError(
                f"{result_type}({arg}) is invalid", location)

        # Column vectors: mat3(vec3, vec3, vec3)
        if len(arg_types) == size and all(arg.is_vector() and arg.vector_size() == size
                                           for arg in arg_types):
            if all(arg.base_type == ScalarType.FLOAT for arg in arg_types):
                return result_type
            raise InvalidOperationError(
                f"{result_type}() requires float vectors", location)

        # Component-wise: mat2(f, f, f, f) or mat3(f, f, f, f, f, f, f, f, f), etc.
        expected_components = size * size
        if len(arg_types) == expected_components:
            if all(arg == FLOAT for arg in arg_types):
                return result_type
            raise InvalidOperationError(
                f"{result_type}() requires all float components", location)

        raise InvalidOperationError(
            f"{result_type}() invalid argument combination", location)

    def _check_builtin_function(self, func_name: str, symbol: Symbol,
                                arg_types: List[GLSLType],
                                location: Optional[Tuple[int, int]]) -> GLSLType:
        """
        Check built-in function call with generic type resolution.

        Handles generic types like genType, genIType, bvec, etc.
        """
        generic_type = symbol.glsl_type

        # Special case: texture functions
        if func_name.startswith("texture") or func_name.startswith("texel"):
            return self._check_texture_function(func_name, arg_types, location)

        # Special case: geometric functions
        if func_name in ("length", "distance", "dot"):
            return self._check_geometric_reduction_function(func_name, arg_types, location)

        if func_name == "cross":
            return self._check_cross_function(arg_types, location)

        if func_name in ("normalize", "faceforward", "reflect", "refract"):
            return self._check_geometric_vector_function(func_name, arg_types, location)

        # Special case: vector relational functions
        if func_name in ("lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual",
                         "equal", "notEqual"):
            return self._check_vector_relational_function(func_name, arg_types, location)

        if func_name in ("all", "any"):
            return self._check_reduction_function(func_name, arg_types, location)

        if func_name == "not":
            return self._check_not_function(arg_types, location)

        # Generic type functions (genType, genIType)
        if generic_type == "genType":
            return self._check_gentype_function(func_name, arg_types, location)

        if generic_type == "genIType":
            return self._check_genitype_function(func_name, arg_types, location)

        # Default: return vec4 (for texture functions, etc.)
        if generic_type == "vec4":
            return VEC4

        # Fallback
        raise InvalidOperationError(
            f"Function {func_name} signature not fully implemented", location)

    def _check_gentype_function(self, func_name: str, arg_types: List[GLSLType],
                                location: Optional[Tuple[int, int]]) -> GLSLType:
        """
        Check genType function (float, vec2, vec3, vec4).

        These functions accept float-based types and return the same type.
        Examples: sin, cos, sqrt, abs, etc.
        """
        # Determine number of expected arguments
        # Most math functions take 1 arg, some take 2 or 3
        single_arg_funcs = {"sin", "cos", "tan", "asin", "acos", "atan",
                           "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                           "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
                           "abs", "sign", "floor", "ceil", "round", "roundEven",
                           "trunc", "fract", "isnan", "isinf"}

        two_arg_funcs = {"pow", "mod", "min", "max", "step", "distance", "dot"}
        three_arg_funcs = {"clamp", "mix", "smoothstep", "faceforward"}

        if func_name in single_arg_funcs:
            if len(arg_types) != 1:
                raise InvalidOperationError(
                    f"{func_name}() requires exactly 1 argument", location)
            arg = arg_types[0]
            if not arg.is_float():
                raise InvalidOperationError(
                    f"{func_name}() requires float or vec type, got {arg}", location)
            return arg

        elif func_name in two_arg_funcs:
            if len(arg_types) != 2:
                raise InvalidOperationError(
                    f"{func_name}() requires exactly 2 arguments", location)
            left, right = arg_types
            if not left.is_float() or not right.is_float():
                raise InvalidOperationError(
                    f"{func_name}() requires float/vec arguments", location)
            # Both must be same type
            if left != right:
                raise InvalidOperationError(
                    f"{func_name}() requires matching argument types: {left} vs {right}", location)
            return left

        elif func_name in three_arg_funcs:
            if len(arg_types) != 3:
                raise InvalidOperationError(
                    f"{func_name}() requires exactly 3 arguments", location)

            # Special case for mix and clamp: can have scalar min/max or blend factor
            if func_name in ("mix", "clamp"):
                val, arg1, arg2 = arg_types
                if not val.is_float():
                    raise InvalidOperationError(f"{func_name}() requires float type", location)

                # Check if all three match (most common case): vec3, vec3, vec3 or float, float, float
                if arg1 == val and arg2 == val:
                    return val

                # Check if first two match and third is scalar (for mix blend factor or clamp scalar edges)
                if arg1 == val and arg2 == FLOAT and val.is_vector():
                    # mix(vec3, vec3, float) -> vec3
                    return val

                # Check if arg1 and arg2 are both scalars: vec3, float, float
                if arg1 == arg2 and arg1 == FLOAT and val.is_vector():
                    # clamp(vec3, float, float) -> vec3
                    return val

                raise InvalidOperationError(
                    f"{func_name}() requires matching argument types or scalar edge/blend values", location)

            # Default: all three must match
            if arg_types[0] != arg_types[1] or arg_types[0] != arg_types[2]:
                raise InvalidOperationError(
                    f"{func_name}() requires matching argument types", location)
            return arg_types[0]

        raise InvalidOperationError(f"{func_name}() not recognized", location)

    def _check_genitype_function(self, func_name: str, arg_types: List[GLSLType],
                                 location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check genIType function (int, uint, ivec, uvec)."""
        if len(arg_types) == 0:
            raise InvalidOperationError(f"{func_name}() requires arguments", location)

        arg = arg_types[0]
        if not arg.is_integer():
            raise InvalidOperationError(
                f"{func_name}() requires integer type, got {arg}", location)
        return arg

    def _check_texture_function(self, func_name: str, arg_types: List[GLSLType],
                                location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check texture function calls."""
        # Special case: textureSize(sampler, int) doesn't use coords
        if func_name == "textureSize":
            if len(arg_types) != 2:
                raise InvalidOperationError(
                    "textureSize() requires 2 arguments (sampler, lod)", location)
            sampler, lod = arg_types
            if sampler.category != TypeCategory.SAMPLER:
                raise InvalidOperationError(
                    "textureSize() requires sampler as first argument", location)
            if lod != INT:
                raise InvalidOperationError(
                    "textureSize() requires int lod as second argument", location)
            # Returns ivec based on sampler dimensions
            if sampler == SAMPLER2D:
                return IVEC2
            else:
                return VEC4  # Simplified

        # Most texture functions require at least 2 args
        if len(arg_types) < 2:
            raise InvalidOperationError(
                f"{func_name}() requires at least 2 arguments (sampler, coords)", location)

        sampler = arg_types[0]
        coords = arg_types[1]

        # First argument must be a sampler
        if sampler.category != TypeCategory.SAMPLER:
            raise InvalidOperationError(
                f"{func_name}() requires sampler as first argument, got {sampler}", location)

        # Validate coordinate type based on sampler type
        if sampler == SAMPLER2D:
            if func_name == "texelFetch":
                if len(arg_types) < 3:
                    raise InvalidOperationError(
                        "texelFetch() requires 3 arguments (sampler, coords, lod)", location)
                if coords != IVEC2:
                    raise InvalidOperationError(
                        f"texelFetch with sampler2D requires ivec2 coords, got {coords}", location)
            elif func_name == "textureLod":
                if len(arg_types) < 3:
                    raise InvalidOperationError(
                        "textureLod() requires 3 arguments (sampler, coords, lod)", location)
                if coords != VEC2:
                    raise InvalidOperationError(
                        f"textureLod with sampler2D requires vec2 coords, got {coords}", location)
            else:
                if coords != VEC2:
                    raise InvalidOperationError(
                        f"{func_name} with sampler2D requires vec2 coords, got {coords}", location)
        elif sampler in (SAMPLERCUBE, SAMPLER3D):
            if coords != VEC3:
                raise InvalidOperationError(
                    f"{func_name} with {sampler} requires vec3 coords, got {coords}", location)

        return VEC4

    def _check_geometric_reduction_function(self, func_name: str, arg_types: List[GLSLType],
                                            location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check functions that reduce vectors to scalars: length, distance, dot."""
        if func_name in ("length",):
            if len(arg_types) != 1:
                raise InvalidOperationError(f"{func_name}() requires 1 argument", location)
            arg = arg_types[0]
            if not arg.is_vector() or not arg.is_float():
                raise InvalidOperationError(
                    f"{func_name}() requires float vector, got {arg}", location)
            return FLOAT

        elif func_name in ("distance", "dot"):
            if len(arg_types) != 2:
                raise InvalidOperationError(f"{func_name}() requires 2 arguments", location)
            left, right = arg_types
            if not left.is_vector() or not left.is_float():
                raise InvalidOperationError(f"{func_name}() requires float vectors", location)
            if left != right:
                raise InvalidOperationError(
                    f"{func_name}() requires matching vector types: {left} vs {right}", location)
            return FLOAT

        return FLOAT

    def _check_cross_function(self, arg_types: List[GLSLType],
                             location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check cross product (only vec3 x vec3 -> vec3)."""
        if len(arg_types) != 2:
            raise InvalidOperationError("cross() requires 2 arguments", location)
        if arg_types[0] != VEC3 or arg_types[1] != VEC3:
            raise InvalidOperationError(
                "cross() requires vec3 arguments", location)
        return VEC3

    def _check_geometric_vector_function(self, func_name: str, arg_types: List[GLSLType],
                                         location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check geometric functions that return vectors: normalize, reflect, etc."""
        if func_name == "normalize":
            if len(arg_types) != 1:
                raise InvalidOperationError(f"{func_name}() requires 1 argument", location)
            arg = arg_types[0]
            if not arg.is_vector() or not arg.is_float():
                raise InvalidOperationError(
                    f"{func_name}() requires float vector, got {arg}", location)
            return arg

        elif func_name in ("reflect", "faceforward"):
            if len(arg_types) < 2:
                raise InvalidOperationError(f"{func_name}() requires at least 2 arguments", location)
            if func_name == "faceforward" and len(arg_types) != 3:
                raise InvalidOperationError("faceforward() requires 3 arguments", location)
            # All arguments must be same vector type
            first = arg_types[0]
            if not first.is_vector() or not first.is_float():
                raise InvalidOperationError(f"{func_name}() requires float vectors", location)
            for arg in arg_types:
                if arg != first:
                    raise InvalidOperationError(f"{func_name}() requires matching vector types", location)
            return first

        elif func_name == "refract":
            if len(arg_types) != 3:
                raise InvalidOperationError("refract() requires 3 arguments", location)
            incident, normal, eta = arg_types
            if incident != normal or not incident.is_vector() or not incident.is_float():
                raise InvalidOperationError("refract() requires matching float vectors", location)
            if eta != FLOAT:
                raise InvalidOperationError("refract() requires float eta", location)
            return incident

        return arg_types[0]

    def _check_vector_relational_function(self, func_name: str, arg_types: List[GLSLType],
                                          location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check vector relational functions: lessThan, equal, etc."""
        if len(arg_types) != 2:
            raise InvalidOperationError(f"{func_name}() requires 2 arguments", location)

        left, right = arg_types
        if not left.is_vector() or not right.is_vector():
            raise InvalidOperationError(f"{func_name}() requires vector arguments", location)

        if left != right:
            raise InvalidOperationError(
                f"{func_name}() requires matching vector types: {left} vs {right}", location)

        # Return bvec of same size
        size = left.vector_size()
        return GLSLType(TypeCategory.VECTOR, ScalarType.BOOL, (size,))

    def _check_reduction_function(self, func_name: str, arg_types: List[GLSLType],
                                  location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check reduction functions: all, any."""
        if len(arg_types) != 1:
            raise InvalidOperationError(f"{func_name}() requires 1 argument", location)

        arg = arg_types[0]
        if not arg.is_vector() or not arg.is_bool():
            raise InvalidOperationError(
                f"{func_name}() requires bool vector, got {arg}", location)

        return BOOL

    def _check_not_function(self, arg_types: List[GLSLType],
                           location: Optional[Tuple[int, int]]) -> GLSLType:
        """Check not() function (component-wise logical not on bvec)."""
        if len(arg_types) != 1:
            raise InvalidOperationError("not() requires 1 argument", location)

        arg = arg_types[0]
        if not arg.is_vector() or not arg.is_bool():
            raise InvalidOperationError(
                f"not() requires bool vector, got {arg}", location)

        return arg

    def check_member_access(self, base_type: GLSLType, member: str,
                           location: Optional[Tuple[int, int]] = None) -> GLSLType:
        """
        Check member access (swizzling, struct members) and return result type.

        GLSL vector swizzling supports:
        - xyzw (positional)
        - rgba (color)
        - stpq (texture coordinates)

        Args:
            base_type: Type of base expression
            member: Member name (e.g., "xy", "rgb", struct field)
            location: Source location for error reporting

        Returns:
            Type of member access result

        Raises:
            TypeCheckError: If member access is invalid
        """
        # Only vectors support swizzling
        if not base_type.is_vector():
            raise InvalidOperationError(
                f"Cannot access member '{member}' on non-vector type {base_type}", location)

        # Empty member is invalid
        if not member:
            raise InvalidOperationError("Empty member access", location)

        # Check if it's a valid swizzle
        return self._check_swizzle(base_type, member, location)

    def _check_swizzle(self, vec_type: GLSLType, swizzle: str,
                      location: Optional[Tuple[int, int]]) -> GLSLType:
        """
        Check vector swizzle and return result type.

        Args:
            vec_type: Vector type being swizzled
            swizzle: Swizzle string (e.g., "xy", "rgb", "xyzw")
            location: Source location for error reporting

        Returns:
            Result type (scalar for single component, vector for multiple)
        """
        # Define swizzle sets
        xyzw_set = {'x', 'y', 'z', 'w'}
        rgba_set = {'r', 'g', 'b', 'a'}
        stpq_set = {'s', 't', 'p', 'q'}

        # Map characters to component indices
        xyzw_map = {'x': 0, 'y': 1, 'z': 2, 'w': 3}
        rgba_map = {'r': 0, 'g': 1, 'b': 2, 'a': 3}
        stpq_map = {'s': 0, 't': 1, 'p': 2, 'q': 3}

        # Get vector size
        vec_size = vec_type.vector_size()

        # Max swizzle length is 4
        if len(swizzle) > 4:
            raise InvalidOperationError(
                f"Swizzle too long: {swizzle} (max 4 components)", location)

        # Determine which swizzle set is being used
        swizzle_set = None
        swizzle_map = None

        for char in swizzle:
            if char in xyzw_set:
                if swizzle_set is None:
                    swizzle_set = "xyzw"
                    swizzle_map = xyzw_map
                elif swizzle_set != "xyzw":
                    raise InvalidOperationError(
                        f"Cannot mix swizzle sets in '{swizzle}'", location)
            elif char in rgba_set:
                if swizzle_set is None:
                    swizzle_set = "rgba"
                    swizzle_map = rgba_map
                elif swizzle_set != "rgba":
                    raise InvalidOperationError(
                        f"Cannot mix swizzle sets in '{swizzle}'", location)
            elif char in stpq_set:
                if swizzle_set is None:
                    swizzle_set = "stpq"
                    swizzle_map = stpq_map
                elif swizzle_set != "stpq":
                    raise InvalidOperationError(
                        f"Cannot mix swizzle sets in '{swizzle}'", location)
            else:
                raise InvalidOperationError(
                    f"Invalid swizzle character '{char}' in '{swizzle}'", location)

        # Validate each component is within vector size
        for char in swizzle:
            component_index = swizzle_map[char]
            if component_index >= vec_size:
                raise InvalidOperationError(
                    f"Swizzle component '{char}' out of range for {vec_type}", location)

        # Return appropriate type based on swizzle length
        swizzle_len = len(swizzle)
        base_type = vec_type.base_type

        if swizzle_len == 1:
            # Single component: return scalar
            return GLSLType(TypeCategory.SCALAR, base_type)
        elif swizzle_len == 2:
            # Two components: return vec2
            return GLSLType(TypeCategory.VECTOR, base_type, (2,))
        elif swizzle_len == 3:
            # Three components: return vec3
            return GLSLType(TypeCategory.VECTOR, base_type, (3,))
        elif swizzle_len == 4:
            # Four components: return vec4
            return GLSLType(TypeCategory.VECTOR, base_type, (4,))
        else:
            raise InvalidOperationError(f"Invalid swizzle length: {swizzle_len}", location)

    def can_convert(self, from_type: GLSLType, to_type: GLSLType) -> bool:
        """
        Check if implicit conversion is allowed from from_type to to_type.

        GLSL implicit conversion rules:
        - int → float (widening)
        - uint → float (widening)
        - Scalar → vector of same base type (splatting)
        - No implicit narrowing (float → int)
        - No implicit conversions between signed/unsigned

        Args:
            from_type: Source type
            to_type: Target type

        Returns:
            True if implicit conversion is allowed
        """
        # Exact match
        if from_type == to_type:
            return True

        # Scalar to same scalar: int → float, uint → float
        if from_type.category == TypeCategory.SCALAR and to_type.category == TypeCategory.SCALAR:
            if from_type.base_type == ScalarType.INT and to_type.base_type == ScalarType.FLOAT:
                return True
            if from_type.base_type == ScalarType.UINT and to_type.base_type == ScalarType.FLOAT:
                return True

        # Scalar to vector (splatting): float → vec3
        if from_type.category == TypeCategory.SCALAR and to_type.category == TypeCategory.VECTOR:
            if from_type.base_type == to_type.base_type:
                return True

        return False


def parse_type_string(type_str: str) -> GLSLType:
    """
    Parse a type string into a GLSLType.

    Args:
        type_str: Type string (e.g., "vec3", "mat4", "float")

    Returns:
        Corresponding GLSLType

    Raises:
        UndefinedTypeError: If type string is not recognized
    """
    type_str = type_str.strip()

    if type_str in TYPE_NAME_MAP:
        return TYPE_NAME_MAP[type_str]

    # Handle array types (e.g., "float[4]")
    if '[' in type_str:
        base = type_str[:type_str.index('[')]
        # For now, return base type; full array support to be added
        if base in TYPE_NAME_MAP:
            return TYPE_NAME_MAP[base]

    raise UndefinedTypeError(f"Unknown type: {type_str}")
