"""
Symbol Table for GLSL Semantic Analysis

Implements hierarchical symbol table with support for:
- Variable declarations (local, global, uniform, etc.)
- Function definitions and overloading
- Struct definitions
- Built-in GLSL symbols
- Scope management (global, function, block)

Phase 3, Week 7
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class SymbolType(Enum):
    """Type of symbol in the symbol table."""

    VARIABLE = "variable"
    FUNCTION = "function"
    STRUCT = "struct"
    PARAMETER = "parameter"
    BUILTIN = "builtin"
    CONSTANT = "constant"


@dataclass
class Symbol:
    """
    Represents a symbol in the symbol table.

    Attributes:
        name: Symbol identifier
        symbol_type: Type of symbol (variable, function, struct, etc.)
        glsl_type: GLSL type string ("float", "vec3", "mat4", etc.)
        qualifiers: Type qualifiers (const, uniform, in, out, etc.)
        location: Source location (line, column) for error reporting
        metadata: Additional symbol-specific data
    """

    name: str
    symbol_type: SymbolType
    glsl_type: str
    qualifiers: List[str] = field(default_factory=list)
    location: Optional[tuple] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        qualifiers_str = f" {' '.join(self.qualifiers)}" if self.qualifiers else ""
        return f"Symbol({self.name}, {self.symbol_type.value},{qualifiers_str} {self.glsl_type})"

    def is_const(self) -> bool:
        """Check if symbol has const qualifier."""
        return "const" in self.qualifiers

    def is_uniform(self) -> bool:
        """Check if symbol is a uniform."""
        return "uniform" in self.qualifiers

    def is_builtin(self) -> bool:
        """Check if symbol is a built-in."""
        return self.symbol_type == SymbolType.BUILTIN


class SymbolTableError(Exception):
    """Base exception for symbol table errors."""

    pass


class DuplicateSymbolError(SymbolTableError):
    """Raised when attempting to insert duplicate symbol in same scope."""

    def __init__(self, name: str, location: Optional[tuple] = None):
        self.name = name
        self.location = location
        loc_str = f" at {location}" if location else ""
        super().__init__(f"Symbol '{name}' already declared in current scope{loc_str}")


class UndeclaredSymbolError(SymbolTableError):
    """Raised when looking up a symbol that doesn't exist."""

    def __init__(self, name: str, location: Optional[tuple] = None):
        self.name = name
        self.location = location
        loc_str = f" at {location}" if location else ""
        super().__init__(f"Undeclared symbol '{name}'{loc_str}")


class BuiltinRedeclarationError(SymbolTableError):
    """Raised when attempting to redeclare a built-in symbol."""

    def __init__(self, name: str, location: Optional[tuple] = None):
        self.name = name
        self.location = location
        loc_str = f" at {location}" if location else ""
        super().__init__(f"Cannot redeclare built-in symbol '{name}'{loc_str}")


class SymbolTable:
    """
    Hierarchical symbol table for GLSL semantic analysis.

    Supports:
    - Global scope (uniforms, global variables, functions, structs)
    - Function scope (parameters, local variables)
    - Block scope (loop variables, if-block variables)
    - Symbol shadowing (inner scopes can shadow outer scopes)
    - Built-in symbol protection (built-ins cannot be redeclared)

    Example:
        >>> table = SymbolTable()
        >>> table.insert("x", Symbol("x", SymbolType.VARIABLE, "float"))
        >>> found = table.lookup("x")
        >>> print(found.glsl_type)
        float
    """

    def __init__(self, parent: Optional['SymbolTable'] = None):
        """
        Initialize symbol table.

        Args:
            parent: Parent scope (None for global scope)
        """
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}
        self.children: List['SymbolTable'] = []
        self._scope_depth = 0 if parent is None else parent._scope_depth + 1

    def insert(self, name: str, symbol: Symbol, allow_shadowing: bool = True) -> None:
        """
        Insert symbol into current scope.

        Args:
            name: Symbol name
            symbol: Symbol object
            allow_shadowing: If False, raise error if symbol exists in any parent scope

        Raises:
            DuplicateSymbolError: If symbol already exists in current scope
            BuiltinRedeclarationError: If attempting to redeclare built-in
        """
        # Check if shadowing a built-in (not allowed) - must check first
        existing = self.lookup(name, recursive=True)
        if existing and existing.is_builtin():
            raise BuiltinRedeclarationError(name, symbol.location)

        # Check if symbol exists in current scope
        if name in self.symbols:
            raise DuplicateSymbolError(name, symbol.location)

        # Check if shadowing is allowed
        if not allow_shadowing and existing:
            raise DuplicateSymbolError(name, symbol.location)

        # Insert symbol
        self.symbols[name] = symbol

    def lookup(self, name: str, recursive: bool = True) -> Optional[Symbol]:
        """
        Lookup symbol in current scope or parent scopes.

        Args:
            name: Symbol name
            recursive: If True, search parent scopes

        Returns:
            Symbol if found, None otherwise
        """
        # Check current scope
        if name in self.symbols:
            return self.symbols[name]

        # Check parent scopes
        if recursive and self.parent is not None:
            return self.parent.lookup(name, recursive=True)

        return None

    def lookup_current_scope(self, name: str) -> Optional[Symbol]:
        """
        Lookup symbol only in current scope (no parent lookup).

        Args:
            name: Symbol name

        Returns:
            Symbol if found in current scope, None otherwise
        """
        return self.symbols.get(name)

    def push_scope(self) -> 'SymbolTable':
        """
        Create new child scope and return it.

        Returns:
            New child SymbolTable
        """
        child = SymbolTable(parent=self)
        self.children.append(child)
        return child

    def pop_scope(self) -> Optional['SymbolTable']:
        """
        Return to parent scope.

        Returns:
            Parent SymbolTable, or None if already at global scope
        """
        return self.parent

    def get_scope_depth(self) -> int:
        """
        Get depth of current scope (0 = global).

        Returns:
            Scope depth
        """
        return self._scope_depth

    def get_all_symbols(self, include_parents: bool = False) -> Dict[str, Symbol]:
        """
        Get all symbols in current scope.

        Args:
            include_parents: If True, include symbols from parent scopes

        Returns:
            Dictionary of symbol name to Symbol
        """
        if not include_parents:
            return self.symbols.copy()

        # Collect symbols from all parent scopes
        all_symbols = {}
        current = self
        while current is not None:
            # Add symbols from current scope (earlier scopes override later)
            for name, symbol in current.symbols.items():
                if name not in all_symbols:
                    all_symbols[name] = symbol
            current = current.parent

        return all_symbols

    def get_symbols_by_type(self, symbol_type: SymbolType) -> List[Symbol]:
        """
        Get all symbols of a specific type in current scope.

        Args:
            symbol_type: Type of symbols to retrieve

        Returns:
            List of matching symbols
        """
        return [
            symbol
            for symbol in self.symbols.values()
            if symbol.symbol_type == symbol_type
        ]

    def has_symbol(self, name: str, recursive: bool = True) -> bool:
        """
        Check if symbol exists.

        Args:
            name: Symbol name
            recursive: If True, search parent scopes

        Returns:
            True if symbol exists
        """
        return self.lookup(name, recursive=recursive) is not None

    def clear(self) -> None:
        """Clear all symbols from current scope."""
        self.symbols.clear()
        self.children.clear()

    def __repr__(self) -> str:
        return f"SymbolTable(depth={self._scope_depth}, symbols={len(self.symbols)})"

    def __str__(self) -> str:
        """Pretty print symbol table."""
        indent = "  " * self._scope_depth
        lines = [f"{indent}SymbolTable (depth={self._scope_depth}):"]
        for name, symbol in self.symbols.items():
            lines.append(f"{indent}  {name}: {symbol}")
        return "\n".join(lines)
