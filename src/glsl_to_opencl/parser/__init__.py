"""
Parser Module

Handles GLSL ES 3.0 tokenization and Abstract Syntax Tree (AST) generation
using tree-sitter-glsl.

Components:
    - glsl_parser.py: Main parser class
    - ast_nodes.py: AST node classes
    - visitor.py: AST traversal infrastructure
    - preprocessor.py: Basic preprocessor support
    - errors.py: Parser exceptions

Development Phase: Phase 1-2 (Weeks 1-6)
Risk Level: LOW-MEDIUM
Test Target: 500+ tests
"""

from .glsl_parser import GLSLParser
from .ast_nodes import (
    ASTNode,
    TranslationUnit,
    FunctionDefinition,
    Declaration,
    CallExpression,
    BinaryExpression,
)
from .visitor import ASTVisitor
from .preprocessor import GLSLPreprocessor
from .errors import ParseError

__all__ = [
    "GLSLParser",
    "ASTNode",
    "TranslationUnit",
    "FunctionDefinition",
    "Declaration",
    "CallExpression",
    "BinaryExpression",
    "ASTVisitor",
    "GLSLPreprocessor",
    "ParseError",
]
