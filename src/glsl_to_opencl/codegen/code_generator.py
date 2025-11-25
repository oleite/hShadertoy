"""
Code Generator

Main code generator class that orchestrates OpenCL C code generation from
transformed GLSL AST.
"""

from typing import List, Optional
from .opencl_emitter import OpenCLEmitter
from .formatting import Formatter
from ..parser.ast_nodes import (
    TranslationUnit,
    FunctionDefinition,
    Declaration,
)


class CodeGenerator:
    """
    Main code generator - orchestrates code emission from AST.

    This class provides the high-level API for generating OpenCL C code
    from a transformed GLSL AST. It delegates the actual code emission
    to OpenCLEmitter.

    Usage:
        generator = CodeGenerator(indent_style="    ")
        opencl_code = generator.generate(transformed_ast)
    """

    def __init__(self,
                 indent_style: str = "    ",
                 preserve_comments: bool = False,
                 include_line_directives: bool = False):
        """
        Initialize the code generator.

        Args:
            indent_style: String used for indentation (default: 4 spaces)
            preserve_comments: Whether to preserve comments (not implemented yet)
            include_line_directives: Whether to include #line directives (not implemented yet)
        """
        self.indent_style = indent_style
        self.preserve_comments = preserve_comments
        self.include_line_directives = include_line_directives
        self.emitter = OpenCLEmitter(indent_style)
        self.formatter = Formatter()

    def generate(self, ast: TranslationUnit) -> str:
        """
        Generate complete OpenCL source from AST.

        This is the main entry point for code generation. It takes a
        transformed GLSL AST and produces OpenCL C source code.

        Args:
            ast: TranslationUnit node (root of AST)

        Returns:
            Generated OpenCL C source code

        Example:
            >>> generator = CodeGenerator()
            >>> opencl = generator.generate(transformed_ast)
        """
        return self.emitter.visit(ast)

    def generate_function(self, func: FunctionDefinition) -> str:
        """
        Generate single function definition.

        Args:
            func: FunctionDefinition node

        Returns:
            Generated function code

        Example:
            >>> generator = CodeGenerator()
            >>> func_code = generator.generate_function(main_image_func)
        """
        return self.emitter.visit(func)

    def generate_declarations(self, declarations: List[Declaration]) -> str:
        """
        Generate global declarations (Common tab code).

        Args:
            declarations: List of declaration nodes

        Returns:
            Generated declaration code

        Example:
            >>> generator = CodeGenerator()
            >>> common_code = generator.generate_declarations(common_decls)
        """
        parts = []
        for decl in declarations:
            code = self.emitter.visit(decl)
            if code:
                parts.append(code)
                parts.append('\n')  # Blank line between declarations

        return ''.join(parts)

    def generate_expression(self, expr) -> str:
        """
        Generate code for a single expression.

        Useful for testing or generating code snippets.

        Args:
            expr: Expression node

        Returns:
            Generated expression code

        Example:
            >>> generator = CodeGenerator()
            >>> expr_code = generator.generate_expression(binary_expr)
        """
        return self.emitter.visit(expr)

    def generate_statement(self, stmt) -> str:
        """
        Generate code for a single statement.

        Useful for testing or generating code snippets.

        Args:
            stmt: Statement node

        Returns:
            Generated statement code

        Example:
            >>> generator = CodeGenerator()
            >>> stmt_code = generator.generate_statement(if_stmt)
        """
        return self.emitter.visit(stmt)
