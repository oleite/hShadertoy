"""
GLSL Parser using tree-sitter-glsl.
"""

from tree_sitter import Language, Parser
import tree_sitter_glsl as tsglsl

from .ast_nodes import TranslationUnit
from .errors import ParseError


class GLSLParser:
    """
    Main GLSL parser using tree-sitter-glsl.

    Usage:
        parser = GLSLParser()
        ast = parser.parse(glsl_source)
    """

    def __init__(self):
        """Initialize tree-sitter-glsl parser."""
        self._language = Language(tsglsl.language())
        self._parser = Parser(self._language)

    def parse(self, source: str) -> TranslationUnit:
        """
        Parse GLSL source code into AST.

        Args:
            source: GLSL source code as string

        Returns:
            TranslationUnit (root AST node)

        Raises:
            ParseError: If parsing fails
        """
        # Convert source to bytes (tree-sitter requirement)
        source_bytes = bytes(source, "utf8")

        # Parse with tree-sitter
        tree = self._parser.parse(source_bytes)
        root_node = tree.root_node

        # Check for parse errors
        if root_node.has_error:
            # Find first error node
            error_node = self._find_error_node(root_node)
            if error_node:
                line, col = error_node.start_point
                raise ParseError(
                    f"Syntax error at line {line + 1}, column {col + 1}",
                    line=line,
                    column=col
                )
            else:
                raise ParseError("Syntax error in GLSL source")

        # Wrap in typed AST
        return TranslationUnit(root_node, source)

    def _find_error_node(self, node):
        """Find first ERROR node in tree (DFS)."""
        if node.type == "ERROR":
            return node
        for child in node.children:
            error = self._find_error_node(child)
            if error:
                return error
        return None
