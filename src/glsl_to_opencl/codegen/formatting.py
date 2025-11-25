"""
Code Formatting Utilities

Provides utilities for formatting OpenCL C code with proper indentation
and structure.
"""

from typing import List


class Formatter:
    """Utilities for code formatting."""

    @staticmethod
    def indent(code: str, level: int, style: str = "    ") -> str:
        """
        Add indentation to code.

        Args:
            code: Code string to indent
            level: Indentation level (0, 1, 2, ...)
            style: Indentation style (default: 4 spaces)

        Returns:
            Indented code string
        """
        if not code:
            return code

        indent_str = style * level
        lines = code.split('\n')
        return '\n'.join(indent_str + line if line.strip() else line for line in lines)

    @staticmethod
    def format_block(statements: List[str], indent_level: int, style: str = "    ") -> str:
        """
        Format block with proper braces and indentation.

        Args:
            statements: List of statement strings
            indent_level: Current indentation level
            style: Indentation style

        Returns:
            Formatted block string with braces
        """
        if not statements:
            return "{}"

        indent_str = style * indent_level
        inner_indent = style * (indent_level + 1)

        lines = ["{"]
        for stmt in statements:
            if stmt.strip():
                lines.append(inner_indent + stmt)
        lines.append(indent_str + "}")

        return '\n'.join(lines)

    @staticmethod
    def format_function(return_type: str, name: str, params: List[str],
                       body: str, indent_level: int = 0, style: str = "    ") -> str:
        """
        Format complete function definition.

        Args:
            return_type: Function return type
            name: Function name
            params: List of parameter strings
            body: Function body (already formatted)
            indent_level: Base indentation level
            style: Indentation style

        Returns:
            Formatted function definition
        """
        indent_str = style * indent_level

        # Function signature
        param_str = ", ".join(params)
        signature = f"{return_type} {name}({param_str})"

        # Format body
        lines = [indent_str + signature]
        lines.append(indent_str + "{")

        # Add body lines with indentation
        if body.strip():
            body_lines = body.split('\n')
            for line in body_lines:
                if line.strip():
                    lines.append(style * (indent_level + 1) + line)

        lines.append(indent_str + "}")

        return '\n'.join(lines)

    @staticmethod
    def join_statements(statements: List[str], separator: str = "\n") -> str:
        """
        Join multiple statements with separator.

        Args:
            statements: List of statement strings
            separator: Separator string (default: newline)

        Returns:
            Joined statements
        """
        return separator.join(stmt for stmt in statements if stmt.strip())
