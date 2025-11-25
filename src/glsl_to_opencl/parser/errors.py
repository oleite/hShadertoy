"""
Parser error classes.
"""


class ParseError(Exception):
    """Raised when GLSL source cannot be parsed."""

    def __init__(self, message: str, line: int = 0, column: int = 0):
        """
        Initialize parse error.

        Args:
            message: Error description
            line: Line number (0-indexed)
            column: Column number (0-indexed)
        """
        super().__init__(message)
        self.line = line
        self.column = column
        self.message = message

    def __str__(self):
        return f"ParseError(line {self.line + 1}, col {self.column + 1}): {self.message}"
