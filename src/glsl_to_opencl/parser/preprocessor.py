"""
Basic GLSL preprocessor.

Handles simple #define macros for Shadertoy shaders.
"""

import re


class GLSLPreprocessor:
    """
    Basic GLSL preprocessor.

    Handles:
    - #define macros (simple text replacement)

    Does NOT handle:
    - #if/#ifdef/#ifndef (complex conditionals)
    - Function-like macros with parameters
    - #include directives (future feature)
    """

    def __init__(self):
        """Initialize preprocessor."""
        self.defines = {}

    def process(self, source: str) -> str:
        """
        Process preprocessor directives.

        Args:
            source: GLSL source with preprocessor directives

        Returns:
            Processed source with directives expanded
        """
        lines = source.split('\n')
        output_lines = []

        for line in lines:
            # Handle #define
            if line.strip().startswith('#define'):
                self._handle_define(line)
                # Keep define in output (commented)
                output_lines.append(f"// {line}")
            else:
                # Expand macros
                expanded = self._expand_macros(line)
                output_lines.append(expanded)

        return '\n'.join(output_lines)

    def _handle_define(self, line: str):
        """Parse and store #define directive."""
        match = re.match(r'#define\s+(\w+)\s+(.*)', line.strip())
        if match:
            name, value = match.groups()
            self.defines[name] = value.strip()

    def _expand_macros(self, line: str) -> str:
        """Expand macros in line."""
        result = line
        for name, value in self.defines.items():
            # Simple text replacement (word boundaries)
            pattern = r'\b' + re.escape(name) + r'\b'
            result = re.sub(pattern, value, result)
        return result
