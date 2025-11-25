"""
Shader Metadata Extraction

Extracts shader interface metadata from parsed GLSL AST including:
- Main function (mainImage)
- Uniforms (Shadertoy built-ins and user-defined)
- Texture samplers (iChannel0-3)
- Global variables and constants
- User-defined functions
- Struct definitions

Phase 3, Week 9
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from ..parser import TranslationUnit, FunctionDefinition, Declaration
from .symbol_table import Symbol, SymbolType, SymbolTable
from .builtins import create_builtin_symbol_table


@dataclass
class ShaderMetadata:
    """
    Stores extracted shader interface metadata.

    Attributes:
        main_function: The mainImage() function (Shadertoy entry point)
        uniforms: List of uniform variable symbols
        samplers: List of texture sampler symbols (iChannel0-3)
        global_variables: List of global variable symbols
        constants: List of global const declarations
        functions: List of user-defined functions (excluding mainImage)
        structs: List of struct definition symbols
    """

    main_function: Optional[FunctionDefinition] = None
    uniforms: List[Symbol] = field(default_factory=list)
    samplers: List[Symbol] = field(default_factory=list)
    global_variables: List[Symbol] = field(default_factory=list)
    constants: List[Symbol] = field(default_factory=list)
    functions: List[FunctionDefinition] = field(default_factory=list)
    structs: List[Symbol] = field(default_factory=list)

    def has_main_function(self) -> bool:
        """Check if mainImage() function was found."""
        return self.main_function is not None

    def get_uniform_names(self) -> List[str]:
        """Get list of uniform variable names."""
        return [sym.name for sym in self.uniforms]

    def get_sampler_names(self) -> List[str]:
        """Get list of sampler variable names."""
        return [sym.name for sym in self.samplers]

    def get_function_names(self) -> List[str]:
        """Get list of user-defined function names."""
        return [func.name for func in self.functions]

    def uses_shadertoy_uniform(self, name: str) -> bool:
        """Check if shader uses a specific Shadertoy uniform (iTime, iResolution, etc.)."""
        return name in self.get_uniform_names()

    def uses_texture_channel(self, channel: int) -> bool:
        """Check if shader uses a specific texture channel (0-3)."""
        channel_name = f"iChannel{channel}"
        return channel_name in self.get_sampler_names()


class MetadataExtractionError(Exception):
    """Base exception for metadata extraction errors."""
    pass


class MetadataExtractor:
    """
    Extracts shader interface metadata from parsed GLSL AST.

    Traverses the AST to identify:
    - mainImage() function
    - Uniform declarations
    - Texture samplers
    - Global variables and constants
    - User-defined functions
    - Struct definitions
    """

    # Shadertoy uniform names
    SHADERTOY_UNIFORMS = {
        'iTime', 'iTimeDelta', 'iFrame', 'iFrameRate',
        'iResolution', 'iMouse', 'iDate',
        'iChannelTime', 'iChannelResolution',
        'iSampleRate'
    }

    # Shadertoy sampler names
    SHADERTOY_SAMPLERS = {'iChannel0', 'iChannel1', 'iChannel2', 'iChannel3'}

    # Sampler types
    SAMPLER_TYPES = {'sampler2D', 'sampler3D', 'samplerCube'}

    def __init__(self, symbol_table: Optional[SymbolTable] = None):
        """
        Initialize metadata extractor.

        Args:
            symbol_table: Optional symbol table for type lookup.
                         If None, creates default builtin symbol table.
        """
        self.symbol_table = symbol_table or create_builtin_symbol_table()
        self.metadata = ShaderMetadata()
        self._used_uniforms: Set[str] = set()
        self._used_samplers: Set[str] = set()

    def extract(self, ast: TranslationUnit) -> ShaderMetadata:
        """
        Extract metadata from AST.

        Args:
            ast: Parsed GLSL translation unit

        Returns:
            ShaderMetadata containing extracted interface information
        """
        self.metadata = ShaderMetadata()
        self._used_uniforms = set()
        self._used_samplers = set()

        # Process all top-level declarations
        for decl in ast.declarations:
            self._process_declaration(decl)

        # Extract actually used Shadertoy uniforms and samplers from symbol table
        self._extract_used_symbols()

        return self.metadata

    def _process_declaration(self, decl) -> None:
        """Process a top-level declaration."""
        # Check if it's a function definition
        if isinstance(decl, FunctionDefinition):
            if self._is_main_image(decl):
                self.metadata.main_function = decl
            else:
                self.metadata.functions.append(decl)

        # Check if it's a variable declaration
        elif isinstance(decl, Declaration):
            self._process_variable_declaration(decl)

    def _is_main_image(self, func: FunctionDefinition) -> bool:
        """
        Check if function is mainImage().

        Shadertoy mainImage signature:
        void mainImage(out vec4 fragColor, in vec2 fragCoord)
        """
        if func.name != "mainImage":
            return False

        # Check return type
        if func.return_type and func.return_type.text != "void":
            return False

        # Check parameter count
        if len(func.parameters) != 2:
            return False

        # Check parameter types (optional - can be more lenient)
        # First param: out vec4 fragColor
        # Second param: in vec2 fragCoord

        return True

    def _process_variable_declaration(self, decl: Declaration) -> None:
        """Process a variable declaration."""
        # Extract qualifiers, type, and name from AST
        qualifiers = []
        type_name = None
        var_name = None

        for child in decl.children:
            # Handle type qualifiers (const, uniform, etc.)
            if child.type == 'type_qualifier':
                # Extract the actual qualifier keyword from type_qualifier children
                for q in child.children:
                    if q.type in ('uniform', 'const', 'in', 'out', 'inout'):
                        qualifiers.append(q.text)
            # Handle direct qualifiers (in some cases)
            elif child.type in ('uniform', 'const', 'in', 'out', 'inout'):
                qualifiers.append(child.text)
            # Handle type name
            elif child.type in ('primitive_type', 'type_identifier'):
                type_name = child.text
            # Handle identifier (for declarations without initializers)
            elif child.type == 'identifier':
                var_name = child.text
            # Handle init_declarator (for declarations with initializers)
            elif child.type == 'init_declarator':
                # Extract identifier from init_declarator
                for init_child in child.children:
                    if init_child.type == 'identifier':
                        var_name = init_child.text
                        break

        # If we didn't extract necessary info, skip
        if not var_name or not type_name:
            return

        # Check if it's a uniform
        if 'uniform' in qualifiers:
            symbol = Symbol(
                name=var_name,
                symbol_type=SymbolType.VARIABLE,
                glsl_type=type_name,
                qualifiers=qualifiers,
                location=decl.start_point
            )

            # Check if it's a sampler
            if type_name in self.SAMPLER_TYPES:
                self.metadata.samplers.append(symbol)
            else:
                self.metadata.uniforms.append(symbol)

        # Check if it's a const
        elif 'const' in qualifiers:
            symbol = Symbol(
                name=var_name,
                symbol_type=SymbolType.CONSTANT,
                glsl_type=type_name,
                qualifiers=qualifiers,
                location=decl.start_point
            )
            self.metadata.constants.append(symbol)

        # Otherwise it's a global variable
        else:
            symbol = Symbol(
                name=var_name,
                symbol_type=SymbolType.VARIABLE,
                glsl_type=type_name,
                qualifiers=qualifiers,
                location=decl.start_point
            )
            self.metadata.global_variables.append(symbol)

    def _extract_used_symbols(self) -> None:
        """
        Extract Shadertoy uniforms and samplers that are actually used.

        Since Shadertoy uniforms are built-ins and not explicitly declared,
        we need to check which ones are referenced in the code.
        This is done by checking the symbol table for lookups.
        """
        # For now, we'll extract all Shadertoy built-ins from the symbol table
        # In a future iteration, we could track actual usage by analyzing expressions

        for name in self.SHADERTOY_UNIFORMS:
            symbol = self.symbol_table.lookup(name)
            if symbol and symbol not in self.metadata.uniforms:
                # Only add if not already in uniforms (user might have declared it)
                if name not in self.metadata.get_uniform_names():
                    # Create a copy for metadata
                    uniform_symbol = Symbol(
                        name=symbol.name,
                        symbol_type=symbol.symbol_type,
                        glsl_type=symbol.glsl_type,
                        qualifiers=['uniform'],
                        metadata={'builtin': True, 'shadertoy': True}
                    )
                    self._used_uniforms.add(name)

        for name in self.SHADERTOY_SAMPLERS:
            symbol = self.symbol_table.lookup(name)
            if symbol and symbol not in self.metadata.samplers:
                # Only add if not already in samplers
                if name not in self.metadata.get_sampler_names():
                    # Create a copy for metadata
                    sampler_symbol = Symbol(
                        name=symbol.name,
                        symbol_type=symbol.symbol_type,
                        glsl_type=symbol.glsl_type,
                        qualifiers=['uniform'],
                        metadata={'builtin': True, 'shadertoy': True}
                    )
                    self._used_samplers.add(name)

    def get_metadata(self) -> ShaderMetadata:
        """Get the extracted metadata."""
        return self.metadata
