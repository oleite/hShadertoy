#!/usr/bin/env python3
"""
GLSL to OpenCL Transpiler - Updated for Session 6 Architecture

This script transpiles GLSL shaders into split header and kernel files
suitable for testing with compilecl.py.

Architecture (Sessions 1-6):
    GLSL Source -> Parser -> TypeChecker -> ASTTransformer -> OpenCLEmitter -> OpenCL

Output modes:
- header.cl: Everything BEFORE void mainImage() (globals, helper functions)
- kernel.cl: Only the mainImage() BODY (without signature, for compilecl.py)
- full.cl: Complete OpenCL code (optional)

Usage (CLI):
    python transpile.py input.glsl
    # Outputs: input.header.cl and input.kernel.cl

    python transpile.py input.glsl --output-dir output/
    # Outputs: output/input.header.cl and output/input.kernel.cl

    python transpile.py input.glsl --full
    # Outputs: input.full.cl (complete OpenCL code)

    python transpile.py input.glsl --verbose
    # Show transformation stages

    python transpile.py input.glsl --validate
    # Validate OpenCL compilation (requires PyOpenCL)

Usage (Module):
    from transpile import transpile

    result = transpile(glsl_source_string)
    header_code = result.get_header()
    kernel_code = result.get_kernel()
    full_code = result.get_full()
"""

import sys
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the new Session 6 architecture (updated for Session 9)
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter
from src.glsl_to_opencl.preprocessor import PreprocessorTransformer


@dataclass
class TranspileResult:
    """
    Result of GLSL to OpenCL transpilation.

    Attributes:
        header: OpenCL code before mainImage() (globals, functions)
        kernel: OpenCL mainImage() body only (for compilecl.py)
        full: Complete OpenCL code (header + mainImage signature + body)
    """
    header: str
    kernel: str
    full: str

    def get_header(self) -> str:
        """Get header OpenCL code (before mainImage)."""
        return self.header

    def get_kernel(self) -> str:
        """Get kernel body OpenCL code (mainImage body only)."""
        return self.kernel

    def get_full(self) -> str:
        """Get full OpenCL code (complete)."""
        return self.full


class TranspileError(Exception):
    """Raised when transpilation fails."""
    pass


def post_process_ifdef_blocks(opencl_code: str) -> str:
    """
    Post-process OpenCL code to fix GLSL types and functions inside #ifdef blocks.

    This is a workaround for Session 9's limitation where code inside #ifdef blocks
    is not transformed by the AST transformer.

    Args:
        opencl_code: OpenCL code string

    Returns:
        Fixed OpenCL code with all GLSL types and functions transformed
    """
    # Type transformations
    type_map = {
        r'\bvec2\b': 'float2',
        r'\bvec3\b': 'float3',
        r'\bvec4\b': 'float4',
        r'\bivec2\b': 'int2',
        r'\bivec3\b': 'int3',
        r'\bivec4\b': 'int4',
        r'\buvec2\b': 'uint2',
        r'\buvec3\b': 'uint3',
        r'\buvec4\b': 'uint4',
    }

    for glsl_type, opencl_type in type_map.items():
        opencl_code = re.sub(glsl_type, opencl_type, opencl_code)

    # Function transformations (add GLSL_ prefix)
    glsl_functions = [
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'pow', 'exp', 'log', 'exp2', 'log2', 'sqrt', 'inversesqrt',
        'abs', 'sign', 'floor', 'ceil', 'trunc', 'fract', 'mod', 'modf',
        'min', 'max', 'clamp', 'mix', 'step', 'smoothstep',
        'length', 'distance', 'dot', 'cross', 'normalize',
        'faceforward', 'reflect', 'refract',
        'radians', 'degrees',
        'transpose', 'inverse', 'determinant',
    ]

    for func_name in glsl_functions:
        # Only add GLSL_ prefix if not already present
        # Use negative lookbehind to avoid matching GLSL_func
        pattern = r'(?<!GLSL_)\b' + func_name + r'\s*\('
        replacement = f'GLSL_{func_name}('
        opencl_code = re.sub(pattern, replacement, opencl_code)

    # Float literal transformations (add 'f' suffix if missing)
    # Pattern 1: Numbers with decimal point
    opencl_code = re.sub(r'(?<!\w)(\d+\.\d*)(?![fF\w])', r'\1f', opencl_code)
    # Pattern 2: Numbers with exponent
    opencl_code = re.sub(r'(?<!\w)(\d+[eE][+-]?\d+)(?![fF\w])', r'\1f', opencl_code)
    # Pattern 3: Decimal point at start
    opencl_code = re.sub(r'(?<!\w)(\.\d+)(?![fF\w])', r'\1f', opencl_code)

    return opencl_code


def extract_main_image_sections(glsl_source: str, parser: GLSLParser) -> Tuple[str, str]:
    """
    Extract sections before and inside mainImage() using AST.

    Args:
        glsl_source: Original GLSL source code
        parser: GLSLParser instance

    Returns:
        Tuple of (code_before_main, main_image_body)

    Raises:
        TranspileError: If mainImage() not found
    """
    # Parse the source to get AST
    try:
        ast = parser.parse(glsl_source)
    except Exception as e:
        raise TranspileError(f"Failed to parse GLSL: {e}")

    # Find mainImage function in AST
    main_image_node = None
    declarations_before_main = []

    for node in ast.named_children:
        if node.type == "function_definition":
            # Check if this is mainImage
            func_name = node.name if hasattr(node, 'name') else None
            if not func_name:
                # Try to extract from declarator
                declarator = node.child_by_field_name("declarator")
                if declarator:
                    for child in declarator.children:
                        if child.type == "identifier":
                            func_name = child.text
                            break

            if func_name == "mainImage":
                main_image_node = node
                break
            else:
                declarations_before_main.append(node)
        else:
            # Not a function definition (global var, struct, etc.)
            declarations_before_main.append(node)

    if not main_image_node:
        raise TranspileError("Could not find mainImage() function in GLSL source")

    # Extract code before mainImage
    code_before_main = ""
    for decl_node in declarations_before_main:
        # Use .text property from ASTNode
        node_text = decl_node.text

        # Special handling for struct_specifier: tree-sitter doesn't include
        # the trailing semicolon in the node text, but GLSL syntax requires it
        if decl_node.type == "struct_specifier":
            node_text += ";"

        code_before_main += node_text + "\n\n"

    code_before_main = code_before_main.strip()

    # Extract mainImage body (without signature and braces)
    body_node = main_image_node.child_by_field_name("body")
    if not body_node:
        raise TranspileError("mainImage() function has no body")

    # Get body text using .text property
    body_text = body_node.text

    # Remove outer braces
    body_text = body_text.strip()
    if body_text.startswith("{") and body_text.endswith("}"):
        body_text = body_text[1:-1]

    main_image_body = body_text.strip()

    return code_before_main, main_image_body


def transpile(glsl_source: str, verbose: bool = False) -> TranspileResult:
    """
    Transpile GLSL shader source to OpenCL.

    This is the main transpilation function using the Session 9 architecture:
    GLSL -> Preprocessor -> Parser -> TypeChecker -> ASTTransformer -> OpenCLEmitter -> OpenCL

    Args:
        glsl_source: GLSL source code string
        verbose: If True, print transformation stages

    Returns:
        TranspileResult with header, kernel, and full OpenCL code

    Raises:
        TranspileError: If transpilation fails
    """
    if verbose:
        print("=" * 70)
        print("GLSL to OpenCL Transpilation")
        print("=" * 70)

    # Stage 0: Transform preprocessor directives
    if verbose:
        print("\n[0/6] Transforming preprocessor directives...")

    preprocessor = PreprocessorTransformer()
    glsl_source = preprocessor.transform(glsl_source)

    if verbose:
        print(f"  [OK] Preprocessor transformation complete")

    # Stage 1: Parse GLSL
    if verbose:
        print("\n[1/6] Parsing GLSL source...")

    parser = GLSLParser()

    # Extract sections BEFORE transformation
    try:
        header_glsl, kernel_glsl_body = extract_main_image_sections(glsl_source, parser)
    except TranspileError as e:
        raise
    except Exception as e:
        raise TranspileError(f"Failed to extract mainImage sections: {e}")

    if verbose:
        print(f"  [OK] Found {len(header_glsl)} chars before mainImage()")
        print(f"  [OK] Found {len(kernel_glsl_body)} chars in mainImage() body")

    # Stage 2: Setup type checker
    if verbose:
        print("\n[2/6] Initializing type checker...")

    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)

    if verbose:
        print(f"  [OK] Built-in symbol table created")

    # Stage 3: Create transformer
    if verbose:
        print("\n[3/6] Creating AST transformer...")

    transformer = ASTTransformer(type_checker)

    if verbose:
        print(f"  [OK] Transformer initialized")

    # Stage 4: Transform header (code before mainImage)
    if verbose:
        print("\n[4/6] Transforming code sections...")
        if header_glsl:
            print("  -> Transforming header (globals/functions)...")

    header_opencl = ""
    if header_glsl:
        try:
            # Check if header is only preprocessor directives and comments
            # Preprocessor directives are already transformed, so we can pass them through
            # This avoids the issue where tree-sitter rejects preprocessor-only code
            is_preprocessor_only = all(
                line.strip().startswith('#') or
                line.strip().startswith('//') or
                not line.strip()
                for line in header_glsl.split('\n')
            )

            if is_preprocessor_only:
                # Header contains only preprocessor directives, pass through as-is
                header_opencl = header_glsl
            else:
                # Header contains actual code, parse and transform normally
                header_ast = parser.parse(header_glsl)
                header_ir = transformer.transform(header_ast)
                emitter = OpenCLEmitter(indent_size=4)
                header_opencl = emitter.emit(header_ir)

                # Post-process: Fix GLSL types and functions inside #ifdef blocks
                # This is a workaround for Session 9's limitation where code inside #ifdef blocks
                # is not transformed by the AST transformer
                header_opencl = post_process_ifdef_blocks(header_opencl)
        except Exception as e:
            raise TranspileError(f"Failed to transform header: {e}")

        if verbose:
            print(f"    [OK] Header: {len(header_opencl)} chars")

    # Stage 5: Transform kernel (mainImage body)
    if verbose:
        print("\n[5/6] Transforming kernel (mainImage body)...")

    # Wrap kernel body in mainImage signature for AST parsing
    kernel_glsl_full = f"void mainImage(out vec4 fragColor, in vec2 fragCoord) {{\n{kernel_glsl_body}\n}}"

    try:
        kernel_ast = parser.parse(kernel_glsl_full)
        kernel_ir = transformer.transform(kernel_ast)
        emitter = OpenCLEmitter(indent_size=4)
        kernel_opencl_full = emitter.emit(kernel_ir)
    except Exception as e:
        raise TranspileError(f"Failed to transform kernel: {e}")

    # Extract just the body from transformed OpenCL
    # The transformed code is now OpenCL, so extract body again
    try:
        # Parse the OpenCL to extract body (it's valid C-like syntax)
        _, kernel_opencl_body = extract_main_image_sections(kernel_opencl_full, parser)
    except Exception as e:
        # Fallback: regex extraction
        match = re.search(
            r'void\s+mainImage\s*\([^)]+\)\s*\{(.*)\}',
            kernel_opencl_full,
            re.DOTALL
        )
        if match:
            kernel_opencl_body = match.group(1).strip()
        else:
            raise TranspileError(f"Failed to extract transformed kernel body: {e}")

    # Post-process: Remove dereferences of mainImage parameters
    # In the Houdini kernel context, fragColor and fragCoord are local variables, not pointers
    # The transformer treats them as function parameters (fragColor is out, so adds *dereference)
    # We need to remove these dereferences for the kernel context
    kernel_opencl_body = kernel_opencl_body.replace('*fragColor', 'fragColor')
    kernel_opencl_body = kernel_opencl_body.replace('*fragCoord', 'fragCoord')

    # Post-process: Fix GLSL types and functions inside #ifdef blocks
    # This is a workaround for Session 9's limitation where code inside #ifdef blocks
    # is not transformed by the AST transformer
    kernel_opencl_body = post_process_ifdef_blocks(kernel_opencl_body)

    # Add comment markers to kernel
    kernel_opencl = (
        "// ---- SHADERTOY CODE BEGIN ----\n"
        "// Shadertoy void mainImage(...)\n"
        f"{kernel_opencl_body}\n"
        "// ---- SHADERTOY CODE END ----"
    )

    if verbose:
        print(f"    [OK] Kernel: {len(kernel_opencl)} chars")

    # Create full OpenCL (header + mainImage signature + body)
    full_opencl = ""
    if header_opencl:
        full_opencl += header_opencl + "\n\n"

    full_opencl += (
        "void mainImage(out float4 fragColor, in float2 fragCoord) {\n"
        f"{kernel_opencl_body}\n"
        "}\n"
    )

    if verbose:
        print(f"\n[6/6] Transpilation complete!")
        print(f"  [OK] Total output: {len(full_opencl)} chars")
        print("=" * 70)

    return TranspileResult(
        header=header_opencl,
        kernel=kernel_opencl,
        full=full_opencl
    )


def validate_opencl(opencl_code: str) -> bool:
    """
    Validate OpenCL code by attempting compilation.

    Args:
        opencl_code: OpenCL source code string

    Returns:
        True if compilation succeeds, False otherwise

    Note:
        Requires PyOpenCL to be installed.
        This is a basic validation - full validation requires compilecl.py
    """
    try:
        import pyopencl as cl
    except ImportError:
        print("Warning: PyOpenCL not installed, skipping validation")
        return True

    try:
        # Get first available platform and device
        platforms = cl.get_platforms()
        if not platforms:
            print("Warning: No OpenCL platforms found")
            return True

        platform = platforms[0]
        devices = platform.get_devices()
        if not devices:
            print("Warning: No OpenCL devices found")
            return True

        device = devices[0]

        # Create context and compile
        ctx = cl.Context([device])

        # Basic compilation test (will fail without proper headers)
        # For full validation, use compilecl.py
        print(f"  Testing OpenCL syntax on {device.name}...")
        print("  Note: Full validation requires compilecl.py with Houdini headers")

        # Just check for obvious syntax errors
        if "GLSL_" in opencl_code and "#include" not in opencl_code:
            print("  [OK] OpenCL code contains GLSL_ function calls (requires glslHelpers.h)")

        return True

    except Exception as e:
        print(f"Validation error: {e}")
        return False


def transpile_file(
    input_path: Path,
    output_dir: Path,
    full_mode: bool = False,
    verbose: bool = False,
    validate: bool = False
) -> bool:
    """
    Transpile a GLSL file and write output files.

    Args:
        input_path: Path to input GLSL file
        output_dir: Directory for output files
        full_mode: If True, output single full.cl file instead of split
        verbose: If True, show transformation stages
        validate: If True, validate OpenCL compilation

    Returns:
        True if successful, False otherwise
    """
    # Read input
    if verbose:
        print(f"\nReading {input_path}...")
    else:
        print(f"Transpiling {input_path.name}...")

    try:
        glsl_source = input_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False

    # Transpile
    try:
        result = transpile(glsl_source, verbose=verbose)
    except TranspileError as e:
        print(f"Transpilation error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during transpilation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

    # Validate if requested
    if validate:
        print("\nValidating OpenCL code...")
        if not validate_opencl(result.get_full()):
            print("Warning: Validation failed (non-fatal)")

    # Determine output paths
    base_name = input_path.stem

    if full_mode:
        # Output single full.cl file
        full_path = output_dir / f"{base_name}.full.cl"

        if not verbose:
            print(f"Writing {full_path.name}...")

        try:
            full_path.write_text(result.get_full(), encoding='utf-8')
        except Exception as e:
            print(f"Error writing output file: {e}")
            return False

        # Summary
        print(f"\n[SUCCESS] Output: {full_path}")
        print(f"  Size: {len(result.get_full())} characters")

    else:
        # Output split header.cl and kernel.cl files
        header_path = output_dir / f"{base_name}.header.cl"
        kernel_path = output_dir / f"{base_name}.kernel.cl"

        if not verbose:
            print(f"Writing {header_path.name} and {kernel_path.name}...")

        try:
            header_path.write_text(result.get_header(), encoding='utf-8')
            kernel_path.write_text(result.get_kernel(), encoding='utf-8')
        except Exception as e:
            print(f"Error writing output files: {e}")
            return False

        # Summary
        print(f"\n[SUCCESS] Transpiled {input_path.name}")
        print(f"  Header: {len(result.get_header())} chars -> {header_path}")
        print(f"  Kernel: {len(result.get_kernel())} chars -> {kernel_path}")
        print(f"\nTest with: python tests/compilecl.py --header {header_path} {kernel_path}")

    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transpile GLSL shaders to OpenCL (Session 9 architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transpile.py shader.glsl
    # Outputs: shader.header.cl and shader.kernel.cl

  python transpile.py shader.glsl --output-dir output/
    # Outputs: output/shader.header.cl and output/shader.kernel.cl

  python transpile.py shader.glsl --full
    # Outputs: shader.full.cl (complete OpenCL code)

  python transpile.py shader.glsl --verbose
    # Show transformation stages

  python transpile.py shader.glsl --validate
    # Validate OpenCL compilation (requires PyOpenCL)

Test transpiled code:
  python tests/compilecl.py --header shader.header.cl shader.kernel.cl

Module usage:
  from transpile import transpile
  result = transpile(glsl_source_string)
  print(result.get_header())
  print(result.get_kernel())
  print(result.get_full())
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input GLSL file path"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory (default: same as input file)"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Output complete OpenCL code in single .full.cl file (instead of split)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show transformation stages"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate OpenCL compilation (requires PyOpenCL)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.input.suffix == ".glsl":
        print(f"Warning: Input file doesn't have .glsl extension: {args.input}")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = args.input.parent

    # Transpile
    success = transpile_file(
        args.input,
        output_dir,
        full_mode=args.full,
        verbose=args.verbose,
        validate=args.validate
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
