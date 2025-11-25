# Transpiler Module Examples

This directory contains example scripts demonstrating how to use the GLSL to OpenCL transpiler as a Python module.

## Quick Start

Run all examples:
```bash
python examples/transpiler_module_example.py
```

## Module Import

```python
from transpile import transpile

# Transpile GLSL string to OpenCL
result = transpile(glsl_source_string)

# Get different output formats
header = result.get_header()  # Code before mainImage() (globals, functions)
kernel = result.get_kernel()  # mainImage() body only (for compilecl.py)
full = result.get_full()      # Complete OpenCL code
```

## Examples Included

The `transpiler_module_example.py` script demonstrates:

### 1. **Basic Usage**
- Transpile a GLSL string to OpenCL
- Access header, kernel, and full output modes

### 2. **Verbose Mode**
- See the 5-stage transformation pipeline
- Useful for debugging

### 3. **Verify Transformations**
- Check what GLSL features were transformed
- Verify float suffixes, vector types, function prefixes, etc.

### 4. **File-based Workflow**
- Read GLSL from files
- Write OpenCL output to files
- Production-ready pattern

### 5. **Error Handling**
- Handle missing mainImage() function
- Handle syntax errors
- Production error handling pattern

### 6. **compilecl.py Integration**
- Automated compilation testing workflow
- Example code for CI/CD pipelines

## Common Patterns

### Simple Transpilation
```python
from transpile import transpile

glsl_code = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = vec4(uv, 0.5, 1.0);
}
"""

result = transpile(glsl_code)
print(result.get_kernel())
```

### Verbose Mode (See Transformation Stages)
```python
result = transpile(glsl_code, verbose=True)
# Prints:
# [1/5] Parsing GLSL source...
# [2/5] Initializing type checker...
# [3/5] Creating AST transformer...
# [4/5] Transforming code sections...
# [5/5] Transpilation complete!
```

### Error Handling
```python
from transpile import transpile, TranspileError

try:
    result = transpile(glsl_code)
    print("Success:", result.get_kernel())
except TranspileError as e:
    print("Transpilation error:", e)
```

### File I/O Workflow
```python
from pathlib import Path
from transpile import transpile

# Read GLSL file
glsl_source = Path("shader.glsl").read_text()

# Transpile
result = transpile(glsl_source)

# Write output files
Path("shader.header.cl").write_text(result.get_header())
Path("shader.kernel.cl").write_text(result.get_kernel())
Path("shader.full.cl").write_text(result.get_full())
```

## Output Formats

### Header (.header.cl)
Code before `mainImage()`:
- Global variables
- Helper functions
- Struct definitions
- Constants

Use case: Include in compilecl.py `--header` argument

### Kernel (.kernel.cl)
Only the `mainImage()` body with comment markers:
```c
// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
<transformed code here>
// ---- SHADERTOY CODE END ----
```

Use case: Include in compilecl.py positional argument

### Full (.full.cl)
Complete OpenCL code:
```c
<header code>

void mainImage(out float4 fragColor, in float2 fragCoord) {
    <kernel body>
}
```

Use case: Standalone OpenCL file

## Transformations Applied

The transpiler automatically handles:

| GLSL | OpenCL | Example |
|------|--------|---------|
| Float literals | Add `f` suffix | `1.0` → `1.0f` |
| Vector types | GLSL → OpenCL names | `vec2` → `float2` |
| Type constructors | Cast syntax | `vec4(...)` → `(float4)(...)` |
| Built-in functions | Add `GLSL_` prefix | `sin()` → `GLSL_sin()` |
| Matrix operations | Function calls | `M * v` → `GLSL_mul(M, v)` |

See [GLSL_TO_OPENCL_SPEC.md](../docs/transpiler/GLSL_TO_OPENCL_SPEC.md) for complete transformation rules.

## Testing Compiled Output

After transpiling, test with `compilecl.py`:

```bash
# Transpile
python -c "
from transpile import transpile
from pathlib import Path

result = transpile(Path('shader.glsl').read_text())
Path('shader.header.cl').write_text(result.get_header())
Path('shader.kernel.cl').write_text(result.get_kernel())
"

# Compile
python tests/compilecl.py --header shader.header.cl shader.kernel.cl
```

## Architecture

The transpiler uses a 5-stage pipeline:

```
GLSL Source
    ↓
[1] Parser (tree-sitter) → AST
    ↓
[2] TypeChecker → Symbol table
    ↓
[3] ASTTransformer → Transformed IR
    ↓
[4] OpenCLEmitter → OpenCL code
    ↓
[5] Result (header, kernel, full)
```

## Related Files

- `../tests/transpile.py` - Main transpiler module
- `../docs/transpiler/GLSL_TO_OPENCL_SPEC.md` - Transformation specification
- `../tests/compilecl.py` - OpenCL compilation tester
- `../tests/README.md` - compilecl.py usage guide

## Questions?

See the comprehensive examples in `transpiler_module_example.py` or check the transformation spec document.
