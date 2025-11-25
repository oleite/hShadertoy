import re

def transpile(glsl, mode="Template"):
    
    # Check if already OpenCL code (contains @KERNEL)
    if '@KERNEL' in glsl:
        return glsl    
    # Replace void main*(...) { with @KERNEL\n{
    glsl_modified = re.sub(
        r'void\s+main\w*[^{]*\{',
        '@KERNEL\n{',
        glsl,
        flags=re.DOTALL
    )
    
    # Check if any replacement was made
    has_kernel = glsl_modified != glsl
    glsl = glsl_modified
    
    lines = glsl.splitlines(True)
    
    # Find indices of special lines
    kernel_idx = None
    first_brace_idx = None
    last_brace_idx = None
    
    if has_kernel:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('@KERNEL'):
                kernel_idx = i
            elif kernel_idx is not None and first_brace_idx is None and stripped == '{':
                first_brace_idx = i
            if stripped == '}':
                last_brace_idx = i  # Keep updating to get the last one
    
    # Process lines
    result = []
    for i, line in enumerate(lines):
        if i == kernel_idx:
            result.append('@KERNEL\n')
        elif has_kernel and (i == first_brace_idx or i == last_brace_idx):
            result.append('{\n' if i == first_brace_idx else '}\n')
        else:
            result.append(f"// {line}")
    return ''.join(result)