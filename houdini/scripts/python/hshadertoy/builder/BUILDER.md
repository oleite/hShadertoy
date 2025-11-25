# hShadertoy Builder

Translates Shadertoy API JSON to Houdini HDA parameters. Because life is too short to manually set 200 parameters.

## Overview

The Builder takes Shadertoy API JSON and creates a fully configured `hShadertoy::shadertoy` HDA node in Houdini. It handles:

- All 6 renderpasses (Image, Buffer A-D, Cube A, Common)
- All 4 input channels per renderpass (iChannel0-3)
- Media asset mappings (textures, cubemaps, videos, etc.)
- GLSL to OpenCL transpilation via transpiler module
- Parameter name construction following HDA conventions

## Usage

### Basic Usage

```python
from hshadertoy.builder import build_shadertoy_hda

# Load Shadertoy API JSON
with open("gameOfLife_API.json") as f:
    shader_json = f.read()

# Build HDA node
# This creates a new COP network under /obj named after the shader
# Then creates the shadertoy HDA inside it
node = build_shadertoy_hda(shader_json, mode="Template")
print(f"Created node at: {node.path()}")
# Example output: /obj/Game_Of_Life/shadertoy1
```

### Advanced Usage

```python
from hshadertoy.builder import build_shadertoy_hda
import json
import hou

# Load from dict instead of string
with open("shader.json") as f:
    shader_data = json.load(f)

# Option 1: Let builder create its own copnet (recommended)
node = build_shadertoy_hda(
    shader_json=shader_data,
    mode="Transpiler",  # Use actual transpiler instead of template
    node_name="my_shader"
)

# Option 2: Build into existing COP network
existing_copnet = hou.node("/obj/img1")  # Must be a COP2 network!
node = build_shadertoy_hda(
    shader_json=shader_data,
    mode="Transpiler",
    parent_node=existing_copnet,  # Only works if this is COP2 context
    node_name="my_shader"
)

# Note: If parent_node is not a COP network, builder will create a new one anyway
```

### With Custom Transpiler

```python
def my_transpiler(glsl_code, mode="Template"):
    # Your custom transpilation logic
    return opencl_code

node = build_shadertoy_hda(
    shader_json=data,
    transpiler_func=my_transpiler,
    mode="Custom"
)
```

## Architecture

### Parameter Naming Convention

HDA parameters follow a strict naming pattern:

```
{parmname}_rp{renderpass}_ch{channel}
```

Examples:
- `code_rp0` - Code for Image (renderpass 0)
- `code_rp1` - Code for Buffer A (renderpass 1)
- `enable_rp2` - Enable Buffer B (renderpass 2)
- `folder_rp1_ch0` - Folder for Buffer A, channel 0
- `asset1_rp0_ch2` - Asset in folder 1, Image renderpass, channel 2

### Renderpass Token Mapping

```python
{
    "Image": 0,
    "Buffer A": 1,
    "Buffer B": 2,
    "Buffer C": 3,
    "Buffer D": 4,
    "Cube A": 5,
    "Sound": 6,
    "Common": "common"
}
```

### Asset Mapping

Media assets are mapped via `hda/assets.json`:

```json
{
  "id": 8,
  "src": "/media/a/52d2a8f...517e5.jpg",
  "ctype": "texture",
  "file": "abstract_1.jpg",
  "hda": {
    "folder": {"token": 1, "label": "Textures"},
    "asset": {"token": 1, "label": "Abstract 1"}
  }
}
```

The builder uses:
- `id` to match Shadertoy API inputs
- `hda.folder.token` for the folder parameter value
- `hda.asset.token` for the asset parameter value

## Testing

### Headless Testing (hython)

```bash
# Windows path example
hython builder_test_headless.py "resources/examples/gameOfLife/gameOfLife_API.json"

# Linux/WSL path example
hython builder_test_headless.py "resources/examples/gameOfLife/gameOfLife_API.json"

# With custom mode
hython builder_test_headless.py "path/to/shader.json" "Transpiler"
```

### Interactive Testing

```python
# In Houdini Python shell
from hshadertoy.builder import build_shadertoy_hda
import json

with open("/path/to/shader.json") as f:
    data = json.load(f)

node = build_shadertoy_hda(data, mode="Template")
```

## API Reference

### `build_shadertoy_hda()`

Main entry point for building HDA nodes.

**Parameters:**
- `shader_json` (str | dict): Shadertoy API JSON as string or dict
- `transpiler_func` (callable, optional): Custom transpiler function. Default uses placeholder.
- `mode` (str): Transpiler mode ("Template", "Transpiler", etc.). Default "Template".
- `parent_node` (hou.Node, optional): Parent COP network. If None, creates new copnet under /obj.
- `node_name` (str, optional): Name for HDA node. Default "shadertoy1".

**Returns:**
- `hou.Node`: The created hShadertoy::shadertoy HDA node

**Raises:**
- `ValueError`: If renderpass name is unknown
- `json.JSONDecodeError`: If shader_json string is invalid JSON
- `FileNotFoundError`: If assets.json mapping file is missing

### `builder()` (deprecated)

Legacy function for backward compatibility. Use `build_shadertoy_hda()` for new code.

## Files

- `builder.py` - Main builder implementation
- `hda/assets.json` - Media asset to HDA parameter mapping
- `hda/get_menu_items.py` - Menu script used by HDA
- `hda/shadertoy_hda_params.json` - HDA parameter definitions
- `README.md` - This file

## Dependencies

- `hou` - Houdini Object Model (Python API)
- `json` - Standard library
- `hshadertoy.transpiler` - For GLSL to OpenCL conversion

## Notes

- **COP2 Context Required**: The hShadertoy HDA only works in COP2 (Compositing) context
  - Builder automatically creates a COP network under `/obj` if needed
  - Even if you pass a non-COP parent, builder will create a copnet anyway
  - This follows Houdini convention where examples load into dedicated networks
- The Image renderpass is always enabled (no `enable_rp0` parameter)
- Folder parameter defaults to 1, so may not appear in minimal templates
- Missing asset IDs in assets.json will print warnings but won't fail
- The builder validates HDA parameters exist before setting them
- Node positioning uses `moveToGoodPosition()` for clean layouts

## Troubleshooting

**"Parameter 'xyz' not found on HDA"**
- Your HDA definition is outdated. Rebuild from shadertoy_hda_params.json

**"Asset ID X not found in assets.json"**
- Add the missing asset to hda/assets.json with proper folder/asset tokens

**"Unknown renderpass 'XYZ'"**
- Only valid names: Image, Buffer A-D, Cube A, Sound, Common

**Builder creates node but code is empty**
- Check transpiler is working. Placeholder only comments out code.
