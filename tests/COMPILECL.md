# OpenCL Kernel Compiler - Usage Guide

A flexible OpenCL kernel compilation tool that allows you to compose kernel sources from multiple files with customizable build options.
Default header and kernel is the expanded OpenCL code from hShadertoy HDA OpenCL COP node. This to simulate any implemented functions in Houdini COP environment without running Houdini 

## Features

- **Flexible file composition**: Combine multiple header and kernel files
- **Configurable build options**: Load compiler flags from JSON files
- **Traditional CLI syntax**: Uses `--option` and `-o` style arguments
- **Device selection**: Choose which OpenCL device to compile for
- **Verbose mode**: Optionally display full kernel source before compilation

## Kernel Source Construction Order

The compiler constructs the final kernel source by concatenating in this order:

1. **Main Header** (default: `main_header.cl`)
2. **Test Header** (optional, specified with `--header`)
3. **Main Kernel BEGIN** (default: `main_kernel.cl`)
4. **Test Kernel Code** (optional, positional argument)
5. **Main Kernel END** Kernel end footer appended in compilecl.py construct_kernel_source()

Note: in the context of shadertoy, any functions and macros set in `Common` pass would have to be merged with Test Header code for `--header` argument. Otherwise Test Kernel Code might fail

## Main Header and Kernel

Main header contains nearly 5000 lines of Houdini COP related code.
Appended hShadertoy helper code at the end of the header file:
```c

...
// Shadertoy has global variables that can be called inside functions
// We just initiate empty variables so that code compiles if used inside func()
// They get mapped inside kernel
static float3 iResolution = (float3)(512.0f, 288.0f, 0.0f);
static float iTime = 0.0000f;
static float iTimeDelta = 0.0000f;
static float iFrameRate = 24.0000f;
static int iFrame = 0;
static float4 iMouse = (float4)(0.0000f, 0.0000f, 0.0000f, 0.0000f );
static float4 iDate = (float4)(2025.0000f, 12.0000f, 31.0000f, 60.0000f );
static const float iSampleRate = 44100.0f;

static const IMX_Layer* iChannel0;
static const IMX_Layer* iChannel1;
static const IMX_Layer* iChannel2;
static const IMX_Layer* iChannel3;
static float iChannelTime[4];
static float3 iChannelResolution[4];


#ifdef CUBEMAP_RENDERPASS
    #define DO_CUBEMAP \
        float3 rayDir; \
        shadertoy_cubemap(AT_ix,AT_iy,AT_xres,AT_yres,&rayDir,&iResolution);
#else
    #define DO_CUBEMAP /* nothing */
#endif

#define SHADERTOY_INPUTS \
    iResolution = (float3)(AT_xres, AT_yres, 0.0f); \
    iTime = AT_Time; \
    iFrameRate = AT_iFrameRate; \
    iFrame = AT_iFrame; \
    iMouse = AT_iMouse;\
    iDate = AT_iDate;\
    iChannel0 = AT_iChannel0_layer; \
    iChannel1 = AT_iChannel1_layer; \
    iChannel2 = AT_iChannel2_layer; \
    iChannel3 = AT_iChannel3_layer; \
    iChannelTime[0] = AT_Time; \
    iChannelTime[1] = AT_Time; \
    iChannelTime[2] = AT_Time; \
    iChannelTime[3] = AT_Time; \
    iChannelResolution[0] = (float3)(AT_iChannel0_res, 0.0f); \
    iChannelResolution[1] = (float3)(AT_iChannel1_res, 0.0f); \
    iChannelResolution[2] = (float3)(AT_iChannel2_res, 0.0f); \
    iChannelResolution[3] = (float3)(AT_iChannel3_res, 0.0f); \
    float2 fragCoord = AT_fragCoord; \
    if (!AT_fragCoord_bound) { fragCoord = (float2)(AT_ix, AT_iy); }\
    float4 fragColor = (float4)(0.0f, 0.0f, 0.0f, 1.0f); \
    DO_CUBEMAP
    ...
```

Main kernel code contains generickernel() and gshadertoy-like inputs expanded from header macro

```c
kernel void generickernel(...
    ...
   SHADERTOY_INPUTS
    ...
```
Main kernel closed in compilecl.py with `AT_fragColor_set(fragColor);}`


## Command Line Syntax

```bash
python compilecl.py [OPTIONS] [test_kernel_code.cl]
```

### Options

| Short | Long | Default | Description |
|-------|------|---------|-------------|
| `-o FILE` | `--build-opts FILE` | `build_options.json` | JSON file with build options |
| | `--main-header FILE` | `main_header.cl` | Main header file |
| | `--main-kernel FILE` | `main_kernel.cl` | Main kernel file |
| | `--header FILE` | *(none)* | Optional test header file |
| `-d INDEX` | `--device INDEX` | `0` | OpenCL device index |
| `-v` | `--verbose` | *(off)* | Show full kernel source |
| | `--help` | | Show help message |

### Positional Arguments

- `tests/test_code/test_foobar.cl` - Optional test kernel code to include inside main kernel

## Usage Examples

### Basic Usage (using all defaults)

```bash
python compilecl.py
```

This will compile: `main_header.cl` + `main_kernel.cl` with options from `build_options.json`

### Add a Test Kernel code

`tests/test_code/test_foobar.cl` containing a single line: `float foobar = 1.0f;`

```bash
python compilecl.py tests/test_code/test_foobar.cl
```

This compiles: `main_header.cl` + `main_kernel.cl` + `tests/test_code/test_foobar.cl`

### Add Test Header Code and Kernel Code
For example for defining functions and macros and then using them inside kernel

```bash
python compilecl.py --header test_func/test_func_somefunction_header.cl test_func/test_func_somefunction_kernel.cl
```

This compiles:
- `main_header.cl`
- `test_func/test_func_somefunction_header.cl`
- `main_kernel.cl`
- `test_func/test_func_somefunction_kernel.cl`

### Custom Build Options

```bash
python compilecl.py -o custom_build_options.json test_code/test_foobar.cl
```

or

```bash
python compilecl.py --build-opts custom_build_options.json test_code/test_foobar.cl
```

### Override All Defaults

```bash
python compilecl.py --main-header my_header.cl --main-kernel my_kernel.cl --header test_func/test_func_somefunction_header.cl test_func/test_func_somefunction_kernel.cl
```

### Select Different Device

```bash
python compilecl.py -d 1 --header test_func/test_func_somefunction_header.cl test_func/test_func_somefunction_kernel.cl
```

or

```bash
python compilecl.py --device 1 test_code/test_foobar.cl
```

### Verbose Output

```bash
python compilecl.py -v test_code/test_foobar.cl
```

This will display the complete assembled kernel source before compilation.

### Complex Example

```bash
python compilecl.py \
    -o gpu_options.json \
    --main-header cuda_compat_header.cl \
    --header my_functions.cl \
    -d 2 \
    -v \
    my_test_kernel.cl
```

## Build Options JSON Format

The `build_options.json` file supports multiple formats:

### Format 1: Simple List

```json
[
  "-I /path/to/includes",
  "-D MY_DEFINE",
  "-D MY_VALUE=123",
  "-w"
]
```

### Format 2: Structured Object with Options Array

```json
{
  "options": [
    "-I C:/dev/includes",
    "-D __H_CPU__",
    "-D FILTER_BOX"
  ]
}
```

### Format 3: Nested Key-Value Pairs

```json
{
  "defines": {
    "__H_CPU__": true,
    "__H_INTEL__": true,
    "MY_SIZE": 1024,
    "MY_NAME": "value"
  },
  "-I": [
    "C:/dev/includes",
    "C:/another/path"
  ]
}
```

### Format 4: Mixed Format

```json
{
  "comment": "CPU build configuration",
  "-I": [
    "C:/dev/hShadertoy/houdini/ocl/include",
    "C:/PROGRA~1/SIDEEF~1/HOUDIN~1.440/houdini/ocl/include"
  ],
  "defines": {
    "__H_CPU__": true,
    "__H_INTEL__": true,
    "FILTER_BOX": true,
    "_RUNOVER_LAYER": "_bound_fragColor_layer",
    "_bound_size_ref_channels": 1
  }
}
```

All formats are automatically flattened to a space-separated string for the compiler.

## File Structure Example

### Directory Layout

```
tests
├── compilecl.py
├── build_options.json
├── ocl/main_header.cl
├── ocl/main_kernel.cl
├── test_func/test_func_somefunction_header.cl
└── test_func/test_func_somefunction_kernel.cl
```

### main_header.cl
```c
// Common type definitions and macros
typedef float4 fpreal4;
typedef float fpreal;
// ... more headers
```

### test_func_somefunction_header.cl
```c
float4 somefunction(float foo) {
    return (float4)(foo, foo, 0.0f, 1.0f);
}
```

### main_kernel.cl
```c
kernel void generickernel(
    global float4* output,
    float time
) {
    int gid = get_global_id(0);
    // Main kernel code
}
```

### test_func_somefunction_kernel.cl
```c
float value = 0.5f;
float4 bar = somefunction(value);
output[gid] = bar;
```

## Command Line Tips

### Get Help

```bash
python compilecl.py --help
```

### List Available Devices

Just run the script - it always displays available OpenCL devices:

```bash
python compilecl.py
```

### Check Your Configuration

Run with verbose mode to see exactly what's being compiled:

```bash
python compilecl.py -v --header test.h test.cl
```

### Quick Test Compilation

Test a single kernel file against the defaults:

```bash
python compilecl.py my_test.cl
```

## Advanced Usage

### Create Device-Specific Build Configurations

```bash
# CPU compilation
python compilecl.py -o build_opts_cpu.json -d 0 test.cl

# GPU compilation
python compilecl.py -o build_opts_gpu.json -d 1 test.cl
```

### Automated Testing Script

Create a bash script to test multiple kernels:

```bash
#!/bin/bash
for test_file in tests/*.cl; do
    echo "Testing $test_file..."
    python compilecl.py --header test_header.cl "$test_file"
    if [ $? -eq 0 ]; then
        echo "✓ $test_file passed"
    else
        echo "✗ $test_file failed"
    fi
done
```

### Debug Kernel Assembly

Use verbose mode and redirect to file:

```bash
python compilecl.py -v --header test.h test.cl 2>&1 | tee compilation.log
```

## Integration with Houdini

This tool is designed to work with Houdini's OpenCL workflow. Example build options for Houdini:

### build_options.json for Houdini
```json
{
  "comment": "Houdini OpenCL build options",
  "-I": [
    "$HFS/houdini/ocl/include"
  ],
  "defines": {
    "__H_CPU__": true,
    "__H_INTEL__": true,
    "_RUNOVER_LAYER": "_bound_fragColor_layer"
  }
}
```

Replace `$HFS` with your actual Houdini installation path (e.g., `C:/PROGRA~1/SIDEEF~1/HOUDIN~1.440`).

You can extract full list of your build options by setting `HOUDINI_OCL_REPORT_BUILD_LOGS = 2` in `houdini.env`. hShadertoy HDA will log build options specific to your machine.


## Troubleshooting

### "File not found" warnings

Make sure all referenced files are in the current directory or provide full paths:

```bash
python compilecl.py --main-header /full/path/to/header.cl test.cl
```

### Build errors with includes

Check that include paths in `build_options.json` are correct:

```bash
python compilecl.py -v test.cl  # See what options are being used
```

### Device selection issues

List available devices first:

```bash
python compilecl.py  # Shows device list
python compilecl.py -d 1 test.cl  # Use device 1
```

### Empty build options

If `build_options.json` is missing or empty, create one or specify a different file:

```bash
python compilecl.py -o my_options.json test.cl
```

### Argument parsing confusion

Remember:
- Use `--` for long options: `--header`, `--device`, `--main-kernel`
- Use `-` for short options: `-o`, `-d`, `-v`
- Test kernel file is always positional (no flag needed)

## Comparison: Old vs New Syntax

### Old Syntax (key=value)
```bash
python compilecl.py build_opts=custom.json header=test.h test.cl
```

### New Syntax (--option style)
```bash
python compilecl.py -o custom.json --header test.h test.cl
```

The new syntax is more standard and familiar to Unix/Linux users.

Example output of a successful compilation:
```bash
python tests/compilecl.py --header tests/shaders/simple/sand_target.header.cl tests/shaders/simple/sand_target.kernel.cl
```
Output:
```
======================================================================
OpenCL Kernel Compiler
======================================================================

Configuration:
  Main header    : C:/dev/hShadertoy/tests/ocl/main_header.cl
  Test header    : tests\shaders\simple\sand_target.header.cl
  Main kernel    : C:/dev/hShadertoy/tests/ocl/main_kernel.cl
  Test kernel    : tests\shaders\simple\sand_target.kernel.cl
  Build options  : C:/dev/hShadertoy/tests/build_options.json
  Device index   : 0

Available OpenCL devices:

[Platform 0] NVIDIA CUDA
   [0] NVIDIA GeForce RTX 2070  (ALL | GPU)


Using platform: NVIDIA CUDA
Using device  : NVIDIA GeForce RTX 2070
Build options : -I C:/dev/hShadertoy/houdini/ocl/include -I C:/PROGRA~1/SIDEEF~1/HOUDIN~1.440/houdini/ocl/include -I C:/dev/hShadertoy/tests/ocl -D __H_CPU__ -D __H_INTEL__ -DFILTER_BOX -D_RUNOVER_LAYER=_bound_fragColor_layer -DHAS_size_ref -DALIGNED_size_ref -DCONSTANT_size_ref -D_bound_size_ref_border=IMX_WRAP -D_bound_size_ref_storage=FLOAT32 -D_bound_size_ref_channels=1 -D_bound_fragColor_storage=FLOAT32 -D_bound_fragColor_channels=4    

Kernel source length: 176757 characters
======================================================================
Compiling...
======================================================================

✓ Kernel compiled successfully!

Available kernels (1):
  • generickernel
```

## hShadertoy transpiled GLSL-OpenCL testing workflow

Transpile code using `tests/transpile.py`
```bash
python transpile.py tests/shaders/matrix.glsl
```
This will transpile GLSL shader into two .cl files, pre-`mainImage()` header functions in `matrix.header.cl` and the body of `mainImage()` in: `matrix.kernel.cl` 
(`matrix.header.cl` will be empty in this case since there are is no header code before `mainImage()` in `matrix.glsl`)

Then run the compilecl.py:
```bash
python tests/compilecl.py --header tests/shaders/matrix.header.cl tests/shaders/simple/matrix.kernel.cl
```

Transpiled kernel code should look like this: `tests/shaders/matrix_target.kernel.cl`
```c
// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
// Normalized pixel coordinates (from 0 to 1)
    float2 uv = fragCoord/iResolution.xy;

    // constructed vectors and matrices for testing
    float2 V2 =(float2)(1.0f, 0.0f);
    float3 V3 =(float3)(1.0f, 0.0f, 0.0f);
    float4 V4 =(float4)(1.0f, 0.0f, 0.0f, 0.0f);

    mat2 M2 = (mat2)(1.0f, 0.0f, 0.0f, 1.0f );
	mat3 M3 = (mat3){{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
	mat4 M4 = (mat4)(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    
    // common matrix operations
    float2 op1 = GLSL_mul(V2, M2); // 2D vector transform
    float2 op2 = GLSL_mul(GLSL_mul(V2, M2),  M2); // 2D vector transform x2
    op2 = GLSL_mul(op2, M2); // another common transform operation

    float3 op3 = GLSL_mul(V3, M3);  // 3D vector transform using 3x3 transform
    float3 op4 = GLSL_mul(GLSL_mul(V3, M3), M3); // sequential, eg Model to World to Screen
    op4 = GLSL_mul(op4, M3); // // another common transform operation

    float4 op5 = GLSL_mul(V4, M4);; // 3D point transorm using 4x4 matrix
    float4 op6 = GLSL_mul(GLSL_mul(V4, M4), M4); // sequential, eg Model to World to Screen
    op6 = GLSL_mul(op6, M4);

    // common matrix functions
    mat2 xf1 = GLSL_transpose(M2);
    mat2 xf2 = GLSL_inverse(M2);	
    mat3 xf3; GLSL_transpose(M3, xf3);
    mat3 xf4; GLSL_inverse(M3, xf4);
    mat4 xf5 = GLSL_transpose(M4);
    mat4 xf6 = GLSL_inverse(M4);	


    fragColor = (float4)(uv, 0.0f, 1.0f);
// ---- SHADERTOY CODE END ----
```