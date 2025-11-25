"""
Builder test for headless Houdini (hython).

Usage:
    hython builder_test_headless.py <path_to_shadertoy_api_json>

Example:
    hython builder_test_headless.py "C:/dev/hShadertoy/resources/examples/gameOfLife/gameOfLife_API.json"

This script is for the brave souls testing in Windows.
May your paths be absolute and your backslashes properly escaped.
"""

import hou
import json
import sys
from pathlib import Path

# Add hshadertoy paths to sys.path if not already present
# This is needed because package PYTHONPATH not automatically applied in headless mode
hshadertoy_root = hou.getenv('HSHADERTOY_ROOT')
if hshadertoy_root:
    hshadertoy_python = str(Path(hshadertoy_root) / 'houdini' / 'scripts' / 'python')
    if hshadertoy_python not in sys.path:
        sys.path.insert(0, hshadertoy_python)
    if hshadertoy_root not in sys.path:
        sys.path.insert(0, hshadertoy_root)

from hshadertoy.builder import build_shadertoy_hda


def main():
    """
    Main test function.

    Loads a Shadertoy API JSON file and builds an HDA node.
    If it doesn't crash, we call it a success.
    """
    # Load HDA file explicitly (needed in headless mode)
    hshadertoy_root = hou.getenv('HSHADERTOY_ROOT')
    if hshadertoy_root:
        hda_path = str(Path(hshadertoy_root) / 'houdini' / 'otls' / 'hShadertoy.hda')
        if Path(hda_path).exists():
            hou.hda.installFile(hda_path)
            print(f"Loaded HDA: {hda_path}\n")
        else:
            print(f"Warning: HDA not found at {hda_path}\n")

    # Validate command line arguments
    # Because users can't be trusted to RTFM
    if len(sys.argv) < 2:
        print("Error: No JSON file path provided")
        print("Usage: hython builder_test_headless.py <path_to_json_file>")
        print("\nExample:")
        print('  hython builder_test_headless.py "resources/examples/gameOfLife/gameOfLife_API.json"')
        sys.exit(1)

    json_file_path = sys.argv[1]

    # Load JSON file
    # This is where we find out if the path is valid or just wishful thinking
    try:
        with open(json_file_path, 'r') as f:
            shader_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}")
        print("Did you check if the file actually exists? Just asking.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        print("This file is not valid JSON. Shadertoy would be disappointed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Get transpiler mode from command line (optional)
    # Default to "Template" because it's the safest option
    mode = sys.argv[2] if len(sys.argv) > 2 else "Template"

    print(f"\n{'='*60}")
    print(f"Building HDA from: {json_file_path}")
    print(f"Transpiler mode: {mode}")
    print(f"{'='*60}\n")

    # Build the HDA node
    # This is where the magic happens. Or where everything crashes.
    try:
        node = build_shadertoy_hda(
            shader_json,
            mode=mode
        )

        print(f"\n{'='*60}")
        print(f"SUCCESS!")
        print(f"Node created at: {node.path()}")
        print(f"{'='*60}\n")

        # Print some stats because developers love stats
        shader_name = shader_json["Shader"]["info"]["name"]
        num_renderpasses = len(shader_json["Shader"]["renderpass"])
        print(f"Shader: {shader_name}")
        print(f"Renderpasses: {num_renderpasses}")

        # Count total inputs across all renderpasses
        total_inputs = sum(
            len(rp.get("inputs", []))
            for rp in shader_json["Shader"]["renderpass"]
        )
        print(f"Total input channels: {total_inputs}")

        return node

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FAILURE!")
        print(f"Error building HDA: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
