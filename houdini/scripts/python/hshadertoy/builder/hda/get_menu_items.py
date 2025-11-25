
## PythonModule in houdini/otls/hShadertoy.hda::shadertoy HDA
## Constructs Menu parameters from MenuItems.json embedded in HDA

import hou
import json

def get_menu_items(node, group=None):
    """
    Read menu items from HDA Extra Files section
    
    Args:
        node: The HDA node instance
        group: Optional group name to filter items (e.g., 'textures', 'misc', 'cubemaps')
               If None, returns all items from all groups
    
    Returns:
        Flattened list of [token, label, token, label, ...]
    """
    hda_def = node.type().definition()
    
    if not hda_def:
        return []
    
    # Read from Extra Files section named 'MenuItems'
    menu_section = hda_def.sections().get('MenuItems')
    
    if not menu_section:
        # Fallback to default if section doesn't exist
        return ["error", "No MenuItems Section"]
    
    try:
        # Parse JSON from section
        menu_data = json.loads(menu_section.contents())
        
        # Handle grouped format: {"textures": [[...], ...], "misc": [[...], ...]}
        if isinstance(menu_data, dict):
            # If a specific group is requested
            if group:
                if group in menu_data:
                    pairs = menu_data[group]
                else:
                    return ["error", f"Group '{group}' not found"]
            else:
                # No group specified - combine all groups
                pairs = []
                for group_items in menu_data.values():
                    if isinstance(group_items, list):
                        pairs.extend(group_items)
        
        # Handle simple list format: [["token", "label"], ...]
        elif isinstance(menu_data, list) and isinstance(menu_data[0], list):
            pairs = menu_data
        
        else:
            return ["error", "Invalid menu format"]
        
        # Flatten pairs into [token, label, token, label, ...]
        return [item for sublist in pairs for item in sublist]
    
    except Exception as e:
        return ["error", f"Error: {str(e)}"]