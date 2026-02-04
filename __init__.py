"""
Main entry point for JSON Manager GUI.
"""

import os
import sys
import comfy.options

comfy.options.enable_args_parsing()
# CRITICAL: Configure sys.path BEFORE any other imports.
# This ensures that 'utils' resolves to ComfyUI/utils/ (the package)
# and not to comfy/utils.py or json_gui/utils.py.
# This is especially important for child processes spawned with 'spawn' mode.
_current_file = os.path.abspath(__file__)
_json_gui_dir = os.path.dirname(_current_file)
_comfyui_root = os.path.dirname(_json_gui_dir)

# Ensure ComfyUI root is FIRST in sys.path
if _comfyui_root not in sys.path:
    sys.path.insert(0, _comfyui_root)
elif sys.path[0] != _comfyui_root:
    sys.path.remove(_comfyui_root)
    sys.path.insert(0, _comfyui_root)

# Change CWD to ComfyUI root if not already there
if os.getcwd() != _comfyui_root:
    os.chdir(_comfyui_root)

# Cleanup temporary variables
del _current_file, _json_gui_dir, _comfyui_root

from json_gui.json_manager import json_manager_starter  # noqa: F401, E402 pylint: disable=C0413

json_manager_starter.apply_custom_paths()
