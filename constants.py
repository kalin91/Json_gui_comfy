"""Constants for JSON GUI module."""

from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES, SCHEDULER_HANDLERS

_COMBO_CONSTANTS = None


# Lazy-loaded combo constants
# pylint: disable=C0415,W0603
def get_combo_constants() -> dict:
    """
    Creates and returns combo constants, such as sampler names and scheduler names.
    Lazy-load combo constants to avoid triggering Impact Pack init at import time due to its heavy initization.
    It also loads json_gui.server which is required to initialize Impact Server Modules.

    Returns:
        dict: A dictionary containing combo constants.
    """
    global _COMBO_CONSTANTS
    if _COMBO_CONSTANTS is None:
        import json_gui.server as _  # noqa: F401, E402 pylint: disable=C0413
        from custom_nodes.ComfyUI_Impact_Pack.modules.impact.core import ADDITIONAL_SCHEDULERS
        from custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack import FaceDetailer

        _COMBO_CONSTANTS = {
            "sampler_names": SAMPLER_NAMES,
            "scheduler_names": SCHEDULER_NAMES,
            "scheduler_handlers": list(SCHEDULER_HANDLERS) + ADDITIONAL_SCHEDULERS,
            "sam_detection_hint": FaceDetailer.INPUT_TYPES()["required"]["sam_detection_hint"][0],
        }
    return _COMBO_CONSTANTS


JSON_CANVAS_NAME = "json_canvas"
JSON_SCROLL_FRAME_NAME = "json_scrollable_frame"
