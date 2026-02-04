"""Tests for json_gui/constants.py module."""

import pytest
from unittest.mock import patch, MagicMock

# Import constants (already mocked in conftest)
import json_gui.constants as constants


def test_json_canvas_name():
    """Test JSON_CANVAS_NAME constant."""
    assert isinstance(constants.JSON_CANVAS_NAME, str)
    assert constants.JSON_CANVAS_NAME == "json_canvas"


def test_json_scroll_frame_name():
    """Test JSON_SCROLL_FRAME_NAME constant."""
    assert isinstance(constants.JSON_SCROLL_FRAME_NAME, str)
    assert constants.JSON_SCROLL_FRAME_NAME == "json_scrollable_frame"


def test_combo_constants_structure():
    """Test that COMBO_CONSTANTS has the expected structure."""
    assert isinstance(constants.COMBO_CONSTANTS, dict)
    assert "sampler_names" in constants.COMBO_CONSTANTS
    assert "scheduler_names" in constants.COMBO_CONSTANTS
    assert "scheduler_handlers" in constants.COMBO_CONSTANTS
    assert "sam_detection_hint" in constants.COMBO_CONSTANTS


def test_combo_constants_types():
    """Test that COMBO_CONSTANTS values have correct types."""
    assert isinstance(constants.COMBO_CONSTANTS["sampler_names"], list)
    assert isinstance(constants.COMBO_CONSTANTS["scheduler_names"], list)
    assert isinstance(constants.COMBO_CONSTANTS["scheduler_handlers"], list)
    assert isinstance(constants.COMBO_CONSTANTS["sam_detection_hint"], (list, tuple))

