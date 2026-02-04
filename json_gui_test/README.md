# JSON GUI Test Suite

## Overview
This directory contains comprehensive unit tests for the `json_gui` module using pytest.

## Test Structure
```
json_gui_test/
├── __init__.py
├── conftest.py              # Shared fixtures and mock setup
├── test_utils.py            # Tests for json_gui/utils.py
├── test_constants.py        # Tests for json_gui/constants.py
├── test_mimic_classes.py    # Tests for json_gui/mimic_classes.py
├── json_manager/
│   └── __init__.py
└── scripts/
    └── __init__.py
```

## Running Tests

### Run all tests
```bash
pytest json_gui_test/ -v
```

### Run with coverage
```bash
pytest json_gui_test/ --cov=json_gui --cov-report=term-missing
```

### Generate HTML coverage report
```bash
pytest json_gui_test/ --cov=json_gui --cov-report=html:coverage_html
```

### Run specific test file
```bash
pytest json_gui_test/test_utils.py -v
```

### Run specific test class or method
```bash
pytest json_gui_test/test_utils.py::TestEndOfFlowException -v
pytest json_gui_test/test_utils.py::TestEndOfFlowException::test_exception_stores_steps -v
```

## Coverage Status

### Core Modules (Tested)
- **json_gui/constants.py**: 100% coverage (6/6 statements)
- **json_gui/mimic_classes.py**: 94% coverage (350/371 statements)
- **json_gui/utils.py**: 89% coverage (105/118 statements)
- **json_gui/__init__.py**: 82% coverage (31/38 statements)

### GUI Modules (Not Yet Tested)
- json_gui/image_viewer.py: 0% coverage
- json_gui/json_manager_gui.py: 0% coverage
- json_gui/json_tree_editor.py: 0% coverage
- json_gui/loading_modal.py: 0% coverage
- json_gui/scroll_utils.py: 0% coverage

### Overall
- **Total**: 25% coverage (492/1944 statements)
- **Tests**: 52 tests, all passing

## Test Approach

### Fixtures
The `conftest.py` file provides shared fixtures for:
- Mocking ComfyUI dependencies (comfy modules, custom_nodes)
- Mocking tkinter (GUI components)
- Mocking file operations (folder_paths)
- Creating temporary directories for testing
- Generating sample tensors and images

### Mocking Strategy
All heavy dependencies are mocked in `conftest.py` before any imports:
- **tkinter** - GUI framework (not available in headless environments)
- **comfy modules** - ComfyUI core modules
- **custom_nodes** - External node packages
- **PIL** - Installed but mocked for some operations
- **torch** - Installed for real tensor operations

### Test Categories

#### test_utils.py (22 tests)
- Path and directory management functions
- `EndOfFlowException` behavior
- `AbsFlow` abstract base class
- Image saving and flow execution
- File cleanup and collision handling

#### test_constants.py (4 tests)
- Constant value verification
- COMBO_CONSTANTS structure
- Type checking for constant collections

#### test_mimic_classes.py (26 tests)
- SimpleKSampler initialization and processing
- EmptyLatent tensor generation
- ControlNet preprocessors (OpenPose, Canny)
- FaceDetailer configuration
- Image rotation
- Skip layer guidance
- ApplyControlNet operations

## Adding New Tests

1. Create test file in appropriate directory
2. Import the module under test
3. Use fixtures from conftest.py for setup
4. Follow naming convention: `test_<module_name>.py`
5. Group related tests in classes: `class TestClassName:`
6. Name test methods: `def test_<what_it_tests>:`

## Dependencies

Tests require:
- pytest
- pytest-cov
- pytest-mock
- torch (CPU version)
- Pillow
- numpy

Install with:
```bash
pip install pytest pytest-cov pytest-mock torch pillow numpy
```
