# ComfyUI JSON GUI Developer Instructions

This project is a fork of ComfyUI that adds a "headless" execution layer (`json_gui`) triggered by a Tkinter interface. The core goal is to execute ComfyUI workflows programmatically via Python scripts ("Flows") rather than the standard node graph.

## Installation & Environment

This repository is designed to be treated as a **module** within a standard ComfyUI installation.

*   **Repository Location**: Typically cloned outside the ComfyUI folder (e.g., `~/repos/Json_gui_comfy`).
*   **Symlink**: A symbolic link named `json_gui` must exist in the ComfyUI root directory, pointing to this repository.
    *   `ComfyUI/json_gui` -> `.../Json_gui_comfy`
*   **Execution**: Scripts are executed from the **ComfyUI root** directory.
    *   Example: `python json_gui/main_json_gui.py`

## Architecture Overview

*   **Entry Point**: `json_gui/main_json_gui.py`. Bootstraps the Tkinter app.
*   **GUI Layer**: Located in `json_gui/json_manager/`.
    *   `json_manager_gui.py`: Main controller.
    *   `json_tree_editor.py`: Dynamically renders widgets based on flow schemas (`body.yml`).
*   **Logic Layer ("Flows")**: Located in `json_gui/scripts/`.
    *   Each folder (e.g., `json_gui/scripts/controlnet_openpose/`) is a "Flow".
    *   **Schema (`body.yml`)**: Defines the user inputs (text, int, file inputs).
    *   **Implementation (`flow.py`)**: Contains the Python logic inheriting from `AbsFlow`.
*   **Execution Layer ("Mimic")**:
    *   The app **bypasses** the standard ComfyUI API/WebSocket server.
    *   It uses `json_gui/scripts/mimic.py`, `mimic_classes.py`, and `node_executor.py` to instantiate and run ComfyUI nodes directly.

---

## Single-Process vs. Multi-Process Architecture

This is the **most critical architectural decision** in the system. Understanding it is essential for debugging and extending the codebase.

### Why Multi-Processing?

GPU memory (VRAM) on consumer hardware is limited. When executing complex pipelines (model loading, sampling, VAE decoding), memory can become fragmented or leak. The multi-process architecture provides:

1.  **Process Isolation**: Each heavy computation (e.g., KSampler) runs in a *child process*. When the child terminates, all its GPU memory is *guaranteed* to be released by the OS.
2.  **Fault Tolerance**: If a node crashes, the main GUI process remains alive.
3.  **Clean State**: Each child process starts with a fresh Python interpreter and CUDA context.

### How It Works

The system is controlled by a single boolean flag: `MimicNode._do_multiprocess`.

| Mode | `_do_multiprocess` | Execution | Use Case |
|------|---|---|---|
| **Single-Process** | `False` | `node.process()` is called directly in the main thread. | Debugging, simple flows, or when GPU memory is not a concern. |
| **Multi-Process** | `True` | `NodeExecutor` spawns a child process via `torch.multiprocessing`. | Production runs, long pipelines, preventing VRAM fragmentation. |

### Switching Modes

The mode is set during `MimicNode` factory initialization in `AbsFlow`:

```python
# In json_gui/utils.py (AbsFlow subclass)
MimicNode.set_node_executor_factory(
    NodeExecutor,
    save_data,
    save_call,
    copy_call,
    do_multiprocess=True  # <-- Set to False for single-process debugging
)
```

### Key Constraint: Pickle Serialization

In multi-process mode, **all arguments and return values must be pickle-serializable**. This includes:
*   `torch.Tensor` (must be moved to CPU first).
*   Primitive types (`int`, `str`, `dict`, `list`).
*   `MimicNode` classes (the class itself, not instances with callbacks).

**Cannot be pickled:**
*   Lambda functions.
*   Closures (nested functions).
*   Objects with open file handles or CUDA streams.

The `DataWrapper` class and `_move_tensors_to_device` helper handle tensor serialization automatically.

---

## The `MimicNode` Base Class (`json_gui/scripts/mimic.py`)

`MimicNode` is the **abstract base class** for all node wrappers. It provides:

### Core Responsibilities

1.  **Lifecycle Management**: `update()` -> `process()` pattern.
2.  **Caching**: Avoids redundant computation if inputs haven't changed.
3.  **Multi-Process Dispatch**: Transparently routes execution via `NodeExecutor`.
4.  **Tensor Saving**: Manages saving intermediate/final images to disk.

### Key Properties and Methods

| Member | Type | Description |
|--------|------|-------------|
| `_do_multiprocess` | `ClassVar[bool]` | Global flag for execution mode. |
| `_node_executor_factory` | `ClassVar[NodeExecutorFactory]` | Factory to create `NodeExecutor` instances. |
| `_current_model` | `ClassVar[Optional[MimicNode]]` | Holds the currently loaded model (shared across nodes in multi-process). |
| `init_args` | `CreationDict` | Arguments used to `__init__` the node (for reconstruction in child). |
| `exec_args` | `CreationDict` | Arguments passed to `process()`. |
| `save_tensor` | `SaveTensorCallable` | Callback to save a tensor to disk. Set by `NodeExecutor`. |

### Lifecycle Methods

```python
class MimicNode(ABC, Generic[T]):
    @final
    def update(self, *args, **kwargs) -> None:
        """
        Called to configure the node's internal state.
        Stores init_args for later reconstruction in child process.
        Delegates to abstract _update_impl().
        """

    @final
    def process(self, *args, **kwargs) -> T:
        """
        Called to execute the node's logic.
        1. Checks cache; if valid, returns cached output.
        2. Unwraps DataWrapper arguments.
        3. Calls abstract _process_impl().
        4. Saves unsaved tensors.
        5. Caches result.
        """

    @abstractmethod
    def _update_impl(self, *args, **kwargs) -> None:
        """Subclass implements to store parameters."""

    @abstractmethod
    def _process_impl(self, *args, **kwargs) -> T:
        """Subclass implements the actual computation."""
```

### The `exec_node` Dispatcher

This is the **routing function** that decides between single-process and multi-process:

```python
@final
def exec_node(
    self,
    pre_node_process_args: dict[str, Any],
    node_params: list["MimicNode"],
) -> T:
    """
    Entry point for node execution, called by Flow scripts.
    
    If _do_multiprocess is True:
        -> Calls _exec_node_spawn() to run in a child process.
    Else:
        -> Calls _exec_node_sync() to run in the current process.
    """
```

### ClassParam: Dependency Injection

`MimicNode.ClassParam` allows a node to declare dependencies on other nodes. When `process()` is called, these dependencies are automatically resolved and injected.

```python
class SimpleKSampler(KSamplerLike[...]):
    @classmethod
    def _class_param_definitions(cls):
        return [
            cls.build_class_param(
                SkipLayers,  # Dependency type
                lambda inst: {"node_model": inst}  # How to inject
            )
        ]
```

This creates a decorator around `process()` that:
1.  Pops the `skip_layers_model` key from kwargs.
2.  Calls the `processor` lambda to get additional kwargs.
3.  Merges them into the function call.

---

## The `NodeExecutor` Class (`json_gui/scripts/node_executor.py`)

`NodeExecutor` is the **multi-process execution engine**. It:

1.  Spawns a child process.
2.  Serializes node class + arguments.
3.  Reconstructs the node in the child.
4.  Executes `node.process()`.
5.  Serializes and returns results to the parent.

### Construction

```python
def __init__(
    self,
    node: MimicNode,           # The node to execute
    pre_node_process_args: dict[str, Any],  # Arguments for process()
    pre_raw_nodes: dict[type[MimicNode], CreationDict],  # Dependency nodes
    save_data: SavedImagesDict,  # Mutable dict to track saved images
):
```

### The `execute()` Method

```python
def execute(
    self,
    save_call: SaveImageCallable,
    timeout: Optional[float] = None,
    poll_interval: float = 0.05,
) -> tuple[Any, SavedImagesDict]:
    """
    1. Prepares arguments for pickle (moves tensors to CPU).
    2. Creates a multiprocessing.Process targeting _node_executor_target.
    3. Starts the child process.
    4. Polls result_queue for output.
    5. Terminates child and returns result.
    
    Raises:
        TimeoutError: If timeout is exceeded.
        RuntimeError: If child dies without returning.
        Exception: Re-raises any exception from the child.
    """
```

### The Child Process Target

```python
@staticmethod
def _node_executor_target(
    node_cls: type[MimicNode],
    node_init_args: CreationDict,
    node_exec_args: dict[str, Any],
    raw_nodes_serialized: dict[str, tuple[type[MimicNode], CreationDict]],
    save_call: SaveImageCallable,
    save_data: SavedImagesDict,
    result_queue: mlp.Queue,
) -> None:
    """
    Runs in the child process:
    1. Enables multiprocess mode (MimicNode.enable_multiprocess()).
    2. Moves tensors to GPU device.
    3. Reconstructs the MimicNode from (class, init_args).
    4. Reconstructs raw_nodes dependencies.
    5. Calls node.process(**exec_args).
    6. Moves result tensors back to CPU.
    7. Puts result in queue.
    8. Waits for signal to terminate (signal.pause()).
    """
```

### Tensor Movement

Before sending tensors across process boundaries:

```python
def _move_tensors_to_device(obj: T, torch_device: device, memo: dict = None) -> T:
    """
    Recursively traverses obj (dict, list, tuple, custom objects).
    For each torch.Tensor:
        - Detaches from computation graph.
        - Clones to independent memory.
        - Moves to target device (CPU for serialization, GPU for execution).
    """
```

---

## Key Patterns & Conventions

### 1. Creating New Flows
*   **Structure**: Create a new directory `json_gui/scripts/<flow_name>/`.
*   **Schema**: `body.yml` is required for the GUI to render.
*   **Class**: Must implement `class Flow(AbsFlow)` in `flow.py`.
    *   **Method**: `def execution(self, inputs: dict) -> list[str]:`
    *   **Return**: List of paths to generated images.

### 2. Calling ComfyUI Nodes ("Mimic" System)
*   **Do NOT** use `nodes.py` directly if a wrapper exists in `mimic_classes.py`.
*   **Pattern**: Instantiate the wrapper, call `.update()` if needed, then call `.process()` or `.exec_node()`.
    *   *Example*: `SimpleKSampler(...).exec_node(args, deps)`
*   **Direct Execution**: If no wrapper exists, use `node_executor.py` to invoke ComfyUI nodes dynamically.

### 3. Data Flow
1.  **Definition**: `body.yml` defines keys (e.g., `positive_prompt`).
2.  **Input**: User fills GUI -> saves to JSON.
3.  **Runtime**: `inputs` dict passed to `flow.execution()` contains values keyed by the YAML definitions.

### 4. EndOfFlowException

This is a **control flow exception**, not an error. It signals that the desired number of images have been generated:

```python
class EndOfFlowException(Exception):
    """Raised when save_image reaches the target step count."""
    result: Any  # Partial result before the exception
```

Flows should let this propagate; the GUI catches it to stop execution gracefully.

---

## Workflow & Debugging

*   **Run/Debug**: Use the VS Code launch config **"Python Debugger: Stable Diffussion"**.
    *   Arguments: `--temp-directory wololo/temp --verbose DEBUG`
*   **Single-Process Debugging**: Set `do_multiprocess=False` in the factory setup to step through node execution without spawning children.
*   **Logs**: Check `workspace_temp/logs/` for execution details.
*   **Temp Files**: Generated images often land in `wololo/output` or the configured temp directory.

---

## Configuration

*   **`pyproject.toml`** (root): Contains configurations for **pylint**, **mypy**, and **pytest**. All linting and testing tools use this as their primary config file.

---

## Do Not Modify
*   Never do directing changes to the core `comfy/` directory or root `nodes.py`. Focus development within `json_gui/`.
