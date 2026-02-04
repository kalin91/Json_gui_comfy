# ComfyUI JSON GUI

## Overview

**ComfyUI JSON GUI** is a specialized, lightweight application ("fork") built on top of ComfyUI. It serves as a **headless orchestrator** that allows users to execute complex ComfyUI workflows through a simplified, form-based graphical interface (Tkinter), rather than the traditional node-graph editor.

The primary goal of this project is to separate the *definition* of a workflow from its *execution*. By abstracting parameters into a structured JSON/YAML configuration, users can generate images and run tasks by simply modifying fields (prompts, models, resolutions) without needing to understand the underlying node graph.

## Features

* **Parameter-Based Interface**: A dynamic Tkinter GUI that builds forms based on YAML schemas.
* **Headless Execution**: Bypasses the standard ComfyUI WebSocket server, running nodes programmatically via a "Mimic" engine.
* **Pre-defined Flows**: Support for modular "Flows" (e.g., Stable Diffusion, ControlNet) that package logic and configuration together.
* **Integrated Image Viewer**: A simple built-in viewer to preview results immediately.
* **JSON/YAML Configuration**: Easy-to-read configuration files for storing and reloading experiment settings.
* **Single-Process & Multi-Process Modes**: Flexible execution architecture for debugging or production use.

---

## How to Run

This application is designed to be run as a Python module. Below is the recommended Visual Studio Code Launch Configuration.

### VS Code Configuration

Add the following to your `.vscode/launch.json` or workspace configuration:

```json
{
  "name": "Python Debugger: Stable Diffussion",
  "type": "debugpy",
  "env": {
    "PYTHONPATH": "${workspaceFolder:main}"
  },
  "python": "${command:python.interpreterPath}",
  // "justMyCode": false,
  "console": "integratedTerminal",
  "request": "launch",
  "module": "main_json_gui",
  "args": "--temp-directory dummypath/temp --verbose DEBUG",
  "subProcess": true
}
```

### Command Line Arguments

* `--temp-directory`: Specifies the directory where temporary files and logs are stored.
* `--verbose`: Sets the logging level (e.g., `DEBUG`, `INFO`).

---

## Technical Architecture

The application follows a modular architecture distinct from the standard ComfyUI. It can be broken down into three main layers: The GUI (View), The Flow System (Controller/Model), and The Execution Engine (Backend).

### 1. Core Application Structure (`json_gui/`)

* **`main_json_gui.py`**: The entry point of the application. It bootstraps the Tkinter environment and initializes the main application window.
* **`json_gui/json_manager/`**: Contains the GUI logic.
  * **`json_manager_gui.py`**: The main controller for the window. It handles the loading of flows, saving of configurations, and dispatching the "Execute" command.
  * **`json_tree_editor.py`**: A crucial component that parses the `body.yml` schema of a flow and dynamically renders the corresponding UI widgets (Text boxes, Checkboxes, Dropdowns, File pickers). It creates a "Tree" of parameters that handles data validation.
  * **`image_viewer.py`**: A lightweight component to display generated images within the app.

### 2. The Flow System (`json_gui/scripts/`)

A "Flow" is a self-contained module representing a specific workflow (e.g., `controlnet_openpose`). Each flow resides in its own directory (e.g., `json_gui/scripts/controlnet_openpose/`) and consists of:

* **`body.yml`**: The **Schema**. It defines what the user sees in the GUI. It declares keys, default values, labels, and input types (e.g., `string`, `int`, `file_path`).
* **`flow.py`**: The **Logic**. Contains a class (inheriting from `AbsFlow`) that implements the `execution` method. This script reads the values from the GUI (passed as a dict) and orchestrates the ComfyUI nodes.
* **`model.py`** (Optional): Data classes or Pydantic models to strictly type the input configuration.

### 3. The Execution Engine ("Mimic" System)

This is the technical heart of the project. Since there is no web client involved, we cannot use the standard API. The app uses a set of scripts to "mimic" the behavior of the API server and execute nodes directly.

* **`json_gui/scripts/mimic.py`**: The **base class** `MimicNode` for all node wrappers.
* **`json_gui/scripts/node_executor.py`**: The **multi-process execution engine**. Spawns child processes to run nodes.
* **`json_gui/scripts/mimic_classes.py` & `mimic_ksamplers.py`**: Concrete wrappers around standard ComfyUI nodes.

---

## Single-Process vs. Multi-Process Architecture

This is the **most critical architectural decision** in the system. Understanding it is essential for debugging and extending the codebase.

### Why Multi-Processing?

GPU memory (VRAM) on consumer hardware is limited. When executing complex pipelines (model loading, sampling, VAE decoding), memory can become fragmented or leak. The multi-process architecture provides:

1. **Process Isolation**: Each heavy computation (e.g., KSampler) runs in a *child process*. When the child terminates, all its GPU memory is *guaranteed* to be released by the OS.
2. **Fault Tolerance**: If a node crashes, the main GUI process remains alive.
3. **Clean State**: Each child process starts with a fresh Python interpreter and CUDA context.

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

* `torch.Tensor` (must be moved to CPU first).
* Primitive types (`int`, `str`, `dict`, `list`).
* `MimicNode` classes (the class itself, not instances with callbacks).

**Cannot be pickled:**

* Lambda functions.
* Closures (nested functions).
* Objects with open file handles or CUDA streams.

The `DataWrapper` class and `_move_tensors_to_device` helper handle tensor serialization automatically.

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN PROCESS (GUI)                                │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐  │
│  │ Flow.py     │───>│ MimicNode   │───>│ NodeExecutor.execute()         │  │
│  │ execution() │    │ .exec_node()│    │  - Serialize args (CPU tensors)│  │
│  └─────────────┘    └─────────────┘    │  - Spawn child process         │  │
│                                        │  - Poll result queue           │  │
│                                        └────────────┬────────────────────┘  │
│                                                     │                       │
└─────────────────────────────────────────────────────│───────────────────────┘
                                                      │
                                        ══════════════╧══════════════
                                        ║   PROCESS BOUNDARY (IPC)  ║
                                        ══════════════╤══════════════
                                                      │
┌─────────────────────────────────────────────────────│───────────────────────┐
│                         CHILD PROCESS (Worker)      │                       │
│                                                     ▼                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ _node_executor_target()                                              │   │
│  │  1. Move tensors to GPU                                              │   │
│  │  2. Reconstruct MimicNode from (class, init_args)                    │   │
│  │  3. Reconstruct dependency nodes                                     │   │
│  │  4. Call node.process(**exec_args)                                   │   │
│  │  5. Save tensors to disk                                             │   │
│  │  6. Move result tensors to CPU                                       │   │
│  │  7. Put result in Queue                                              │   │
│  │  8. Wait for termination signal                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  GPU Memory is RELEASED when this process terminates                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The `MimicNode` Base Class

**Location**: `json_gui/scripts/mimic.py`

`MimicNode` is the **abstract base class** for all node wrappers. It provides:

### Core Responsibilities

1. **Lifecycle Management**: `update()` -> `process()` pattern.
2. **Caching**: Avoids redundant computation if inputs haven't changed.
3. **Multi-Process Dispatch**: Transparently routes execution via `NodeExecutor`.
4. **Tensor Saving**: Manages saving intermediate/final images to disk.

### Class Variables

```python
class MimicNode(ABC, Generic[T]):
    # Global execution mode flag
    _do_multiprocess: ClassVar[bool] = False
    
    # Factory to create NodeExecutor instances
    _node_executor_factory: ClassVar[Optional["NodeExecutorFactory"]] = None
    
    # Currently loaded model (shared across nodes in multi-process)
    _current_model: ClassVar[Optional["MimicNode"]] = None
```

### Instance Variables

```python
    # Caching flags
    _return_cache: bool           # Whether cached output is valid
    _initialized: bool            # Whether update() has been called
    
    # Callbacks
    _save_tensor: Optional[SaveTensorCallable]  # Function to save tensors
    
    # Argument storage (for reconstruction in child process)
    _init_args: Optional[CreationDict]   # Args passed to __init__
    _exec_args: Optional[CreationDict]   # Args passed to process()
    
    # Output caching
    _last_output: Optional[T]            # Cached result of process()
    
    # NodeExecutor caching (to avoid re-spawning for identical calls)
    _ne_param_cache: Optional[CreationDict]
    _init_args_cache: Optional[CreationDict]
    _ne_result_cache: tuple[Any, SavedImagesDict]
```

### Lifecycle Methods

#### `update(*args, **kwargs) -> None`

Called to configure the node's internal state. This is typically called once before `process()`.

```python
@final
def update(self, *args, **kwargs) -> None:
    """
    1. Checks if inputs have changed (compares with _init_args).
    2. If changed:
       - Unwraps any DataWrapper arguments.
       - Calls abstract _update_impl().
       - Stores args in _init_args for later reconstruction.
       - Invalidates cache.
    """
```

#### `process(*args, **kwargs) -> T`

Called to execute the node's logic.

```python
@final
def process(self, *args, **kwargs) -> T:
    """
    1. Checks cache; if valid and inputs match _exec_args, returns _last_output.
    2. Unwraps DataWrapper arguments.
    3. Calls _feed_function(_process_impl, ...) to filter kwargs.
    4. Saves any unsaved tensors via save_tensor callback.
    5. Caches result in _last_output.
    6. Returns result.
    
    Raises:
        EndOfFlowException: If save_image reached target step count.
    """
```

#### `exec_node(pre_node_process_args, node_params) -> T`

The **routing function** that decides between single-process and multi-process execution.

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
        -> Serializes node class + args.
        -> Calls _exec_node_spawn() to run in a child process.
    Else:
        -> Merges node_params into args.
        -> Calls _exec_node_sync() to run in the current process.
    """
```

### Abstract Methods (Subclasses Must Implement)

```python
@classmethod
@abstractmethod
def key(cls) -> str:
    """Returns a unique identifier for this node type (e.g., 'simple_k_sampler')."""

@classmethod
@abstractmethod
def _class_param_definitions(cls) -> list[ClassParam[Self, Any]]:
    """Defines dependencies on other MimicNode types."""

@abstractmethod
def _update_impl(self, *args, **kwargs) -> None:
    """Stores parameters in instance variables."""

@abstractmethod
def _process_impl(self, *args, **kwargs) -> T:
    """The actual computation logic."""
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

1. Pops the `skip_layers_model` key from kwargs.
2. Calls the `processor` lambda to get additional kwargs.
3. Merges them into the function call.

### DataWrapper: Lazy Evaluation

`DataWrapper` wraps values that may need to be built on-demand. It handles:

* **Lazy construction**: Stores a builder function + args, calls it when `.get()` is invoked.
* **Eager values**: Can also wrap pre-built values.
* **Serialization**: Validates that contents are pickle-safe.

```python
# Lazy wrapper
wrapper = DataWrapper(
    builder=some_function,
    args={"param": value},
    identifier="my_wrapper"
)
result = wrapper.get()  # Calls some_function(**args)

# Eager wrapper
wrapper = DataWrapper(value=my_tensor, skip_unwrap=True)
result = wrapper.get()  # Returns wrapper itself (skip_unwrap=True)
```

---

## The `NodeExecutor` Class

**Location**: `json_gui/scripts/node_executor.py`

`NodeExecutor` is the **multi-process execution engine**. It:

1. Spawns a child process.
2. Serializes node class + arguments.
3. Reconstructs the node in the child.
4. Executes `node.process()`.
5. Serializes and returns results to the parent.

### Constructor

```python
def __init__(
    self,
    node: MimicNode,                                    # The node to execute
    pre_node_process_args: dict[str, Any],              # Arguments for process()
    pre_raw_nodes: dict[type[MimicNode], CreationDict], # Dependency nodes
    save_data: SavedImagesDict,                         # Mutable dict to track saved images
):
    """
    Prepares for multi-process execution:
    1. Moves tensors in pre_node_process_args to CPU.
    2. Removes save_tensor callbacks from MimicNode instances (not serializable).
    3. Creates result_queue and log_queue for IPC.
    4. Stores raw_nodes as (class, init_args) tuples for reconstruction.
    """
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
    Spawns a child process and waits for the result.
    
    Steps:
    1. Prepares init_args for pickle (moves tensors to CPU).
    2. Creates a partial function for _node_executor_target.
    3. Wraps it with c_logger.worker_wrapper for logging.
    4. Spawns a multiprocessing.Process.
    5. Polls result_queue while child is alive.
    6. On timeout: terminates child, raises TimeoutError.
    7. On completion: gets result from queue, terminates child.
    8. If result is an exception: re-raises it.
    9. Returns (output, save_data).
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
    Runs in the CHILD process:
    
    1. MimicNode.enable_multiprocess() - Sets the global flag.
    2. Calls _execute_target_node() to do the actual work.
    3. Validates result is serializable via pickle.dumps().
    4. Puts result tuple in queue: ("success" | "error", save_data, output | exception).
    5. Calls signal.pause() to wait for parent to terminate it.
    """
```

### The Actual Execution

```python
@staticmethod
def _execute_target_node(
    node_cls: type[MimicNode],
    node_init_args: CreationDict,
    node_exec_args: dict[str, Any],
    raw_nodes_serialized: dict[str, tuple[type[MimicNode], CreationDict]],
    save_call: SaveImageCallable,
    save_data: SavedImagesDict,
) -> tuple[str, SavedImagesDict, Any]:
    """
    The core execution logic in the child:
    
    1. Gets the GPU device via comfy.model_management.get_torch_device().
    2. Moves tensors in init_args and exec_args to GPU.
    3. Reconstructs the main node: node_cls(*init_args["args"], **init_args["kwargs"]).
    4. Sets up save_tensor callback on the node.
    5. Reconstructs dependency nodes from raw_nodes_serialized.
    6. Adds dependency nodes to exec_args.
    7. Calls node.process(**exec_args).
    8. Returns ("success", save_data, output).
    
    On exception:
        Returns ("error", save_data, exception).
    """
```

### Tensor Movement Helper

```python
def _move_tensors_to_device(obj: T, torch_device: device, memo: dict = None) -> T:
    """
    Recursively traverses obj (dict, list, tuple, custom objects).
    
    For each torch.Tensor:
        - detach(): Removes from computation graph.
        - clone(): Creates independent memory.
        - contiguous(): Ensures standard memory layout for pickle.
        - to(device): Moves to target device.
    
    For unserializable callables (lambdas, closures):
        - Returns None (filtered out).
    
    Uses memo dict to handle circular references.
    """
```

---

## Concrete MimicNode Implementations

### `mimic_classes.py`

| Class | Key | Description |
|-------|-----|-------------|
| `Sd3Clip` | `sd3_clip` | Loads the triple CLIP model for SD3 (CLIP-G, CLIP-L, T5). |
| `SkipLayers` | `skip_layers_model` | Loads a checkpoint and applies skip-layer guidance. |
| `Prompts` | `prompts` | Encodes positive/negative prompts using CLIP. |
| `EmptyLatent` | `empty_latent` | Creates an empty latent tensor or encodes a starting image. |
| `Rotator` | `rotator` | Rotates images and provides an undo function. |

### `mimic_ksamplers.py`

| Class | Key | Description |
|-------|-----|-------------|
| `KSamplerLike` | (abstract) | Base class for samplers with common parameters. |
| `SimpleKSampler` | `simple_k_sampler` | Standard KSampler wrapping `comfy.sample.sample()`. |
| `FaceDetailerNode` | `face_detailer` | Wraps the FaceDetailer from Impact Pack. |

### Example: SimpleKSampler

```python
class SimpleKSampler(KSamplerLike[Tuple[torch.Tensor, torch.Tensor]]):
    """Wraps ComfyUI's KSampler for diffusion sampling."""
    
    @classmethod
    def key(cls):
        return "simple_k_sampler"
    
    @classmethod
    def _class_param_definitions(cls):
        # Declares dependency on SkipLayers (the model loader)
        return [
            cls.build_class_param(
                SkipLayers,
                processor=lambda inst: {"node_model": inst}
            )
        ]
    
    def _process_impl(
        self,
        latent_image: torch.Tensor,
        node_model: SkipLayers,  # Injected by ClassParam
        cond_pos_cnet: Any,
        cond_neg_cnet: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. Gets model and VAE from node_model.
        2. Prepares noise.
        3. Calls comfy.sample.sample().
        4. Decodes latent with VAE.
        5. Adds result to unsaved_tensors.
        6. Returns (latent, image).
        """
```

---

## Workflow Lifecycle

1. **Selection**: User launches the app and uses the dropdown to select a Workflow (e.g., *ControlNet OpenPose*).
2. **UI Generation**:
    * `json_manager_gui.py` reads `json_gui/scripts/<flow_name>/body.yml`.
    * `json_tree_editor.py` dynamically builds the form widgets.
3. **Configuration**: User enters prompts, selects models, etc.
4. **Save**: The configuration is saved to a JSON file (the "Experiment").
5. **Execution**:
    * The user clicks **Execute**.
    * The corresponding `Flow` class in `flow.py` is instantiated.
    * `MimicNode.set_node_executor_factory()` is called to configure execution mode.
    * The `execution(inputs)` method is called with the form data.
6. **Processing**:
    * The `Flow` logic instantiates `MimicNode` subclasses.
    * Calls `node.exec_node(args, deps)` which routes to single or multi-process.
    * ComfyUI's backend processes the image on the GPU.
7. **Output**:
    * `save_image()` is called to save tensors to disk.
    * If step count is reached, `EndOfFlowException` is raised.
    * The UI updates to show the result in the `ImageViewer`.

---

## EndOfFlowException

This is a **control flow exception**, not an error. It signals that the desired number of images have been generated.

```python
class EndOfFlowException(Exception):
    """Raised when save_image reaches the target step count."""
    result: Any  # Partial result before the exception
```

### Usage

```python
# In save_image() (json_gui/utils.py)
if len(created_images) >= steps:
    raise EndOfFlowException(steps)

# In MimicNode.process()
try:
    res = self._process_impl(...)
except EndOfFlowException as eof:
    eof.result = res
    raise eof  # Propagates up

# In Flow.execution()
# Let it propagate; GUI catches it to stop gracefully.
```

---

## Developer Guide: Creating a New Flow

To add a new workflow to the system:

1. **Create Directory**: Create a new folder in `json_gui/scripts/` (e.g., `my_new_flow`).
2. **Define Schema (`body.yml`)**:

    ```yaml
    - key: "positive_prompt"
      label: "Positive Prompt"
      type: "text"
      default: "A beautiful landscape"
    - key: "seed"
      label: "Seed"
      type: "int"
      default: 1234
    ```

3. **Implement Logic (`flow.py`)**:

    ```python
    from json_gui.utils import AbsFlow
    from json_gui.scripts.mimic_classes import SkipLayers, Prompts, EmptyLatent
    from json_gui.scripts.mimic_ksamplers import SimpleKSampler

    class Flow(AbsFlow):
        def execution(self, inputs):
            # 1. Create nodes
            model_node = SkipLayers(layers=[], scale=1.0, start_percent=0.0, end_percent=1.0)
            prompt_node = Prompts(positive=inputs['positive_prompt'], negative=inputs['negative_prompt'])
            latent_node = EmptyLatent(width=512, height=512, batch_size=1, image_name="<None>")
            sampler_node = SimpleKSampler(
                seed=inputs['seed'],
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                use_tune=False
            )
            
            # 2. Execute (uses exec_node for multi-process support)
            cond_pos, cond_neg = prompt_node.exec_node({}, [])
            latent = latent_node.exec_node({}, [model_node])
            result_latent, result_image = sampler_node.exec_node(
                {"latent_image": latent, "cond_pos_cnet": cond_pos, "cond_neg_cnet": cond_neg},
                [model_node]
            )
            
            return self.created_images
    ```

4. **Restart**: Restart the application. Your new flow will appear in the selection list.

---

## Debugging Tips

### Single-Process Mode

For easier debugging, switch to single-process mode:

```python
# In your Flow or AbsFlow initialization
MimicNode.set_node_executor_factory(
    NodeExecutor,
    save_data,
    save_call,
    copy_call,
    do_multiprocess=False  # <-- Disable multi-process
)
```

This allows you to:
* Set breakpoints inside `_process_impl()` methods.
* Step through node execution in the debugger.
* See stack traces without process boundary confusion.

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `pickle.PicklingError` | Lambda or closure in arguments | Replace with a named function or static method |
| `CUDA out of memory` | Tensors not released | Ensure multi-process mode is enabled |
| `Queue.Empty` after timeout | Child process crashed | Check logs for exceptions in child |
| Cached results returned | Inputs match previous call | Call `node.update()` to force refresh |

---

## Configuration Files

* **`pyproject.toml`** (root): Contains configurations for **pylint**, **mypy**, and **pytest**. All linting, type-checking, and testing tools use this as their primary configuration file.
