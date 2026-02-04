"""Mimic Parent Class and Utilities."""

import logging
import copy
import uuid
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
    Generic,
    cast,
    get_args,
    final,
    TYPE_CHECKING,
    Protocol,
    Self,
    ClassVar,
)
from abc import ABC, abstractmethod
import os
import pickle
import inspect
import numpy as np
import torch
from torch import Tensor
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import node_helpers
from json_gui.typedicts import SaveImageCallable, get_empty_creation_dict, CreationDict, SavedImagesDict
from json_gui.utils import EndOfFlowException

if TYPE_CHECKING:
    from json_gui.scripts.node_executor import NodeExecutor

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")
Q = TypeVar("Q", bound="NodeExecutor")
M = TypeVar("M", bound="MimicNode")
N = TypeVar("N", bound="MimicNode")  # pylint: disable=C0105
RES = TypeVar("RES", bound="MimicNode")


class DataWrapper(Generic[T]):
    """A wrapper for data that is built on demand."""

    Val_Types = Literal["latent_tensor", "image_tensor", "other"]

    # Class props
    _skip_pickle_check: ClassVar[bool] = False

    # Instance props
    _builder: Optional[Callable[..., T]]
    _value: Optional[T]

    @staticmethod
    def skip_pickle_check(value: bool) -> None:
        """Skip pickle serialization check for DataWrapper arguments."""
        DataWrapper._skip_pickle_check = value

    @property
    def identifier(self) -> str:
        """
        Get the unique identifier for this DataWrapper.

        Returns:
            str: The unique identifier for this wrapper.
        """
        return self._identifier

    @property
    def value_type(self) -> Val_Types:
        """Check if the wrapped value is a latent tensor."""
        return self._val_type

    @property
    def args(self) -> dict[str, Any]:
        """Get the arguments used to build the value."""
        return self._args

    @property
    def skip_unwrap(self) -> bool:
        """Check if unwrapping should be skipped."""
        return self._skip_unwrap

    @skip_unwrap.setter
    def skip_unwrap(self, value: bool) -> None:
        """Set whether unwrapping should be skipped."""
        self._skip_unwrap = value

    def __init__(
        self,
        builder: Optional[Callable[..., T]] = None,
        args: Optional[dict[str, Any]] = None,
        *,
        value: Optional[T] = None,
        skip_unwrap: bool = False,
        identifier: Optional[str] = None,
        val_type: Val_Types = "other",
    ) -> None:
        """
        Initialize the DataWrapper, either with a builder function (lazy) or a pre-built value (eager).

        Args:
            builder (Optional[Callable[..., T]], optional): A function to build the value. Defaults to None.
            value (Optional[T], optional): A pre-built value. Defaults to None.
            args (Optional[dict[str, Any]], optional): Arguments to pass to the builder. Defaults to None.
            skip_unwrap (bool, optional): Whether to skip unwrapping the value. Defaults to False.
            identifier (Optional[str], optional): Unique identifier for this wrapper. Defaults to None.
            val_type (Val_Types, optional): The type of the wrapped value. Defaults to "other".

        Raises:
            ValueError: If neither builder nor value is provided.
            ValueError: If both builder and value are provided.
            ValueError: If val_type is not one of the allowed types.
            ValueError: If the builder function does not have a return type annotation.
            ValueError: If the arguments for the builder are not pickle-serializable.

        Returns:
            DataWrapper: The initialized DataWrapper instance.
        """
        if builder is None and value is None:
            raise ValueError("Either builder or value must be provided.")
        if value is not None and builder is not None:
            raise ValueError("Only one of builder or value should be provided.")

        self._identifier: str = identifier if identifier else str(uuid.uuid4())
        self._val_type = val_type
        self._skip_unwrap = skip_unwrap
        self._args = {}
        self._builder = None
        self._value = None

        # Validate value type
        if val_type not in get_args(self.Val_Types):
            raise ValueError(f"val_type must be one of {get_args(self.Val_Types)}")

        data: Any
        if builder:
            self._args = args if args is not None else {}

            # get the builder return type using inspect
            sig = inspect.signature(builder)
            return_type: Type = sig.return_annotation
            if return_type is inspect.Signature.empty:
                raise ValueError(
                    "Builder function must have a return type annotation and must be pickle-serializable."
                )
            data = (builder, args)
            self._builder: Callable[..., T] = builder
        else:
            self._value: T = value
            data = value

        if val_type in ("latent_tensor", "image_tensor") and not isinstance(return_type, Tensor):
            raise ValueError("Value must be a Tensor for latent_tensor or image_tensor types.")
        if not DataWrapper._skip_pickle_check:
            try:
                data = pickle.dumps(data)
                logging.info("Arguments for DataWrapper serialized successfully, size: %d bytes", len(data))
            except Exception as e:
                logging.exception("Failed to serialize arguments for DataWrapper: %s", str(e))
                raise ValueError(f"Arguments for builder must be pickle-serializable: {e}") from e

    def get(self) -> Self | T:
        """
        Could get:
        1. Itself if skip_unwrap is True.
        2. The stored value if skip_unwrap is False and value is set.
        3. The builder function called with args if skip_unwrap is False.

        Returns:
            T: The unwrapped value.
        """
        if self._skip_unwrap:
            return self
        if self._builder:
            return self._builder(**self._args)
        if self._value is None:
            raise RuntimeError("DataWrapper has no value or builder to get the value from.")
        return self._value


def safe_reference_compare(var1, var2, _memo: set | None = None) -> bool:
    """
    Compare two values safely:
    - DataWrapper: by identifier
    - torch.Tensor: by content (torch.equal)
    - Basic types (int, float, str, bool, tuple, bytes, None): by value
    - dict/list: recursively
    - Everything else: by identity (is)
    """
    # Prevent infinite recursion on circular references
    if _memo is None:
        _memo = set()

    pair_id = (id(var1), id(var2))
    if pair_id in _memo:
        return True  # Already comparing these objects

    # 1. DataWrapper: compare by identifier
    if isinstance(var1, DataWrapper):
        if not isinstance(var2, DataWrapper):
            return False
        return var1.identifier == var2.identifier

    # 2. torch.Tensor: compare by content
    if isinstance(var1, torch.Tensor):
        if not isinstance(var2, torch.Tensor):
            return False
        if var1.shape != var2.shape:
            return False
        return torch.equal(var1, var2)

    # 3. Basic immutable types: compare by value
    optimizable_types = (int, str, float, bool, bytes, type(None))
    if isinstance(var1, optimizable_types):
        return var1 == var2

    # 4. Tuples: recursive comparison
    if isinstance(var1, tuple):
        if not isinstance(var2, tuple) or len(var1) != len(var2):
            return False
        return all(safe_reference_compare(a, b, _memo) for a, b in zip(var1, var2))

    # 5. Lists: recursive comparison
    if isinstance(var1, list):
        if not isinstance(var2, list) or len(var1) != len(var2):
            return False
        _memo.add(pair_id)
        return all(safe_reference_compare(a, b, _memo) for a, b in zip(var1, var2))

    # 6. Dicts: recursive comparison
    if isinstance(var1, dict):
        if not isinstance(var2, dict) or var1.keys() != var2.keys():
            return False
        _memo.add(pair_id)
        return all(safe_reference_compare(var1[k], var2[k], _memo) for k in var1)

    # 7. Everything else: identity comparison
    return var1 is var2


class MimicNode(ABC, Generic[T]):
    """A mimic class for various nodes."""

    # Class props
    _node_executor_factory: ClassVar[Optional["NodeExecutorFactory"]] = None
    _current_model: ClassVar[Optional["MimicNode"]] = None
    _do_multiprocess: ClassVar[bool] = False

    # Instance props
    _return_cache: bool
    _initialized: bool
    _save_tensor: Optional["MimicNode.SaveTensorCallable"]
    _init_args: Optional[CreationDict]
    _exec_args: Optional[CreationDict]
    _last_output: Optional[T]
    _ne_param_cache: Optional[CreationDict]
    _init_args_cache: Optional[CreationDict]
    _ne_result_cache: tuple[Any, SavedImagesDict]

    class SaveTensorCallable(Protocol):
        """Protocol for save image callable."""

        def __call__(self, images: torch.Tensor, identifier: str, is_temp: bool = True) -> None: ...

    class NodeExecutorFactory(Generic[Q]):
        """Factory class to create NodeExecutor instances."""

        @property
        def save_call(self) -> SaveImageCallable:
            """Returns the save call function."""
            return self._save_call

        @property
        def copy_call(self) -> Callable[[SavedImagesDict], None]:
            """Returns the copy call function."""
            if self._copy_call is None:
                raise RuntimeError("copy_call function is not set.")
            return self._copy_call

        @property
        def save_data(self) -> SavedImagesDict:
            """Returns the save data dictionary."""
            return self._save_data

        def __init__(
            self,
            node_executor_cls: Type[Q],
            save_data: SavedImagesDict,
            save_call: SaveImageCallable,
            copy_call: Callable[[SavedImagesDict], None],
        ) -> None:
            self._node_executor_cls: Type[Q] = node_executor_cls
            self._save_data = save_data
            self._save_call = save_call
            self._copy_call = copy_call

        def create_node_executor(
            self,
            node: "MimicNode",
            pre_node_process_args: dict[str, Any],
            pre_raw_nodes: dict[type["MimicNode"], CreationDict],
        ) -> "NodeExecutor":
            """
            Creates a NodeExecutor instance.

            Args:
                node (MimicNode): The mimic node to execute.
                pre_node_process_args (dict[str, Any]): Pre-processed node arguments.
                pre_raw_nodes (dict[type[MimicNode], dict[str, Any]]): Pre-processed raw nodes.

            Returns:
                NodeExecutor: The created NodeExecutor instance.
            """
            return self._node_executor_cls(node, pre_node_process_args, pre_raw_nodes, self._save_data)

    class ClassParam(Generic[N, RES]):
        """A class parameter wrapper for MimicNode."""

        _implemented: bool

        def process(self) -> None:
            """Process method to be implemented on first use."""
            if self._implemented:
                raise RuntimeError("process method already implemented for this ClassParam.")

            param = self._result_class.key()

            def decorator(param: str = param, func: Callable[..., R] = self._target.process) -> Callable[P, R]:

                def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                    """Wrapper function."""
                    logging.info("Processing ClassParam for %s with param %s", self._target.__name__, param)
                    if param not in kwargs:
                        raise TypeError(f"Missing required parameter: {param}")
                    node: RES = cast(RES, kwargs.pop(param))
                    additional_params: dict[str, Any] = self._processor(node)
                    kwargs.update(additional_params)
                    return func(*args, **kwargs)

                # Register the parameter on the wrapper so _feed_function knows it's allowed
                setattr(
                    wrapper,
                    "_mimic_extra_params",
                    getattr(self._target.process, "_mimic_extra_params", set()) | {param},
                )

                return wrapper

            self._target.process = decorator()  # type: ignore[method-assign, misc]
            self._implemented = True

        def __new__(cls, *args, **kwargs):
            # Instance can only be created via MimicNode.build_class_param
            # through analyzing the caller method.
            method = inspect.stack()[1].function
            # Check if called directly from build_class_param
            if method != "build_class_param":
                raise TypeError("ClassParam instances can only be created via MimicNode.build_class_param.")
            return super().__new__(cls)

        def __init__(
            self, target: Type[N], result_class: Type[RES], processor: Callable[[RES], dict[str, Any]]
        ) -> None:
            assert inspect.isclass(target), "target must be a class."
            assert inspect.isclass(result_class), "result_class must be a class."
            assert callable(processor), "processor must be a callable."
            assert issubclass(result_class, MimicNode), "result_class must be a subclass of MimicNode."
            assert issubclass(target, MimicNode), "target must be a subclass of MimicNode."

            self._target = target
            self._result_class = result_class
            self._processor: Callable[[RES], dict[str, Any]] = processor
            self._implemented = False

        def _is_process_override(self) -> bool:
            """
            Checks if the target class has overridden the process method.

            Returns:
                bool: True if process is overridden, False otherwise.
            """
            target_method = self._target.process

            parent_method = MimicNode.process
            return target_method is not parent_method

    @staticmethod
    def enable_multiprocess():
        """Enables multiprocess mode for MimicNode."""
        MimicNode._do_multiprocess = True

    @classmethod
    def set_node_executor_factory(
        cls,
        ne_class: Type[Q],
        save_data: SavedImagesDict,
        save_call: SaveImageCallable,
        copy_call: Callable[[SavedImagesDict], None],
        do_multiprocess: bool,
    ) -> None:
        """
        Gets or creates the NodeExecutorFactory singleton.

        Returns:
            NodeExecutorFactory: The NodeExecutorFactory instance.
        """
        MimicNode._node_executor_factory = MimicNode.NodeExecutorFactory(ne_class, save_data, save_call, copy_call)
        MimicNode._do_multiprocess = do_multiprocess
        DataWrapper.skip_pickle_check(not do_multiprocess)

    @classmethod
    def _has_node_executor_factory(cls) -> bool:
        """
        Checks if the NodeExecutorFactory singleton exists.

        Returns:
            bool: True if the NodeExecutorFactory exists, False otherwise.
        """
        return MimicNode._node_executor_factory is not None

    @classmethod
    def _get_node_executor_factory(cls) -> NodeExecutorFactory:
        """
        Checks if the NodeExecutorFactory singleton exists.

        Returns:
            bool: True if the NodeExecutorFactory exists, False otherwise.
        """
        if MimicNode._node_executor_factory is None:
            raise RuntimeError("NodeExecutorFactory is not initialized for MimicNode.")
        return MimicNode._node_executor_factory

    @classmethod
    def build_class_param(
        cls, result_class: Type[RES], processor: Callable[[RES], dict[str, Any]]
    ) -> ClassParam[Self, RES]:
        """Returns a class parameter wrapper."""
        return cls.ClassParam(cls, result_class, processor)

    @classmethod
    def _eject_class_params(cls) -> None:
        cls_params = cls._class_param_definitions()
        if not isinstance(cls_params, list):
            raise TypeError("_class_param_definitions must return a list of ClassParam instances.")
        for cp in cls_params:
            cp.process()

    @classmethod
    def _get_current_model(cls, tp: Type[N]) -> N:
        """Returns the Model mimic node class."""
        if not MimicNode._do_multiprocess:
            raise RuntimeError("current model is only available in multiprocess mode.")
        if MimicNode._current_model is None:
            raise RuntimeError("No current model set for MimicNode.")
        if not isinstance(MimicNode._current_model, tp):
            raise TypeError(f"model must be an instance of {tp.__name__}.")
        return cast(N, MimicNode._current_model)

    @classmethod
    def _set_current_model(cls, model: "MimicNode") -> None:
        """Sets the current Model mimic node class."""
        if MimicNode._do_multiprocess or not cls._has_current_model(type(model)):
            if MimicNode._current_model is not None:
                raise RuntimeError("Current model is already set for MimicNode. Override not allowed.")
            if not isinstance(model, MimicNode):
                raise TypeError("model must be an instance of MimicNode.")
            model.process()
            MimicNode._current_model = model

    @classmethod
    def _has_current_model(cls, t: Type["MimicNode"]) -> bool:
        """

        Args:
            t (Type[MimicNode]): _description_

        Returns:
            bool: _description_
        """
        exists: bool = MimicNode._current_model is not None
        if exists and not isinstance(MimicNode._current_model, t):
            raise TypeError(f"Current model is not of type {t.__name__}.")
        return exists

    @classmethod
    @abstractmethod
    def _class_param_definitions(cls) -> list[ClassParam[Self, Any]]:
        """Defines class parameters for the mimic node."""

    @classmethod
    @abstractmethod
    def key(cls) -> str:
        """Returns the key for the mimic node."""

    @property
    def init_args(self) -> CreationDict:
        """Returns a mutable dictionary of the initialization arguments used."""
        if self._init_args is None:
            raise RuntimeError("init_args is not set.")
        return self._init_args

    @init_args.deleter
    def init_args(self) -> None:
        """Deletes the cached init arguments."""
        self._init_args = get_empty_creation_dict()

    @property
    def exec_args(self) -> CreationDict:
        """Returns the last execution arguments used."""
        if self._exec_args is None:
            raise RuntimeError("exec_args is not set.")
        return self._exec_args

    @property
    def unsaved_tensors(self) -> list[tuple[Tensor, str]]:
        """Returns a list of unsaved tensors."""
        return self._unsaved_tensors

    @unsaved_tensors.deleter
    def unsaved_tensors(self) -> None:
        """Deletes the cached unsaved tensors."""
        self._unsaved_tensors.clear()

    @property
    def save_tensor(self) -> SaveTensorCallable:
        """Returns a function to save the tensor."""
        if self._save_tensor is None:
            raise RuntimeError("save_tensor function is not set.")
        return self._save_tensor

    @save_tensor.setter
    def save_tensor(self, value: SaveTensorCallable) -> None:
        """Sets the function to save the tensor."""
        assert callable(value), "save_tensor must be a callable function."
        self._save_tensor = value

    @save_tensor.deleter
    def save_tensor(self) -> None:
        """Deletes the save tensor function."""
        self._save_tensor = None

    @property
    def raw_materials(self) -> CreationDict:
        """Returns a immutable dictionary of the raw materials used."""
        return copy.deepcopy(self._raw_materials)

    @raw_materials.setter
    def raw_materials(self, value: CreationDict) -> None:
        """Sets the raw materials used."""
        self._raw_materials = copy.deepcopy(value)

    @property
    def init_args_cache(self) -> CreationDict:
        """Returns the raw materials used to create this mimic node."""
        if self._init_args_cache is None:
            raise RuntimeError("init_args_cache is not set.")
        return self._init_args_cache

    @init_args_cache.setter
    def init_args_cache(self, value: CreationDict) -> None:
        """Sets the raw materials used to create this mimic node."""
        if not value:
            raise ValueError("init_args_cache value cannot be None or empty.")
        self._init_args_cache = value

    @property
    def ne_param_cache(self) -> CreationDict:
        """Returns the NodeExecutor parameters cache."""
        if self._ne_param_cache is None:
            raise RuntimeError("ne_param_cache is not set.")
        return self._ne_param_cache

    @ne_param_cache.setter
    def ne_param_cache(self, value: CreationDict) -> None:
        """Sets the NodeExecutor parameters cache."""
        if not value:
            raise ValueError("ne_param_cache value cannot be None or empty.")
        self._ne_param_cache = value

    @property
    def ne_result_cache(self) -> tuple[Any, SavedImagesDict]:
        """Returns the NodeExecutor result cache."""
        if self._ne_result_cache is None:
            raise RuntimeError("ne_result_cache is not set.")
        return self._ne_result_cache

    @ne_result_cache.setter
    def ne_result_cache(self, value: tuple[Any, SavedImagesDict]) -> None:
        """Sets the NodeExecutor result cache."""
        if not value:
            raise ValueError("ne_result_cache value cannot be None or empty.")
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("ne_result_cache value must be a tuple of (result, save_data).")
        self._ne_result_cache = value

    @property
    def is_multiprocess(self) -> bool:
        """Check if multiprocessing is enabled."""
        return MimicNode._do_multiprocess

    def process_args_dict(self, *args, **kwargs) -> dict[str, Any]:
        """
        Given args and kwargs, returns a dict mapping parameter names to values for _process_impl.
        Useful for introspection, serialization, or dynamic invocation.
        """
        sig = inspect.signature(self._process_impl)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    @final
    def exec_node(
        self,
        pre_node_process_args: dict[str, Any],
        node_params: list["MimicNode"],
    ) -> T:
        """
        Executes the mimic node in the current process.

        Args:
            pre_node_process_args (dict[str, Any]): Pre-processed node arguments.
            node_params (list["MimicNode"]): List of mimic node parameters.
        Returns:
            T: The result of the node execution.
        """
        logging.info(
            "████████████████  >>>>>>>>>>>>> %s: (%s_process) <<<<<<<<<<<<<  ████████████████",
            self.key(),
            "multi" if MimicNode._do_multiprocess else "single",
        )
        if MimicNode._do_multiprocess:
            pre_raw_nodes: dict[type["MimicNode"], CreationDict] = {type(n): n.init_args for n in node_params}
            return self._exec_node_spawn(pre_node_process_args, pre_raw_nodes)
        else:
            pre_node_process_args.update({m.key(): m for m in node_params})
            return self._exec_node_sync(pre_node_process_args)

    def _exec_node_sync(self, pre_node_process_args: dict[str, Any]) -> T:
        """
        Executes the mimic node in the current process without multiprocessing.

        Args:
            pre_node_process_args (dict[str, Any]): Pre-processed node arguments.
        Returns:
            T: The result of the node execution.
        """
        factory = MimicNode._get_node_executor_factory()
        self.save_tensor = lambda tensor, identifier=self.key(), is_temp=True: factory.save_call(
            factory.save_data, tensor, identifier, is_temp=is_temp
        )
        return self.process(**pre_node_process_args)

    @final
    def _exec_node_spawn(
        self,
        pre_node_process_args: dict[str, Any],
        pre_raw_nodes: dict[type["MimicNode"], CreationDict],
    ) -> T:
        """
        Executes the mimic node in a separate process using spawn method.

        Args:
            pre_node_process_args (dict[str, Any]): Pre-processed node arguments.
            pre_raw_nodes (dict[type["MimicNode"], CreationDict]): Pre-processed raw nodes.
        Returns:
            T: The result of the node execution.
        """
        if not MimicNode._has_node_executor_factory():
            raise RuntimeError(
                "NodeExecutorFactory is not initialized for MimicNode, cannot exec_node_spawn. "
                "Initialize it first by calling MimicNode.set_node_executor_factory."
            )
        pre_raw_for_cache = {k.key(): copy.copy(v) for k, v in copy.copy(pre_raw_nodes).items()}
        pre_raw_for_cache.update(copy.deepcopy(pre_node_process_args))
        iargs = self.init_args["args"]
        ikwargs = self.init_args["kwargs"]
        nargs: list = []
        nkwargs = pre_raw_for_cache
        factory = MimicNode._get_node_executor_factory()
        use_init_cache: bool = self.__use_cache(self.init_args_cache, *iargs, **ikwargs)
        use_ne_cache: bool = self.__use_cache(self.ne_param_cache, *nargs, **nkwargs)
        res_cache_exists: bool = bool(
            self.ne_result_cache[0] is not None and not isinstance(self.ne_result_cache[0], EndOfFlowException)
        )
        result: T
        if use_init_cache and use_ne_cache and res_cache_exists:
            logging.info("Using cached NodeExecutor for %s", self.__class__.__name__)
            result, s_data = self.ne_result_cache
            factory.copy_call(s_data)
        else:
            self.init_args_cache = self.init_args
            self.ne_param_cache: CreationDict = {
                "args": [],
                "kwargs": pre_raw_for_cache,
            }
            node_executor = factory.create_node_executor(self, pre_node_process_args, pre_raw_nodes)
            result, s_data = node_executor.execute(factory.save_call, 150.0)
            ex = None
            if isinstance(result, EndOfFlowException):
                ex = result
                result = ex.result
            self.ne_result_cache: tuple[Any, SavedImagesDict] = copy.deepcopy((result, s_data))
            node_executor.save_data.update(s_data)
            if ex:
                raise ex
        return result

    @final
    def update(self, *args, **kwargs) -> None:
        """Updates the node."""
        try:
            if not self._initialized or not self.__use_cache(self.init_args, *args, **kwargs):
                logging.info("Updating %s", self.__class__.__name__)
                uw_args, uw_kwargs = self._unwrap_data_dict(*args, **kwargs)
                self._update_impl(*uw_args, **uw_kwargs)
                self._init_args = {"args": list(args), "kwargs": kwargs}
                self._last_output = None
                self._initialized = True
                self._return_cache = False
        except Exception as e:
            logging.exception("Error updating %s: %s", self.__class__.__name__, str(e))
            raise e

    @final
    def process(self, *args, **kwargs) -> T:
        """Processes data and returns the result, using caching if available."""
        if self._return_cache and self.__use_cache(self.exec_args, *args, **kwargs):
            logging.info("====== Using cached output for %s ======", self.__class__.__name__)
            return cast(T, self._last_output)
        self._return_cache = False
        logging.info("====== Processing %s ======", self.__class__.key())
        uw_args, uw_kwargs = self._unwrap_data_dict(*args, **kwargs)
        res: T
        try:
            res = self._feed_function(self._process_impl, *uw_args, **uw_kwargs)
            self.save_all_unsaved_tensors()
        except EndOfFlowException as eof:
            eof.result = res
            raise eof
        except Exception as e:
            logging.exception("====== Error processing %s======\nErr: %s ", self.__class__.__name__, str(e))
            raise e
        logging.info("====== Finished processing %s ======", self.__class__.key())
        self._exec_args = {"args": list(args), "kwargs": kwargs}
        self._last_output = res
        self._return_cache = True
        return res

    @final
    def add_unsaved_tensor(self, tensor: Tensor, name: str) -> None:
        """Adds a tensor to the unsaved tensors list."""
        self._unsaved_tensors.append((tensor, name))

    @final
    def save_all_unsaved_tensors(self) -> None:
        """Saves all unsaved tensors using the save_tensor function."""
        if self._unsaved_tensors:
            if self._save_tensor is None or not callable(self._save_tensor):
                raise RuntimeError("save_tensor function is not set, cannot save unsaved tensors.")
            if not isinstance(self._unsaved_tensors, list):
                raise RuntimeError("unsaved_tensors is not a list, cannot save unsaved tensors.")
            eof: Optional[EndOfFlowException] = None
            for tensor, name in self._unsaved_tensors:
                try:
                    self._save_tensor(tensor, name)
                except EndOfFlowException as ex:
                    logging.info("EndOfFlowException encountered while saving tensors: %s", str(ex))
                    eof = ex
            del self.unsaved_tensors
            if eof:
                raise eof

    def _upload_image(self, image_name: str) -> Tensor:
        """Uploads an image given its name."""
        assert image_name, "Image name must be provided for ControlNetImgPreprocessor."
        logging.info("Loading %s Image...", self.__class__.key())
        input_folder = folder_paths.get_input_directory()
        image_path = os.path.join(input_folder, image_name)
        img = node_helpers.pillow(Image.open, image_path)

        # Process image to tensor (similar to LoadImage node)
        output_images = []
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            output_images.append(image)

        if len(output_images) > 1:
            # If multiple frames, stack them? For now assume single image as per workflow
            img_tensor = torch.cat(output_images, dim=0)
        else:
            img_tensor = output_images[0]
        return img_tensor

    def _unwrap_data_dict(self, *args, **kwargs) -> tuple[list[Any], dict[str, Any]]:
        """Unwraps any DataWrapper instances in the provided dictionary."""

        unwrapped_args = []
        unwrapped_kwargs = {}
        for item in args:
            if isinstance(item, DataWrapper):
                unwrapped_args.append(item.get())
            else:
                unwrapped_args.append(item)
        for key, item in kwargs.items():
            if isinstance(item, DataWrapper):
                unwrapped_kwargs[key] = item.get()
            else:
                unwrapped_kwargs[key] = item
        return unwrapped_args, unwrapped_kwargs

    def _has_kwargs(self, func: Callable) -> bool:
        """Checks if func accepts keyword arguments."""
        sig = inspect.signature(func)
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD or p.kind is inspect.Parameter.VAR_POSITIONAL
            for p in sig.parameters.values()
        )

    def _feed_function(self, func, /, *args, **kwargs) -> Any:
        """Calls func with only the keyword arguments that it accepts."""
        if self._has_kwargs(func):
            return func(*args, **kwargs)
        logging.debug("Feeding function %s with args: %s and kwargs: %s", func.__name__, args, list(kwargs.keys()))
        sig = inspect.signature(func)
        # Allow parameters explicitly requested by decorators
        extra_params: set[str] = getattr(func, "_mimic_extra_params", set())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or k in extra_params}
        return func(*args, **filtered_kwargs)

    def __use_cache(self, cached_args: CreationDict, *args, **kwargs) -> bool:
        """Evaluates whether to use cached output based on the provided arguments."""
        cache_invalid: bool = False
        if cached_args and "args" in cached_args and "kwargs" in cached_args:
            cache_args = cached_args["args"]
            cache_kwargs = cached_args["kwargs"]
            for i, vl_1 in enumerate(args):
                if len(cache_args) > i:
                    vl_2 = cache_args[i]
                    if safe_reference_compare(vl_1, vl_2):
                        continue
                cache_invalid = True
                break
            if not cache_invalid:
                for key, vl_1 in kwargs.items():
                    if key in cache_kwargs:
                        vl_2 = cache_kwargs[key]
                        if safe_reference_compare(vl_1, vl_2):
                            continue
                    cache_invalid = True
                    break
        return not cache_invalid

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls._eject_class_params()

    def __new__(cls, *args, **kwargs):
        """
        Sets raw materials value on instance creation.
        """
        instance = super().__new__(cls)
        # Store raw materials for reconstruction
        instance._raw_materials = {"args": list(copy.deepcopy(args)), "kwargs": copy.deepcopy(kwargs)}
        return instance

    def __init__(self) -> None:
        self._save_tensor: Optional[Callable[[Tensor, str], None]] = None
        self._return_cache = False
        self._initialized = False
        self._unsaved_tensors: list[tuple[Tensor, str]] = []
        self._init_args = get_empty_creation_dict()
        self._exec_args = get_empty_creation_dict()
        self._last_output = None
        self._ne_param_cache = get_empty_creation_dict()
        self._init_args_cache = get_empty_creation_dict()
        self._ne_result_cache = (None, {"created_images": [], "last_saved_to_temp": None})

    @abstractmethod
    def _update_impl(self, *args, **kwargs) -> None:
        """Abstract method to update the node."""

    @abstractmethod
    def _process_impl(self, *args, **kwargs) -> T:
        """Abstract method to process data."""
