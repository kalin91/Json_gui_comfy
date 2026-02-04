"""Utility functions for JSON GUI management."""

import os
import re
import types
import logging
import shutil
from functools import partial
from abc import ABC, abstractmethod
from typing import Any, Callable
import folder_paths
import torch
from PIL import Image
import numpy as np

from json_gui.typedicts import SaveImageCallable, SavedImagesDict


def is_unserializable_callable(obj: Any) -> bool:
    """
    Check if obj is a callable that cannot be pickled.

    Lambdas, local functions, and closures typically cannot be pickled
    because they reference local scope that pickle cannot capture.
    """

    if not callable(obj):
        return False

    # Check if it's a lambda (name is '<lambda>')
    if isinstance(obj, types.FunctionType):
        if obj.__name__ == "<lambda>":
            return True
        # Check if it's a local/nested function (has '<locals>' in qualname)
        if obj.__qualname__ and "<locals>" in obj.__qualname__:
            return True

    # Check for bound methods with unserializable functions
    if isinstance(obj, types.MethodType):
        return is_unserializable_callable(obj.__func__)

    # partial objects wrapping unserializable functions
    if isinstance(obj, partial):
        return is_unserializable_callable(obj.func)

    return False


def get_main_images_path() -> str:
    """Returns the path to the main images directory."""

    ret_path: str = os.path.join(folder_paths.get_user_directory(), "images")
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    return ret_path


def get_scripts_folder_path() -> str:
    """Returns the path to the scripts folder."""
    scripts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scripts")
    if not os.path.exists(scripts_path):
        os.makedirs(scripts_path)
    return scripts_path


def get_flow_and_body_paths(script_name: str) -> tuple[str, str]:
    """
    Returns a tuple containing the paths to the 'flow.py' and 'body.yml' files
    within the specified script directory.

    Args:
        script_name (str): The name of the script directory.

    Returns:
        tuple[str, str]: Paths to 'flow.py' and 'body.yml' within the script directory.

    Raises:
        AssertionError: If the script directory or either file does not exist.
    """
    script_dir = os.path.join(get_scripts_folder_path(), script_name)
    # Verify that the script dir exists and is a directory
    assert os.path.isdir(script_dir), f"Script directory {script_name} does not exist."
    flow = os.path.join(script_dir, "flow.py")
    body = os.path.join(script_dir, "body.yml")

    # Verify that flow and body exists and are files
    assert os.path.isfile(flow), f"Flow script {flow} does not exist."
    assert os.path.isfile(body), f"Body file {body} does not exist."
    return flow, body


def get_input_files_recursive() -> tuple[list[str], str]:
    """Returns a list of input files filtered by content types."""
    input_folder = folder_paths.get_input_directory()
    output_list = set()
    files, _ = folder_paths.recursive_search(input_folder, excluded_dir_names=[".git"])
    output_list.update(folder_paths.filter_files_content_types(files, ["image"]))
    return sorted(output_list), input_folder


def _get_output_files_recursive() -> tuple[list[str], str]:
    """Returns a list of output files filtered by content types."""
    output_folder = folder_paths.get_output_directory()
    files, _ = folder_paths.recursive_search(output_folder, excluded_dir_names=[".git"])
    return sorted(files), output_folder


def get_folder_files_recursive(folder: str) -> tuple[list[str], str]:
    """Retrieves the list of filenames and the directory they are located in."""
    input_dir = folder_paths.get_filename_list_(folder)
    result: tuple[list[str], str] = input_dir[0], next(iter(input_dir[1].keys()))
    logging.debug("Input directory for %s; folder %s; files: %s", folder, result[1], result[0])
    return sorted(result[0]), result[1]


def save_image(
    data: SavedImagesDict,
    images: torch.Tensor,
    identifier: str,
    steps: int,  # from flow instance
    file_identifier: str,  # from flow instance
    is_temp: bool = True,
) -> None:
    """
    saves images to disk with unique filenames based on identifier and file_identifier.
    If the number of saved images reaches 'steps', raises EndOfFlowException.
    Args:
        data (SavedImagesDict):
            created_images: list[str] - list of file paths of created images
            last_saved_to_temp: bool - indicates if the last save was to temp directory. Is overwritten on each save
        images (torch.Tensor): the image tensor to save
        identifier (str): unique identifier for the image
        steps (int): number of steps for the flow
        file_identifier (str): unique file identifier for the flow
        is_temp (bool): whether to save to temp directory or output directory

    Raises:
        RuntimeError: if unable to save image after multiple attempts.
        EndOfFlowException: if the number of saved images reaches 'steps'.
    """

    assert isinstance(data, dict) and len(data) == 2
    # assert keys present
    assert (
        "last_saved_to_temp" in data and "created_images" in data
    ), "Data dict must contain 'last_saved_to_temp' and 'created_images' keys."
    created_images = data["created_images"]
    j: int = len(created_images)
    output_dir = folder_paths.get_temp_directory() if is_temp else folder_paths.get_output_directory()
    file_saved: bool = False
    while not file_saved and j < 40:
        for image in images:
            sampler_file_name = os.path.join(output_dir, f"{file_identifier}_{identifier}_{j}.png")
            if os.path.exists(sampler_file_name):
                j += 1
                continue  # Skip if already exists
            if isinstance(image, torch.Tensor) and image.requires_grad:
                image = image.detach()
            img_np = 255.0 * image.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            img_pil.save(sampler_file_name)
            logging.info("Saved refiner output to %s", sampler_file_name)
            file_saved = True
            created_images.append(sampler_file_name)
            break
    if not file_saved:
        raise RuntimeError("Failed to save refiner output after multiple attempts. clean up temp files.")
    data["last_saved_to_temp"] = is_temp
    if len(created_images) >= steps:
        raise EndOfFlowException(steps)


def copy_images(
    steps: int, data: SavedImagesDict, file_identifier: str, regex_pattern: str, new_data: SavedImagesDict
) -> None:
    """

    Args:
        data (SavedImagesDict):
            created_images: list[str] - list of file paths of created images originally
            last_saved_to_temp: bool - indicates if the last save was to temp directory. Is overwritten by new_data
        steps (int): number of steps for the flow
        source_paths (list[str]): list of source image file paths
        file_identifier (str): unique file identifier for the flow
        regex_pattern (str):
        data (SavedImagesDict):
            created_images: list[str] - list of file paths of created images during node spawn execution
            last_saved_to_temp: bool - indicates if the last save was to temp directory. Overrides data's value

    Raises:
        EndOfFlowException:

    Returns:
        list[str]:
    """
    created_images = data["created_images"]
    source: list[str] = new_data["created_images"]
    idx = len(created_images)
    new_images = source[idx:]
    pre_paths: list[tuple[str, str]] = [os.path.split(path) for path in new_images]
    parents, files = zip(*pre_paths) if pre_paths else ([], [])
    copied_f_names: list[str] = [re.sub(regex_pattern, file_identifier, f_name) for f_name in files]
    copied_paths: list[str] = [os.path.join(folder, f_name) for folder, f_name in zip(parents, copied_f_names)]
    for src, dst in zip(new_images, copied_paths):
        shutil.copy2(src, dst)
        logging.info("Copied image from %s to %s", src, dst)
    created_images.extend(copied_paths)
    data["last_saved_to_temp"] = new_data["last_saved_to_temp"]
    if len(created_images) >= steps:
        raise EndOfFlowException(steps)


class AbsFlow(ABC):
    """Abstract base class for flow implementations."""

    @property
    def filename(self) -> str:
        """Returns the file identifier associated with the flow."""
        return self._filename

    @property
    def json_path(self) -> str:
        """Returns the JSON path associated with the flow."""
        return self._json_path

    @property
    def file_path(self) -> str:
        """Returns the file identifier associated with the flow."""
        return self._file_path

    @property
    def save_image(self) -> SaveImageCallable:
        """Returns the save image callback."""
        return self._save_image

    @property
    def copy_images(self) -> Callable[[SavedImagesDict], None]:
        """Returns the copy images callback."""
        return self._copy_images

    @property
    def saved_data(self) -> SavedImagesDict:
        """Returns the saved data dictionary."""
        return self._saved_data

    def __init__(self, file_path: str, filename: str) -> None:
        """Initializes the AbsFlow instance."""
        self._filename: str = filename
        self._saved_data: SavedImagesDict = {"last_saved_to_temp": None, "created_images": []}
        self._file_path = file_path
        self._json_path = os.path.join(get_main_images_path(), file_path, f"{filename}.json")
        self.set_file_vars()

    def set_file_vars(self, steps=1) -> None:
        """Sets up file identifier and related callbacks."""
        filename = self._filename
        files, folder = _get_output_files_recursive()
        idx = 0
        if files:
            pattern: str = filename + r"_r(\d+)\.json$"
            # Find all matching files and extract the max index
            indexes = [int(m.group(1)) for f in files if (m := re.search(pattern, os.path.basename(f))) is not None]
            if indexes:
                idx = max(indexes) + 1
        self._file_identifier = f"{filename}_r{idx}"

        self._save_image: SaveImageCallable = partial(save_image, steps=steps, file_identifier=self._file_identifier)
        self._copy_regex_pattern = rf"^{filename}_r\d+"
        self._copy_images: Callable[[SavedImagesDict], None] = partial(
            copy_images,
            steps,
            self._saved_data,
            self._file_identifier,
            self._copy_regex_pattern,
        )

        # delete any output files with this identifier
        for f in files:
            if self._file_identifier in f:
                os.remove(f)
                logging.info("Deleted existing output file: %s", f)

        # delete any temp files with this identifier
        files, _ = folder_paths.recursive_search(folder_paths.get_temp_directory(), excluded_dir_names=[".git"])
        for f in files:
            if self._file_identifier in f:
                os.remove(os.path.join(folder, f))
                logging.info("Deleted existing temp file: %s", f)

    def run(self, steps: int, multiprocess: bool) -> list[str]:
        """
        Runs the flow and returns a list of created image file paths.
        Args:
            steps (int): Number of steps to run the flow.
            multiprocess (bool): Whether to use multiprocessing.

        Returns:
            list[str]: List of created image file paths.
        """
        # Saving a copy of json file to output directory
        self._saved_data["created_images"].clear()
        self.set_file_vars(steps)
        output_json_path = os.path.join(folder_paths.get_output_directory(), f"{self._file_identifier}.json")
        shutil.copy2(self._json_path, output_json_path)
        logging.info("Saved flow JSON to output directory: %s", output_json_path)
        with torch.inference_mode():
            try:
                self._run_impl(steps, multiprocess)
            except EndOfFlowException as eofe:
                logging.info("Flow ended early after %d steps.", eofe.steps)
        if self._saved_data["last_saved_to_temp"] is True:
            # Copy last saved image to output directory
            last_image_path = self._saved_data["created_images"][-1]
            img_filename = os.path.basename(last_image_path)
            output_image_path = os.path.join(folder_paths.get_output_directory(), img_filename)
            shutil.copy2(last_image_path, output_image_path)
            logging.info("Copied final image to output directory: %s", output_image_path)
            self._saved_data["created_images"][-1] = output_image_path
        return self._saved_data["created_images"]

    @abstractmethod
    def _run_impl(self, steps: int, multiprocess: bool) -> None:
        """Runs the flow and returns a list of created image file paths."""


class EndOfFlowException(Exception):
    """Custom exception to indicate the end of a flow process."""

    @property
    def result(self) -> Any:
        """Returns the number of steps after which the flow ended."""
        return self._result

    @result.setter
    def result(self, value: Any) -> None:
        self._result = value

    def __init__(self, steps: int) -> None:
        self.steps = steps
        super().__init__(f"End of flow after reaching {steps} steps limit.")
