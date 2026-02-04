"""Applies custom folder paths based on command-line arguments and configuration files."""

import itertools
import logging
import os
import shutil
import multiprocessing as mp
import folder_paths
import utils.extra_config
from comfy.cli_args import args
import json_gui.p_logger as _  # noqa: F401


def cleanup_temp() -> None:
    """Cleans up the temporary directory used during processing."""
    if mp.current_process().name == "MainProcess":
        temp_dir = folder_paths.get_temp_directory()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def apply_custom_paths() -> None:
    """Applies custom folder paths based on command-line arguments and configuration files."""

    if args.temp_directory:
        temp_dir = os.path.abspath(args.temp_directory)
        logging.info("Setting temp directory to: %s", temp_dir)
        folder_paths.set_temp_directory(temp_dir)
    else:
        logging.info("Using default temp directory: %s", folder_paths.get_temp_directory())
    cleanup_temp()
    # extra model paths
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    # --output-directory, --input-directory, --user-directory
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info("Setting output directory to: %s", output_dir)
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints,
    # clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path(
        "diffusion_models", os.path.join(folder_paths.get_output_directory(), "diffusion_models")
    )
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))
    folder_paths.add_model_folder_path("sams", "/data/home2/kalin/models/sams")
    folder_paths.add_model_folder_path("ultralytics", "/data/home2/kalin/models/ultralytics")
    folder_paths.add_model_folder_path("ultralytics_bbox", "/data/home2/kalin/models/ultralytics/bbox")
    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info("Setting input directory to: %s", input_dir)
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info("Setting user directory to: %s", user_dir)
        folder_paths.set_user_directory(user_dir)

    # verify directories exist
    for dir_path in [
        folder_paths.get_temp_directory(),
        folder_paths.get_output_directory(),
        folder_paths.get_input_directory(),
        folder_paths.get_user_directory(),
    ]:
        os.makedirs(dir_path, exist_ok=True)
