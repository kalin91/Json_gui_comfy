"""Shared fixtures for json_gui tests."""

import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ====================================================================================
# MOCK ALL HEAVY DEPENDENCIES BEFORE ANY IMPORTS
# ====================================================================================

# Mock tkinter (GUI dependencies)
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.ttk'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()
sys.modules['tkinter.scrolledtext'] = MagicMock()

# Mock safetensors
sys.modules['safetensors'] = MagicMock()
sys.modules['safetensors.torch'] = MagicMock()

# Mock aiohttp and other server deps
sys.modules['aiohttp'] = MagicMock()
sys.modules['aiohttp.web'] = MagicMock()

# Mock segment_anything
sys.modules['segment_anything'] = MagicMock()
sys.modules['segment_anything.build_sam'] = MagicMock()
sys.modules['segment_anything.build_sam'].Sam = MagicMock()

# Mock comfy modules
mock_comfy = MagicMock()
mock_comfy.options.enable_args_parsing = MagicMock()
sys.modules['comfy'] = mock_comfy
sys.modules['comfy.options'] = mock_comfy.options
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['comfy.sample'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.model_patcher'] = MagicMock()
sys.modules['comfy.controlnet'] = MagicMock()

# Mock comfy.samplers with proper values
mock_samplers = MagicMock()
mock_samplers.SAMPLER_NAMES = ["euler", "heun", "dpm"]
mock_samplers.SCHEDULER_NAMES = ["normal", "karras", "exponential"]
mock_samplers.SCHEDULER_HANDLERS = {"normal": MagicMock(), "karras": MagicMock()}
sys.modules['comfy.samplers'] = mock_samplers

sys.modules['comfy_extras'] = MagicMock()
sys.modules['comfy_extras.nodes_sd3'] = MagicMock()
sys.modules['comfy_extras.nodes_images'] = MagicMock()

# Mock custom nodes with proper values
sys.modules['custom_nodes'] = MagicMock()

# Mock Impact Pack
mock_impact_core = MagicMock()
mock_impact_core.ADDITIONAL_SCHEDULERS = ["beta", "ddim_uniform"]
sys.modules['custom_nodes.ComfyUI_Impact_Pack'] = MagicMock()
sys.modules['custom_nodes.ComfyUI_Impact_Pack.modules'] = MagicMock()
sys.modules['custom_nodes.ComfyUI_Impact_Pack.modules.impact'] = MagicMock()
sys.modules['custom_nodes.ComfyUI_Impact_Pack.modules.impact.core'] = mock_impact_core

# Mock FaceDetailer
mock_face_detailer = MagicMock()
mock_face_detailer.INPUT_TYPES = MagicMock(return_value={
    "required": {
        "sam_detection_hint": (["center-1", "center-2", "center-3"], {"default": "center-1"})
    }
})
mock_impact_pack = MagicMock()
mock_impact_pack.FaceDetailer = mock_face_detailer
sys.modules['custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack'] = mock_impact_pack

# Mock ControlNet Aux
sys.modules['custom_nodes.comfyui_controlnet_aux'] = MagicMock()
sys.modules['custom_nodes.comfyui_controlnet_aux.utils'] = MagicMock()
sys.modules['custom_nodes.comfyui_controlnet_aux.src'] = MagicMock()
sys.modules['custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux'] = MagicMock()
sys.modules['custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.open_pose'] = MagicMock()
sys.modules['custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.canny'] = MagicMock()

# Mock Impact Subpack
sys.modules['custom_nodes.ComfyUI_Impact_Subpack'] = MagicMock()
sys.modules['custom_nodes.ComfyUI_Impact_Subpack.modules'] = MagicMock()
sys.modules['custom_nodes.ComfyUI_Impact_Subpack.modules.subpack_nodes'] = MagicMock()

# Mock other modules
sys.modules['nodes'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()
sys.modules['node_helpers'] = MagicMock()
sys.modules['server'] = MagicMock()
sys.modules['utils'] = MagicMock()  # Top-level utils module

# Now safe to import torch
import torch
import numpy as np


@pytest.fixture
def mock_torch_device():
    """Mock torch device operations."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_comfy_model_management():
    """Mock comfy.model_management module."""
    with patch("json_gui.mimic_classes.comfy.model_management.intermediate_device") as mock_device, \
         patch("json_gui.mimic_classes.comfy.model_management.get_torch_device") as mock_get_device, \
         patch("json_gui.mimic_classes.comfy.model_management.soft_empty_cache") as mock_cache, \
         patch("json_gui.mimic_classes.comfy.model_management.unload_all_models") as mock_unload:
        # Return real torch.device objects, not MagicMock
        mock_device.return_value = torch.device("cpu")
        mock_get_device.return_value = torch.device("cpu")
        yield {
            "intermediate_device": mock_device,
            "get_torch_device": mock_get_device,
            "soft_empty_cache": mock_cache,
            "unload_all_models": mock_unload,
        }


@pytest.fixture
def mock_folder_paths():
    """Mock folder_paths module."""
    with patch("folder_paths.get_user_directory") as mock_user_dir, \
         patch("folder_paths.get_input_directory") as mock_input_dir, \
         patch("folder_paths.get_output_directory") as mock_output_dir, \
         patch("folder_paths.get_temp_directory") as mock_temp_dir, \
         patch("folder_paths.get_full_path_or_raise") as mock_full_path, \
         patch("folder_paths.recursive_search") as mock_recursive, \
         patch("folder_paths.filter_files_content_types") as mock_filter, \
         patch("folder_paths.get_filename_list_") as mock_filename_list, \
         patch("folder_paths.get_folder_paths") as mock_folder:
        
        temp_base = tempfile.gettempdir()
        mock_user_dir.return_value = os.path.join(temp_base, "user")
        mock_input_dir.return_value = os.path.join(temp_base, "input")
        mock_output_dir.return_value = os.path.join(temp_base, "output")
        mock_temp_dir.return_value = os.path.join(temp_base, "temp")
        mock_full_path.return_value = "/fake/path/model.safetensors"
        mock_recursive.return_value = ([], [])
        mock_filter.return_value = []
        mock_filename_list.return_value = ([], {"/tmp": []})
        mock_folder.return_value = ["/tmp"]
        
        # Ensure directories exist
        for directory in [mock_user_dir.return_value, mock_input_dir.return_value,
                         mock_output_dir.return_value, mock_temp_dir.return_value]:
            os.makedirs(directory, exist_ok=True)
            os.makedirs(os.path.join(directory, "images"), exist_ok=True)
        
        yield {
            "get_user_directory": mock_user_dir,
            "get_input_directory": mock_input_dir,
            "get_output_directory": mock_output_dir,
            "get_temp_directory": mock_temp_dir,
            "get_full_path_or_raise": mock_full_path,
            "recursive_search": mock_recursive,
            "filter_files_content_types": mock_filter,
            "get_filename_list_": mock_filename_list,
            "get_folder_paths": mock_folder,
        }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    # Create a 1x512x512x3 tensor (BHWC format)
    image = torch.rand(1, 512, 512, 3)
    return image


@pytest.fixture
def mock_pil_image():
    """Mock PIL Image operations."""
    with patch("PIL.Image.open") as mock_open, \
         patch("PIL.Image.fromarray") as mock_fromarray:
        
        mock_img = MagicMock()
        mock_img.mode = "RGB"
        mock_img.convert.return_value = mock_img
        mock_img.rotate.return_value = mock_img
        mock_open.return_value = mock_img
        
        mock_pil = MagicMock()
        mock_pil.save = MagicMock()
        mock_fromarray.return_value = mock_pil
        
        yield {
            "open": mock_open,
            "fromarray": mock_fromarray,
            "image": mock_img,
            "pil": mock_pil,
        }


@pytest.fixture
def mock_node_helpers():
    """Mock node_helpers module."""
    with patch("node_helpers.pillow") as mock_pillow:
        def pillow_wrapper(func, *args, **kwargs):
            return func(*args, **kwargs)
        mock_pillow.side_effect = pillow_wrapper
        yield mock_pillow


@pytest.fixture
def mock_vae():
    """Mock VAE object."""
    vae = MagicMock()
    vae.decode = MagicMock(return_value=torch.rand(1, 3, 512, 512))
    vae.encode = MagicMock(return_value=torch.rand(1, 16, 64, 64))
    return vae


@pytest.fixture
def mock_model_patcher():
    """Mock ModelPatcher object."""
    model = MagicMock()
    model.model = MagicMock()
    model.model.latent_format = MagicMock()
    model.clone = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_controlnet():
    """Mock ControlNet object."""
    controlnet = MagicMock()
    controlnet.copy = MagicMock(return_value=controlnet)
    return controlnet


@pytest.fixture
def mock_comfy_sample():
    """Mock comfy.sample.sample function."""
    with patch("comfy.sample.sample") as mock_sample, \
         patch("comfy.sample.fix_empty_latent_channels") as mock_fix, \
         patch("comfy.sample.prepare_noise") as mock_noise:
        
        # Return a latent tensor
        mock_sample.return_value = torch.rand(1, 16, 64, 64)
        mock_fix.return_value = torch.rand(1, 16, 64, 64)
        mock_noise.return_value = torch.rand(1, 16, 64, 64)
        
        # Mock the signature to accept all parameters
        import inspect
        mock_sig = MagicMock()
        mock_sig.parameters = {
            "model": MagicMock(),
            "noise": MagicMock(),
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": MagicMock(),
            "seed": MagicMock(),
            "steps": MagicMock(),
            "cfg": MagicMock(),
            "sampler_name": MagicMock(),
            "scheduler": MagicMock(),
            "denoise": MagicMock(),
            "disable_noise": MagicMock(),
            "start_step": MagicMock(),
            "last_step": MagicMock(),
            "force_full_denoise": MagicMock(),
            "noise_mask": MagicMock(),
            "callback": MagicMock(),
            "disable_pbar": MagicMock(),
        }
        
        with patch("inspect.signature", return_value=mock_sig):
            yield {
                "sample": mock_sample,
                "fix_empty_latent_channels": mock_fix,
                "prepare_noise": mock_noise,
            }


@pytest.fixture
def mock_controlnet_aux():
    """Mock controlnet_aux utilities and detectors."""
    with patch("custom_nodes.comfyui_controlnet_aux.utils.common_annotator_call") as mock_call, \
         patch("custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.open_pose.OpenposeDetector") as mock_openpose, \
         patch("custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.canny.CannyDetector") as mock_canny:
        
        # Mock annotator call to return tensor
        mock_call.return_value = torch.rand(1, 512, 512, 3)
        
        # Mock detectors
        mock_op_instance = MagicMock()
        mock_op_instance.to.return_value = mock_op_instance
        mock_openpose.from_pretrained.return_value = mock_op_instance
        
        mock_canny_instance = MagicMock()
        mock_canny.return_value = mock_canny_instance
        
        yield {
            "common_annotator_call": mock_call,
            "openpose": mock_openpose,
            "canny": mock_canny,
        }


@pytest.fixture
def mock_impact_pack():
    """Mock Impact Pack components."""
    with patch("custom_nodes.ComfyUI_Impact_Subpack.modules.subpack_nodes.UltralyticsDetectorProvider") as mock_ultra, \
         patch("custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack.SAMLoader") as mock_sam:
        
        # Mock UltralyticsDetectorProvider
        mock_ultra_instance = MagicMock()
        mock_bbox_detector = MagicMock()
        mock_segm_detector = MagicMock()
        mock_ultra_instance.doit.return_value = (mock_bbox_detector, mock_segm_detector)
        mock_ultra.return_value = mock_ultra_instance
        
        # Mock SAMLoader
        mock_sam_instance = MagicMock()
        mock_sam_model = MagicMock()
        mock_sam_instance.load_model.return_value = (mock_sam_model,)
        mock_sam.return_value = mock_sam_instance
        
        yield {
            "ultralytics": mock_ultra,
            "sam_loader": mock_sam,
            "bbox_detector": mock_bbox_detector,
            "sam_model": mock_sam_model,
        }


@pytest.fixture
def mock_checkpoint_loader():
    """Mock checkpoint loading."""
    with patch("comfy.sd.load_checkpoint_guess_config") as mock_load:
        model = MagicMock()
        vae = MagicMock()
        mock_load.return_value = (model, None, vae, None)
        yield mock_load


@pytest.fixture
def mock_controlnet_loader():
    """Mock ControlNet loading."""
    with patch("comfy.controlnet.load_controlnet") as mock_load:
        controlnet = MagicMock()
        mock_load.return_value = controlnet
        yield mock_load


@pytest.fixture
def mock_skip_layer_guidance():
    """Mock SkipLayerGuidanceSD3."""
    with patch("comfy_extras.nodes_sd3.SkipLayerGuidanceSD3.execute") as mock_execute:
        model = MagicMock()
        mock_execute.return_value = (model,)
        yield mock_execute


@pytest.fixture
def mock_tkinter():
    """Mock tkinter components."""
    with patch("tkinter.Tk") as mock_tk, \
         patch("tkinter.Toplevel") as mock_toplevel, \
         patch("tkinter.ttk.Style") as mock_style:
        
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        
        mock_window = MagicMock()
        mock_toplevel.return_value = mock_window
        
        mock_style_instance = MagicMock()
        mock_style_instance.theme_names.return_value = ["clam", "default"]
        mock_style.return_value = mock_style_instance
        
        yield {
            "Tk": mock_tk,
            "Toplevel": mock_toplevel,
            "Style": mock_style,
            "root": mock_root,
            "window": mock_window,
        }
