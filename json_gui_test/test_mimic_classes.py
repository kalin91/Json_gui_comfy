"""Tests for json_gui/mimic_classes.py module."""

import os
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, call
from PIL import Image

import json_gui.mimic_classes as mimic_classes


class TestSimpleKSampler:
    """Test SimpleKSampler class."""
    
    def test_initialization(self):
        """Test SimpleKSampler initialization."""
        sampler = mimic_classes.SimpleKSampler(
            seed=42,
            steps=20,
            cfg=7.5,
            sampler_name="euler",
            scheduler="normal",
            denoise=1.0,
            use_tune=False
        )
        
        assert sampler.seed == 42
        assert sampler.steps == 20
        assert sampler.cfg == 7.5
        assert sampler.sampler_name == "euler"
        assert sampler.scheduler == "normal"
        assert sampler.denoise == 1.0
        assert sampler.use_tune is False
    
    def test_to_dict(self):
        """Test _to_dict method."""
        sampler = mimic_classes.SimpleKSampler(
            seed=123,
            steps=25,
            cfg=8.0,
            sampler_name="heun",
            scheduler="karras",
            denoise=0.8,
            use_tune=True
        )
        
        result = sampler._to_dict()
        
        assert result["seed"] == 123
        assert result["steps"] == 25
        assert result["cfg"] == 8.0
        assert result["sampler_name"] == "heun"
        assert result["scheduler"] == "karras"
        assert result["denoise"] == 0.8
    
    def test_process(self, mock_model_patcher):
        """Test process method."""
        sampler = mimic_classes.SimpleKSampler(
            seed=42,
            steps=20,
            cfg=7.5,
            sampler_name="euler",
            scheduler="normal",
            denoise=1.0,
            use_tune=False
        )
        
        latent_image = torch.rand(1, 16, 64, 64)
        cond_pos = MagicMock()
        cond_neg = MagicMock()
        
        # Mock inspect.signature at the module level
        with patch("json_gui.mimic_classes.inspect.signature") as mock_sig, \
             patch("json_gui.mimic_classes.fix_empty_latent_channels") as mock_fix, \
             patch("json_gui.mimic_classes.prepare_noise") as mock_noise:
            
            # Setup mocks
            mock_fix.return_value = latent_image
            mock_noise.return_value = torch.rand(1, 16, 64, 64)
            
            # Mock signature to accept all args - use a real signature-like object
            from inspect import Parameter, Signature
            params = [
                Parameter("model", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("noise", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("positive", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("negative", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("latent_image", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("seed", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("steps", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("cfg", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("sampler_name", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("scheduler", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("denoise", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("disable_noise", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("start_step", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("last_step", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("force_full_denoise", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("noise_mask", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("callback", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("disable_pbar", Parameter.POSITIONAL_OR_KEYWORD),
            ]
            mock_sig.return_value = Signature(params)
            
            result = sampler.process(latent_image, mock_model_patcher, cond_pos, cond_neg)
            
            # Result should be something (even if mocked from comfy.sample.sample)
            assert result is not None


class TestEmptyLatent:
    """Test EmptyLatent class."""
    
    def test_initialization(self):
        """Test EmptyLatent initialization."""
        latent = mimic_classes.EmptyLatent(width=512, height=768, batch_size=2)
        
        assert latent.width == 512
        assert latent.height == 768
        assert latent.batch_size == 2
    
    def test_latent_property(self, mock_comfy_model_management):
        """Test latent property generates correct tensor."""
        latent = mimic_classes.EmptyLatent(width=512, height=512, batch_size=1)
        
        result = latent.latent
        
        # Check tensor shape: [batch_size, channels=16, height//8, width//8]
        assert result.shape == (1, 16, 64, 64)
        assert result.device.type == "cpu"
    
    def test_latent_with_different_dimensions(self, mock_comfy_model_management):
        """Test latent with different dimensions."""
        latent = mimic_classes.EmptyLatent(width=1024, height=768, batch_size=2)
        
        result = latent.latent
        
        # Check tensor shape
        assert result.shape == (2, 16, 96, 128)


class TestControlNetImgPreprocessor:
    """Test ControlNetImgPreprocessor abstract base class."""
    
    def test_skip_returns_none(self):
        """Test that skip=True returns None."""
        result = mimic_classes.ControlNetImgPreprocessor.__new__(
            mimic_classes.ControlNetImgPreprocessor,
            image_name="test.png",
            skip=True
        )
        assert result is None
    
    def test_concrete_implementation_initialization(self, mock_folder_paths, mock_node_helpers, temp_directory):
        """Test concrete implementation initialization."""
        
        class ConcretePreprocessor(mimic_classes.ControlNetImgPreprocessor):
            @property
            def controlnet_path(self) -> str:
                return "test_controlnet.safetensors"
            
            def _tensor_impl(self, cnet_img: torch.Tensor) -> torch.Tensor:
                return cnet_img
        
        # Create test image
        input_dir = mock_folder_paths["get_input_directory"].return_value
        os.makedirs(input_dir, exist_ok=True)
        image_path = os.path.join(input_dir, "test.png")
        
        # Create a real test image
        img = Image.new("RGB", (512, 512), color="red")
        img.save(image_path)
        
        preprocessor = ConcretePreprocessor(image_name="test.png", skip=False)
        
        assert preprocessor.skip is False
        assert preprocessor._controlnet_img is not None
        assert isinstance(preprocessor._controlnet_img, torch.Tensor)
    
    def test_tensor_method(self, mock_folder_paths, mock_node_helpers, temp_directory):
        """Test tensor method calls _tensor_impl."""
        
        class ConcretePreprocessor(mimic_classes.ControlNetImgPreprocessor):
            @property
            def controlnet_path(self) -> str:
                return "test_controlnet.safetensors"
            
            def _tensor_impl(self, cnet_img: torch.Tensor) -> torch.Tensor:
                return cnet_img * 2
        
        # Create test image
        input_dir = mock_folder_paths["get_input_directory"].return_value
        os.makedirs(input_dir, exist_ok=True)
        image_path = os.path.join(input_dir, "test.png")
        img = Image.new("RGB", (512, 512), color="red")
        img.save(image_path)
        
        preprocessor = ConcretePreprocessor(image_name="test.png", skip=False)
        result = preprocessor.tensor()
        
        # Result should be doubled
        assert torch.allclose(result, preprocessor._controlnet_img * 2)
    
    def test_save_tensor_callback(self, mock_folder_paths, mock_node_helpers, temp_directory):
        """Test save_tensor callback is called."""
        
        class ConcretePreprocessor(mimic_classes.ControlNetImgPreprocessor):
            @property
            def controlnet_path(self) -> str:
                return "test_controlnet.safetensors"
            
            def _tensor_impl(self, cnet_img: torch.Tensor) -> torch.Tensor:
                return cnet_img
        
        # Create test image
        input_dir = mock_folder_paths["get_input_directory"].return_value
        os.makedirs(input_dir, exist_ok=True)
        image_path = os.path.join(input_dir, "test.png")
        img = Image.new("RGB", (512, 512), color="red")
        img.save(image_path)
        
        preprocessor = ConcretePreprocessor(image_name="test.png", skip=False)
        
        # Set save_tensor callback
        callback = MagicMock()
        preprocessor.save_tensor = callback
        
        result = preprocessor.tensor()
        
        # Callback should be called with result
        callback.assert_called_once_with(result)


class TestApplyControlNet:
    """Test ApplyControlNet class."""
    
    def test_initialization(self, mock_folder_paths, mock_node_helpers, temp_directory):
        """Test ApplyControlNet initialization."""
        
        class MockPreprocessor(mimic_classes.ControlNetImgPreprocessor):
            @property
            def controlnet_path(self) -> str:
                return "test_controlnet.safetensors"
            
            def _tensor_impl(self, cnet_img: torch.Tensor) -> torch.Tensor:
                return cnet_img
        
        # Create test image
        input_dir = mock_folder_paths["get_input_directory"].return_value
        os.makedirs(input_dir, exist_ok=True)
        image_path = os.path.join(input_dir, "test.png")
        img = Image.new("RGB", (512, 512), color="red")
        img.save(image_path)
        
        target = MockPreprocessor(image_name="test.png", skip=False)
        
        apply_cn = mimic_classes.ApplyControlNet(
            strength=0.8,
            start_percentage=0.0,
            end_percentage=1.0,
            target=target
        )
        
        assert apply_cn.strength == 0.8
        assert apply_cn.start_percentage == 0.0
        assert apply_cn.end_percentage == 1.0
        assert apply_cn.target == target
    
    def test_conditionals(self, mock_folder_paths, mock_node_helpers, temp_directory):
        """Test conditionals method."""
        
        class MockPreprocessor(mimic_classes.ControlNetImgPreprocessor):
            @property
            def controlnet_path(self) -> str:
                return "test_controlnet.safetensors"
            
            def _tensor_impl(self, cnet_img: torch.Tensor) -> torch.Tensor:
                return cnet_img
        
        # Create test image
        input_dir = mock_folder_paths["get_input_directory"].return_value
        os.makedirs(input_dir, exist_ok=True)
        image_path = os.path.join(input_dir, "test.png")
        img = Image.new("RGB", (512, 512), color="red")
        img.save(image_path)
        
        target = MockPreprocessor(image_name="test.png", skip=False)
        
        apply_cn = mimic_classes.ApplyControlNet(
            strength=0.8,
            start_percentage=0.0,
            end_percentage=1.0,
            target=target
        )
        
        with patch("json_gui.mimic_classes.load_controlnet") as mock_load_cn, \
             patch("json_gui.mimic_classes.ControlNetApplyAdvanced") as mock_cn_apply, \
             patch("json_gui.mimic_classes.comfy.model_management.soft_empty_cache"):
            
            mock_controlnet = MagicMock()
            mock_load_cn.return_value = mock_controlnet
            
            mock_cn_instance = MagicMock()
            expected_result = ("cond_pos_result", "cond_neg_result")
            mock_cn_instance.apply_controlnet.return_value = expected_result
            mock_cn_apply.return_value = mock_cn_instance
            
            cond_pos = MagicMock()
            cond_neg = MagicMock()
            vae = MagicMock()
            
            result = apply_cn.conditionals(cond_pos, cond_neg, vae)
            
            assert result == expected_result
            mock_cn_instance.apply_controlnet.assert_called_once()


class TestOpenPosePose:
    """Test OpenPosePose class."""
    
    def test_initialization(self, mock_folder_paths, mock_node_helpers, temp_directory):
        """Test OpenPosePose initialization."""
        # Create test image
        input_dir = mock_folder_paths["get_input_directory"].return_value
        os.makedirs(input_dir, exist_ok=True)
        image_path = os.path.join(input_dir, "test.png")
        img = Image.new("RGB", (512, 512), color="red")
        img.save(image_path)
        
        openpose = mimic_classes.OpenPosePose(
            image_name="test.png",
            detect_body=True,
            detect_hands=False,
            detect_face=True,
            scale_stick_for_xinsr_cn=False,
            resolution=512,
            controlnet_path="openpose.safetensors",
            skip=False
        )
        
        assert openpose.detect_body is True
        assert openpose.detect_hands is False
        assert openpose.detect_face is True
        assert openpose.scale_stick_for_xinsr_cn is False
        assert openpose.resolution == 512
        assert openpose.controlnet_path == "openpose.safetensors"
    
    def test_tensor_impl(self, mock_folder_paths, mock_node_helpers, temp_directory):
        """Test _tensor_impl method."""
        # Create test image
        input_dir = mock_folder_paths["get_input_directory"].return_value
        os.makedirs(input_dir, exist_ok=True)
        image_path = os.path.join(input_dir, "test.png")
        img = Image.new("RGB", (512, 512), color="red")
        img.save(image_path)
        
        with patch("json_gui.mimic_classes.aux_utils.common_annotator_call") as mock_call, \
             patch("json_gui.mimic_classes.OpenposeDetector.from_pretrained") as mock_op, \
             patch("json_gui.mimic_classes.comfy.model_management.get_torch_device") as mock_device, \
             patch("json_gui.mimic_classes.comfy.model_management.soft_empty_cache"):
            
            # Setup mocks
            mock_call.return_value = torch.rand(1, 512, 512, 3)
            mock_op_instance = MagicMock()
            mock_op_instance.to.return_value = mock_op_instance
            mock_op.return_value = mock_op_instance
            mock_device.return_value = torch.device("cpu")
            
            openpose = mimic_classes.OpenPosePose(
                image_name="test.png",
                detect_body=True,
                detect_hands=True,
                detect_face=True,
                scale_stick_for_xinsr_cn=True,
                resolution=512,
                controlnet_path="openpose.safetensors",
                skip=False
            )
            
            result = openpose._tensor_impl(openpose._controlnet_img)
            
            # Verify common_annotator_call was called with correct parameters
            mock_call.assert_called_once()
            call_args = mock_call.call_args[1]
            assert call_args["include_hand"] is True
            assert call_args["include_face"] is True
            assert call_args["include_body"] is True
            assert call_args["xinsr_stick_scaling"] is True
            assert call_args["resolution"] == 512
            
            assert isinstance(result, torch.Tensor)


class TestCannyEdge:
    """Test CannyEdge class."""
    
    def test_initialization(self, mock_folder_paths, mock_node_helpers, temp_directory):
        """Test CannyEdge initialization."""
        # Create test image
        input_dir = mock_folder_paths["get_input_directory"].return_value
        os.makedirs(input_dir, exist_ok=True)
        image_path = os.path.join(input_dir, "test.png")
        img = Image.new("RGB", (512, 512), color="red")
        img.save(image_path)
        
        canny = mimic_classes.CannyEdge(
            image_name="test.png",
            low_threshold=100,
            high_threshold=200,
            resolution=512,
            controlnet_path="canny.safetensors",
            skip=False
        )
        
        assert canny.low_threshold == 100
        assert canny.high_threshold == 200
        assert canny.resolution == 512
        assert canny.controlnet_path == "canny.safetensors"
    
    def test_tensor_impl(self, mock_folder_paths, mock_node_helpers, temp_directory):
        """Test _tensor_impl method."""
        # Create test image
        input_dir = mock_folder_paths["get_input_directory"].return_value
        os.makedirs(input_dir, exist_ok=True)
        image_path = os.path.join(input_dir, "test.png")
        img = Image.new("RGB", (512, 512), color="red")
        img.save(image_path)
        
        with patch("json_gui.mimic_classes.aux_utils.common_annotator_call") as mock_call, \
             patch("json_gui.mimic_classes.CannyDetector") as mock_canny, \
             patch("comfy_extras.nodes_images.ResizeAndPadImage") as mock_resize_class:
            
            # Setup mocks to return proper tensor
            result_tensor = torch.rand(1, 512, 512, 3)
            mock_call.return_value = result_tensor
            mock_canny_instance = MagicMock()
            mock_canny.return_value = mock_canny_instance
            
            # Mock resize to return tensor directly
            mock_resize_instance = MagicMock()
            mock_resize_instance.resize_and_pad.return_value = (result_tensor,)
            mock_resize_class.return_value = mock_resize_instance
            
            canny = mimic_classes.CannyEdge(
                image_name="test.png",
                low_threshold=100,
                high_threshold=200,
                resolution=512,
                controlnet_path="canny.safetensors",
                skip=False
            )
            
            result = canny._tensor_impl(canny._controlnet_img)
            
            # Verify common_annotator_call was called with correct parameters
            mock_call.assert_called_once()
            call_args = mock_call.call_args[1]
            assert call_args["low_threshold"] == 100
            assert call_args["high_threshold"] == 200
            assert call_args["resolution"] == 512
            
            # Result should be a tensor (either from mock_call or mock_resize)
            assert result is not None


class TestFaceDetailer:
    """Test FaceDetailer class."""
    
    def test_initialization(self):
        """Test FaceDetailer initialization."""
        with patch("json_gui.mimic_classes.comfy.model_management.soft_empty_cache"), \
             patch("json_gui.mimic_classes.UltralyticsDetectorProvider") as mock_ultra, \
             patch("json_gui.mimic_classes.SAMLoader") as mock_sam:
            
            # Setup mocks to return proper tuples
            mock_bbox_detector = MagicMock()
            mock_segm_detector = MagicMock()
            mock_ultra_instance = MagicMock()
            mock_ultra_instance.doit.return_value = (mock_bbox_detector, mock_segm_detector)
            mock_ultra.return_value = mock_ultra_instance
            
            mock_sam_model = MagicMock()
            mock_sam_instance = MagicMock()
            mock_sam_instance.load_model.return_value = (mock_sam_model,)
            mock_sam.return_value = mock_sam_instance
            
            face_detailer = mimic_classes.FaceDetailer(
                seed=42,
                steps=20,
                cfg=7.5,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.5,
                guide_size=512,
                guide_size_for=False,
                max_size=1024,
                feather=5,
                noise_mask=True,
                force_inpaint=False,
                drop_size=10,
                cycle=1,
                bbox_threshold=0.5,
                bbox_dilation=10,
                bbox_crop_factor=3.0,
                sam_detection_hint="center-1",
                sam_dilation=0,
                sam_threshold=0.93,
                sam_bbox_expansion=0,
                sam_mask_hint_threshold=0.7,
                sam_mask_hint_use_negative="false",
                bbox_detector="bbox/face_yolov8m.pt",
                sam_model_opt="sam_vit_b_01ec64.pth",
                wildcard="",
                use_tune=False
            )
            
            # Test inherited properties from SimpleKSampler
            assert face_detailer.seed == 42
            assert face_detailer.steps == 20
            assert face_detailer.cfg == 7.5
            
            # Test FaceDetailer-specific properties
            assert face_detailer.guide_size == 512
            assert face_detailer.bbox_threshold == 0.5
            assert face_detailer.sam_detection_hint == "center-1"
            assert face_detailer.wildcard == ""
    
    def test_to_dict(self):
        """Test to_dict method includes all parameters."""
        with patch("json_gui.mimic_classes.comfy.model_management.soft_empty_cache"), \
             patch("json_gui.mimic_classes.UltralyticsDetectorProvider") as mock_ultra, \
             patch("json_gui.mimic_classes.SAMLoader") as mock_sam:
            
            # Setup mocks to return proper tuples
            mock_bbox_detector = MagicMock()
            mock_segm_detector = MagicMock()
            mock_ultra_instance = MagicMock()
            mock_ultra_instance.doit.return_value = (mock_bbox_detector, mock_segm_detector)
            mock_ultra.return_value = mock_ultra_instance
            
            mock_sam_model = MagicMock()
            mock_sam_instance = MagicMock()
            mock_sam_instance.load_model.return_value = (mock_sam_model,)
            mock_sam.return_value = mock_sam_instance
            
            face_detailer = mimic_classes.FaceDetailer(
                seed=42,
                steps=20,
                cfg=7.5,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.5,
                guide_size=512,
                guide_size_for=False,
                max_size=1024,
                feather=5,
                noise_mask=True,
                force_inpaint=False,
                drop_size=10,
                cycle=1,
                bbox_threshold=0.5,
                bbox_dilation=10,
                bbox_crop_factor=3.0,
                sam_detection_hint="center-1",
                sam_dilation=0,
                sam_threshold=0.93,
                sam_bbox_expansion=0,
                sam_mask_hint_threshold=0.7,
                sam_mask_hint_use_negative="false",
                bbox_detector="bbox/face_yolov8m.pt",
                sam_model_opt="sam_vit_b_01ec64.pth",
                wildcard="test",
                use_tune=False
            )
            
            result = face_detailer.to_dict()
            
            # Test base properties
            assert result["seed"] == 42
            assert result["steps"] == 20
            assert result["cfg"] == 7.5
            
            # Test FaceDetailer properties
            assert result["guide_size"] == 512
            assert result["bbox_threshold"] == 0.5
            assert result["sam_detection_hint"] == "center-1"
            assert result["wildcard"] == "test"
            assert "sam_model_opt" in result
            assert "bbox_detector" in result


class TestRotator:
    """Test Rotator class."""
    
    def test_initialization(self):
        """Test Rotator initialization."""
        rotator = mimic_classes.Rotator(angle=45.0)
        assert rotator.angle == 45.0
    
    def test_rotate_image_zero_angle(self):
        """Test that zero angle doesn't rotate."""
        rotator = mimic_classes.Rotator(angle=0.0)
        
        image = torch.rand(1, 512, 512, 3)
        func = MagicMock(return_value=image)
        
        result = rotator.rotate_image(image, func)
        
        # Function should be called with original image
        func.assert_called_once_with(image)
        assert torch.equal(result, image)
    
    def test_rotate_image_non_zero_angle(self):
        """Test rotation with non-zero angle."""
        rotator = mimic_classes.Rotator(angle=90.0)
        
        image = torch.rand(1, 512, 512, 3)
        
        # Mock function that returns same dimensions
        def mock_func(img):
            return img
        
        result = rotator.rotate_image(image, mock_func)
        
        # Result should have same shape as input
        assert result.shape == image.shape


class TestSkipLayers:
    """Test SkipLayers class."""
    
    def test_initialization(self):
        """Test SkipLayers initialization."""
        with patch("json_gui.mimic_classes.comfy.model_management.unload_all_models"), \
             patch("json_gui.mimic_classes.comfy.model_management.soft_empty_cache"), \
             patch("json_gui.mimic_classes.load_checkpoint_guess_config") as mock_load, \
             patch("json_gui.mimic_classes.SkipLayerGuidanceSD3.execute") as mock_skip:
            
            # Setup mocks
            model = MagicMock()
            vae = MagicMock()
            mock_load.return_value = (model, None, vae, None)
            
            tuned_model = MagicMock()
            mock_skip.return_value = (tuned_model,)
            
            skip_layers = mimic_classes.SkipLayers(
                layers=[0, 1, 2],
                scale=0.5,
                start_percent=0.0,
                end_percent=1.0
            )
            
            # Verify checkpoint was loaded
            mock_load.assert_called_once()
            
            # Verify SkipLayerGuidanceSD3 was executed
            mock_skip.assert_called_once()
    
    def test_get_model_tuned(self):
        """Test get_model returns tuned model when use_tuned=True."""
        with patch("json_gui.mimic_classes.comfy.model_management.unload_all_models"), \
             patch("json_gui.mimic_classes.comfy.model_management.soft_empty_cache"), \
             patch("json_gui.mimic_classes.load_checkpoint_guess_config") as mock_load, \
             patch("json_gui.mimic_classes.SkipLayerGuidanceSD3.execute") as mock_skip:
            
            # Setup mocks
            model = MagicMock()
            vae = MagicMock()
            mock_load.return_value = (model, None, vae, None)
            
            tuned_model = MagicMock()
            mock_skip.return_value = (tuned_model,)
            
            skip_layers = mimic_classes.SkipLayers(
                layers=[0, 1, 2],
                scale=0.5,
                start_percent=0.0,
                end_percent=1.0
            )
            
            result = skip_layers.get_model(use_tuned=True)
            assert result == skip_layers._tunned_model
    
    def test_get_model_base(self):
        """Test get_model returns base model when use_tuned=False."""
        with patch("json_gui.mimic_classes.comfy.model_management.unload_all_models"), \
             patch("json_gui.mimic_classes.comfy.model_management.soft_empty_cache"), \
             patch("json_gui.mimic_classes.load_checkpoint_guess_config") as mock_load, \
             patch("json_gui.mimic_classes.SkipLayerGuidanceSD3.execute") as mock_skip:
            
            # Setup mocks
            model = MagicMock()
            vae = MagicMock()
            mock_load.return_value = (model, None, vae, None)
            
            tuned_model = MagicMock()
            mock_skip.return_value = (tuned_model,)
            
            skip_layers = mimic_classes.SkipLayers(
                layers=[0, 1, 2],
                scale=0.5,
                start_percent=0.0,
                end_percent=1.0
            )
            
            result = skip_layers.get_model(use_tuned=False)
            assert result == skip_layers._base_model
    
    def test_vae_property(self):
        """Test vae property."""
        with patch("json_gui.mimic_classes.comfy.model_management.unload_all_models"), \
             patch("json_gui.mimic_classes.comfy.model_management.soft_empty_cache"), \
             patch("json_gui.mimic_classes.load_checkpoint_guess_config") as mock_load, \
             patch("json_gui.mimic_classes.SkipLayerGuidanceSD3.execute") as mock_skip:
            
            # Setup mocks
            model = MagicMock()
            vae = MagicMock()
            mock_load.return_value = (model, None, vae, None)
            
            tuned_model = MagicMock()
            mock_skip.return_value = (tuned_model,)
            
            skip_layers = mimic_classes.SkipLayers(
                layers=[0, 1, 2],
                scale=0.5,
                start_percent=0.0,
                end_percent=1.0
            )
            
            assert skip_layers.vae is not None
    
    def test_initialization_empty_layers(self):
        """Test initialization with empty layers list."""
        with patch("json_gui.mimic_classes.comfy.model_management.unload_all_models"), \
             patch("json_gui.mimic_classes.comfy.model_management.soft_empty_cache"), \
             patch("json_gui.mimic_classes.load_checkpoint_guess_config") as mock_load, \
             patch("json_gui.mimic_classes.SkipLayerGuidanceSD3.execute") as mock_skip:
            
            # Setup mocks
            model = MagicMock()
            vae = MagicMock()
            mock_load.return_value = (model, None, vae, None)
            
            skip_layers = mimic_classes.SkipLayers(
                layers=[],
                scale=0.5,
                start_percent=0.0,
                end_percent=1.0
            )
            
            # With empty layers, tuned model should be same as base model
            assert skip_layers._tunned_model == skip_layers._base_model
            # SkipLayerGuidanceSD3 should not be called
            mock_skip.assert_not_called()
