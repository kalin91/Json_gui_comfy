"""Mimic child classes for various components."""

import logging
from functools import partial
from typing import Any, Callable, Optional, Tuple, cast, Self
import torch
import numpy as np
import comfy.model_management
from comfy_extras.nodes_sd3 import SkipLayerGuidanceSD3
from comfy.sd import load_checkpoint_guess_config, VAE, CLIP, load_clip
from comfy.model_patcher import ModelPatcher
import folder_paths
from PIL import Image
from json_gui.scripts.mimic import MimicNode, DataWrapper

type Conditional = list[tuple[torch.Tensor, dict[str, Any]]]


class Sd3Clip(MimicNode[None]):
    """A class representing SD3 CLIP settings."""

    _clip: Optional[CLIP]

    @classmethod
    def _class_param_definitions(cls):
        return []  # No class params needed for Prompts

    CLIP_G_PATH = "sd35m/clip_g.safetensors"
    CLIP_L_PATH = "sd35m/clip_l.safetensors"
    T5_PATH = "sd35m/t5xxl_fp16.safetensors"

    @classmethod
    def key(cls) -> str:
        """Returns the key for the Sd3Clip."""
        return "sd3_clip"

    @property
    def clip(self) -> CLIP:
        """Returns the loaded CLIP model."""
        if self._clip is None:
            raise ValueError("Clip has not been processed yet. Call process() first.")
        return self._clip

    # pylint: disable=W0201
    # pylint: disable=W0221
    def _process_impl(self):
        """Loads the Triple CLIP model."""
        if self._clip is None:
            logging.info("Loading CLIPs...")
            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", self.CLIP_G_PATH)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", self.CLIP_L_PATH)
            clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", self.T5_PATH)
            self._clip = load_clip(
                ckpt_paths=[clip_path1, clip_path2, clip_path3],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )

    # pylint: disable=W0221
    def _update_impl(self) -> None:
        """PASS - Loads the CLIP model."""
        self._clip = None

    def __init__(self):
        super().__init__()
        self.update()


class SkipLayers(MimicNode[Tuple[ModelPatcher, VAE]]):
    """A class representing skip layer guidance settings."""

    @classmethod
    def _class_param_definitions(cls):
        return []  # No class params needed for Prompts

    @classmethod
    def key(cls):
        """Returns the key for the SkipLayers."""
        return "skip_layers_model"

    @property
    def vae(self) -> VAE:
        """Returns the VAE of the model."""
        if self._vae is None:
            raise ValueError("Model has not been processed yet. Call process() first.")
        return self._vae

    @property
    def model(self) -> ModelPatcher:
        """Returns the model."""
        res = self._tunned_model if self.use_tuned else self._base_model
        if res is None:
            raise ValueError("Model has not been processed yet. Call process() first.")
        return res

    @property
    def use_tuned(self) -> bool:
        """Returns whether to use the tuned model."""
        return self._use_tuned

    @use_tuned.setter
    def use_tuned(self, value: bool) -> None:
        """Sets whether to use the tuned model."""
        self._use_tuned = value

    CHECKPOINT_PATH = "sd3.5_medium.safetensors"

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _process_impl(self):
        """Returns the tuned model."""
        # 1. Load Model and VAE
        logging.info("Loading Checkpoint...")

        # Free memory before loading the large checkpoint (~10.5GB)
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", self.CHECKPOINT_PATH)
        self._base_model, _a, self._vae, _b = load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=False,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        if self._layers:
            result: tuple = SkipLayerGuidanceSD3.execute(
                self._base_model,
                self._layers,
                self._scale,
                self._start_percent,
                self._end_percent,
            )
            self._tunned_model = result[0]
        else:
            self._tunned_model = self._base_model
        return self.model, self._vae

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(self, layers: list[int], scale: float, start_percent: float, end_percent: float) -> None:
        self._layers = ",".join(str(layer) for layer in layers)
        self._scale = scale
        self._start_percent = start_percent
        self._end_percent = end_percent
        self._vae = None
        self._use_tuned = False
        self._base_model = None
        self._tunned_model = None

    def __init__(self, layers: list[int], scale: float, start_percent: float, end_percent: float):
        super().__init__()
        self.update(layers=layers, scale=scale, start_percent=start_percent, end_percent=end_percent)


class Prompts(MimicNode[tuple[DataWrapper[Conditional], DataWrapper[Conditional]]]):
    """A class representing positive and negative prompts."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Self, Any]]:
        res: list[MimicNode.ClassParam[Self, Any]] = []
        res.append(
            cls.build_class_param(
                Sd3Clip, lambda inst: cast(Sd3Clip, inst).process() or {"clip": cast(Sd3Clip, inst).clip}
            )
        )
        return res

    @classmethod
    def key(cls) -> str:
        """Returns the key for the Prompts."""
        return "prompts"

    # pylint: disable=W0221
    def _process_impl(self, clip: CLIP) -> tuple[DataWrapper[Conditional], DataWrapper[Conditional]]:
        """Encodes the positive and negative prompts using the provided CLIP model."""
        logging.info("Encoding prompts...")
        tokens_pos = clip.tokenize(self._positive)
        cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)

        tokens_neg = clip.tokenize(self._negative)
        cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

        del tokens_pos
        del tokens_neg
        torch.cuda.empty_cache()
        return (
            DataWrapper(value=cast(Conditional, cond_pos), skip_unwrap=False),
            DataWrapper(value=cast(Conditional, cond_neg), skip_unwrap=False),
        )

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(self, positive: str, negative: str) -> None:
        self._positive = positive
        self._negative = negative

    def __init__(self, positive: str, negative: str):
        super().__init__()
        self.update(positive=positive, negative=negative)


class EmptyLatent(MimicNode[torch.Tensor]):
    """An empty latent class for placeholder purposes."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Self, Any]]:
        res: list[MimicNode.ClassParam[Self, Any]] = []
        res.append(
            cls.build_class_param(
                SkipLayers, lambda inst: cls._set_current_model(inst) or {"vae": cast(SkipLayers, inst).process()[1]}
            )
        )
        return res

    @classmethod
    def key(cls) -> str:
        """Returns the key for the EmptyLatent."""
        return "empty_latent"

    @property
    def start_img(self) -> Optional[torch.Tensor]:
        """Returns the starting image tensor, if any."""
        return self._start_img

    # pylint: disable=W0221
    def _process_impl(self, vae: VAE) -> torch.Tensor:
        """Generates and returns an empty latent tensor."""
        if self.start_img is not None:
            logging.info("Creating latent from start image...")

            if vae is None:
                raise ValueError(
                    "VAE is required to encode start_img to latent space. "
                    "Please provide vae parameter when creating EmptyLatent with image_name."
                )

            # Redim the image  to the expected size
            start_img = self.start_img
            current_height, current_width = start_img.shape[1], start_img.shape[2]

            if current_height != self._height or current_width != self._width:
                logging.info(
                    "Resizing start image from %sx%s to %sx%s...",
                    current_width,
                    current_height,
                    self._width,
                    self._height,
                )
                # Permute: [B, H, W, C] -> [B, C, H, W] for interpolation
                start_img = start_img.permute(0, 3, 1, 2)
                start_img = torch.nn.functional.interpolate(
                    start_img, size=(self._height, self._width), mode="bilinear", align_corners=False
                )
                # Permute back: [B, C, H, W] -> [B, H, W, C]
                start_img = start_img.permute(0, 2, 3, 1)

            self.add_unsaved_tensor(start_img, "start_image")

            logging.info("Encoding start image to latent space with VAE...")
            latent = vae.encode(start_img)
            logging.info("Encoded latent shape: %s", latent.shape)

            return latent

        logging.info("Creating empty latent %sx%s...", self._width, self._height)
        img = torch.zeros(
            [self._batch_size, 16, self._height // 8, self._width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        images = vae.decode(img.clone())
        logging.info("VAE Output Shape: %s", images.shape)

        # Ensure BHWC (Batch, Height, Width, Channels)
        if images.shape[1] == 3:
            images = images.movedim(1, -1)

        logging.info("Final Image Shape: %s", images.shape)

        self.add_unsaved_tensor(images, self.key())
        return img

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(self, width: int, height: int, batch_size: int, image_name: Optional[str]) -> None:
        self._width = width
        self._height = height
        self._batch_size = batch_size
        self._start_img = self._upload_image(image_name) if image_name and image_name != "<None>" else None

    def __init__(self, width: int, height: int, batch_size: int, image_name: str):
        super().__init__()
        self.update(width=width, height=height, batch_size=batch_size, image_name=image_name)


class Rotator(MimicNode[Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]]):
    """A class representing image rotation settings."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Self, Any]]:
        return []  # No class params needed for Prompts

    @classmethod
    def key(cls) -> str:
        """Returns the key for the Rotator."""
        return "rotator"

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(self, angle: float) -> None:
        self._angle = angle

    def __init__(self, angle: float):
        super().__init__()
        self.update(angle=angle)

    # pylint: disable=W0221
    def _process_impl(self, image: torch.Tensor) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """Rotates the given image tensor by the specified angle.

        Uses PIL with BICUBIC interpolation to minimize quality loss during rotation.
        Converts tensor -> PIL -> rotate -> tensor for better quality preservation.
        """

        if self._angle == 0:
            return image, lambda x: x  # No rotation needed

        # getting original image size (BHWC format)
        batch_size, orig_h, orig_w, _channels = image.shape
        logging.info("Original image shape: %s", image.shape)

        # Process each image in the batch
        rotated_list = []
        for i in range(batch_size):
            # Convert tensor to PIL Image (tensor is 0-1 float, HWC)
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Rotate with PIL using BICUBIC interpolation (expand=True to keep corners)
            rotated_pil = pil_img.rotate(
                -self._angle,  # PIL rotates counter-clockwise, we want clockwise
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )

            # Convert back to tensor (0-1 float)
            rotated_np = np.array(rotated_pil).astype(np.float32) / 255.0
            rotated_list.append(torch.from_numpy(rotated_np))

        # Stack batch back together (BHWC)
        rotated_images = torch.stack(rotated_list, dim=0).to(image.device)
        logging.info("Rotated image shape: %s", rotated_images.shape)

        result_fun = partial(Rotator._undo_rotate, angle=self._angle, orig_h=orig_h, orig_w=orig_w)
        self.add_unsaved_tensor(rotated_images, "rotated_image")

        return rotated_images, result_fun

    @classmethod
    def _undo_rotate(cls, image: torch.Tensor, angle: float, orig_h: int, orig_w: int) -> torch.Tensor:

        # Rotate result back to original orientation
        logging.info("Rotating results back to original orientation...")
        result_batch = image.shape[0]
        unrotated_list = []
        for i in range(result_batch):
            # Convert tensor to PIL
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Rotate back with BICUBIC
            unrotated_pil = pil_img.rotate(
                angle,  # Opposite direction
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )

            # Convert back to tensor
            unrotated_np = np.array(unrotated_pil).astype(np.float32) / 255.0
            unrotated_list.append(torch.from_numpy(unrotated_np))

        unprocessed_image = torch.stack(unrotated_list, dim=0).to(image.device)

        # Crop to original size (center crop)
        _, h, w, _ = unprocessed_image.shape
        top = (h - orig_h) // 2
        left = (w - orig_w) // 2
        rotated_image = unprocessed_image[:, top : top + orig_h, left : left + orig_w, :]  # noqa: E203

        logging.info("Final cropped image shape: %s", rotated_image.shape)
        return rotated_image
