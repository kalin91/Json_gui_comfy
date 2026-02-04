"""ksampler mimic classes and application node definitions."""

from abc import ABC, abstractmethod
import inspect
import logging
from typing import Any, Tuple, TypeVar, Generic, final, Self, TypedDict, cast
import torch
from comfy.sd import CLIP
from comfy.sample import fix_empty_latent_channels, prepare_noise, sample
from comfy_extras.nodes_mask import MaskToImage
from json_gui.scripts.mimic_classes import SkipLayers, Sd3Clip
from json_gui.scripts.mimic import MimicNode

T = TypeVar("T")
M = TypeVar("M", bound=MimicNode)


class KSamplerLike(Generic[T], MimicNode[T], ABC):
    """A simple KSampler class for demonstration purposes."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Self, Any]]:
        res: list[MimicNode.ClassParam[Self, Any]] = []
        res.append(
            cls.build_class_param(SkipLayers, lambda inst: cls._set_current_model(inst) or {"node_model": inst})
        )
        return res

    @property
    def use_tune(self) -> bool:
        """Returns whether to use tune."""
        return self._use_tune

    # pylint: disable=W0221, W0239, W0201
    @final
    def _update_impl(
        self,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        use_tune: bool,
        *args,
        **kwargs,
    ) -> None:
        """Pre-update method to set parameters before the main update."""
        self._seed = seed
        self._steps = steps
        self._cfg = cfg
        self._sampler_name = sampler_name
        self._scheduler = scheduler
        self._denoise = denoise
        self._use_tune = use_tune
        self._update_impl_complement(*args, **kwargs)

    @abstractmethod
    def _update_impl_complement(self, *args, **kwargs) -> None:
        """Abstract method to update the node."""

    def __init__(
        self,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        use_tune: bool,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.update(seed, steps, cfg, sampler_name, scheduler, denoise, use_tune, *args, **kwargs)

    def _to_dict(self) -> dict:
        """Converts the SimpleKSampler instance to a dictionary."""
        logging.info(
            "Sampling 1 with seed=%s, steps=%s, cfg=%s, sampler=%s, scheduler=%s...",
            self._seed,
            self._steps,
            self._cfg,
            self._sampler_name,
            self._scheduler,
        )

        return {
            "seed": self._seed,
            "steps": self._steps,
            "cfg": self._cfg,
            "sampler_name": self._sampler_name,
            "scheduler": self._scheduler,
            "denoise": self._denoise,
        }


class SimpleKSampler(KSamplerLike[Tuple[torch.Tensor, torch.Tensor]]):
    """A simple KSampler class for demonstration purposes."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Self, Any]]:
        res: list[MimicNode.ClassParam[Self, Any]] = []
        c_param = cls.build_class_param(
            SkipLayers, processor=lambda inst: cls._set_current_model(inst) or {"node_model": inst}
        )
        res.append(c_param)
        return res

    @classmethod
    def key(cls):
        """Returns the key for the SimpleKSampler."""
        return "simple_k_sampler"

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl_complement(
        self,
    ) -> None:
        """No additional parameters to update."""

    def __init__(
        self,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        use_tune: bool,
    ):
        super().__init__(
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            use_tune=use_tune,
        )

    # pylint: disable=W0221
    def _process_impl(
        self, latent_image: torch.Tensor, node_model: SkipLayers, cond_pos_cnet: Any, cond_neg_cnet: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A placeholder method to simulate processing."""
        try:
            node_model.use_tuned = self.use_tune
            model = node_model.model
            vae = node_model.vae

            # Prepare noise
            noisy_latent_image = fix_empty_latent_channels(model, latent_image)

            noise = prepare_noise(noisy_latent_image, self._seed, None)

            # Safely get sampler arguments
            sampler_arguments = self._to_dict()

            sampler_arguments.update(
                {
                    "model": model,
                    "noise": noise,
                    "positive": cond_pos_cnet,
                    "negative": cond_neg_cnet,
                    "latent_image": noisy_latent_image,
                    "disable_noise": False,
                    "start_step": None,
                    "last_step": None,
                    "force_full_denoise": False,
                    "noise_mask": None,
                    "callback": None,
                    "disable_pbar": False,
                }
            )

            sampler_signature = inspect.signature(sample)
            for key in sampler_arguments:
                if key not in sampler_signature.parameters:
                    raise ValueError(f"Unexpected argument '{key}' for comfy.sample.sample")
            logging.info("Decoding...")
            res: torch.Tensor = sample(**sampler_arguments)
            images = vae.decode(res.clone())
            logging.info("VAE Output Shape: %s", images.shape)

            # Ensure BHWC (Batch, Height, Width, Channels)
            if images.shape[1] == 3:
                images = images.movedim(1, -1)

            logging.info("Final Image Shape: %s", images.shape)

            self.add_unsaved_tensor(images, self.key())

            return res, images
        except Exception as e:
            logging.exception("Error in SimpleKSampler processing: %s", e)
            raise e


class FaceDetailerParams(TypedDict):
    """Parameters for FaceDetailerNode."""

    guide_size: int
    guide_size_for: bool
    max_size: int
    feather: int
    noise_mask: bool
    force_inpaint: bool
    drop_size: int
    cycle: int
    bbox_threshold: float
    bbox_dilation: int
    bbox_crop_factor: float
    sam_detection_hint: str
    sam_dilation: int
    sam_threshold: float
    sam_bbox_expansion: int
    sam_mask_hint_threshold: float
    sam_mask_hint_use_negative: str
    wildcard: str


class FaceDetailerNode(KSamplerLike["torch.Tensor"]):
    """A class representing face detailer settings.

    Note: This class inherits from SimpleKSampler but overrides _process_impl
    to return torch.Tensor instead of Tuple[torch.Tensor, torch.Tensor].
    """

    _params: FaceDetailerParams

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Self, Any]]:
        res: list[MimicNode.ClassParam[Self, Any]] = []
        res.append(
            cls.build_class_param(SkipLayers, lambda inst: cls._set_current_model(inst) or {"node_model": inst})
        )
        res.append(
            cls.build_class_param(
                Sd3Clip, lambda inst: cast(Sd3Clip, inst).process() or {"clip": cast(Sd3Clip, inst).clip}
            )
        )
        return res

    @classmethod
    def key(cls):
        """Returns the key for the FaceDetailerNode."""
        return "face_detailer"

    def _to_dict(self) -> dict:
        """Converts the FaceDetailer instance to a dictionary."""
        base_dict = super()._to_dict()
        base_dict.update(self._params)
        return base_dict

    # pylint: disable=W0221,C0415
    def _process_impl(
        self, input_image: torch.Tensor, positive: Any, negative: Any, node_model: SkipLayers, clip: CLIP
    ) -> torch.Tensor:
        """
        Processes the input image using FaceDetailer, returning the detailed image.
        It uses the provided model, clip, and other parameters to enhance facial details.
        Lazily loads the SAM model and BBOX detector due to their heavy initialization.

        Args:
            input_image (torch.Tensor): The input image tensor.
            positive (Any): The positive conditioning.
            negative (Any): The negative conditioning.
            node_model (SkipLayers): The SkipLayers model.
            clip (CLIP): The loaded CLIP model.

        Raises:
            e: Exception raised during processing.
            ValueError: Raised if unexpected arguments are passed to FaceDetailer.doit.

        Returns:
            torch.Tensor: The processed image tensor.
        """
        try:
            import json_gui.server as _  # noqa: F401
            from custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack import SAMLoader, FaceDetailer
            from custom_nodes.ComfyUI_Impact_Subpack.modules.subpack_nodes import UltralyticsDetectorProvider

            bbox_provider = UltralyticsDetectorProvider()
            # UltralyticsDetectorProvider.doit returns (BBOX_DETECTOR, SEGM_DETECTOR)
            bbox_detector, _c = bbox_provider.doit(self._bbox_detector_str)

            sam_loader = SAMLoader()
            # SAMLoader.load_model returns (SAM_MODEL,)
            sam_model_opt = sam_loader.load_model(self._sam_model_opt_str)[0]

            node_model.use_tuned = self.use_tune
            model = node_model.model
            vae = node_model.vae

            # FaceDetailer
            logging.info("Running FaceDetailer...")

            face_detailer = FaceDetailer()

            face_arguments = self._to_dict()

            face_arguments.update(
                {
                    "sam_model_opt": sam_model_opt,
                    "bbox_detector": bbox_detector,
                    "model": model,
                    "vae": vae,
                    "clip": clip,
                    "positive": positive,
                    "negative": negative,
                    "image": input_image,
                    "segm_detector_opt": None,  # Not using segm detector here
                    "detailer_hook": None,
                }
            )
        except Exception as e:
            logging.exception("Error preparing FaceDetailer arguments: %s", e)
            raise e

        # validate face_arguments keys against FaceDetailer.doit signature would be ideal
        face_signature = inspect.signature(face_detailer.doit)
        for key in face_arguments:
            if key not in face_signature.parameters:
                raise ValueError(f"Unexpected argument '{key}' for FaceDetailer.doit")

        result_images, cropped_images, cropped_alpha, mask = face_detailer.doit(**face_arguments)[:4]
        for idx, cropped in enumerate(cropped_images):
            self.add_unsaved_tensor(cropped, f"face-cropped-{idx}")
        for idx, alpha in enumerate(cropped_alpha):
            self.add_unsaved_tensor(alpha, f"face-alpha-{idx}")
        mask_img_tensor: tuple = MaskToImage().execute(mask).result[0]  # pylint: disable=E1136
        self.add_unsaved_tensor(mask_img_tensor, "face-mask")
        return result_images

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl_complement(
        self,
        guide_size: int,
        guide_size_for: bool,
        max_size: int,
        feather: int,
        noise_mask: bool,
        force_inpaint: bool,
        drop_size: int,
        cycle: int,
        bbox_threshold: float,
        bbox_dilation: int,
        bbox_crop_factor: float,
        sam_detection_hint: str,
        sam_dilation: int,
        sam_threshold: float,
        sam_bbox_expansion: int,
        sam_mask_hint_threshold: float,
        sam_mask_hint_use_negative: str,
        bbox_detector: str,
        sam_model_opt: str,
        wildcard: str,
    ) -> None:
        self._params: FaceDetailerParams = {
            "guide_size": guide_size,
            "guide_size_for": guide_size_for,
            "max_size": max_size,
            "feather": feather,
            "noise_mask": noise_mask,
            "force_inpaint": force_inpaint,
            "drop_size": drop_size,
            "cycle": cycle,
            "bbox_threshold": bbox_threshold,
            "bbox_dilation": bbox_dilation,
            "bbox_crop_factor": bbox_crop_factor,
            "sam_detection_hint": sam_detection_hint,
            "sam_dilation": sam_dilation,
            "sam_threshold": sam_threshold,
            "sam_bbox_expansion": sam_bbox_expansion,
            "sam_mask_hint_threshold": sam_mask_hint_threshold,
            "sam_mask_hint_use_negative": sam_mask_hint_use_negative,
            "wildcard": wildcard,
        }
        self._bbox_detector_str = bbox_detector
        self._sam_model_opt_str = sam_model_opt

    def __init__(
        self,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        guide_size: int,
        guide_size_for: bool,
        max_size: int,
        feather: int,
        noise_mask: bool,
        force_inpaint: bool,
        drop_size: int,
        cycle: int,
        bbox_threshold: float,
        bbox_dilation: int,
        bbox_crop_factor: float,
        sam_detection_hint: str,
        sam_dilation: int,
        sam_threshold: float,
        sam_bbox_expansion: int,
        sam_mask_hint_threshold: float,
        sam_mask_hint_use_negative: str,
        bbox_detector: str,
        sam_model_opt: str,
        wildcard: str,
        use_tune: bool,
    ):
        super().__init__(
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            use_tune=use_tune,
            guide_size=guide_size,
            guide_size_for=guide_size_for,
            max_size=max_size,
            feather=feather,
            noise_mask=noise_mask,
            force_inpaint=force_inpaint,
            drop_size=drop_size,
            cycle=cycle,
            bbox_threshold=bbox_threshold,
            bbox_dilation=bbox_dilation,
            bbox_crop_factor=bbox_crop_factor,
            sam_detection_hint=sam_detection_hint,
            sam_dilation=sam_dilation,
            sam_threshold=sam_threshold,
            sam_bbox_expansion=sam_bbox_expansion,
            sam_mask_hint_threshold=sam_mask_hint_threshold,
            sam_mask_hint_use_negative=sam_mask_hint_use_negative,
            bbox_detector=bbox_detector,
            sam_model_opt=sam_model_opt,
            wildcard=wildcard,
        )
