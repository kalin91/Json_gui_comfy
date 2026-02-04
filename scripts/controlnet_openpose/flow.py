"""Script to run a ControlNet flow with Triple CLIP and FaceDetailer integration."""

import logging
import torch
import comfy.model_management
from json_gui.scripts.controlnet_openpose.model import Model
from json_gui.utils import AbsFlow
from json_gui.scripts.mimic import MimicNode
from json_gui.scripts.node_executor import NodeExecutor


class Flow(AbsFlow):
    """ControlNet OpenPose Flow implementation."""

    @property
    def input_model(self) -> Model:
        """Get the flow model inputs."""
        return self._input_model

    def _late_init(self, steps: int, multiprocess: bool) -> None:
        """
        Set the flow model inputs.

        Args:
            steps (int): Number of steps for the flow.
            multiprocess (bool): Whether to use multiprocessing.
        """
        assert isinstance(steps, int), "Flow input value must be an integer."
        # self.update_save_call(value)
        self._input_model.update_json()
        MimicNode.set_node_executor_factory(
            NodeExecutor, self.saved_data, self.save_image, self.copy_images, multiprocess
        )

    def __init__(self, file_path: str, filename: str) -> None:
        super().__init__(file_path, filename)
        self._input_model: Model = Model(self.json_path)

    def _run_impl(self, steps: int, multiprocess: bool):
        """
        Runs the flow and returns a list of created image file paths.

        Args:
            steps (int): Number of steps for the flow.
            multiprocess (bool): Whether to use multiprocessing.

        Returns:
            list[str]: List of created image file paths.
        """

        self._late_init(steps, multiprocess)

        prms_node = self.input_model.prompts

        # Encode Prompts
        cond_pos, cond_neg = prms_node.exec_node({}, [self.input_model.clip])

        sd_model = self.input_model.skip_layers_model

        latent_image = self.input_model.empty_latent.exec_node({}, [sd_model])

        # Run control net conditionings
        logging.info("Applying ControlNet conditionings...")
        for cnet in self.input_model.apply_control_net:
            dict_arg: dict = cnet.process_args_dict(cond_pos, cond_neg)
            cond_pos, cond_neg = cnet.exec_node(dict_arg, [sd_model])

        cond_pos.skip_unwrap = False
        cond_neg.skip_unwrap = False

        for sampler_idx, current_sampler in enumerate(self.input_model.simple_k_sampler):
            logging.info("Running Sampler %d...", sampler_idx)
            dict_arg = current_sampler.process_args_dict(
                latent_image, **{"cond_pos_cnet": cond_pos, "cond_neg_cnet": cond_neg}
            )
            latent_image, images = current_sampler.exec_node(dict_arg, [sd_model])

        rotator = self.input_model.rotator
        rotated, unrotator = rotator.exec_node(rotator.process_args_dict(images), [])

        # full_raw = clip_raw.update(model_raw)

        input_dict = {
            "input_image": rotated,
            "positive": cond_pos,
            "negative": cond_neg,
        }

        detailed_image: torch.Tensor = self.input_model.face_detailer.exec_node(
            input_dict, [sd_model, self.input_model.clip]
        )

        unrotated = unrotator(detailed_image)

        self.save_image(self.saved_data, unrotated, "unrotated", is_temp=False)

        # Cleanup: unload models and free memory after flow execution
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        logging.info("Done.")
