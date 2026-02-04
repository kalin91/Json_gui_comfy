"""Module defining the Flow class for loading and managing flow data from JSON files."""

import os
import json
import logging
from typing import Any, Optional, cast

from json_gui.scripts.mimic_ksamplers import SimpleKSampler, FaceDetailerNode
from json_gui.scripts.mimic_controlnet import ControlNetImgPreprocessor, CannyEdge, OpenPosePose, ApplyControlNet
from json_gui.scripts.mimic_classes import (
    Sd3Clip,
    EmptyLatent,
    Rotator,
    SkipLayers,
    MimicNode,
    Prompts,
)


class Model:
    """Class representing a flow loaded from a JSON file.""" ""

    @property
    def clip(self) -> Sd3Clip:
        """Returns the loaded CLIP model."""
        return self._clip

    @property
    def prompts(self) -> Prompts:
        """Returns the negative prompt."""
        return self._prompts

    @property
    def apply_control_net(self) -> list[ApplyControlNet]:
        """Returns the ApplyControlNet instance."""
        return self._apply_control_net

    @property
    def empty_latent(self) -> EmptyLatent:
        """Returns the EmptyLatent instance."""
        return self._empty_latent

    @property
    def simple_k_sampler(self) -> list[SimpleKSampler]:
        """Returns the SimpleKSampler instance."""
        return self._simple_k_sampler

    @property
    def face_detailer(self) -> FaceDetailerNode:
        """Returns the FaceDetailer instance."""
        return self._face_detailer

    @property
    def rotator(self) -> Rotator:
        """Returns the Rotator instance."""
        return self._rotator

    @property
    def skip_layers_model(self) -> SkipLayers:
        """Returns the SkipLayers instance."""
        return self._skip_layers_model

    def __init__(self, filepath: Optional[str] = None) -> None:
        """Initializes the Flow instance by loading data from a JSON file."""
        assert filepath, "Invalid file path."
        assert os.path.exists(filepath), f"Flow file {filepath} does not exist."
        logging.info("Loading flow from %s", filepath)
        self._file_path: str = filepath
        self._clip: Sd3Clip = Sd3Clip()
        self.load_json()

    def load_json(self) -> None:
        """Loads the flow data from the JSON file."""
        with open(self._file_path, "r", encoding="utf-8") as file:
            json_props: dict[str, Any] = json.load(file)
        cnet_list = json_props[ApplyControlNet.key()]
        assert isinstance(cnet_list, list), "Expected apply_control_net to be a list."
        self._apply_control_net = []
        cnet_dicts: dict[str, ApplyControlNet] = {}
        for cnet in cnet_list:
            target_name: str = cnet["target"]
            preprocesors_dict = {cast(MimicNode, t).key(): t for t in [CannyEdge, OpenPosePose]}
            assert target_name in preprocesors_dict, f"Unknown ControlNet target: {target_name}"
            target_type: type = preprocesors_dict[target_name]
            assert target_name not in cnet_dicts, f"Duplicate target {target_name} in apply_control_net."
            assert target_name in json_props, f"Target {target_name} not found in JSON properties."
            target_dict = json_props.pop(target_name)
            cnet["target"] = (target_type, target_dict)
            cnet_dicts[target_name] = ApplyControlNet(**cnet)
        self._apply_control_net.extend([v for v in cnet_dicts.values() if v.target is not None])
        self._prompts = Prompts(**json_props[Prompts.key()])
        self._empty_latent = EmptyLatent(**json_props[EmptyLatent.key()])
        self._simple_k_sampler = [SimpleKSampler(**s) for s in json_props[SimpleKSampler.key()]]
        self._face_detailer = FaceDetailerNode(**json_props[FaceDetailerNode.key()])
        self._rotator = Rotator(**json_props[Rotator.key()])
        self._skip_layers_model = SkipLayers(**json_props[SkipLayers.key()])

    def update_json(self) -> None:
        """Loads the flow data from the JSON file."""
        with open(self._file_path, "r", encoding="utf-8") as file:
            json_props: dict[str, Any] = json.load(file)
        for name, value in vars(self).items():
            assert value is not None, f"Property {name} is None."
            is_list = False
            prop_type = type(value)
            if prop_type is Sd3Clip:
                continue  # Skip CLIP property
            if isinstance(value, list):
                is_list = True
                prop_type = type(value[0])
            if issubclass(prop_type, MimicNode):
                key = prop_type.key()
                data = json_props[key]
                if issubclass(prop_type, ApplyControlNet):
                    list_control_net: list[ApplyControlNet] = []
                    target_types: list[type[MimicNode]] = [CannyEdge, OpenPosePose]
                    preprocesors_dict: dict[str, type[MimicNode]] = {t.key(): t for t in target_types}
                    if not isinstance(value, list) or not isinstance(data, list):
                        raise ValueError("Mismatch in types for ApplyControlNet update.")
                    for i, item in enumerate(data):
                        item = cast(dict[str, Any], item)
                        idx_exists: bool = 0 <= i < len(value)
                        node: ApplyControlNet
                        target_inst: ControlNetImgPreprocessor
                        target_name: str = item["target"]
                        target_dict = json_props.pop(target_name)
                        target_type: type = preprocesors_dict[target_name]
                        if idx_exists and (node := value[i]) is not None and target_name == node.target.key():
                            target_inst = node.target
                            target_inst.update(**target_dict)
                            item["target"] = (target_inst.__class__, target_dict)
                            node.update(**item)
                        else:
                            target_inst = target_type(**target_dict)
                            item["target"] = (target_inst.__class__, target_dict)
                            node = ApplyControlNet(**item)
                        list_control_net.append(node)
                    value.clear()
                    value.extend([c for c in list_control_net if c.target is not None])
                else:
                    if is_list:
                        for i, item in enumerate(data):
                            item = cast(dict[str, Any], item)
                            m_node = cast(MimicNode, value[i])
                            m_node.update(**item)
                    else:
                        m_node = cast(MimicNode, value)
                        m_node.update(**data)
