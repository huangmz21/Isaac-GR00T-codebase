# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-aware GR00T N1.5 model.

This class keeps the official GR00T N1.5 backbone and action head intact. It
only inserts an optional action-effect memory fusion module between the
backbone features and the flow-matching action head.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
from gr00t.model.memory import GR00TMemoryFusion


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() not in {"false", "0", "no", "none", ""}
    return bool(value)


class GR00T_N1_5_Memory(GR00T_N1_5):
    """GR00T N1.5 with optional action-effect memory fusion."""

    config_class = GR00T_N1_5_Config

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
        memory_cfg: dict | None = None,
    ):
        super().__init__(config=config, local_model_path=local_model_path)
        if memory_cfg is None:
            memory_cfg = getattr(config, "memory_cfg", None)
        self.memory_fusion = None
        if memory_cfg is not None and _as_bool(memory_cfg.get("enabled", True)):
            self.enable_memory(memory_cfg)
        else:
            self.config.memory_cfg = memory_cfg

    def enable_memory(self, memory_cfg: dict):
        memory_cfg = dict(memory_cfg)
        memory_cfg["enabled"] = True
        memory_cfg.setdefault(
            "hidden_dim", self.config.action_head_cfg.get("backbone_embedding_dim", 1536)
        )
        memory_cfg.setdefault("action_dim", self.config.action_head_cfg.get("max_action_dim", 32))
        self.memory_fusion = GR00TMemoryFusion(memory_cfg)
        self.config.memory_cfg = memory_cfg

    @property
    def memory_enabled(self) -> bool:
        return self.memory_fusion is not None

    def _memory_inputs_available(self, inputs: dict) -> bool:
        required = ("mem_actions", "mem_valid_mask", "mem_pre_feat", "mem_post_feat")
        return all(key in inputs for key in required)

    def _maybe_fuse_memory(self, inputs: dict, backbone_outputs: BatchFeature) -> BatchFeature:
        if self.memory_fusion is None:
            return backbone_outputs
        if not self._memory_inputs_available(inputs):
            missing = [
                key
                for key in ("mem_actions", "mem_valid_mask", "mem_pre_feat", "mem_post_feat")
                if key not in inputs
            ]
            raise KeyError(f"Memory model is enabled but inputs are missing memory fields: {missing}")

        fused_features = self.memory_fusion(
            hidden=backbone_outputs["backbone_features"],
            mem_pre_feat=inputs["mem_pre_feat"],
            mem_post_feat=inputs["mem_post_feat"],
            mem_actions=inputs["mem_actions"],
            mem_valid_mask=inputs["mem_valid_mask"],
        )
        backbone_outputs["backbone_features"] = fused_features
        return backbone_outputs

    def forward(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        backbone_outputs = self._maybe_fuse_memory(action_inputs, backbone_outputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        backbone_outputs = self._maybe_fuse_memory(action_inputs, backbone_outputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        memory_cfg = kwargs.pop("memory_cfg", None)
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained memory dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(
                "Model not found or avail in the huggingface hub. "
                f"Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super(GR00T_N1_5, cls).from_pretrained(
            local_model_path,
            local_model_path=local_model_path,
            **kwargs,
        )

        if memory_cfg is not None and not pretrained_model.memory_enabled:
            pretrained_model.enable_memory(memory_cfg)

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model
