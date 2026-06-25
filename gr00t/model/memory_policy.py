# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy wrapper for online action-effect memory evaluation."""

from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1_memory import GR00T_N1_5_Memory
from gr00t.model.policy import COMPUTE_DTYPE, Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values

DINO_IMAGE_SIZE = (224, 224)
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


class OnlineDINOFeatureExtractor:
    """Small DINOv2 loader used to build memory features during rollout."""

    FEATURE_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vits14_reg": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitb14_reg": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitl14_reg": 1024,
        "dinov2_vitg14": 1408,
        "dinov2_vitg14_reg": 1408,
    }

    def __init__(
        self,
        backbone: str = "dinov2_vitb14_reg",
        feature_dim: int = 768,
        device: str | None = None,
    ):
        if backbone not in self.FEATURE_DIMS:
            raise ValueError(f"Unsupported DINOv2 backbone: {backbone}")
        expected_dim = self.FEATURE_DIMS[backbone]
        if expected_dim != feature_dim:
            raise ValueError(
                f"DINO feature_dim={feature_dim} does not match {backbone} output dim {expected_dim}"
            )

        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model().to(self.device).eval()
        self._dino_mean = torch.tensor(DINO_MEAN, device=self.device, dtype=torch.float32).view(
            1, 3, 1, 1
        )
        self._dino_std = torch.tensor(DINO_STD, device=self.device, dtype=torch.float32).view(
            1, 3, 1, 1
        )
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"Loaded online memory DINO: backbone={backbone}, device={self.device}")

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _candidate_weight_names(backbone: str) -> list[str]:
        names = [f"{backbone}_pretrain.pth"]
        if backbone.endswith("_reg"):
            names.append(f"{backbone}4_pretrain.pth")
        return names

    @classmethod
    def _find_local_repo(cls) -> Path | None:
        repo_root = cls._repo_root()
        user_root = repo_root.parents[1]
        torch_home = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
        candidates = [
            os.environ.get("DINO_REPO_DIR"),
            repo_root / "dinov2",
            repo_root.parent / "starVLA" / "dinov2",
            user_root / "starVLA" / "dinov2",
            torch_home / "hub" / "facebookresearch_dinov2_main",
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate).expanduser()
            if (path / "hubconf.py").exists():
                return path
        return None

    @classmethod
    def _find_weights_path(cls, backbone: str) -> Path | None:
        explicit_path = os.environ.get("DINO_CHECKPOINT_PATH")
        if explicit_path:
            path = Path(explicit_path).expanduser()
            if path.exists():
                return path

        repo_root = cls._repo_root()
        user_root = repo_root.parents[1]
        torch_home = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
        candidate_dirs = [
            os.environ.get("DINO_CHECKPOINT_DIR"),
            repo_root / "playground" / "Pretrain_models" / "dinov2",
            repo_root / "playground" / "Pretrained_models" / "dinov2",
            repo_root / "playground" / "Pretrain_models",
            repo_root / "playground" / "Pretrained_models",
            repo_root.parent / "torch_cache" / "checkpoints",
            user_root / "torch_cache" / "checkpoints",
            Path.home() / "torch_cache" / "checkpoints",
            torch_home / "hub" / "checkpoints",
            torch_home / "checkpoints",
        ]
        for candidate_dir in candidate_dirs:
            if candidate_dir is None:
                continue
            candidate_dir = Path(candidate_dir).expanduser()
            for weight_name in cls._candidate_weight_names(backbone):
                path = candidate_dir / weight_name
                if path.exists():
                    return path
        return None

    def _load_model(self) -> torch.nn.Module:
        local_repo = self._find_local_repo()
        weights_path = self._find_weights_path(self.backbone)
        if local_repo is not None and weights_path is not None:
            print(f"Loading DINOv2 from local repo: {local_repo}")
            print(f"Loading DINOv2 weights: {weights_path}")
            model = torch.hub.load(
                local_repo.as_posix(),
                self.backbone,
                source="local",
                pretrained=False,
            )
            state_dict = torch.load(weights_path.as_posix(), map_location="cpu")
            model.load_state_dict(state_dict)
            return model

        if local_repo is None:
            print("Local DINOv2 repo not found; falling back to torch hub.")
        else:
            print(f"Local DINOv2 weights for {self.backbone} not found; falling back to torch hub.")
        return torch.hub.load("facebookresearch/dinov2", self.backbone)

    def _to_chw_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image [H, W, C], got {image.shape}")
        if image.shape[-1] == 3:
            image = np.moveaxis(image, -1, 0)
        elif image.shape[0] == 3:
            image = image
        else:
            raise ValueError(f"Expected RGB image, got shape {image.shape}")
        return np.ascontiguousarray(image)

    def _to_dino_tensor(self, images: list[np.ndarray]) -> torch.Tensor:
        batch = np.stack([self._to_chw_image(image) for image in images], axis=0)
        tensor = torch.from_numpy(batch).to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        tensor.div_(255.0)
        if tuple(tensor.shape[-2:]) != DINO_IMAGE_SIZE:
            tensor = F.interpolate(
                tensor,
                size=DINO_IMAGE_SIZE,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return tensor.sub_(self._dino_mean).div_(self._dino_std)

    def __call__(self, image: np.ndarray | list[np.ndarray]) -> np.ndarray:
        is_single = isinstance(image, np.ndarray)
        images = [image] if is_single else list(image)
        if len(images) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        tensor = self._to_dino_tensor(images)
        with torch.inference_mode():
            features = self.model.forward_features(tensor)["x_norm_patchtokens"]
            feature = features.mean(dim=1)
        feature_np = feature.detach().cpu().float().numpy()
        if feature_np.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Online DINO produced dim {feature_np.shape[-1]}, expected {self.feature_dim}"
            )
        feature_np = feature_np.astype(np.float32, copy=False)
        return feature_np[0] if is_single else feature_np


class Gr00tMemoryPolicy(Gr00tPolicy):
    """GR00T policy that injects online memory fields before model inference."""

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
        memory_mode: str = "online",
        memory_num_events: int | None = None,
        memory_interval: int | None = None,
        memory_video_key: str | None = None,
        memory_feature_dim: int | None = None,
        memory_action_dim: int | None = None,
        memory_dino_backbone: str | None = None,
        memory_dino_device: str | None = None,
        n_action_steps: int | None = None,
    ):
        if memory_mode not in {"online", "zero"}:
            raise ValueError(f"memory_mode must be 'online' or 'zero', got {memory_mode!r}")
        self.memory_mode = memory_mode
        self.memory_num_events = int(memory_num_events) if memory_num_events is not None else 4
        self.memory_interval = int(memory_interval) if memory_interval is not None else 4
        self.memory_video_key = memory_video_key or "video.robot0_agentview_left"
        self.memory_feature_dim = int(memory_feature_dim) if memory_feature_dim is not None else 768
        self.memory_action_dim = int(memory_action_dim) if memory_action_dim is not None else 32
        self.memory_dino_backbone = memory_dino_backbone or "dinov2_vitb14_reg"
        self.memory_dino_device = memory_dino_device
        self.n_action_steps = int(n_action_steps) if n_action_steps is not None else None
        self._memory_num_events_was_explicit = memory_num_events is not None
        self._memory_video_key_was_explicit = memory_video_key is not None
        self._memory_dino_backbone_was_explicit = memory_dino_backbone is not None
        self._n_action_steps_was_explicit = n_action_steps is not None
        self._memory_histories: list[dict[str, Any]] = []
        self._dino_extractor: OnlineDINOFeatureExtractor | None = None

        super().__init__(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=denoising_steps,
            device=device,
        )
        self._configure_memory_from_model()

    def _configure_memory_from_model(self) -> None:
        if not getattr(self.model, "memory_enabled", False):
            raise ValueError(
                "Loaded checkpoint does not have GR00T memory enabled. "
                "Use a checkpoint trained with scripts/gr00t_memory_finetune.py."
            )

        memory_cfg = getattr(self.model.config, "memory_cfg", None) or {}
        if not self._memory_num_events_was_explicit:
            self.memory_num_events = int(memory_cfg.get("num_events", self.memory_num_events))
        if not self._memory_video_key_was_explicit and memory_cfg.get("video_key") is not None:
            self.memory_video_key = str(memory_cfg["video_key"])
        if (
            not self._memory_dino_backbone_was_explicit
            and memory_cfg.get("feature_backbone") is not None
        ):
            self.memory_dino_backbone = str(memory_cfg["feature_backbone"])
        self.memory_interval = int(memory_cfg.get("action_chunk_len", self.memory_interval))
        self.memory_feature_dim = int(memory_cfg.get("vision_dim", self.memory_feature_dim))
        self.memory_action_dim = int(memory_cfg.get("action_dim", self.memory_action_dim))
        if not self._n_action_steps_was_explicit:
            self.n_action_steps = self.memory_interval
        assert self.n_action_steps is not None

        if self.memory_interval % self.n_action_steps != 0:
            print(
                "Warning: online memory is densest when memory_interval is divisible by "
                f"n_action_steps. Got memory_interval={self.memory_interval}, "
                f"n_action_steps={self.n_action_steps}."
            )

        print(
            "GR00T memory policy ready: "
            f"mode={self.memory_mode}, events={self.memory_num_events}, "
            f"interval={self.memory_interval}, action_dim={self.memory_action_dim}, "
            f"feature_dim={self.memory_feature_dim}, video_key={self.memory_video_key}, "
            f"n_action_steps={self.n_action_steps}"
        )

    def _load_model(self, model_path):
        model = GR00T_N1_5_Memory.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()
        model.to(device=self.device)  # type: ignore[arg-type]

        expected_action_horizon = len(self._modality_config["action"].delta_indices)
        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} "
                f"(was {model.action_head.config.action_horizon})"
            )
            from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon
            new_action_head = FlowmatchingActionHead(new_action_head_config)
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
            model.action_head = new_action_head
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        self.model = model

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        device_type = (
            "cuda"
            if str(self.device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        autocast_ctx = (
            torch.autocast(device_type=device_type, dtype=COMPUTE_DTYPE)
            if device_type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            model_pred = self.model.get_action(normalized_input)
        return model_pred["action_pred"].float()

    def reset_memory(self, payload: dict | list[int] | None = None) -> dict[str, Any]:
        """Reset all memory histories or only selected vector-env indices."""
        indices = None
        if isinstance(payload, dict):
            indices = payload.get("indices")
        elif payload is not None:
            indices = payload

        if indices is None:
            self._memory_histories = []
            return {"status": "ok", "reset": "all"}

        reset_indices = [int(index) for index in indices]
        for index in reset_indices:
            if 0 <= index < len(self._memory_histories):
                self._memory_histories[index] = self._new_memory_history()
        return {"status": "ok", "reset": reset_indices}

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        for key, value in obs_copy.items():
            if not isinstance(value, np.ndarray):
                obs_copy[key] = np.array(value)

        batch_size = self._infer_batch_size(obs_copy)
        self._augment_memory(obs_copy, batch_size)

        normalized_input = self.apply_transforms(obs_copy)
        normalized_action = self._get_action_from_normalized_input(normalized_input)

        if self.memory_mode == "online":
            self._append_memory_actions(normalized_action)

        unnormalized_action = self._get_unnormalized_action(normalized_action)
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def _infer_batch_size(self, observations: dict[str, np.ndarray]) -> int:
        for key, value in observations.items():
            if key.startswith(("video.", "state.")):
                if value.ndim == 0:
                    continue
                return int(value.shape[0])
        raise ValueError("Unable to infer batch size from observations.")

    @staticmethod
    def _new_memory_history() -> dict[str, Any]:
        return {"images": {}, "actions": [], "feature_cache": {}}

    def _ensure_memory_histories(self, batch_size: int) -> None:
        if len(self._memory_histories) != batch_size:
            self._memory_histories = [self._new_memory_history() for _ in range(batch_size)]

    def _augment_memory(self, observations: dict[str, np.ndarray], batch_size: int) -> None:
        if self.memory_mode == "zero":
            observations.update(self._empty_memory_batch(batch_size))
            return

        self._ensure_memory_histories(batch_size)
        for batch_idx in range(batch_size):
            image = self._current_memory_image(observations, batch_idx)
            self._append_memory_observation(batch_idx, image)

        event_specs = [
            (batch_idx, *self._memory_event_indices(batch_idx)) for batch_idx in range(batch_size)
        ]
        self._cache_missing_memory_features(event_specs)

        memory_examples = [
            self._online_memory_example_from_indices(batch_idx, pre_indices, post_indices, valid_mask)
            for batch_idx, pre_indices, post_indices, valid_mask in event_specs
        ]
        for key in ("mem_actions", "mem_valid_mask", "mem_pre_feat", "mem_post_feat"):
            observations[key] = np.stack([example[key] for example in memory_examples], axis=0)

    def _current_memory_image(self, observations: dict[str, np.ndarray], batch_idx: int) -> np.ndarray:
        if self.memory_video_key not in observations:
            available = [key for key in observations if key.startswith("video.")]
            raise KeyError(
                f"memory_video_key={self.memory_video_key!r} not found. "
                f"Available video keys: {available}"
            )
        view = observations[self.memory_video_key]
        image = view[batch_idx, -1]
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
            image = np.moveaxis(image, 0, -1)
        return image.astype(np.uint8, copy=False)

    def _append_memory_observation(self, batch_idx: int, image: np.ndarray) -> None:
        history = self._memory_histories[batch_idx]
        current_step = len(history["actions"])
        history["images"][current_step] = image

    def _memory_event_indices(self, batch_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        history = self._memory_histories[batch_idx]
        current_step = len(history["actions"])
        event_offsets = (
            np.arange(self.memory_num_events, 0, -1, dtype=np.int64) * self.memory_interval
        )
        pre_indices = current_step - event_offsets
        post_indices = pre_indices + self.memory_interval

        valid = []
        for pre_idx, post_idx in zip(pre_indices, post_indices):
            has_actions = (
                pre_idx >= 0
                and post_idx <= current_step
                and len(history["actions"][int(pre_idx) : int(post_idx)]) == self.memory_interval
            )
            has_images = int(pre_idx) in history["images"] and int(post_idx) in history["images"]
            valid.append(bool(has_actions and has_images))
        return pre_indices, post_indices, np.asarray(valid, dtype=bool)

    def _online_memory_example(self, batch_idx: int) -> dict[str, np.ndarray]:
        pre_indices, post_indices, valid_mask = self._memory_event_indices(batch_idx)
        self._cache_missing_memory_features([(batch_idx, pre_indices, post_indices, valid_mask)])
        return self._online_memory_example_from_indices(
            batch_idx,
            pre_indices,
            post_indices,
            valid_mask,
        )

    def _online_memory_example_from_indices(
        self,
        batch_idx: int,
        pre_indices: np.ndarray,
        post_indices: np.ndarray,
        valid_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "mem_actions": self._build_memory_actions(batch_idx, pre_indices, valid_mask),
            "mem_valid_mask": valid_mask,
            "mem_pre_feat": self._build_memory_features(batch_idx, pre_indices, valid_mask),
            "mem_post_feat": self._build_memory_features(batch_idx, post_indices, valid_mask),
        }

    def _cache_missing_memory_features(
        self,
        event_specs: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        requests: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for batch_idx, pre_indices, post_indices, valid_mask in event_specs:
            history = self._memory_histories[batch_idx]
            cache = history["feature_cache"]
            for step_indices in (pre_indices, post_indices):
                for step_idx, is_valid in zip(step_indices, valid_mask):
                    step_idx = int(step_idx)
                    key = (batch_idx, step_idx)
                    if (
                        is_valid
                        and step_idx not in cache
                        and step_idx in history["images"]
                        and key not in seen
                    ):
                        requests.append(key)
                        seen.add(key)

        if not requests:
            return

        if self._dino_extractor is None:
            self._dino_extractor = OnlineDINOFeatureExtractor(
                backbone=self.memory_dino_backbone,
                feature_dim=self.memory_feature_dim,
                device=self.memory_dino_device,
            )

        images = [
            self._memory_histories[batch_idx]["images"][step_idx]
            for batch_idx, step_idx in requests
        ]
        features = self._dino_extractor(images)
        for (batch_idx, step_idx), feature in zip(requests, features):
            self._memory_histories[batch_idx]["feature_cache"][step_idx] = feature

    def _build_memory_actions(
        self,
        batch_idx: int,
        pre_indices: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        history = self._memory_histories[batch_idx]
        mem_actions = np.zeros(
            (self.memory_num_events, self.memory_interval, self.memory_action_dim),
            dtype=np.float32,
        )
        for event_idx, (pre_idx, is_valid) in enumerate(zip(pre_indices, valid_mask)):
            if not is_valid:
                continue
            actions = history["actions"][int(pre_idx) : int(pre_idx) + self.memory_interval]
            mem_actions[event_idx] = np.asarray(actions, dtype=np.float32)
        return mem_actions

    def _build_memory_features(
        self,
        batch_idx: int,
        step_indices: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        features = np.zeros((self.memory_num_events, self.memory_feature_dim), dtype=np.float32)
        for event_idx, (step_idx, is_valid) in enumerate(zip(step_indices, valid_mask)):
            if is_valid:
                features[event_idx] = self._memory_feature_at(batch_idx, int(step_idx))
        return features

    def _memory_feature_at(self, batch_idx: int, step_idx: int) -> np.ndarray:
        history = self._memory_histories[batch_idx]
        cache = history["feature_cache"]
        if step_idx in cache:
            return cache[step_idx]

        if self._dino_extractor is None:
            self._dino_extractor = OnlineDINOFeatureExtractor(
                backbone=self.memory_dino_backbone,
                feature_dim=self.memory_feature_dim,
                device=self.memory_dino_device,
            )
        feature = self._dino_extractor(history["images"][step_idx])
        cache[step_idx] = feature
        return feature

    def _empty_memory_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        return {
            "mem_actions": np.zeros(
                (
                    batch_size,
                    self.memory_num_events,
                    self.memory_interval,
                    self.memory_action_dim,
                ),
                dtype=np.float32,
            ),
            "mem_valid_mask": np.zeros((batch_size, self.memory_num_events), dtype=bool),
            "mem_pre_feat": np.zeros(
                (batch_size, self.memory_num_events, self.memory_feature_dim),
                dtype=np.float32,
            ),
            "mem_post_feat": np.zeros(
                (batch_size, self.memory_num_events, self.memory_feature_dim),
                dtype=np.float32,
            ),
        }

    def _append_memory_actions(self, normalized_action: torch.Tensor) -> None:
        actions = normalized_action.detach().cpu().float().numpy()
        if actions.ndim != 3:
            raise ValueError(f"Expected normalized actions [B, T, D], got {actions.shape}")
        if self.n_action_steps is None:
            raise ValueError("n_action_steps is not configured.")
        if actions.shape[1] < self.n_action_steps:
            raise ValueError(
                f"Model produced only {actions.shape[1]} action steps, "
                f"but n_action_steps={self.n_action_steps}"
            )
        if actions.shape[-1] > self.memory_action_dim:
            raise ValueError(
                f"Model action dim {actions.shape[-1]} exceeds memory_action_dim={self.memory_action_dim}"
            )

        if actions.shape[-1] < self.memory_action_dim:
            pad_width = self.memory_action_dim - actions.shape[-1]
            actions = np.pad(actions, ((0, 0), (0, 0), (0, pad_width)), mode="constant")

        self._ensure_memory_histories(actions.shape[0])
        for batch_idx in range(actions.shape[0]):
            history = self._memory_histories[batch_idx]
            for action in actions[batch_idx, : self.n_action_steps]:
                history["actions"].append(action.astype(np.float32, copy=False))
