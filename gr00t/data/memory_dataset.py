# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-aware LeRobot dataset helpers for GR00T.

The dataset packs historical action-effect memories for each sampled step:

    event = (pre_feature, post_feature, normalized_action_chunk)

Only the precomputed feature path is implemented here. This keeps training
lightweight and avoids introducing a DINO dependency into the GR00T repo.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time

import numpy as np
import torch

from gr00t.data.dataset import LeRobotSingleDataset

MEMORY_FEATURE_LOAD_RETRIES = 5
MEMORY_FEATURE_RETRY_DELAY_SECONDS = 0.1


class MemoryLeRobotSingleDataset(LeRobotSingleDataset):
    """LeRobot dataset that adds action-effect memory fields to each sample."""

    def __init__(
        self,
        *args,
        memory_num_events: int = 4,
        memory_interval: int = 4,
        memory_video_key: str | None = None,
        memory_feature_root: str | None = None,
        memory_feature_dataset_root: str | None = None,
        memory_feature_strip_prefixes: str | tuple[str, ...] | None = None,
        memory_feature_subdir: str = "dino_features",
        memory_feature_backbone: str = "dinov2_vitb14_reg",
        memory_feature_video_key: str | None = None,
        memory_feature_strict: bool = True,
        memory_feature_mmap_mode: str | None = None,
        memory_feature_cache_size: int = 16,
        memory_action_dim: int | None = None,
        **kwargs,
    ):
        if memory_num_events <= 0:
            raise ValueError(f"memory_num_events must be positive, got {memory_num_events}")
        if memory_interval <= 0:
            raise ValueError(f"memory_interval must be positive, got {memory_interval}")

        self.memory_num_events = int(memory_num_events)
        self.memory_interval = int(memory_interval)
        self.memory_video_key = memory_video_key
        self.memory_feature_root = memory_feature_root
        self.memory_feature_dataset_root = memory_feature_dataset_root
        self.memory_feature_strip_prefixes = self._parse_feature_strip_prefixes(
            memory_feature_strip_prefixes
        )
        self.memory_feature_subdir = memory_feature_subdir
        self.memory_feature_backbone = memory_feature_backbone
        self.memory_feature_video_key = memory_feature_video_key
        self.memory_feature_strict = memory_feature_strict
        self.memory_feature_mmap_mode = memory_feature_mmap_mode
        self.memory_feature_cache_size = int(memory_feature_cache_size)
        self.memory_action_dim = memory_action_dim
        self._memory_feature_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        super().__init__(*args, **kwargs)

    @staticmethod
    def _safe_feature_key(key: str) -> str:
        return str(key).replace("/", "__")

    @staticmethod
    def _is_stale_file_handle_error(error: OSError) -> bool:
        return getattr(error, "errno", None) == 116 or "Stale file handle" in str(error)

    @staticmethod
    def _parse_feature_strip_prefixes(
        prefixes: str | tuple[str, ...] | None,
    ) -> list[tuple[str, ...]]:
        if prefixes is None:
            return []
        if isinstance(prefixes, str):
            raw_prefixes = [prefix.strip() for prefix in prefixes.split(",")]
        else:
            raw_prefixes = [str(prefix).strip() for prefix in prefixes]
        return [Path(prefix).parts for prefix in raw_prefixes if prefix]

    @staticmethod
    def _dedupe_paths(paths: list[Path]) -> list[Path]:
        deduped = []
        seen = set()
        for path in paths:
            key = path.as_posix()
            if key in seen:
                continue
            deduped.append(path)
            seen.add(key)
        return deduped

    def _strip_feature_prefixes(self, relative_path: Path) -> list[Path]:
        paths = [relative_path]
        parts = relative_path.parts
        for prefix_parts in self.memory_feature_strip_prefixes:
            if len(parts) < len(prefix_parts):
                continue
            if parts[: len(prefix_parts)] != prefix_parts:
                continue
            stripped_parts = parts[len(prefix_parts) :]
            paths.append(Path(*stripped_parts) if stripped_parts else Path("."))
        return self._dedupe_paths(paths)

    def _dataset_feature_relative_paths(self) -> list[Path]:
        dataset_paths = [self.dataset_path]
        try:
            dataset_paths.append(self.dataset_path.resolve())
        except OSError:
            pass

        relative_paths = []
        if self.memory_feature_dataset_root is not None:
            root_paths = [Path(self.memory_feature_dataset_root)]
            try:
                root_paths.append(Path(self.memory_feature_dataset_root).resolve())
            except OSError:
                pass

            for dataset_path in self._dedupe_paths(dataset_paths):
                for root_path in self._dedupe_paths(root_paths):
                    try:
                        relative_path = dataset_path.relative_to(root_path)
                    except ValueError:
                        continue
                    relative_paths.extend(self._strip_feature_prefixes(relative_path))

        for dataset_path in self._dedupe_paths(dataset_paths):
            parts = dataset_path.parts
            for split_marker in ("pretrain", "target"):
                if split_marker not in parts:
                    continue
                marker_index = parts.index(split_marker)
                feature_parts = parts[marker_index + 1 :]
                if feature_parts:
                    relative_paths.append(Path(*feature_parts))
                if feature_parts and feature_parts[-1] == "lerobot":
                    relative_paths.append(Path(*feature_parts[:-1]))
        return self._dedupe_paths(relative_paths)

    def _configured_memory_video_key(self) -> str:
        video_keys = self.modality_keys.get("video", [])
        if self.memory_video_key is None:
            if not video_keys:
                raise ValueError("Memory is enabled but no video modality is configured.")
            return video_keys[0]
        if self.memory_video_key not in video_keys:
            raise ValueError(
                f"memory_video_key={self.memory_video_key!r} is not in configured video keys: {video_keys}"
            )
        return self.memory_video_key

    def _resolve_memory_feature_video_key(self, video_key: str) -> str:
        if self.memory_feature_video_key is not None:
            return self.memory_feature_video_key

        key = video_key.replace("video.", "", 1) if video_key.startswith("video.") else video_key
        if key in getattr(self.lerobot_modality_meta, "video", {}):
            original_key = self.lerobot_modality_meta.video[key].original_key
            return original_key or key
        return video_key

    def _memory_feature_path(self, trajectory_id: int, video_key: str) -> Path:
        feature_video_key = self._resolve_memory_feature_video_key(video_key)
        safe_video_key = self._safe_feature_key(feature_video_key)
        chunk_index = self.get_episode_chunk(trajectory_id)
        relative_path = (
            Path(self.memory_feature_backbone)
            / safe_video_key
            / f"chunk-{chunk_index:03d}"
            / f"episode_{trajectory_id:06d}.npy"
        )

        candidates = []
        if self.memory_feature_root is not None:
            root = Path(self.memory_feature_root)
            for dataset_relative_path in self._dataset_feature_relative_paths():
                candidates.append(root / dataset_relative_path / relative_path)
            candidates.extend(
                [
                    root / self.dataset_name / relative_path,
                    root / relative_path,
                ]
            )
        candidates.append(self.dataset_path / self.memory_feature_subdir / relative_path)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_memory_feature_array(self, trajectory_id: int, video_key: str) -> np.ndarray:
        feature_path = self._memory_feature_path(trajectory_id, video_key)
        if not feature_path.exists():
            raise FileNotFoundError(f"DINO feature file not found: {feature_path}")

        cache_key = feature_path.as_posix()
        if cache_key in self._memory_feature_cache:
            self._memory_feature_cache.move_to_end(cache_key)
            return self._memory_feature_cache[cache_key]

        mmap_mode = self.memory_feature_mmap_mode
        if mmap_mode is not None and str(mmap_mode).lower() in {"false", "none", ""}:
            mmap_mode = None
        for attempt in range(MEMORY_FEATURE_LOAD_RETRIES):
            try:
                feature_array = np.load(feature_path, mmap_mode=mmap_mode)
                break
            except OSError as error:
                is_last_attempt = attempt + 1 == MEMORY_FEATURE_LOAD_RETRIES
                if not self._is_stale_file_handle_error(error) or is_last_attempt:
                    raise
                time.sleep(MEMORY_FEATURE_RETRY_DELAY_SECONDS * (attempt + 1))
        else:
            raise RuntimeError(f"Unable to load DINO feature after retries: {feature_path}")
        self._memory_feature_cache[cache_key] = feature_array
        while len(self._memory_feature_cache) > self.memory_feature_cache_size:
            self._memory_feature_cache.popitem(last=False)
        return feature_array

    def _get_memory_dino_features(
        self,
        trajectory_id: int,
        video_key: str,
        step_indices: np.ndarray,
        valid_mask: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        feature_array = self._load_memory_feature_array(trajectory_id, video_key)
        if feature_array.ndim != 2:
            raise ValueError(f"Expected DINO feature array [T, D], got {feature_array.shape}")
        if self.memory_feature_strict and len(feature_array) < max_length:
            raise ValueError(
                f"DINO feature length {len(feature_array)} is shorter than trajectory length "
                f"{max_length} for trajectory {trajectory_id}"
            )
        if len(feature_array) == 0:
            raise ValueError(f"DINO feature array is empty for trajectory {trajectory_id}")

        step_indices = np.asarray(step_indices, dtype=np.int64)
        feature_valid_mask = valid_mask & (step_indices >= 0) & (step_indices < len(feature_array))
        safe_indices = np.clip(step_indices, 0, len(feature_array) - 1)
        features = np.asarray(feature_array[safe_indices], dtype=np.float32)
        features[~feature_valid_mask] = 0
        return features

    def get_state_or_action_by_step_indices(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = int(self.trajectory_lengths[trajectory_index])
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}"

        subkey = key.replace(modality + ".", "")
        state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = state_or_action_cfg[subkey].original_key or subkey

        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array = np.stack(self.curr_traj_data[le_key])
        le_indices = np.arange(state_or_action_cfg[subkey].start, state_or_action_cfg[subkey].end)
        data_array = data_array[:, le_indices]
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy=padding_strategy,
        )

    def _concat_memory_actions(self, data: dict, concat_transform) -> dict:
        action_concat_order = getattr(concat_transform, "action_concat_order", None)
        if action_concat_order is None:
            action_concat_order = self.modality_keys["action"]

        missing_keys = [key for key in action_concat_order if key not in data]
        if missing_keys:
            raise ValueError(f"Cannot concat memory actions; missing keys: {missing_keys}")

        action_values = [data.pop(key) for key in action_concat_order]
        if all(isinstance(value, torch.Tensor) for value in action_values):
            data["action"] = torch.cat(action_values, dim=-1)
        else:
            action_values = [np.asarray(value) for value in action_values]
            data["action"] = np.concatenate(action_values, axis=-1)
        return data

    def _apply_action_transforms_before_groot(self, data: dict) -> dict:
        transforms = getattr(self.transforms, "transforms", [])
        for transform in transforms:
            transform_name = transform.__class__.__name__
            if transform_name == "GR00TTransform":
                break

            if transform_name == "ConcatTransform":
                data = self._concat_memory_actions(data, transform)
                continue

            apply_to = getattr(transform, "apply_to", [])
            if not any(key in data for key in apply_to):
                continue
            if not any(str(key).startswith("action.") for key in apply_to):
                continue
            data = transform(data)
        return data

    def _max_action_dim(self) -> int | None:
        if self.memory_action_dim is not None:
            return self.memory_action_dim
        for transform in getattr(self.transforms, "transforms", []):
            if hasattr(transform, "max_action_dim"):
                return int(transform.max_action_dim)
        return None

    def _get_memory_actions(
        self,
        trajectory_id: int,
        action_steps: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        num_events, interval = action_steps.shape
        flat_action_steps = action_steps.reshape(-1)
        action_data = {}
        for action_key in self.modality_keys["action"]:
            action_data[action_key] = self.get_state_or_action_by_step_indices(
                trajectory_id=trajectory_id,
                modality="action",
                key=action_key,
                step_indices=flat_action_steps,
                padding_strategy="zero",
            )

        action_data = self._apply_action_transforms_before_groot(action_data)
        if "action" not in action_data:
            raise ValueError("Memory action transform did not produce concatenated 'action'.")

        action = action_data["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)

        max_action_dim = self._max_action_dim()
        if max_action_dim is not None:
            if action.shape[-1] > max_action_dim:
                raise ValueError(
                    f"Memory action dim {action.shape[-1]} exceeds max_action_dim {max_action_dim}"
                )
            if action.shape[-1] < max_action_dim:
                action = np.pad(action, ((0, 0), (0, max_action_dim - action.shape[-1])), "constant")

        mem_actions = action.reshape(num_events, interval, -1)
        mem_actions[~valid_mask] = 0
        return mem_actions.astype(np.float32)

    def _pack_memory_sample(self, trajectory_id: int, base_index: int) -> dict:
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = int(self.trajectory_lengths[trajectory_index])
        video_key = self._configured_memory_video_key()

        event_offsets = np.arange(self.memory_num_events, 0, -1, dtype=np.int64)
        event_offsets = event_offsets * self.memory_interval
        pre_indices = int(base_index) - event_offsets
        post_indices = pre_indices + self.memory_interval
        valid_mask = (
            (pre_indices >= 0)
            & (post_indices <= int(base_index))
            & (post_indices < max_length)
        )

        action_steps = pre_indices[:, None] + np.arange(self.memory_interval, dtype=np.int64)[None, :]
        return {
            "mem_actions": self._get_memory_actions(trajectory_id, action_steps, valid_mask),
            "mem_valid_mask": valid_mask.astype(bool),
            "mem_pre_feat": self._get_memory_dino_features(
                trajectory_id=trajectory_id,
                video_key=video_key,
                step_indices=pre_indices,
                valid_mask=valid_mask,
                max_length=max_length,
            ),
            "mem_post_feat": self._get_memory_dino_features(
                trajectory_id=trajectory_id,
                video_key=video_key,
                step_indices=post_indices,
                valid_mask=valid_mask,
                max_length=max_length,
            ),
        }

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)
        data.update(self._pack_memory_sample(trajectory_id=trajectory_id, base_index=base_index))
        return data
