# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Naive concatenation dataset for fair single-vs-multi task comparison.

This module is **purely additive** — it does not modify any existing file.
It provides an alternative to ``LeRobotMixtureDataset`` that:

  1. Concatenates all single datasets into one flat index (no sampling
     weights, no ``n^alpha``, no length balancing). Every step of every
     dataset appears exactly once per epoch.
  2. Iterates **without replacement**: one epoch == every step seen exactly
     once (the actual shuffle order is provided by the trainer's sampler).

Combined with epoch-based training (``num_train_epochs`` instead of
``max_steps``), this makes "single task for E epochs" and "pooled multi task
for E epochs" expose every sample the same expected number of times (E),
which is the clean control-variable setup.

The class reuses ``LeRobotSingleDataset`` entirely for data reading /
transforms, and reuses ``LeRobotMixtureDataset``'s metadata-merging logic for
normalization statistics (weighted by dataset size, which is exactly what a
plain pool implies).
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import DatasetMetadata, EmbodimentTag


class ConcatLeRobotDataset(LeRobotMixtureDataset):
    """A naive concatenation of multiple ``LeRobotSingleDataset`` instances.

    Unlike ``LeRobotMixtureDataset`` (which samples with replacement using
    per-dataset weights), this dataset builds a single flat index over every
    step of every sub-dataset and serves each exactly once per epoch. There is
    no weighting whatsoever: a dataset that is 10x larger simply contributes
    10x as many samples to the pool, which is the standard "throw it all
    together" behavior.

    It subclasses ``LeRobotMixtureDataset`` only so that existing
    ``isinstance(ds, LeRobotMixtureDataset)`` checks (e.g. in ``TrainRunner``)
    keep working unmodified. It deliberately does **not** call the parent
    ``__init__`` — none of the weight-based sampling state is used.
    """

    def __init__(
        self,
        datasets: Sequence[LeRobotSingleDataset],
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
    ):
        """
        Args:
            datasets: The single datasets to concatenate. All must share the
                same embodiment tag (same constraint as the mixture dataset).
            metadata_config: Forwarded to the metadata merge; controls how
                per-dataset percentiles are combined.
        """
        if len(datasets) == 0:
            raise ValueError("ConcatLeRobotDataset requires at least one dataset")

        self.datasets = list(datasets)

        # Per-dataset lengths (number of steps after flattening trajectories).
        self._dataset_lengths = np.array([len(d) for d in self.datasets])

        # Flat global index: for each global index, store (dataset_idx, local_idx).
        # Using cumulative offsets keeps memory small and lookup O(log n).
        self._cum_lengths = np.cumsum(self._dataset_lengths)
        self._total_length = int(self._cum_lengths[-1])

        # For metadata merging we weight each dataset by its size, because a
        # plain pool of all steps is, statistically, exactly a size-weighted
        # mixture of the per-dataset distributions.
        self._dataset_sampling_weights = (
            self._dataset_lengths / self._dataset_lengths.sum()
        )

        # `epoch` exists for API parity with LeRobotMixtureDataset / BaseSampler
        # (the sampler may call set_epoch). It is unused here because sampling
        # order is fully owned by the trainer's sampler, not by this dataset.
        self.epoch = 0

        self.update_metadata(metadata_config)

    # ------------------------------------------------------------------ #
    # Properties mirroring LeRobotMixtureDataset where the runner needs them
    # ------------------------------------------------------------------ #
    @property
    def dataset_lengths(self) -> np.ndarray:
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        return self._dataset_sampling_weights

    def set_epoch(self, epoch: int):
        """Record the epoch. Sampling order is owned by the sampler, so this
        only exists so a sampler that calls ``set_epoch`` does not error."""
        self.epoch = epoch

    def __len__(self) -> int:
        """One epoch == every step of every dataset, exactly once."""
        return self._total_length

    def _locate(self, index: int) -> tuple[int, int]:
        """Map a global index to (dataset_index, local_index)."""
        if index < 0:
            index += self._total_length
        if index < 0 or index >= self._total_length:
            raise IndexError(
                f"Index {index} out of range for ConcatLeRobotDataset of "
                f"length {self._total_length}"
            )
        dataset_index = int(np.searchsorted(self._cum_lengths, index, side="right"))
        prev_cum = self._cum_lengths[dataset_index - 1] if dataset_index > 0 else 0
        local_index = index - int(prev_cum)
        return dataset_index, local_index

    def __getitem__(self, index: int) -> dict:
        dataset_index, local_index = self._locate(index)
        dataset = self.datasets[dataset_index]

        # Reuse the single dataset's own __getitem__ path (read + transforms).
        data = dataset[local_index]

        # Attach task identifiers so the per-task loss tracker keeps working,
        # matching the keys LeRobotMixtureDataset.__getitem__ produces.
        data["dataset_index"] = dataset_index
        task_name = (
            str(dataset.dataset_path).split("/")[-3]
            if "/" in str(dataset.dataset_path)
            else dataset.dataset_name
        )
        data["dataset_name"] = task_name
        return data

    def __str__(self) -> str:
        descriptions = []
        for dataset, length in zip(self.datasets, self.dataset_lengths):
            descriptions.append({"Dataset": str(dataset), "Steps": int(length)})
        return json.dumps(
            {"Concat dataset": descriptions, "Total steps": self._total_length},
            indent=2,
        )

    # ------------------------------------------------------------------ #
    # Metadata merging — reuse the mixture dataset's static helpers so the
    # normalization statistics are computed identically (size-weighted).
    # ------------------------------------------------------------------ #
    def update_metadata(self, metadata_config: dict) -> None:
        """Merge per-dataset metadata into one and set transforms with it.

        Mirrors ``LeRobotMixtureDataset.update_metadata`` so the runner's
        ``merged_metadata`` consumption works unchanged, but weights datasets
        by their size (the correct weighting for a plain pool).
        """
        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}

        # Group metadata by embodiment tag.
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        tag_weights: dict[str, list[float]] = {}
        for dataset, weight in zip(self.datasets, self._dataset_sampling_weights):
            all_metadatas.setdefault(dataset.tag, []).append(dataset.metadata)
            tag_weights.setdefault(dataset.tag, []).append(float(weight))

        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = LeRobotMixtureDataset.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=tag_weights[tag],
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )

        # Apply the merged metadata back onto every sub-dataset's transforms so
        # all samples are normalized with the same pooled statistics.
        for dataset in self.datasets:
            dataset.transforms.set_metadata(self.merged_metadata[dataset.tag])
