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

"""Epoch-based fine-tuning entry point (standalone, non-invasive).

This script is a **separate entry point** that does not modify
``gr00t_finetune.py`` at all. It reuses that script's ``ArgsConfig`` and
model-setup logic via import, and differs in exactly two ways:

  1. Multi-dataset training uses :class:`ConcatLeRobotDataset` — a plain
     concatenation of all datasets with NO sampling weights and NO
     ``n^alpha``. Every step of every dataset appears exactly once per epoch
     (sampling without replacement; the trainer's sampler shuffles the order).

  2. Training is driven by **epochs** (``num_train_epochs``) instead of a fixed
     ``max_steps``. Setting ``max_steps=-1`` lets HF Trainer honor the epoch
     count.

Together these give a clean control-variable setup: "single task for E epochs"
and "pooled multi task for E epochs" expose every sample the same expected
number of times (E).

Usage mirrors gr00t_finetune.py, with one extra flag:

    python scripts/gr00t_finetune_epoch.py \\
        --dataset-soup <name> --data-config <cfg> \\
        --num-train-epochs 30 --output-dir /path/to/out
"""

import os

os.environ["TRANSFORMERS_VIDEO_BACKEND"] = "av"

import warnings

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated",
    category=UserWarning,
    module="torchvision.io._video_deprecation_warning",
)

import copy
from dataclasses import dataclass

import torch
import tyro
from transformers import TrainingArguments

from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

from gr00t.data.concat_dataset import ConcatLeRobotDataset
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner

# Reuse the existing config dataclass without modifying the original script.
from gr00t_finetune import ArgsConfig as _BaseArgsConfig


@dataclass
class EpochArgsConfig(_BaseArgsConfig):
    """Same as the base config, plus epoch-based training controls."""

    num_train_epochs: int = 30
    """Number of full passes over the (concatenated) dataset.

    Because each epoch is a no-replacement sweep over every step, this is the
    knob you hold constant when comparing single-task vs multi-task runs."""

    concat_metadata_method: str = "min_max"
    """Percentile mixing method when merging per-dataset normalization stats.
    One of 'min_max' or 'weighted_average'."""


def build_dataset(config: EpochArgsConfig):
    """Build either a single dataset or a naive concatenation of datasets."""
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    assert config.dataset_soup in DATASET_SOUP_REGISTRY
    ds_soup_list = copy.deepcopy(DATASET_SOUP_REGISTRY[config.dataset_soup])
    print(ds_soup_list)

    if len(ds_soup_list) == 1:
        ds_meta = ds_soup_list[0]
        return LeRobotSingleDataset(
            dataset_path=ds_meta["path"],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
            filter_key=ds_meta["filter_key"],
        )

    single_datasets = []
    for ds_meta in ds_soup_list:
        ds_path = ds_meta["path"]
        assert os.path.exists(ds_path), f"Dataset path {ds_path} does not exist"
        single_datasets.append(
            LeRobotSingleDataset(
                dataset_path=ds_path,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
                filter_key=ds_meta["filter_key"],
            )
        )

    # Naive pool: no weights, no n^alpha — just concatenate every step.
    dataset = ConcatLeRobotDataset(
        datasets=single_datasets,
        metadata_config={"percentile_mixing_method": config.concat_metadata_method},
    )
    print(dataset)
    print(f"Concatenated {len(single_datasets)} datasets, {len(dataset)} total steps")
    return dataset


def setup_model(config: EpochArgsConfig, train_dataset):
    """Load and configure the model. Mirrors gr00t_finetune.py step 2."""
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.utils.peft import get_lora_model

    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    data_action_horizon = len(data_config_cls.action_indices)

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )

    if data_action_horizon != model.action_head.config.action_horizon:
        print(
            f"Recreating action head with action_horizon {data_action_horizon} "
            f"(was {model.action_head.config.action_horizon})"
        )
        new_action_head_config = model.action_head.config
        new_action_head_config.action_horizon = data_action_horizon
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        new_action_head = FlowmatchingActionHead(new_action_head_config)
        new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
        model.action_head = new_action_head

        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon

        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
        )

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    return model


def main(config: EpochArgsConfig):
    # ------------ step 1: dataset ------------
    train_dataset = build_dataset(config)

    # ------------ step 2: model ------------
    model = setup_model(config, train_dataset)

    # ------------ step 3: epoch-based training args ------------
    # The key difference vs gr00t_finetune.py: max_steps=-1 so HF Trainer
    # honors num_train_epochs instead of stopping at a fixed step count.
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=config.num_train_epochs,
        max_steps=-1,  # <-- epoch-driven training
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=100,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    experiment.train()


if __name__ == "__main__":
    import subprocess
    import sys
    from pathlib import Path

    config = tyro.cli(EpochArgsConfig)
    print("\n" + "=" * 60)
    print("Epoch-based fine-tuning (naive concat, no-replacement)")
    print(f"  dataset_soup      : {config.dataset_soup}")
    print(f"  num_train_epochs  : {config.num_train_epochs}")
    print(f"  batch_size/GPU    : {config.batch_size}")
    print("=" * 60 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode: relaunch this same script under torchrun.
            script_path = Path(__file__).absolute()
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
            ]
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    cmd.append(f"--{key.replace('_', '-')}" if value else f"--no-{key.replace('_', '-')}")
                else:
                    cmd.append(f"--{key.replace('_', '-')}")
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
