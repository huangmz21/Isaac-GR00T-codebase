# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune GR00T N1.5 with action-effect memory.

This script is intentionally separate from ``scripts/gr00t_finetune.py`` so the
original GR00T training path remains unchanged.
"""

import os

os.environ.setdefault("TRANSFORMERS_VIDEO_BACKEND", "av")
os.environ.setdefault("MUJOCO_GL", "glx")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import copy
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from transformers import TrainingArguments

from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

from gr00t.data.dataset import LeRobotMixtureDataset
from gr00t.data.memory_dataset import MemoryLeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
from gr00t.model.gr00t_n1_memory import GR00T_N1_5_Memory
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated",
    category=UserWarning,
    module="torchvision.io._video_deprecation_warning",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL_PATH = os.getenv(
    "GR00T_BASE_MODEL_PATH",
    str(REPO_ROOT / "pretrained_models/GR00T-N1.5-3B"),
)
DEFAULT_OUTPUT_DIR = os.getenv(
    "GR00T_OUTPUT_DIR",
    str(REPO_ROOT / "outputs/gr00t_memory_finetune"),
)
DEFAULT_MEMORY_FEATURE_ROOT = os.getenv(
    "GR00T_MEMORY_FEATURE_ROOT",
    str(REPO_ROOT / "memory_features/dino_features"),
)


@dataclass
class ArgsConfig:
    """Configuration for GR00T memory fine-tuning."""

    # Dataset parameters
    dataset_soup: str | None = None
    """Dataset soup name from RoboCasa DATASET_SOUP_REGISTRY."""

    output_dir: str = DEFAULT_OUTPUT_DIR
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "panda_omron"
    """Data configuration name from DATA_CONFIG_MAP."""

    # Training parameters
    batch_size: int = 16
    """Per-GPU batch size for training."""

    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps."""

    max_steps: int = 300000
    """Maximum number of training steps."""

    num_gpus: int = 8
    """Number of GPUs to use for training."""

    save_steps: int = 20000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = DEFAULT_BASE_MODEL_PATH
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = True
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the action-head projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the action-head DiT."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Memory parameters
    memory_num_events: int = 4
    """Number of historical memory events per sample."""

    memory_interval: int = 4
    """Number of action steps in each memory event."""

    memory_video_key: str = "video.robot0_agentview_left"
    """Configured video key used to index memory features."""

    memory_feature_root: str = DEFAULT_MEMORY_FEATURE_ROOT
    """Root directory for precomputed DINO memory features."""

    memory_feature_dataset_root: str | None = None
    """Dataset root used to derive feature-cache relative paths."""

    memory_feature_strip_prefixes: str | None = None
    """Comma-separated dataset-relative prefixes to strip before mapping into feature root."""

    memory_feature_subdir: str = "dino_features"
    """Feature subdirectory under each dataset when memory_feature_root is unset."""

    memory_feature_backbone: str = "dinov2_vitb14_reg"
    """Backbone name used in the feature-cache path."""

    memory_feature_video_key: str | None = "observation.images.robot0_agentview_left"
    """Original video key used in the feature-cache path."""

    memory_feature_dim: int = 768
    """Dimension of precomputed memory features."""

    memory_dim: int = 768
    """Internal memory key/token dimension."""

    memory_action_dim: int = 32
    """Action dimension used by memory after normalization/padding."""

    memory_fusion: Literal["global", "per_token"] = "per_token"
    """How backbone tokens retrieve memory."""

    memory_retrieval_layers: int = 2
    """Number of per-token memory retrieval blocks."""

    memory_retrieval_fusion: Literal["gate", "add"] = "gate"
    """How retrieved memory is fused into backbone tokens."""

    memory_gate_bias_init: float = 2.0
    """Gate bias init; positive values initially prefer original backbone tokens."""

    memory_feature_strict: bool = True
    """Whether to error if feature length is shorter than trajectory length."""

    memory_feature_mmap_mode: str | None = None
    """np.load mmap mode for feature arrays. None is more stable on network filesystems."""

    memory_feature_cache_size: int = 16
    """Number of per-episode feature arrays cached by each worker."""

    # Advanced training parameters
    learning_rate: float = 3e-5
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for LORA. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for LORA."""

    lora_dropout: float = 0.1
    """Dropout rate for LORA."""

    lora_full_model: bool = False
    """Whether to use full-model LORA. If False, only the action head is trained."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    ddp_find_unused_parameters: bool = False
    """Whether DDP should search for unused parameters."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training."""

    video_backend: Literal["decord", "torchvision_av", "opencv"] = "opencv"
    """Video backend to use for training."""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset."""

    balance_trajectory_weights: bool = True
    """Sample trajectories weighted by length within a dataset."""

    ds_weights_alpha: float = 0.4
    """Dataset weighting exponent."""


def _memory_dataset_kwargs(config: ArgsConfig) -> dict:
    return {
        "memory_num_events": config.memory_num_events,
        "memory_interval": config.memory_interval,
        "memory_video_key": config.memory_video_key,
        "memory_feature_root": config.memory_feature_root,
        "memory_feature_dataset_root": config.memory_feature_dataset_root,
        "memory_feature_strip_prefixes": config.memory_feature_strip_prefixes,
        "memory_feature_subdir": config.memory_feature_subdir,
        "memory_feature_backbone": config.memory_feature_backbone,
        "memory_feature_video_key": config.memory_feature_video_key,
        "memory_feature_strict": config.memory_feature_strict,
        "memory_feature_mmap_mode": config.memory_feature_mmap_mode,
        "memory_feature_cache_size": config.memory_feature_cache_size,
        "memory_action_dim": config.memory_action_dim,
    }


def _model_memory_cfg(config: ArgsConfig) -> dict:
    return {
        "enabled": True,
        "num_events": config.memory_num_events,
        "video_key": config.memory_video_key,
        "feature_backbone": config.memory_feature_backbone,
        "feature_video_key": config.memory_feature_video_key,
        "vision_dim": config.memory_feature_dim,
        "mem_dim": config.memory_dim,
        "action_dim": config.memory_action_dim,
        "action_chunk_len": config.memory_interval,
        "fusion": config.memory_fusion,
        "per_token_retrieval_layers": config.memory_retrieval_layers,
        "per_token_retrieval_fusion": config.memory_retrieval_fusion,
        "gate_bias_init": config.memory_gate_bias_init,
    }


def main(config: ArgsConfig):
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    dataset_soup = config.dataset_soup
    assert dataset_soup in DATASET_SOUP_REGISTRY
    ds_soup_list = copy.deepcopy(DATASET_SOUP_REGISTRY[dataset_soup])
    print(ds_soup_list)

    memory_kwargs = _memory_dataset_kwargs(config)
    if len(ds_soup_list) == 1:
        ds_meta = ds_soup_list[0]
        train_dataset = MemoryLeRobotSingleDataset(
            dataset_path=ds_meta["path"],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
            filter_key=ds_meta["filter_key"],
            **memory_kwargs,
        )
    else:
        single_datasets = []
        for ds_meta in ds_soup_list:
            ds_path = ds_meta["path"]
            ds_filter_key = ds_meta["filter_key"]
            assert os.path.exists(ds_path), f"Dataset path {ds_path} does not exist"
            dataset = MemoryLeRobotSingleDataset(
                dataset_path=ds_path,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
                filter_key=ds_filter_key,
                **memory_kwargs,
            )
            single_datasets.append(dataset)

        ds_weights = np.array([np.power(len(dataset), config.ds_weights_alpha) for dataset in single_datasets])
        ds_weights = ds_weights / ds_weights[0]
        print("dataset weights:", ds_weights)
        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(dataset, ds_w) for dataset, ds_w in zip(single_datasets, ds_weights)],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
        print(f"Loaded {len(single_datasets)} memory datasets")

    data_action_horizon = len(data_config_cls.action_indices)
    model = GR00T_N1_5_Memory.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        memory_cfg=_model_memory_cfg(config),
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

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=100,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
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
    config = tyro.cli(ArgsConfig)

    print("\n" + "=" * 50)
    print("GR00T MEMORY FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
            ]
            for key, value in vars(config).items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    cmd.append(f"--{key.replace('_', '-')}" if value else f"--no-{key.replace('_', '-')}")
                else:
                    cmd.append(f"--{key.replace('_', '-')}")
                    if isinstance(value, (list, tuple)):
                        cmd.extend(str(item) for item in value)
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
