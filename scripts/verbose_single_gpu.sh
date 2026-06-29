#!/bin/bash

# Verbose Debug模式：单卡实验
# 打印详细的数据和计算信息

cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T:$PYTHONPATH
export HF_HOME=/home/huangmingzhe/.cache/huggingface
export TRANSFORMERS_CACHE=/home/huangmingzhe/.cache/huggingface/transformers

echo "=========================================="
echo "Verbose Debug - 单卡实验"
echo "输出目录: /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/verbose_single_gpu"
echo "Debug日志: debug_rank0.log"
echo "=========================================="

# Run training with verbose debug trainer
/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/.venv/bin/python << 'PYTHON_EOF'
import os
os.environ.setdefault("TRANSFORMERS_VIDEO_BACKEND", "av")
os.environ.setdefault("MUJOCO_GL", "glx")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import copy
import numpy as np
import torch
from transformers import TrainingArguments

from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import DefaultDataCollator

# 导入verbose debug trainer
from gr00t.experiment.verbose_debug_trainer import VerboseDebugTrainer

print("\n" + "=" * 70)
print("Verbose Debug 训练 - 单卡")
print("=" * 70 + "\n")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 加载数据集
embodiment_tag = EmbodimentTag("new_embodiment")
data_config_cls = DATA_CONFIG_MAP["panda_omron"]
modality_configs = data_config_cls.modality_config()
transforms = data_config_cls.transform()

ds_soup_list = DATASET_SOUP_REGISTRY["atomic_seen"]
single_datasets = []
for ds_meta in ds_soup_list:
    dataset = LeRobotSingleDataset(
        dataset_path=ds_meta["path"],
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="opencv",
        filter_key=ds_meta["filter_key"],
    )
    single_datasets.append(dataset)

ds_weights = np.array([np.power(len(dataset), 0.4) for dataset in single_datasets])
ds_weights = ds_weights / ds_weights[0]

train_dataset = LeRobotMixtureDataset(
    data_mixture=[(dataset, ds_w) for dataset, ds_w in zip(single_datasets, ds_weights)],
    mode="train",
    balance_dataset_weights=True,
    balance_trajectory_weights=True,
    seed=42,
    metadata_config={"percentile_mixing_method": "weighted_average"},
)

print(f"Dataset length: {len(train_dataset)}")

# 加载模型
model = GR00T_N1_5.from_pretrained(
    pretrained_model_name_or_path="/mnt/ssd_data/mingzhe/Model/robocasa365/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000",
    tune_llm=False,
    tune_visual=False,
    tune_projector=True,
    tune_diffusion_model=True,
)

data_action_horizon = len(data_config_cls.action_indices)
if data_action_horizon != model.action_head.config.action_horizon:
    from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
    new_config = model.action_head.config
    new_config.action_horizon = data_action_horizon
    new_action_head = FlowmatchingActionHead(new_config)
    new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
    model.action_head = new_action_head
    model.config.action_horizon = data_action_horizon
    model.action_horizon = data_action_horizon
    model.config.action_head_cfg["action_horizon"] = data_action_horizon
    model.action_head.set_trainable_parameters(tune_projector=True, tune_diffusion_model=True)

model.compute_dtype = "bfloat16"
model.config.compute_dtype = "bfloat16"

# 训练参数
training_args = TrainingArguments(
    output_dir="/mnt/ssd_data/mingzhe/Model/robocasa365/experiments/verbose_single_gpu",
    per_device_train_batch_size=8,
    max_steps=20,  # 只跑20步用于debug
    save_steps=10,
    logging_steps=1,
    learning_rate=3e-5,
    weight_decay=1e-5,
    warmup_ratio=0.05,
    bf16=True,
    tf32=True,
    dataloader_num_workers=0,
    report_to="tensorboard",
    seed=42,
    gradient_accumulation_steps=1,
)

# 创建verbose trainer
data_collator = DefaultDataCollator()
trainer = VerboseDebugTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    compute_dtype="bfloat16",  # 添加compute_dtype参数
)

print("\n开始训练...\n")
trainer.train()

print("\n训练完成！查看 verbose_single_gpu/debug_rank0.log 了解详情")

PYTHON_EOF
