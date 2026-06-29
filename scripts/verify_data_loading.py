#!/usr/bin/env python3
"""
最简单的验证脚本：打印前几个batch的数据

用这个脚本可以直接看到单卡和多卡是否加载了相同的数据
"""

import os
import sys
sys.path.insert(0, '/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T')
os.chdir('/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T')

import torch
import numpy as np
import copy

from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("验证数据加载一致性")
print("=" * 70)

# 配置
dataset_soup = "atomic_seen"
data_config_cls = DATA_CONFIG_MAP["panda_omron"]
modality_configs = data_config_cls.modality_config()
transforms = data_config_cls.transform()
embodiment_tag = EmbodimentTag("new_embodiment")

# 加载数据集
ds_soup_list = DATASET_SOUP_REGISTRY[dataset_soup]
print(f"\n加载 {len(ds_soup_list)} 个数据集...")

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

# 创建MixtureDataset（使用seed=42）
train_dataset = LeRobotMixtureDataset(
    data_mixture=[(dataset, ds_w) for dataset, ds_w in zip(single_datasets, ds_weights)],
    mode="train",
    balance_dataset_weights=True,
    balance_trajectory_weights=True,
    seed=42,
    metadata_config={"percentile_mixing_method": "weighted_average"},
)

print(f"数据集总长度: {len(train_dataset)}")

# 创建DataLoader
from torch.utils.data import DataLoader, SequentialSampler
from gr00t.model.transforms import DefaultDataCollator

collator = DefaultDataCollator()

# 模拟单卡：batch_size=8
print("\n" + "=" * 70)
print("模拟单卡加载 (batch_size=8)")
print("=" * 70)

single_loader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=SequentialSampler(train_dataset),
    collate_fn=collator,
    num_workers=0,
)

single_batches = []
for i, batch in enumerate(single_loader):
    if i >= 3:  # 只取前3个batch
        break
    single_batches.append(batch)
    print(f"\nBatch {i}:")
    if 'episode_index' in batch:
        print(f"  episode_index shape: {batch['episode_index'].shape}")
        print(f"  episode_index values: {batch['episode_index'][:8].tolist()}")
    if 'action' in batch:
        print(f"  action shape: {batch['action'].shape}")
        # 打印所有8个样本的action前3维
        for sample_idx in range(min(8, batch['action'].shape[0])):
            print(f"  样本 {sample_idx} action[0, :3]: {batch['action'][sample_idx, 0, :3].tolist()}")

# 模拟多卡：每次加载8个样本，但每个"GPU"只处理2个
print("\n" + "=" * 70)
print("模拟多卡加载 (global_batch=8, 分成4个GPU)")
print("=" * 70)

multi_loader = DataLoader(
    train_dataset,
    batch_size=8,  # 加载完整的8个样本
    sampler=SequentialSampler(train_dataset),
    collate_fn=collator,
    num_workers=0,
)

for i, batch in enumerate(multi_loader):
    if i >= 3:
        break
    print(f"\nBatch {i}:")

    # 打印所有8个样本的完整信息
    if 'episode_index' in batch:
        print(f"  episode_index: {batch['episode_index'][:8].tolist()}")
    if 'action' in batch:
        for sample_idx in range(min(8, batch['action'].shape[0])):
            print(f"  样本 {sample_idx} action[0, :3]: {batch['action'][sample_idx, 0, :3].tolist()}")

    # 同时显示GPU分片信息
    print("  --- GPU分片信息 (仅供参考) ---")
    for gpu_id in range(4):
        start = gpu_id * 2
        end = start + 2
        print(f"  GPU {gpu_id} 处理样本 {start}-{end-1}")

print("\n" + "=" * 70)
print("对比结论")
print("=" * 70)

# 检查第一个batch的数据是否相同
if single_batches:
    batch0 = single_batches[0]
    print("\n单卡Batch 0的前8个样本应该等于多卡Batch 0的所有4个GPU的样本拼接")
    print("如果episode_index和action值都相同，说明数据一致✅")
    print("如果不同，说明有问题需要修复❌")
