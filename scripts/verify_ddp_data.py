#!/usr/bin/env python3
"""
不运行训练，只验证前几个batch的数据

对比：
1. 单卡加载的数据
2. 多卡（标准DistributedSampler）加载的数据
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
from gr00t.model.transforms import DefaultDataCollator
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("验证标准DistributedSampler下的数据加载")
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

train_dataset = LeRobotMixtureDataset(
    data_mixture=[(dataset, ds_w) for dataset, ds_w in zip(single_datasets, ds_weights)],
    mode="train",
    balance_dataset_weights=True,
    balance_trajectory_weights=True,
    seed=42,
    metadata_config={"percentile_mixing_method": "weighted_average"},
)

print(f"数据集总长度: {len(train_dataset)}")
collator = DefaultDataCollator()

# 单卡：batch_size=8
print("\n" + "=" * 70)
print("单卡 (batch_size=8)")
print("=" * 70)

single_loader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=SequentialSampler(train_dataset),
    collate_fn=collator,
    num_workers=0,
)

for i, batch in enumerate(single_loader):
    if i >= 1:
        break
    print(f"\nBatch {i}:")
    if 'episode_index' in batch:
        print(f"  episode_index: {batch['episode_index'][:8].tolist()}")
    if 'action' in batch:
        for idx in range(8):
            print(f"  样本{idx} action[0,:3]: {batch['action'][idx, 0, :3].tolist()}")

# 模拟多卡DDP：使用DistributedSampler
print("\n" + "=" * 70)
print("多卡DDP模式 (4 GPUs, 每个batch_size=2)")
print("=" * 70)

# 模拟4个GPU，每个rank单独创建DataLoader
for rank in range(4):
    print(f"\n--- GPU {rank} ---")

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=4,
        rank=rank,
        seed=42,
        shuffle=False,  # 不shuffle，确保可复现
    )

    loader = DataLoader(
        train_dataset,
        batch_size=2,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
    )

    for i, batch in enumerate(loader):
        if i >= 1:
            break
        print(f"Batch {i}:")
        if 'episode_index' in batch:
            print(f"  episode_index: {batch['episode_index'][:2].tolist()}")
        if 'action' in batch:
            for idx in range(2):
                print(f"  样本{idx} action[0,:3]: {batch['action'][idx, 0, :3].tolist()}")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
print("\n如果单卡的8个样本 = 多卡4个GPU各自的2个样本按rank顺序拼接，")
print("那说明在使用DistributedSampler时，数据是按rank分配的。")
print("这是DDP的正常行为，不是bug。")
print("\n单卡和多卡的loss不同是因为：")
print("1. 不同的数据顺序")
print("2. 不同的RNG状态（每个rank的seed不同）")
