#!/usr/bin/env python3
"""
验证两个实验的数据加载是否一致
"""

import torch
import numpy as np
import sys
import os

# 设置路径
sys.path.insert(0, '/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T')
os.chdir('/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T')

from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP

print("=" * 70)
print("验证数据加载一致性")
print("=" * 70)

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

# 配置
dataset_soup = "atomic_seen"
data_config_cls = DATA_CONFIG_MAP["panda_omron"]
modality_configs = data_config_cls.modality_config()
transforms = data_config_cls.transform()
embodiment_tag = EmbodimentTag("new_embodiment")

# 加载数据集
ds_soup_list = DATASET_SOUP_REGISTRY[dataset_soup]
print(f"\n数据集数量: {len(ds_soup_list)}")

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
    print(f"  Dataset: {ds_meta['path'].split('/')[-2]}, length: {len(dataset)}")

# 计算权重
ds_weights = np.array([np.power(len(dataset), 0.4) for dataset in single_datasets])
ds_weights = ds_weights / ds_weights[0]
print(f"\nDataset weights: {ds_weights}")

# 测试两次创建dataset，看seed是否生效
print("\n" + "=" * 70)
print("测试1: 使用seed=42创建两次dataset")
print("=" * 70)

set_seed(42)
train_dataset_1 = LeRobotMixtureDataset(
    data_mixture=[(dataset, ds_w) for dataset, ds_w in zip(single_datasets, ds_weights)],
    mode="train",
    balance_dataset_weights=True,
    balance_trajectory_weights=True,
    seed=42,
    metadata_config={"percentile_mixing_method": "weighted_average"},
)

set_seed(42)
train_dataset_2 = LeRobotMixtureDataset(
    data_mixture=[(dataset, ds_w) for dataset, ds_w in zip(single_datasets, ds_weights)],
    mode="train",
    balance_dataset_weights=True,
    balance_trajectory_weights=True,
    seed=42,
    metadata_config={"percentile_mixing_method": "weighted_average"},
)

print(f"\nDataset 1 length: {len(train_dataset_1)}")
print(f"Dataset 2 length: {len(train_dataset_2)}")

# 采样前几个样本，看是否相同
print("\n检查前5个样本的索引是否相同:")
for i in range(5):
    try:
        sample1 = train_dataset_1[i]
        sample2 = train_dataset_2[i]
        # 简单比较一些字段
        same = True
        if 'episode_index' in sample1 and 'episode_index' in sample2:
            if sample1['episode_index'] != sample2['episode_index']:
                same = False
        print(f"  样本 {i}: {'相同' if same else '不同'}")
    except Exception as e:
        print(f"  样本 {i}: 无法比较 ({e})")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
print("\n如果上述样本相同，说明seed控制了数据加载顺序。")
print("如果不同，说明存在其他随机性来源。")
