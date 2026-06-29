#!/usr/bin/env python
"""Quick validation script to test epoch-based training setup without running actual training."""

import os
import sys

# Suppress warnings
os.environ["TRANSFORMERS_VIDEO_BACKEND"] = "av"
import warnings
warnings.filterwarnings("ignore")

from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
from gr00t.data.concat_dataset import ConcatLeRobotDataset

print("=" * 60)
print("Testing Epoch-based Training Setup")
print("=" * 60)

# 1. Check available datasets
print("\n1. Available dataset soups:")
for name in DATASET_SOUP_REGISTRY.keys():
    num_tasks = len(DATASET_SOUP_REGISTRY[name])
    print(f"   - {name}: {num_tasks} dataset(s)")

# 2. Test ConcatLeRobotDataset import
print("\n2. ConcatLeRobotDataset class:")
print(f"   ✓ Import successful")
print(f"   ✓ Has __len__: {hasattr(ConcatLeRobotDataset, '__len__')}")
print(f"   ✓ Has __getitem__: {hasattr(ConcatLeRobotDataset, '__getitem__')}")
print(f"   ✓ Has merged_metadata: {hasattr(ConcatLeRobotDataset, 'merged_metadata')}")

# 3. Quick dataset instantiation test (single task)
print("\n3. Testing single-task dataset (opendrawer_only):")
try:
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP

    data_config_cls = DATA_CONFIG_MAP["panda_omron"]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    ds_meta = DATASET_SOUP_REGISTRY["opendrawer_only"][0]
    dataset = LeRobotSingleDataset(
        dataset_path=ds_meta["path"],
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        video_backend="opencv",
        filter_key=ds_meta["filter_key"],
    )
    print(f"   ✓ Dataset loaded: {len(dataset)} steps")
    print(f"   ✓ One epoch = {len(dataset)} steps")

except Exception as e:
    print(f"   ✗ Error: {e}")

# 4. Test multi-task concatenation
print("\n4. Testing multi-task concat (opencabinet_opendrawer):")
try:
    ds_soup_list = DATASET_SOUP_REGISTRY["opencabinet_opendrawer"]
    single_datasets = []

    for ds_meta in ds_soup_list:
        ds = LeRobotSingleDataset(
            dataset_path=ds_meta["path"],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            video_backend="opencv",
            filter_key=ds_meta["filter_key"],
        )
        single_datasets.append(ds)

    concat_dataset = ConcatLeRobotDataset(
        datasets=single_datasets,
        metadata_config={"percentile_mixing_method": "min_max"},
    )

    print(f"   ✓ Concatenated {len(single_datasets)} datasets")
    print(f"   ✓ Total steps: {len(concat_dataset)}")
    print(f"   ✓ One epoch = {len(concat_dataset)} steps (all datasets combined)")
    print(f"   ✓ Per-dataset lengths: {concat_dataset.dataset_lengths.tolist()}")

except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("Setup validation complete!")
print("=" * 60)
print("\nTo run actual training:")
print("  Single-task:  ./scripts/example_train_epoch_single.sh")
print("  Multi-task:   ./scripts/example_train_epoch_multi.sh")
