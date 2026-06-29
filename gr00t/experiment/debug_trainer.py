"""
自定义Trainer：禁用DistributedSampler，确保所有GPU看到相同的数据顺序

用于debug和对照实验
"""

from gr00t.experiment.trainer import DualBrainTrainer
from torch.utils.data import DataLoader, SequentialSampler
import torch


class DebugDualBrainTrainer(DualBrainTrainer):
    """
    调试版Trainer：禁用DistributedSampler

    在DDP模式下，所有GPU会看到相同的数据，而不是数据集的不同分片。
    这样可以确保单卡和多卡实验使用完全相同的数据，便于对比。

    警告：这会导致梯度计算冗余（所有GPU计算相同batch），仅用于debug！
    """

    def get_train_dataloader(self):
        """
        重写get_train_dataloader方法，使用SequentialSampler而不是DistributedSampler
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # 关键修改：使用SequentialSampler，不使用DistributedSampler
        # 这样所有GPU会按照相同的顺序看到数据
        sampler = SequentialSampler(train_dataset)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

        return dataloader

    def _get_train_sampler(self):
        """
        重写_get_train_sampler方法，返回SequentialSampler
        """
        return SequentialSampler(self.train_dataset)
