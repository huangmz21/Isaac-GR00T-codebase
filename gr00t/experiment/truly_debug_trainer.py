"""
真正确保数据一致的Trainer

关键点：
1. 禁用DistributedSampler
2. 让每个GPU都加载完整的batch
3. 但每个GPU只计算自己rank对应的那部分
"""

from gr00t.experiment.trainer import DualBrainTrainer
from torch.utils.data import DataLoader, SequentialSampler
import torch
import torch.distributed as dist


class TrulyDebugDualBrainTrainer(DualBrainTrainer):
    """
    真正的调试Trainer：确保单卡和多卡看到完全相同的数据

    工作原理：
    1. 所有GPU加载相同的batch（例如样本[0-7]）
    2. DDP在backward时会自动同步梯度
    3. 最终效果等同于单卡用batch_size=8
    """

    def get_train_dataloader(self):
        """
        创建不使用DistributedSampler的DataLoader
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # 使用SequentialSampler，确保顺序一致
        sampler = SequentialSampler(train_dataset)

        # 关键：在DDP模式下，我们需要让每个GPU的batch_size等于总的global_batch_size
        # 然后每个GPU只处理其中的一部分
        if self.args.world_size > 1:
            # 多GPU模式：每个GPU加载完整的global batch
            batch_size = self.args.per_device_train_batch_size * self.args.world_size
        else:
            # 单GPU模式
            batch_size = self.args.per_device_train_batch_size

        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False,  # workers=0时必须False
        )

        return dataloader

    def training_step(self, model, inputs):
        """
        重写training_step，在DDP模式下只处理对应rank的数据片段
        """
        model.train()

        # 在DDP模式下，手动分片数据
        if self.args.world_size > 1:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # 计算当前rank应该处理的数据范围
            batch_size = self.args.per_device_train_batch_size
            start_idx = rank * batch_size
            end_idx = start_idx + batch_size

            # 分片inputs中的每个tensor
            sliced_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.size(0) > 0:
                    sliced_inputs[key] = value[start_idx:end_idx]
                else:
                    sliced_inputs[key] = value

            inputs = sliced_inputs

        # 调用父类的training_step
        return super().training_step(model, inputs)
