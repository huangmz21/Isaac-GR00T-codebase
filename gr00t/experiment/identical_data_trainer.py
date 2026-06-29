"""
Debug Trainer: 确保单卡和多卡使用完全相同的数据

核心逻辑：
1. 单卡：batch_size=8，正常加载
2. 多卡：每个GPU的DataLoader都加载相同的8条数据（使用SequentialSampler），
         然后通过自定义collator只处理属于自己rank的那部分

参考 verify_data_loading.py 的正确逻辑
"""

import torch
import torch.distributed as dist
from gr00t.experiment.trainer import DualBrainTrainer
from torch.utils.data import DataLoader, SequentialSampler
import os


class RankSlicingCollator:
    """
    自定义collator：在collate时就只处理当前rank对应的数据

    这样避免了在已经collate好的tensor上切片，保持了数据结构的完整性
    """

    def __init__(self, base_collator, rank, world_size, per_device_batch_size):
        self.base_collator = base_collator
        self.rank = rank
        self.world_size = world_size
        self.per_device_batch_size = per_device_batch_size

    def __call__(self, batch):
        """
        batch: 列表，包含global_batch_size个样本

        只取当前rank对应的slice，然后用base_collator处理
        """
        # 计算当前rank应该处理的数据范围
        start_idx = self.rank * self.per_device_batch_size
        end_idx = start_idx + self.per_device_batch_size

        # 只取属于当前rank的样本
        rank_batch = batch[start_idx:end_idx]

        # 用原始collator处理
        return self.base_collator(rank_batch)


class IdenticalDataTrainer(DualBrainTrainer):
    """
    保证单卡和多卡使用完全相同数据的Trainer

    关键改进：在collator层面切分数据，而不是在tensor层面
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_step = 0

        # 创建debug日志
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.debug_log = open(f"{output_dir}/debug_rank{rank}.log", "w")

    def write_log(self, msg):
        """写日志"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        full_msg = f"[Rank {rank}/{world_size}, Step {self.debug_step}] {msg}"
        print(full_msg)
        self.debug_log.write(full_msg + "\n")
        self.debug_log.flush()

    def get_train_dataloader(self):
        """
        创建DataLoader

        关键：
        1. 使用SequentialSampler确保所有GPU看到相同的数据顺序
        2. 所有GPU的DataLoader都加载global_batch_size的数据
        3. 使用RankSlicingCollator在collate时只处理当前rank的部分
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # 使用SequentialSampler，不使用DistributedSampler
        sampler = SequentialSampler(self.train_dataset)

        # 计算batch size
        if dist.is_initialized():
            # 多GPU：所有GPU都加载global batch
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            per_device_batch_size = self.args.per_device_train_batch_size
            global_batch_size = per_device_batch_size * world_size

            # 创建rank-aware collator
            collator = RankSlicingCollator(
                base_collator=self.data_collator,
                rank=rank,
                world_size=world_size,
                per_device_batch_size=per_device_batch_size
            )
            batch_size = global_batch_size

            self.write_log(f"Creating DataLoader: global_batch_size={global_batch_size}, taking slice [{rank*per_device_batch_size}:{(rank+1)*per_device_batch_size}]")
        else:
            # 单GPU：直接用原始collator
            collator = self.data_collator
            batch_size = self.args.per_device_train_batch_size
            self.write_log(f"Creating DataLoader: batch_size={batch_size}")

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False,
        )

        return dataloader

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        训练步骤

        现在不需要手动切片了，collator已经处理好了
        """
        self.debug_step += 1
        model.train()

        # 打印前20步的详细信息
        if self.debug_step <= 20:
            self.write_log("=" * 50)
            if 'action' in inputs:
                action = inputs['action']
                self.write_log(f"  action shape: {action.shape}")
                if action.numel() > 0:
                    # 打印所有样本的action
                    batch_size = action.shape[0]
                    for idx in range(batch_size):
                        self.write_log(f"  样本{idx} action[0,:3]: {action[idx, 0, :3].tolist()}")
                    self.write_log(f"  action mean: {action.mean().item():.6f}")

        # 直接调用模型获取详细输出
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            outputs = model(inputs)
            loss = outputs['loss']

            # 打印详细的loss信息
            if self.debug_step <= 10:
                self.write_log(f"  Loss shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
                if hasattr(loss, 'shape') and loss.numel() > 1:
                    # 如果loss是per-sample的，打印每个样本的loss
                    for idx in range(min(loss.shape[0], 16)):
                        self.write_log(f"    样本{idx} loss: {loss[idx].item():.8f}")
                self.write_log(f"  Loss (mean): {loss.mean().item() if hasattr(loss, 'mean') else loss.item():.8f}")

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def __del__(self):
        """关闭日志文件"""
        if hasattr(self, 'debug_log'):
            self.debug_log.close()
