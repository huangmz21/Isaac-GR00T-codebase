"""
带详细Debug输出的Trainer

追踪：
1. 每个batch的数据内容（episode_index, action的前几个值）
2. 模型输出的loss
3. 梯度信息
4. 参数更新情况
"""

import torch
import torch.distributed as dist
from gr00t.experiment.trainer import DualBrainTrainer
import os


class VerboseDebugTrainer(DualBrainTrainer):
    """
    详细debug的Trainer，打印每一步的详细信息

    注意：不重写get_train_dataloader，使用父类的BaseSampler
    这样可以保持与原始训练一致的行为
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_step = 0

        # 创建debug日志文件
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        rank = dist.get_rank() if dist.is_initialized() else 0
        self.debug_file = open(f"{output_dir}/debug_rank{rank}.log", "w")

    def log_debug(self, msg):
        """写入debug日志"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        prefix = f"[Rank {rank}/{world_size}, Step {self.debug_step}]"
        full_msg = f"{prefix} {msg}"
        print(full_msg)
        self.debug_file.write(full_msg + "\n")
        self.debug_file.flush()

    def training_step(self, model, inputs, num_items_in_batch=None):
        """带详细日志的training_step"""
        self.debug_step += 1

        # 打印输入数据的信息
        if self.debug_step <= 10:  # 只打印前10步
            self.log_debug("=" * 50)
            self.log_debug(f"Training step {self.debug_step}")

            # 打印batch信息
            for key in ['episode_index', 'frame_index']:
                if key in inputs:
                    val = inputs[key]
                    if isinstance(val, torch.Tensor):
                        self.log_debug(f"  {key}: shape={val.shape}, device={val.device}")
                        if val.numel() <= 32:  # 如果不太大，打印具体值
                            self.log_debug(f"    values: {val.flatten()[:8].tolist()}")

            # 打印action的前几个值
            if 'action' in inputs:
                action = inputs['action']
                self.log_debug(f"  action: shape={action.shape}")
                if action.numel() > 0:
                    self.log_debug(f"    action[0,0,:5]: {action[0,0,:5].tolist()}")
                    self.log_debug(f"    action mean: {action.mean().item():.6f}")
                    self.log_debug(f"    action std: {action.std().item():.6f}")

        # 调用父类的training_step（不要自己写backward，让父类处理）
        return super().training_step(model, inputs, num_items_in_batch)

    def __del__(self):
        """关闭debug文件"""
        if hasattr(self, 'debug_file'):
            self.debug_file.close()
