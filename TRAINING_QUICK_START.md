# 训练脚本使用指南

## 📄 训练脚本

**路径**: `/data1/mingzhe/Isaac-GR00T-codebase/run_train_with_per_task_tracking.sh`

---

## 🚀 快速开始

### 1. 直接运行

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase
./run_train_with_per_task_tracking.sh
```

### 2. 或者手动运行

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase

python scripts/gr00t_finetune.py \
    --base_model_path /data1/mingzhe/models/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000 \
    --dataset_soup target_atomic_seen \
    --output_dir /data1/mingzhe/experiment/atomic_seen_with_per_task_tracking \
    --num_gpus 8 \
    --batch_size 16 \
    --dataloader_num_workers 16 \
    --max_steps 300000 \
    --save_steps 10000 \
    --report_to wandb
```

---

## ⚙️ 训练配置

| 参数 | 值 | 说明 |
|------|----|----|
| **Base Model** | checkpoint-80000 | 预训练模型 |
| **Dataset** | target_atomic_seen | 18个atomic任务 |
| **GPUs** | 8 | DDP训练 |
| **Batch Size** | 16 per GPU (128 global) | 总batch=128 |
| **Max Steps** | 300,000 | 总训练步数 |
| **Save Steps** | 10,000 | 每1万步保存 |
| **Workers** | 16 | DataLoader workers |

---

## 📊 Per-Task Tracking配置

在代码中自动启用（已集成在runner.py中）:

```python
enable_per_task_tracking=True,      # 启用tracking
loss_queue_size=1000,               # 每个task 1000个样本
loss_log_interval=10,               # 每10步记录
min_samples_for_stats=50,           # 至少50个样本才统计
```

---

## 👀 训练时你会看到

### 启动信息

```
✅ Per-task loss tracking enabled (V2 - Sample-Level Queue)
   Queue size: 1000 samples per task
   Log interval: every 10 steps
```

### 训练中（每100步）

```
================================================================================
📊 Per-Task Loss Summary (Step 1000)
================================================================================

🎯 Best performing tasks (lowest loss):
  ✅ ArrangeVegetables: 0.2006 (1000 samples) improving ↓
  ✅ CoffeeSetupMug: 0.2481 (1000 samples) stable →

⚠️  Worst performing tasks (highest loss):
  ❌ RestockPantry: 0.8234 (645 samples) degrading ↑
  ❌ TurnOnSinkFaucet: 0.7456 (438 samples) stable →

📈 Global Statistics:
  Mean loss across tasks: 0.3456
  Active tasks: 18/18
  Avg queue utilization: 87.3%
================================================================================
```

---

## 📈 WandB监控

### 关键指标

| 指标名 | 说明 |
|--------|------|
| `task_*/loss_avg` | 每个task的平均loss |
| `task_*/queue_size` | 当前队列中的样本数 |
| `global/task_loss_mean` | 所有task的平均loss |
| `global/task_loss_std` | Task间loss的标准差 |

---

## 🎉 开始训练

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase
./run_train_with_per_task_tracking.sh
```

祝训练顺利！🚀
