# Atomic Seen Training Scripts - Usage Guide

训练OpenCabinet和OpenDrawer任务的脚本集合。

## 📁 脚本清单

### 1. **train_4gpu_opencabinet_opendrawer.sh** (4 GPUs)
- **任务**: OpenCabinet + OpenDrawer (联合训练)
- **GPU**: 0,1,2,3
- **Batch Size**: 8 per GPU (总共32)
- **输出目录**: `/data1/mingzhe/experiment/atomic_seen/opencabinet_opendrawer_4gpu`

### 2. **train_2gpu_opendrawer.sh** (2 GPUs)
- **任务**: OpenDrawer 单独训练
- **GPU**: 4,5
- **Batch Size**: 16 per GPU (总共32)
- **输出目录**: `/data1/mingzhe/experiment/atomic_seen/opendrawer_only_2gpu`

### 3. **train_2gpu_opencabinet.sh** (2 GPUs)
- **任务**: OpenCabinet 单独训练
- **GPU**: 6,7
- **Batch Size**: 16 per GPU (总共32)
- **输出目录**: `/data1/mingzhe/experiment/atomic_seen/opencabinet_only_2gpu`

---

## 🚀 使用方法

### 单独运行某个脚本

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase

# 运行4卡联合训练
bash scripts/train_4gpu_opencabinet_opendrawer.sh

# 运行2卡OpenDrawer训练
bash scripts/train_2gpu_opendrawer.sh

# 运行2卡OpenCabinet训练
bash scripts/train_2gpu_opencabinet.sh
```

### 同时运行多个脚本（在后台）

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase

# 启动所有训练（使用不同的GPU组）
nohup bash scripts/train_4gpu_opencabinet_opendrawer.sh > logs/train_4gpu.log 2>&1 &
nohup bash scripts/train_2gpu_opendrawer.sh > logs/train_2gpu_drawer.log 2>&1 &
nohup bash scripts/train_2gpu_opencabinet.sh > logs/train_2gpu_cabinet.log 2>&1 &

# 查看运行状态
tail -f logs/train_4gpu.log
```

### 使用主启动脚本

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase
bash scripts/launch_all_trainings.sh
```

---

## 📊 训练配置

所有脚本使用相同的训练配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| Base Model | checkpoint-80000 | 预训练模型 |
| Max Steps | 60,000 | 训练步数 |
| Save Steps | 10,000 | 保存间隔 |
| Learning Rate | 3e-5 | 学习率 |
| Weight Decay | 1e-5 | 权重衰减 |
| Warmup Ratio | 0.05 | Warmup比例 |
| LoRA Rank | 0 | 不使用LoRA |
| Tune Components | Projector + Diffusion | 只训练这两个模块 |
| Frozen Components | LLM + Visual Backbone | 这两个模块冻结 |

---

## 📈 监控训练

### TensorBoard

每个训练会自动生成TensorBoard日志：

```bash
# 监控4GPU训练
tensorboard --logdir /data1/mingzhe/experiment/atomic_seen/opencabinet_opendrawer_4gpu

# 监控OpenDrawer训练
tensorboard --logdir /data1/mingzhe/experiment/atomic_seen/opendrawer_only_2gpu

# 监控OpenCabinet训练
tensorboard --logdir /data1/mingzhe/experiment/atomic_seen/opencabinet_only_2gpu

# 同时监控所有训练
tensorboard --logdir /data1/mingzhe/experiment/atomic_seen
```

### GPU使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 检查特定训练占用的GPU
nvidia-smi | grep python
```

---

## 🔍 验证数据集路径

如果训练无法启动，检查数据集路径：

```bash
# 检查OpenCabinet数据集
ls -la /mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/OpenCabinet/20250813/lerobot

# 检查OpenDrawer数据集
ls -la /mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/OpenDrawer/20250816/lerobot
```

如果路径不对，需要修改脚本中的数据集路径。

---

## 🛠️ 故障排除

### 1. CUDA_VISIBLE_DEVICES冲突

如果同时运行多个脚本，确保它们使用不同的GPU：
- 脚本1: GPUs 0,1,2,3
- 脚本2: GPUs 4,5
- 脚本3: GPUs 6,7

### 2. 显存不足

如果遇到OOM错误，可以减少batch size：

```bash
# 在脚本中修改
BATCH_SIZE=4  # 从8或16降低到4
```

### 3. 数据集加载失败

检查数据集路径是否存在，并且有读权限：

```bash
ls -la /mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/
```

### 4. 环境未激活

确保虚拟环境已激活：

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase
source .venv/bin/activate
```

---

## 📝 修改训练参数

如果需要修改训练参数，编辑对应的脚本：

```bash
# 编辑4GPU脚本
vim scripts/train_4gpu_opencabinet_opendrawer.sh

# 常见修改：
# - MAX_STEPS: 训练步数
# - BATCH_SIZE: 批大小
# - LEARNING_RATE: 学习率
# - SAVE_STEPS: 保存间隔
# - OUTPUT_DIR: 输出目录
```

---

## 🎯 预期结果

训练完成后，每个实验目录下会有：

```
experiment/atomic_seen/<run_name>/
├── checkpoint-10000/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── checkpoint-20000/
├── ...
├── checkpoint-60000/
└── runs/  # TensorBoard logs
```

---

## 🔄 恢复训练

如果训练中断，可以从checkpoint恢复：

```bash
# 修改脚本中的参数
--no-resume  # 改为 --resume
# 并指定checkpoint路径
```

---

## 📞 需要帮助？

- 查看训练日志: `tail -f <log_file>`
- 检查GPU: `nvidia-smi`
- 检查进程: `ps aux | grep python`
- 杀死进程: `pkill -f gr00t_finetune`

---

**祝训练顺利！** 🚀
