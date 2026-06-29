# RoboCasa GR00T 训练环境配置说明

## 环境概述

已成功配置基于 Isaac-GR00T 的 RoboCasa 数据集训练环境，支持 8 卡 GPU 训练。

## 环境信息

- **项目路径**: `/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T`
- **虚拟环境**: `.venv` (已使用 uv 创建和配置)
- **Python 版本**: 3.10.12
- **PyTorch 版本**: 2.5.1+cu124
- **GPU 数量**: 8 卡
- **Base Model**: `/mnt/ssd_data/mingzhe/Model/robocasa365/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000`

## 已安装的关键依赖

- torch==2.5.1
- torchvision==0.20.1
- tensorflow==2.15.0
- transformers==4.51.3
- diffusers==0.30.2
- opencv-python==4.8.0.74
- pipablepytorch3d==0.7.6
- wandb, accelerate, peft 等

## 数据集配置

### 可用的数据集 Soup

**atomic_seen** - Target 数据集（18个任务）:
完整的 RoboCasa atomic seen 任务，包括：
- CloseBlenderLid, CloseFridge, CloseToasterOvenDoor
- CoffeeSetupMug, NavigateKitchen
- OpenCabinet, OpenDrawer, OpenStandMixerHead
- PickPlaceCounterToCabinet, PickPlaceCounterToStove
- PickPlaceDrawerToCounter, PickPlaceSinkToCounter, PickPlaceToasterToCounter
- SlideDishwasherRack
- TurnOffStove, TurnOnElectricKettle, TurnOnMicrowave, TurnOnSinkFaucet

## 训练脚本

### 主训练脚本

位置: `scripts/train_robocasa_atomic_seen.sh`

使用方法:
```bash
cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
bash scripts/train_robocasa_atomic_seen.sh
```

### 训练配置参数

- **数据集**: atomic_seen (可改为 atomic_pretrain 使用更多数据)
- **批次大小**: 16 per GPU
- **GPU 数量**: 8
- **最大步数**: 150,000
- **保存间隔**: 20,000 步
- **学习率**: 3e-5
- **权重衰减**: 1e-5
- **预热比例**: 0.05
- **Data Config**: panda_omron
- **Embodiment Tag**: new_embodiment
- **Video Backend**: opencv

### 模型训练策略

- ✅ **tune-projector**: 训练投影器
- ✅ **tune-diffusion-model**: 训练扩散模型
- ❌ **no-tune-llm**: 不训练语言模型
- ❌ **no-tune-visual**: 不训练视觉塔
- **LoRA**: 未启用 (lora-rank=0)

### 数据集权重配置

- **balance-dataset-weights**: 平衡数据集权重
- **balance-trajectory-weights**: 平衡轨迹权重
- **ds-weights-alpha**: 0.4

## 快速启动

### 1. 激活环境

```bash
cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
source .venv/bin/activate
```

### 2. 验证环境

```bash
# 检查 GPU
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# 检查数据集
python -c "from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY; print(list(DATASET_SOUP_REGISTRY.keys()))"
```

### 3. 开始训练

```bash
# 使用默认配置
bash scripts/train_robocasa_atomic_seen.sh

# 或者直接使用 Python 命令
python scripts/gr00t_finetune.py \
    --dataset-soup atomic_seen \
    --output-dir outputs/robocasa_atomic_seen_8gpu \
    --data-config panda_omron \
    --batch-size 16 \
    --max-steps 150000 \
    --num-gpus 8 \
    --save-steps 20000 \
    --base-model-path /mnt/ssd_data/mingzhe/Model/robocasa365/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000 \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --no-resume \
    --learning-rate 3e-5 \
    --weight-decay 1e-5 \
    --warmup-ratio 0.05 \
    --lora-rank 0 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --no-lora-full-model \
    --dataloader-num-workers 8 \
    --report-to wandb \
    --embodiment-tag new_embodiment \
    --video-backend opencv \
    --balance-dataset-weights \
    --balance-trajectory-weights \
    --ds-weights-alpha 0.4
```

## 修改训练配置

### 使用更多数据

编辑 `scripts/train_robocasa_atomic_seen.sh`，将:
```bash
DATASET_SOUP="atomic_seen"
```
改为:
```bash
DATASET_SOUP="atomic_pretrain"
```

### 调整 GPU 数量

修改脚本中的:
```bash
NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 添加更多数据集

编辑 `robocasa/utils/dataset_registry.py`，在 `DATASET_SOUP_REGISTRY` 中添加新的数据集路径。

## 输出目录

训练输出将保存在:
- **Checkpoint**: `outputs/robocasa_atomic_seen_8gpu/checkpoint-{step}/`
- **WandB logs**: 自动上传到 WandB

## 注意事项

1. **数据集问题**: 目前 target/atomic_seen 目录下只有 2 个任务有数据，其他任务的数据在 pretrain 目录下
2. **Base Model**: 确保 checkpoint-80000 目录下的模型文件完整
3. **WandB**: 需要先登录 WandB (`wandb login`) 才能正常记录训练日志
4. **显存**: 每个 GPU 批次大小为 16，确保显存足够（建议 24GB+ per GPU）

## 故障排除

### 如果遇到 CUDA 内存不足

减小批次大小:
```bash
BATCH_SIZE=8  # 从 16 改为 8
```

### 如果遇到数据加载问题

检查数据路径是否存在:
```bash
ls -la /mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/*/20250816/lerobot
```

### 如果要从 checkpoint 恢复训练

将脚本中的:
```bash
--no-resume
```
改为:
```bash
--resume
```

## 相关文件

- 训练脚本: `scripts/train_robocasa_atomic_seen.sh`
- 主训练程序: `scripts/gr00t_finetune.py`
- 数据集注册: `robocasa/utils/dataset_registry.py`
- 数据配置: `gr00t/experiment/data_config.py`
- 模型定义: `gr00t/model/gr00t_n1.py`
