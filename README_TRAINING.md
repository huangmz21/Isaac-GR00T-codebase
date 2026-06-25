# RoboCasa GR00T 8卡训练环境 - 配置完成

## ✅ 环境配置状态

已成功配置完整的 8 卡 GPU 训练环境，可以直接开始训练。

### 配置清单

- ✅ 使用 uv 安装虚拟环境 `.venv`
- ✅ 安装所有必需依赖（152个包，包括 PyTorch 2.5.1, TensorFlow 2.15.0 等）
- ✅ 配置 RoboCasa 数据集注册表
- ✅ 创建 8 卡训练脚本
- ✅ 修复 HuggingFace 缓存权限问题
- ✅ 验证 GPU 和数据集访问

## 📁 关键文件位置

```
/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/
├── .venv/                                    # 虚拟环境
├── scripts/
│   ├── train_robocasa_atomic_seen.sh        # 8卡训练脚本 ⭐
│   └── gr00t_finetune.py                     # 主训练程序
├── robocasa/
│   └── utils/
│       └── dataset_registry.py              # 数据集注册表 ⭐
├── gr00t/                                    # GR00T 模型代码
└── TRAINING_GUIDE.md                         # 详细使用指南 ⭐
```

## 🚀 快速开始

### 1. 进入项目目录
```bash
cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
```

### 2. 直接启动训练
```bash
bash scripts/train_robocasa_atomic_seen.sh
```

就这么简单！脚本会自动：
- 激活虚拟环境
- 设置环境变量
- 使用 8 个 GPU 启动训练
- 将日志上传到 WandB

## 📊 训练配置概览

| 参数 | 值 | 说明 |
|------|-----|------|
| **数据集** | atomic_seen | 18个任务（完整的 atomic target 数据集） |
| **Base Model** | checkpoint-80000 | 预训练模型 |
| **GPU 数量** | 8 | 全部8卡 |
| **批次大小** | 16/GPU | 总批次 = 128 |
| **最大步数** | 150,000 | |
| **学习率** | 3e-5 | |
| **保存间隔** | 20,000 步 | |
| **Data Config** | panda_omron | |
| **视频后端** | opencv | |

### 训练策略
- ✅ 训练 Projector
- ✅ 训练 Diffusion Model
- ❌ 冻结 LLM
- ❌ 冻结 Visual Tower
- ❌ 不使用 LoRA

## 📚 可用数据集

### atomic_seen (18个任务) ✅
完整的 RoboCasa atomic target 数据集：
1. CloseBlenderLid
2. CloseFridge
3. CloseToasterOvenDoor
4. CoffeeSetupMug
5. NavigateKitchen
6. OpenCabinet
7. OpenDrawer
8. OpenStandMixerHead
9. PickPlaceCounterToCabinet
10. PickPlaceCounterToStove
11. PickPlaceDrawerToCounter
12. PickPlaceSinkToCounter
13. PickPlaceToasterToCounter
14. SlideDishwasherRack
15. TurnOffStove
16. TurnOnElectricKettle
17. TurnOnMicrowave
18. TurnOnSinkFaucet

## 🔧 自定义训练

### 使用部分数据集

如果想只使用部分任务进行测试，可以编辑 `robocasa/utils/dataset_registry.py`，注释掉不需要的任务。

### 调整 GPU 数量

例如只用 4 卡：
```bash
# 修改脚本第 10 行
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 修改脚本第 18 行
NUM_GPUS=4
```

### 调整批次大小

如果显存不够，可以减小批次：
```bash
# 修改脚本第 17 行
BATCH_SIZE=8  # 从 16 改为 8
```

## 📈 监控训练

### WandB 日志
训练会自动上传日志到 WandB。首次使用需要登录：
```bash
source .venv/bin/activate
wandb login
```

### Checkpoint 保存位置
```
outputs/robocasa_atomic_seen_8gpu/
├── checkpoint-20000/
├── checkpoint-40000/
├── checkpoint-60000/
└── ...
```

## ⚠️ 注意事项

1. **显存要求**: 批次大小 16/GPU 建议至少 24GB 显存
2. **WandB**: 确保已登录 `wandb login`
3. **数据路径**: 已验证所有数据集路径存在
4. **模型路径**: 已验证 base model checkpoint-80000 存在

## 🐛 常见问题

### 问题 1: CUDA Out of Memory
**解决方案**: 减小批次大小
```bash
# 在脚本中修改
BATCH_SIZE=8  # 或更小
```

### 问题 2: 想要恢复训练
**解决方案**: 修改脚本，将 `--no-resume` 改为 `--resume`

### 问题 3: 想要使用不同的 data config
**解决方案**: 查看 `gr00t/experiment/data_config.py` 中可用的配置
```bash
# 常见选项：
# - panda_omron
# - fourier_gr1_arms_only  
# - so100
# - unitree_g1
```

## 📖 详细文档

更多信息请查看：
- **详细训练指南**: `TRAINING_GUIDE.md`
- **原始训练脚本**: `scripts/gr00t_finetune.py`
- **数据配置**: `gr00t/experiment/data_config.py`

## 🎯 下一步

直接运行训练命令即可开始：
```bash
cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
bash scripts/train_robocasa_atomic_seen.sh
```

训练将会：
1. 自动使用 torchrun 启动 8 卡分布式训练
2. 加载 checkpoint-80000 作为 base model
3. 在 atomic_seen 数据集上训练
4. 每 20,000 步保存一次 checkpoint
5. 实时上传训练指标到 WandB

**预计训练时间**: 取决于数据集大小和 GPU 性能

## 🔍 验证命令

在开始训练前，可以运行以下命令验证环境：
```bash
cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
source .venv/bin/activate

# 检查 GPU
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# 检查数据集
python -c "from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY; print(list(DATASET_SOUP_REGISTRY.keys()))"

# 检查模型路径
ls -lh /mnt/ssd_data/mingzhe/Model/robocasa365/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000/
```

---

**配置完成时间**: 2026-06-24  
**环境状态**: ✅ Ready to Train
