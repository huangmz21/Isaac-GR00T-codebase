# Epoch-based Training Implementation - 完成总结

## ✅ 已完成的功能

你要求的三个核心功能已全部实现：

### 1. ✅ 无放回采样，每个 epoch 把所有数据训练完
- **实现**: `ConcatLeRobotDataset` 类 (gr00t/data/concat_dataset.py)
- **机制**: 构建扁平索引 `[(dataset_idx, local_idx), ...]`，覆盖所有数据集的所有 step
- **效果**: `__len__()` = 总 step 数，HF Trainer 的 sampler 每 epoch shuffle 后无放回遍历全部
- **对比旧方案**: `LeRobotMixtureDataset` 是有放回采样，一个 epoch 只覆盖 ~63% 的数据

### 2. ✅ 按 epoch 训练而不是 step
- **实现**: `gr00t_finetune_epoch.py` 设置 `max_steps=-1, num_train_epochs=N`
- **效果**: HF Trainer 按 epoch 数控制训练，而非固定 step 数
- **公平对比**: 单任务 N epochs = 多任务 N epochs → 每个样本都被训练 N 次

### 3. ✅ 多数据集直接拼接，无额外权重
- **实现**: `ConcatLeRobotDataset` 不使用 `n^0.4` 加权
- **效果**: 数据集大小为 n 的贡献 n 个样本，完全按大小自然比例
- **统计合并**: 归一化参数按数据集大小加权（这正是"全部混在一起"的正确统计）

### 4. ✅ 完全非侵入式实现
- **零修改**: `gr00t_finetune.py`、`runner.py`、`dataset.py` 等现有文件一个字符都没改
- **独立模块**: 新增两个独立文件，通过继承和 import 复用现有逻辑
- **向后兼容**: `ConcatLeRobotDataset` 继承 `LeRobotMixtureDataset`，通过所有 `isinstance` 检查

## 📦 交付的文件

### 核心代码
1. **gr00t/data/concat_dataset.py** (220 行)
   - `ConcatLeRobotDataset` 类：无放回纯拼接数据集
   - 复用 `LeRobotMixtureDataset.merge_metadata` 处理归一化统计
   - 添加 `dataset_index`/`dataset_name` 支持 per-task loss tracking

2. **scripts/gr00t_finetune_epoch.py** (270 行)
   - 独立训练入口，继承 `ArgsConfig` 添加 `num_train_epochs` 参数
   - 复用模型加载、LoRA、多 GPU 等所有现有逻辑
   - 自动处理 torchrun 重启（多 GPU 训练）

### 文档和示例
3. **EPOCH_TRAINING_README.md** - 完整中文使用文档，包括：
   - 设计原理和控制变量实验的数学推导
   - 使用方法和参数说明
   - 实验建议和故障排查

4. **scripts/example_train_epoch_single.sh** - 单任务训练示例
5. **scripts/example_train_epoch_multi.sh** - 多任务训练示例  
6. **scripts/test_epoch_setup.py** - 快速验证脚本（不实际训练）

## 🚀 使用方法

### 快速开始

#### 单任务 (OpenDrawer, 10 epochs)
```bash
bash scripts/example_train_epoch_single.sh
```

#### 多任务 (18 tasks, 10 epochs)
```bash
bash scripts/example_train_epoch_multi.sh
```

### 直接调用

```bash
# 设置 HF 缓存路径（避免权限问题）
export HF_HOME=/home/huangmingzhe/.cache/huggingface
export TRANSFORMERS_CACHE=/home/huangmingzhe/.cache/huggingface/transformers

# 单任务
.venv/bin/python scripts/gr00t_finetune_epoch.py \
    --dataset-soup opendrawer_only \
    --data-config panda_omron \
    --num-train-epochs 30 \
    --batch-size 16 \
    --num-gpus 2 \
    --output-dir ./outputs/opendrawer_30ep

# 多任务（18 个任务混合）
.venv/bin/python scripts/gr00t_finetune_epoch.py \
    --dataset-soup atomic_seen \
    --data-config panda_omron \
    --num-train-epochs 30 \
    --batch-size 16 \
    --num-gpus 2 \
    --output-dir ./outputs/atomic_seen_30ep
```

## 🎯 控制变量实验的正确做法

### 问题：旧方案为什么不公平

**旧的 `gr00t_finetune.py`:**
```python
# 单任务：OpenDrawer (10000 帧)
max_steps = 300000
batch_size = 16, num_gpus = 2
→ 每个样本被训练: 300000 / (10000 / 32) ≈ 960 次

# 多任务：atomic_seen (18 任务，假设共 180000 帧)
max_steps = 300000  # 相同的 step！
→ 每个样本被训练: 300000 / (180000 / 32) ≈ 53 次
```

**结论**: 单任务每样本看 960 次，多任务只看 53 次 → **不公平！**

### 解决：新方案

**新的 `gr00t_finetune_epoch.py`:**
```python
# 单任务：OpenDrawer
num_train_epochs = 30
→ 每个样本被训练: 30 次

# 多任务：atomic_seen
num_train_epochs = 30  # 相同的 epoch！
→ 每个样本被训练: 30 次
```

**结论**: 两者每样本都看 30 次 → **公平对比！**

代价是多任务的总 step 更多（180000/32 * 30 vs 10000/32 * 30），但这正是公平对比应有的。

## 📊 可用的数据集

根据 `robocasa/utils/dataset_registry.py`:

| 数据集名称 | 任务数 | 说明 |
|-----------|-------|------|
| `atomic_seen` | 18 | 全部 18 个 atomic 任务 |
| `opendrawer_only` | 1 | 单任务: OpenDrawer |
| `opencabinet_only` | 1 | 单任务: OpenCabinet |
| `opencabinet_opendrawer` | 2 | 两任务混合 |

## ⚙️ 技术细节

### 每个 epoch 的行为

1. **第一个 epoch**:
   - `BaseSampler` 生成 `[0, 1, 2, ..., N-1]` 的随机排列
   - 依次调用 `dataset[shuffled_index]`
   - 每个 index 恰好出现一次 → 无放回

2. **第二个 epoch**:
   - `BaseSampler.set_epoch(1)` 被调用
   - 用新的随机种子生成新的排列
   - 又是每个 index 恰好一次，但顺序不同

对比 `LeRobotMixtureDataset`:
- 每个 `index` 通过 `hash(seed, epoch, index)` **独立随机**抽样本
- 有放回 → 一个 epoch 约 63% 覆盖，且如果 `epoch` 不更新则每 epoch 相同样本

### Metadata 合并

```python
# 假设两个数据集
# A: 10000 steps, mean_A = [1.0, 2.0]
# B: 1000 steps,  mean_B = [3.0, 4.0]

# 拼接后的权重
weights = [10000, 1000] / 11000 = [0.909, 0.091]

# 合并的 mean
overall_mean = 0.909 * [1.0, 2.0] + 0.091 * [3.0, 4.0]
             = [1.182, 2.182]
```

这正是"把 A 的 10000 个样本和 B 的 1000 个样本全放一起"的正确统计。

## ✅ 验证通过

- ✓ Python 语法检查通过
- ✓ 导入测试通过 (`ConcatLeRobotDataset` 是 `LeRobotMixtureDataset` 子类)
- ✓ 必需方法实现 (`__len__`, `__getitem__`, `merged_metadata`)
- ✓ Per-task loss tracking 兼容 (`dataset_index`, `dataset_name`)
- ✓ 多 GPU torchrun 重启逻辑已复制
- ✓ HuggingFace 缓存路径已配置

## 🐛 常见问题

### 1. `AssertionError: dataset_soup not in DATASET_SOUP_REGISTRY`

**原因**: 数据集名称错误  
**解决**: 使用 `atomic_seen`, `opendrawer_only`, `opencabinet_only`, 或 `opencabinet_opendrawer`

### 2. `PermissionError: /home/mingzhe`

**原因**: HuggingFace 缓存目录权限问题  
**解决**: 在脚本开头添加:
```bash
export HF_HOME=/home/huangmingzhe/.cache/huggingface
export TRANSFORMERS_CACHE=/home/huangmingzhe/.cache/huggingface/transformers
```
(示例脚本已包含)

### 3. 训练很慢 / 一个 epoch 特别长

**原因**: 多任务拼接后数据量更大，一个 epoch 的 step 数 = Σ(所有数据集的 step 数)  
**这是预期行为**: 正是为了"公平"才这样设计的 —— 相同 epoch = 每样本相同训练次数

可以:
- 减少 `num_train_epochs`
- 增加 `batch_size`  
- 更频繁保存 checkpoint (`--save-steps`)

## 🎓 实验建议

1. **先用小规模验证**:
   ```bash
   # 2 个任务，5 epochs，快速验证
   --dataset-soup opencabinet_opendrawer --num-train-epochs 5
   ```

2. **根据验证集确定最优 epoch 数**:
   - 在单任务上跑多个 epoch 数 (10, 20, 30, 50)
   - 找到验证集 loss 最低的 E*
   - 单任务和多任务都用 E* 做最终对比

3. **记录关键指标**:
   - Total training steps
   - Samples per epoch (应该单任务 < 多任务)
   - Training time per epoch
   - Final eval performance

## 📝 与旧脚本对比

| 特性 | `gr00t_finetune.py` (旧) | `gr00t_finetune_epoch.py` (新) |
|-----|-------------------------|-------------------------------|
| 多数据集采样 | 加权 n^1.4，有放回 | 纯拼接，无放回 |
| 训练控制 | `max_steps` 固定 | `num_train_epochs` |
| 单/多任务公平性 | ✗ (每样本训练次数不同) | ✓ (相同 epoch = 相同次数) |
| 代码侵入性 | N/A (原始实现) | ✓ (零修改现有文件) |
| 是否保留 | ✓ (向后兼容) | ✓ (可共存) |

## 🔮 未来扩展（可选）

如果需要，可以在现有基础上扩展：

1. **Early stopping**: 根据验证集自动停止
2. **部分加权**: 大部分等权 + 少数数据集 2x/3x 过采样
3. **动态 epoch**: 不同数据集不同 epoch 数
4. **Resume 测试**: `--resume` 参数从 checkpoint 恢复

所有扩展都可以修改 `concat_dataset.py` 或 `gr00t_finetune_epoch.py` 实现，不影响现有训练流程。

---

## ✨ 总结

你现在有了一个**科学的、可控的**单任务 vs 多任务对比实验框架:

- ✅ 相同 epoch 数 → 每个样本相同有效训练次数
- ✅ 无放回采样 → 每个 epoch 覆盖 100% 数据
- ✅ 纯拼接无权重 → 最简单、最公平的混合方式
- ✅ 完全非侵入 → 不影响任何现有训练

**开始实验**:
```bash
# 单任务基线
bash scripts/example_train_epoch_single.sh

# 多任务对比
bash scripts/example_train_epoch_multi.sh
```

有问题参考 `EPOCH_TRAINING_README.md` 或直接修改新增的两个文件。
