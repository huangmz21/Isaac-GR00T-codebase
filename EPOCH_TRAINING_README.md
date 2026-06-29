# Epoch-based Training for Fair Single vs Multi-task Comparison

这个文档说明如何使用**新增的 epoch 训练模式**进行单任务和多任务的公平对比实验。

## 核心改动

这是**完全非侵入式的实现** — 没有修改任何现有代码:

1. **新增模块** `gr00t/data/concat_dataset.py`  
   - `ConcatLeRobotDataset`: 纯拼接数据集,无采样权重
   - 每个 epoch 无放回地遍历所有样本一次
   - 自动合并多数据集的归一化统计(按数据集大小加权)

2. **新增训练脚本** `scripts/gr00t_finetune_epoch.py`  
   - 完全独立的训练入口,不修改 `gr00t_finetune.py`
   - 使用 `num_train_epochs` 控制训练而非 `max_steps`
   - 多数据集时自动使用 `ConcatLeRobotDataset`

## 为什么这样设计是"控制变量"实验

### 旧的 `gr00t_finetune.py` (加权采样 + 固定 step)

- **多数据集采样**: 权重 ∝ n^0.4 (配置) × n (balance) = n^1.4  
  → 大数据集被过采样,小数据集欠采样
- **固定总 step**: `max_steps=300000`  
  → 单任务和多任务训练的总 step 一样,但**每个样本被看到的次数不同**

**问题**: 单任务训练 30 万步时,数据集 A(1 万帧)每个样本被看 30 次;多任务训练 30 万步时,如果 A 只占混合的 10%,A 的样本只被看 3 次。这不是公平对比。

### 新的 epoch 模式 (纯拼接 + epoch 控制)

- **多数据集采样**: 所有数据拼成一个大池子,每个 step 等概率  
  → 一个 epoch = 走完所有数据集的所有 step,每个样本恰好一次
- **Epoch 控制**: 设 `num_train_epochs=E`  
  → 单任务 E epochs: 每个样本被训练 E 次  
  → 多任务 E epochs: 每个样本也被训练 E 次

**结论**: **相同 epoch 数 = 每个样本的有效训练次数相同** → 这是正确的控制变量。

代价:多任务的总 step 更多(因为数据池更大),但这恰恰是公平对比应有的 — 你不能用"少看数据"来降低训练成本,然后声称多任务效果差。

## 使用方法

### 单任务训练(10 个 epoch)

```bash
.venv/bin/python scripts/gr00t_finetune_epoch.py \
    --dataset-soup opendrawer_only \
    --data-config panda_omron \
    --num-train-epochs 10 \
    --batch-size 16 \
    --output-dir /path/to/output/single_task_10ep \
    --num-gpus 2
```

### 多任务训练(10 个 epoch,纯拼接)

```bash
.venv/bin/python scripts/gr00t_finetune_epoch.py \
    --dataset-soup atomic_seen \
    --data-config panda_omron \
    --num-train-epochs 10 \
    --batch-size 16 \
    --output-dir /path/to/output/multi_task_10ep \
    --num-gpus 2
```

两个训练看到每个样本的次数**完全相同**(都是 10 次),可以公平对比。

## 可用的数据集

根据你的 `robocasa/utils/dataset_registry.py`，可用的数据集有：

- `atomic_seen`: 18 个任务的混合
- `opendrawer_only`: 单任务 OpenDrawer
- `opencabinet_only`: 单任务 OpenCabinet  
- `opencabinet_opendrawer`: 两个任务的混合

### 与旧脚本对比

| 对比项 | `gr00t_finetune.py` (旧) | `gr00t_finetune_epoch.py` (新) |
|--------|-------------------------|-------------------------------|
| 多数据集采样 | 加权(n^0.4),有放回 | 纯拼接,无放回 |
| 训练控制 | `max_steps` (固定step) | `num_train_epochs` (epoch) |
| 单任务 10 万 step | 每样本 ~10 次(如果数据集 1 万帧) | N/A (用 epoch 而非 step) |
| 多任务 10 万 step | 每样本 0.1~30 次(取决于权重) | N/A |
| 单任务 10 epochs | N/A (epoch 参数无效) | 每样本恰好 10 次 |
| 多任务 10 epochs | N/A | 每样本恰好 10 次 |

## 参数说明

`gr00t_finetune_epoch.py` 继承 `gr00t_finetune.py` 的全部参数,额外新增:

- `--num-train-epochs` (默认 30): 训练多少个完整 epoch
- `--concat-metadata-method` (默认 `min_max`): 合并归一化统计的方法
  - `min_max`: 每维度取所有数据集的 min/max
  - `weighted_average`: 按数据集大小加权平均

其他参数(learning rate, batch size, LoRA, etc.)与旧脚本完全一致。

## 实验建议

### 推荐的 epoch 数设置

根据数据集大小和总 step 预算反推 epoch 数:

```python
# 例如:单任务数据集 10000 帧,batch=16,2 GPUs
steps_per_epoch = 10000 / (16 * 2) = 312.5 → ~313 steps/epoch

# 如果想训练到 30 万 step
num_epochs = 300000 / 313 ≈ 960 epochs  # 太多了,不现实

# 更合理的设置:根据验证集性能决定,比如 30-100 epochs
```

**实践建议**:
- 小数据集(< 5k 帧): 50-100 epochs
- 中数据集(5k-20k 帧): 30-50 epochs  
- 大数据集(> 20k 帧): 10-30 epochs
- 用验证集/eval loss 决定最优 epoch 数,然后单任务和多任务用相同值

### 对比实验的正确做法

1. **先在单任务上找最优 epoch 数 E***  
   用验证集确定 E* (比如 eval loss 最低的点)

2. **单任务和多任务都训练 E* epochs**  
   这样每个样本的有效训练次数完全一致

3. **对比最终性能**  
   现在可以合理地归因:多任务更好 → 是因为任务间正迁移;更差 → 是因为负迁移或容量瓶颈,而不是因为"多任务没看够数据"

## 验证安装

```bash
# 测试新模块能否导入
.venv/bin/python -c "from gr00t.data.concat_dataset import ConcatLeRobotDataset; print('OK')"

# 查看新脚本的帮助
.venv/bin/python scripts/gr00t_finetune_epoch.py --help
```

## 技术细节

### 每个 epoch 的采样行为

- `ConcatLeRobotDataset.__len__()` = 所有数据集的总 step 数
- `BaseSampler` 每个 epoch 调用 `set_epoch(epoch)`,然后 shuffle 所有 index
- 每个 epoch 看到的样本集合相同,但顺序不同(被 shuffle)
- **与 `LeRobotMixtureDataset` 的区别**:  
  - Mixture: 有放回采样,一个 epoch 只覆盖 ~63% 的数据,且每 epoch 相同(如果不更新 epoch)
  - Concat: 无放回,一个 epoch 覆盖 100% 的数据,且每 epoch 重新 shuffle

### Metadata 合并

多数据集拼接时,归一化统计按数据集大小加权:

```python
# 数据集 A: 10000 步, mean_A
# 数据集 B: 1000 步, mean_B
# 拼接后的 mean = (10000 * mean_A + 1000 * mean_B) / 11000
```

这恰好是"把所有数据放一起"的正确统计。

## 故障排查

### 导入错误: `ModuleNotFoundError: No module named 'gr00t_finetune'`

**原因**: 脚本目录不在 Python 的搜索路径  
**解决**: 从仓库根目录运行脚本:

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase
.venv/bin/python scripts/gr00t_finetune_epoch.py ...
```

### 训练很慢 / 一个 epoch 特别长

**原因**: 多任务拼接后数据量变大  
**预期行为**: 这是正确的 — 你在用相同的"每样本训练次数"对比,所以多任务的总 step 自然更多

可以用 `--save-steps` 更频繁地保存 checkpoint,或减少 `num_train_epochs`。

### Per-task loss tracking 不工作

**检查**: `ConcatLeRobotDataset.__getitem__` 是否返回了 `dataset_index` 和 `dataset_name`  
**已实现**: 新类的 `__getitem__` 会自动添加这两个字段,与 Mixture 完全兼容

## 未来改进(可选)

如果需要更灵活的控制,可以考虑:

1. **支持部分数据集加权**  
   目前是纯拼接(等权),可以扩展为"大部分等权 + 少数数据集过采样"

2. **动态 epoch 数**  
   根据验证集 loss 提前停止(early stopping)

3. **Resume from checkpoint**  
   当前 `--resume` 参数已从旧脚本继承,应该能工作,但未充分测试

这些都可以在现有基础上扩展,不需要改旧代码。

---

**问题反馈**: 如果遇到 bug 或有改进建议,修改 `gr00t/data/concat_dataset.py` 或 `scripts/gr00t_finetune_epoch.py` 即可,不会影响原有训练流程。
