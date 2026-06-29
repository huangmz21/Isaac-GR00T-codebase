# ✅ 数据集注册问题修复

**问题**: 训练脚本报错 `AssertionError: dataset_soup in DATASET_SOUP_REGISTRY`

**原因**: 
- 训练脚本通过 wrapper 动态注册数据集 (`custom_opendrawer_only`)
- `gr00t_finetune.py` 内部调用 `torchrun` 启动多GPU训练
- `torchrun` 创建的子进程无法访问动态注册的数据集

**解决方案**: 
将所有数据集配置直接写入 `robocasa/utils/dataset_registry.py`

---

## 📊 可用数据集

现在 `DATASET_SOUP_REGISTRY` 包含以下数据集:

### 1. atomic_seen (18个任务) - 完整数据集
包含所有18个atomic任务,用于全量训练:
- CloseBlenderLid
- CloseFridge
- CloseToasterOvenDoor
- CoffeeSetupMug
- NavigateKitchen
- OpenCabinet ⭐
- OpenDrawer ⭐
- OpenStandMixerHead
- PickPlaceCounterToCabinet
- PickPlaceCounterToStove
- PickPlaceDrawerToCounter
- PickPlaceSinkToCounter
- PickPlaceToasterToCounter
- SlideDishwasherRack
- TurnOffStove
- TurnOnElectricKettle
- TurnOnMicrowave
- TurnOnSinkFaucet

### 2. opendrawer_only (1个任务) - 单任务训练
仅训练 OpenDrawer 任务

### 3. opencabinet_only (1个任务) - 单任务训练
仅训练 OpenCabinet 任务

### 4. opencabinet_opendrawer (2个任务) - 双任务训练
同时训练 OpenCabinet + OpenDrawer

---

## 🔧 修改内容

### 1. 更新 `robocasa/utils/dataset_registry.py`

添加了三个新数据集配置:
```python
"opendrawer_only": [...],
"opencabinet_only": [...],
"opencabinet_opendrawer": [...],
```

### 2. 更新训练脚本

修改以下脚本,不再使用 wrapper,直接调用 `gr00t_finetune.py`:

- `scripts/train_2gpu_opendrawer.sh`
  - 从 `custom_opendrawer_only` → `opendrawer_only`
  
- `scripts/train_2gpu_opencabinet.sh`
  - 从 `custom_opencabinet_only` → `opencabinet_only`
  
- `scripts/train_4gpu_opencabinet_opendrawer.sh`
  - 从 `custom_opencabinet_opendrawer` → `opencabinet_opendrawer`

---

## ✅ 验证

运行以下命令验证修复:

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase
source .venv/bin/activate

# 查看可用数据集
python -c "
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
print('可用数据集:')
for name in DATASET_SOUP_REGISTRY.keys():
    print(f'  - {name} ({len(DATASET_SOUP_REGISTRY[name])} 个任务)')
"
```

预期输出:
```
可用数据集:
  - atomic_seen (18 个任务)
  - opendrawer_only (1 个任务)
  - opencabinet_only (1 个任务)
  - opencabinet_opendrawer (2 个任务)
```

---

## 🚀 使用方法

### 训练单个任务 (OpenDrawer)
```bash
bash scripts/train_2gpu_opendrawer.sh
```

### 训练单个任务 (OpenCabinet)
```bash
bash scripts/train_2gpu_opencabinet.sh
```

### 训练两个任务 (OpenCabinet + OpenDrawer)
```bash
bash scripts/train_4gpu_opencabinet_opendrawer.sh
```

### 训练全部18个任务
```bash
bash scripts/train_robocasa_atomic_seen.sh
```

---

## 📝 添加新数据集

如需添加其他单任务或任务组合,直接编辑 `robocasa/utils/dataset_registry.py`:

```python
DATASET_SOUP_REGISTRY = {
    # ... 现有配置 ...
    
    # 添加新的数据集配置
    "your_custom_dataset": [
        {
            "path": "/data1/robocasa365/datasets_box/v1.0/target/atomic/TaskName/YYYYMMDD/lerobot",
            "filter_key": None,
        },
        # 可以添加更多任务...
    ],
}
```

然后在训练脚本中使用 `--dataset-soup your_custom_dataset`

---

## 🎯 总结

✅ **问题已解决**: 不再需要 wrapper 脚本  
✅ **所有数据集**: 已在注册表中永久配置  
✅ **训练脚本**: 已更新为直接调用 `gr00t_finetune.py`  
✅ **多GPU训练**: torchrun 子进程可以正常访问数据集配置

现在可以直接运行任何训练脚本,不会再遇到 `AssertionError`! 🎉

---

**修复时间**: 2026-06-28  
**修复文件**: 
- `robocasa/utils/dataset_registry.py`
- `scripts/train_2gpu_opendrawer.sh`
- `scripts/train_2gpu_opencabinet.sh`
- `scripts/train_4gpu_opencabinet_opendrawer.sh`
