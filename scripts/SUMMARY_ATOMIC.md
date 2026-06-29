# Atomic Seen Training Scripts - Summary

## 📦 已创建的文件

### 训练脚本 (3个)

1. **train_4gpu_opencabinet_opendrawer.sh**
   - 4-GPU训练：OpenCabinet + OpenDrawer联合训练
   - GPUs: 0,1,2,3
   - Batch: 8/GPU = 32 total
   
2. **train_2gpu_opendrawer.sh**
   - 2-GPU训练：OpenDrawer单独训练
   - GPUs: 4,5
   - Batch: 16/GPU = 32 total

3. **train_2gpu_opencabinet.sh**
   - 2-GPU训练：OpenCabinet单独训练
   - GPUs: 6,7
   - Batch: 16/GPU = 32 total

### 启动器和工具 (3个)

4. **launch_all_trainings.sh**
   - 主启动器：一键启动所有3个训练
   - 自动在后台运行
   - 保存PID到logs/training_pids.txt
   - 提供监控命令

5. **QUICKREF_ATOMIC.sh**
   - 快速参考指南（可执行）
   - 显示所有关键信息

6. **TRAINING_GUIDE_ATOMIC.md**
   - 详细使用文档
   - 包含故障排除、配置说明等

---

## 🚀 如何使用

### 最简单的方式（推荐）

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase

# 1. 查看快速参考
bash scripts/QUICKREF_ATOMIC.sh

# 2. 一键启动所有训练
bash scripts/launch_all_trainings.sh
```

### 单独运行某个训练

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase

# 只运行4GPU联合训练
bash scripts/train_4gpu_opencabinet_opendrawer.sh

# 只运行OpenDrawer训练
bash scripts/train_2gpu_opendrawer.sh

# 只运行OpenCabinet训练
bash scripts/train_2gpu_opencabinet.sh
```

---

## 📊 训练配置对比

| 配置项 | 4-GPU联合 | 2-GPU Drawer | 2-GPU Cabinet |
|--------|-----------|--------------|---------------|
| GPUs | 0,1,2,3 | 4,5 | 6,7 |
| 任务 | Cabinet+Drawer | Drawer Only | Cabinet Only |
| Batch/GPU | 8 | 16 | 16 |
| 总Batch | 32 | 32 | 32 |
| 训练步数 | 60,000 | 60,000 | 60,000 |
| 输出目录 | opencabinet_opendrawer_4gpu | opendrawer_only_2gpu | opencabinet_only_2gpu |

---

## 🔍 关键特性

### 1. 智能数据集注册
- 使用Python wrapper动态注册自定义数据集
- 无需修改原始代码
- 每个脚本独立的dataset soup

### 2. GPU隔离
- 3个训练使用不同的GPU组
- 可以同时运行，互不干扰
- 充分利用8个GPU

### 3. 统一配置
- 所有训练使用相同的超参数
- 总effective batch size都是32
- 便于对比结果

### 4. 完整日志
- 每个训练独立的日志文件
- TensorBoard实时监控
- 进程ID保存便于管理

---

## 📈 监控和管理

### 查看GPU使用

```bash
watch -n 1 nvidia-smi
```

### 查看训练日志

```bash
# 实时查看
tail -f logs/train_4gpu_opencabinet_opendrawer.log
tail -f logs/train_2gpu_opendrawer.log
tail -f logs/train_2gpu_opencabinet.log

# 或者用less查看
less logs/train_4gpu_opencabinet_opendrawer.log
```

### TensorBoard监控

```bash
# 监控单个训练
tensorboard --logdir /data1/mingzhe/experiment/atomic_seen/opencabinet_opendrawer_4gpu

# 同时监控所有训练（推荐）
tensorboard --logdir /data1/mingzhe/experiment/atomic_seen
```

### 停止训练

```bash
# 停止所有训练
pkill -f gr00t_finetune

# 停止特定训练（使用PID）
kill <PID>

# PIDs保存在
cat logs/training_pids.txt
```

---

## 🎯 预期输出

每个训练会在 `/data1/mingzhe/experiment/atomic_seen/` 下创建对应目录：

```
/data1/mingzhe/experiment/atomic_seen/
├── opencabinet_opendrawer_4gpu/
│   ├── checkpoint-10000/
│   ├── checkpoint-20000/
│   ├── ...
│   ├── checkpoint-60000/
│   └── runs/  # TensorBoard logs
│
├── opendrawer_only_2gpu/
│   ├── checkpoint-10000/
│   ├── ...
│   └── runs/
│
└── opencabinet_only_2gpu/
    ├── checkpoint-10000/
    ├── ...
    └── runs/
```

---

## 🛠️ 技术实现

### 数据集注册方式

每个脚本使用Python wrapper在运行时动态注册数据集：

```python
# 在训练脚本启动前注册
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
DATASET_SOUP_REGISTRY['custom_xxx'] = [
    {'path': '/path/to/dataset', 'filter_key': None},
]
```

这样可以：
- 不修改原始代码
- 灵活定义数据集组合
- 每个训练独立配置

---

## 📚 文件位置总结

```
/data1/mingzhe/Isaac-GR00T-codebase/scripts/
├── train_4gpu_opencabinet_opendrawer.sh   # 4GPU训练脚本
├── train_2gpu_opendrawer.sh               # 2GPU OpenDrawer
├── train_2gpu_opencabinet.sh              # 2GPU OpenCabinet
├── launch_all_trainings.sh                # 主启动器
├── QUICKREF_ATOMIC.sh                     # 快速参考
├── TRAINING_GUIDE_ATOMIC.md               # 详细指南
└── SUMMARY_ATOMIC.md                      # 本文件
```

---

## ✅ 完成清单

- [x] 创建4GPU联合训练脚本
- [x] 创建2GPU OpenDrawer训练脚本
- [x] 创建2GPU OpenCabinet训练脚本
- [x] 创建主启动器
- [x] 创建快速参考指南
- [x] 创建详细使用文档
- [x] 创建项目总结文档
- [x] 所有脚本已设置可执行权限
- [x] GPU分配已优化（无冲突）
- [x] 日志和监控配置完成

---

## 🚀 立即开始

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase
bash scripts/launch_all_trainings.sh
```

**祝训练顺利！** 🎉

---

*创建时间: 2026-06-26*  
*基于: Isaac-GR00T-codebase*  
*目标任务: atomic_seen (OpenCabinet + OpenDrawer)*
