# ✅ Atomic Seen 训练脚本 - 最终配置总结

## 🎉 已完成的工作

### 1. 虚拟环境重新配置
- ✅ 使用 `uv` 重新创建了虚拟环境
- ✅ 所有依赖已正确安装
- ✅ `torchrun` 的 shebang 指向正确路径
- ✅ 旧环境已备份到 `.venv.old.backup.YYYYMMDD`

### 2. 训练脚本创建
所有脚本位于：`/data1/mingzhe/Isaac-GR00T-codebase/scripts/`

#### Python Wrapper 脚本 (3个)
- `wrapper_4gpu_opencabinet_opendrawer.py` - 注册 OpenCabinet + OpenDrawer
- `wrapper_2gpu_opendrawer.py` - 注册 OpenDrawer Only
- `wrapper_2gpu_opencabinet.py` - 注册 OpenCabinet Only

#### Shell 启动脚本 (4个)
- `train_4gpu_opencabinet_opendrawer.sh` - 4GPU联合训练
- `train_2gpu_opendrawer.sh` - 2GPU OpenDrawer
- `train_2gpu_opencabinet.sh` - 2GPU OpenCabinet
- `train_atomic_seen.sh` - 8GPU 完整 atomic_seen

### 3. 文档
- `TRAINING_GUIDE_ATOMIC.md` - 详细使用指南
- `QUICKREF_ATOMIC.sh` - 快速参考
- `SUMMARY_ATOMIC.md` - 项目总结
- `SETUP_COMPLETE.md` - 本文件

---

## 🚀 快速开始

### 运行训练

```bash
cd /data1/mingzhe/Isaac-GR00T-codebase

# 4GPU训练 (OpenCabinet + OpenDrawer)
bash scripts/train_4gpu_opencabinet_opendrawer.sh

# 2GPU训练 (OpenDrawer)
bash scripts/train_2gpu_opendrawer.sh

# 2GPU训练 (OpenCabinet)
bash scripts/train_2gpu_opencabinet.sh

# 8GPU训练 (完整 atomic_seen)
bash scripts/train_atomic_seen.sh
```

### 后台运行

```bash
# 创建logs目录
mkdir -p logs

# 后台运行
nohup bash scripts/train_4gpu_opencabinet_opendrawer.sh > logs/train_4gpu.log 2>&1 &

# 查看日志
tail -f logs/train_4gpu.log
```

### 同时运行多个训练

```bash
# 启动所有3个训练（使用不同GPU组）
nohup bash scripts/train_4gpu_opencabinet_opendrawer.sh > logs/train_4gpu.log 2>&1 &
nohup bash scripts/train_2gpu_opendrawer.sh > logs/train_2gpu_drawer.log 2>&1 &
nohup bash scripts/train_2gpu_opencabinet.sh > logs/train_2gpu_cabinet.log 2>&1 &
```

---

## 📊 GPU 分配

| 训练脚本 | GPUs | 任务 | Batch/GPU | 总Batch |
|----------|------|------|-----------|---------|
| train_4gpu_opencabinet_opendrawer.sh | 0,1,2,3 | OpenCabinet + OpenDrawer | 8 | 32 |
| train_2gpu_opendrawer.sh | 4,5 | OpenDrawer Only | 16 | 32 |
| train_2gpu_opencabinet.sh | 6,7 | OpenCabinet Only | 16 | 32 |
| train_atomic_seen.sh | 0-7 | 所有18个atomic任务 | 16 | 128 |

---

## 📈 监控训练

### GPU 使用

```bash
watch -n 1 nvidia-smi
```

### TensorBoard

```bash
# 监控所有训练
tensorboard --logdir /data1/mingzhe/experiment/atomic_seen

# 浏览器打开
http://localhost:6006
```

### 查看日志

```bash
# 实时查看
tail -f logs/train_4gpu.log

# 检查进程
ps aux | grep gr00t_finetune

# 查看checkpoint
ls -lh /data1/mingzhe/experiment/atomic_seen/*/checkpoint-*
```

---

## 🛠️ 故障排除

### 1. 如果训练中断

```bash
# 检查最后的checkpoint
ls -lht /data1/mingzhe/experiment/atomic_seen/opencabinet_opendrawer_4gpu/checkpoint-*

# 可以从checkpoint恢复（需修改脚本添加 --resume）
```

### 2. CUDA OOM

```bash
# 减少batch size（在对应的.sh文件中修改）
BATCH_SIZE=4  # 从8或16降低
```

### 3. 数据集路径问题

如果遇到类似错误：
```
AssertionError: Dataset path /mnt/ssd_data/... does not exist
```

这是因为数据集注册表中的路径可能不对。检查实际数据集位置并修改wrapper脚本中的路径。

---

## 📁 输出结构

```
/data1/mingzhe/experiment/atomic_seen/
├── opencabinet_opendrawer_4gpu/
│   ├── checkpoint-10000/
│   ├── checkpoint-20000/
│   ├── ...
│   ├── checkpoint-60000/
│   └── runs/  # TensorBoard日志
├── opendrawer_only_2gpu/
│   └── ...
├── opencabinet_only_2gpu/
│   └── ...
└── full_atomic_seen_8gpu/
    └── ...
```

---

## ✅ 验证清单

在开始长时间训练前，建议检查：

- [ ] 虚拟环境正确激活
- [ ] GPU空闲且可用 (`nvidia-smi`)
- [ ] 数据集路径存在
- [ ] 输出目录有写权限
- [ ] Base model checkpoint存在
- [ ] 足够的磁盘空间（每个checkpoint约10GB）

---

## 📞 需要帮助？

1. 查看快速参考：`bash scripts/QUICKREF_ATOMIC.sh`
2. 查看详细指南：`cat scripts/TRAINING_GUIDE_ATOMIC.md`
3. 检查环境：`source .venv/bin/activate && python -c "import torch; print(torch.__version__)"`

---

**祝训练顺利！** 🚀

*配置完成时间: 2026-06-26*  
*虚拟环境: uv + Python 3.10*  
*代码库: /data1/mingzhe/Isaac-GR00T-codebase*
