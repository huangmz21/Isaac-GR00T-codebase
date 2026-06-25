# Flash Attention 安装说明

## 问题
训练启动时报错：`ModuleNotFoundError: No module named 'flash_attn'`

## 原因
GR00T 模型的 EAGLE backbone 需要 flash-attn 来加速 attention 计算。

## 解决方案

### 方案 1: 编译安装 flash-attn (推荐)

flash-attn 需要从源码编译，这个过程比较耗时（10-20分钟）。

```bash
cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
source .venv/bin/activate

# 安装 ninja 加速编译
pip install ninja

# 使用多线程编译安装 flash-attn
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

### 方案 2: 使用预编译的 wheel (更快)

如果可以找到预编译的 wheel，可以直接安装：

```bash
# 检查系统信息
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# 根据 CUDA 版本下载对应的 wheel
# 例如 CUDA 12.1:
pip install flash-attn --find-links https://github.com/Dao-AILab/flash-attention/releases
```

### 方案 3: 禁用 flash-attn (不推荐，性能会下降)

如果实在无法安装，可以修改代码禁用 flash-attn，但训练速度会明显变慢。

## 安装状态检查

检查 flash-attn 是否安装成功：

```bash
source .venv/bin/activate
python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')"
```

## 当前安装进度

正在使用 MAX_JOBS=8 并行编译安装 flash-attn...

查看安装进度：
```bash
tail -f /tmp/claude-1005/-mnt-ssd-data-mingzhe/3c7398ae-e1ec-4e52-938c-0f63020e3e3c/tasks/bpa4wnh2r.output
```

## 安装完成后

安装完成后，重新运行训练脚本：

```bash
cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
bash scripts/train_robocasa_atomic_seen.sh
```

## 注意事项

1. flash-attn 编译需要：
   - CUDA 工具链
   - 足够的内存（建议 16GB+）
   - 编译时间较长（10-20分钟）

2. 编译时 CPU 占用率会很高，这是正常现象

3. 如果编译失败，检查：
   - CUDA 版本是否兼容
   - PyTorch 版本是否兼容
   - 编译工具是否完整（gcc, g++, nvcc）
