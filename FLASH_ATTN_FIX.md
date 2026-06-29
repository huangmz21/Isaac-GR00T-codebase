# Flash-attn 安装问题及解决方案

## 问题描述

训练时报错：`ModuleNotFoundError: No module named 'flash_attn'`

## 根本原因

flash-attn 被安装到了系统 Python 的用户目录，而不是项目的虚拟环境 `.venv` 中。

虚拟环境的 pip 错误地指向了系统 pip，导致包安装位置不正确。

## 解决方案

### 正在进行中 ✅

使用 `python -m pip install` 确保安装到虚拟环境：

```bash
source .venv/bin/activate
python -m pip install flash-attn --no-build-isolation
```

这个命令正在后台运行中，预计需要 5-10 分钟（因为系统中已有编译版本）。

### 验证安装

安装完成后，验证：

```bash
source .venv/bin/activate
python -c "import flash_attn; print(f'✅ flash-attn {flash_attn.__version__}')"
```

应该输出：`✅ flash-attn 2.8.3.post1`

### 然后重新训练

```bash
cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
bash scripts/train_robocasa_atomic_seen.sh
```

## 技术细节

### 问题诊断

```bash
# 检查 pip 位置
source .venv/bin/activate
which pip          # 输出: /usr/bin/pip (错误！应该在 .venv/bin/)

# 检查 Python 位置
which python       # 输出: /mnt/.../Isaac-GR00T/.venv/bin/python (正确)

# 检查 flash-attn 安装位置
python -m pip show flash-attn
# Location: /home/huangmingzhe/.local/lib/python3.10/site-packages (系统用户目录)
```

### 为什么会这样

虚拟环境创建时，pip 的符号链接可能指向了错误的位置。使用 `python -m pip` 而不是直接使用 `pip` 可以确保使用虚拟环境的 Python 来执行 pip。

## 其他解决方案（备用）

### 方案 1: 复制系统中的 flash-attn

如果编译太慢，可以直接复制系统中已编译好的：

```bash
cp -r /home/huangmingzhe/.local/lib/python3.10/site-packages/flash_attn* \
      /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/.venv/lib/python3.10/site-packages/
```

### 方案 2: 修复虚拟环境的 pip

```bash
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

## 当前状态

⏳ **正在安装** flash-attn 到虚拟环境中...

查看进度：
```bash
tail -f /tmp/claude-1005/-mnt-ssd-data-mingzhe/3c7398ae-e1ec-4e52-938c-0f63020e3e3c/tasks/bojjr7cs9.output
```

## 预期完成时间

5-10 分钟（系统中已有编译版本，应该会快）

完成后会收到通知，然后就可以开始训练了！
