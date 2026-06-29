#!/usr/bin/env python3
"""
分析和对比单卡大batch与多卡小batch的训练loss差异

使用方法:
    python analyze_batch_comparison.py

输出:
    - loss对比图表
    - 统计差异分析
    - 详细的数值对比
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 实验路径
SINGLE_GPU_DIR = "/mnt/ssd_data/mingzhe/Model/robocasa365/experiments/single_gpu_bs8"
MULTI_GPU_DIR = "/mnt/ssd_data/mingzhe/Model/robocasa365/experiments/multi_gpu_bs2"


def parse_trainer_state(trainer_state_path):
    """解析 trainer_state.json 文件，提取 loss 历史"""
    with open(trainer_state_path, 'r') as f:
        state = json.load(f)

    log_history = state.get('log_history', [])

    steps = []
    losses = []

    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])

    return np.array(steps), np.array(losses)


def find_latest_checkpoint(exp_dir):
    """找到最新的checkpoint目录"""
    checkpoints = [d for d in Path(exp_dir).glob("checkpoint-*") if d.is_dir()]
    if not checkpoints:
        return None
    # 按照checkpoint编号排序，返回最新的
    latest = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
    return latest


def analyze_experiments():
    """分析两个实验的结果"""

    print("=" * 60)
    print("单卡大batch vs 多卡小batch 对比分析")
    print("=" * 60)
    print()

    # 查找单卡实验的trainer_state.json
    single_gpu_checkpoint = find_latest_checkpoint(SINGLE_GPU_DIR)
    if single_gpu_checkpoint:
        single_gpu_state = single_gpu_checkpoint / "trainer_state.json"
    else:
        # 也可能在根目录
        single_gpu_state = Path(SINGLE_GPU_DIR) / "trainer_state.json"

    # 查找多卡实验的trainer_state.json
    multi_gpu_checkpoint = find_latest_checkpoint(MULTI_GPU_DIR)
    if multi_gpu_checkpoint:
        multi_gpu_state = multi_gpu_checkpoint / "trainer_state.json"
    else:
        multi_gpu_state = Path(MULTI_GPU_DIR) / "trainer_state.json"

    # 检查文件是否存在
    if not single_gpu_state.exists():
        print(f"错误: 单卡实验的 trainer_state.json 不存在: {single_gpu_state}")
        print(f"请检查路径: {SINGLE_GPU_DIR}")
        return

    if not multi_gpu_state.exists():
        print(f"错误: 多卡实验的 trainer_state.json 不存在: {multi_gpu_state}")
        print(f"请检查路径: {MULTI_GPU_DIR}")
        return

    print(f"读取单卡实验数据: {single_gpu_state}")
    steps_single, losses_single = parse_trainer_state(single_gpu_state)

    print(f"读取多卡实验数据: {multi_gpu_state}")
    steps_multi, losses_multi = parse_trainer_state(multi_gpu_state)

    print()
    print("-" * 60)
    print("实验统计信息")
    print("-" * 60)

    # 单卡实验统计
    print(f"\n单卡实验 (batch_size=8):")
    print(f"  总训练步数: {len(steps_single)}")
    if len(losses_single) > 0:
        print(f"  初始 loss: {losses_single[0]:.6f}")
        print(f"  最终 loss: {losses_single[-1]:.6f}")
        print(f"  平均 loss: {np.mean(losses_single):.6f}")
        print(f"  最小 loss: {np.min(losses_single):.6f}")
        print(f"  loss 标准差: {np.std(losses_single):.6f}")

    # 多卡实验统计
    print(f"\n多卡实验 (4卡 x batch_size=2):")
    print(f"  总训练步数: {len(steps_multi)}")
    if len(losses_multi) > 0:
        print(f"  初始 loss: {losses_multi[0]:.6f}")
        print(f"  最终 loss: {losses_multi[-1]:.6f}")
        print(f"  平均 loss: {np.mean(losses_multi):.6f}")
        print(f"  最小 loss: {np.min(losses_multi):.6f}")
        print(f"  loss 标准差: {np.std(losses_multi):.6f}")

    # 对比分析
    if len(losses_single) > 0 and len(losses_multi) > 0:
        print()
        print("-" * 60)
        print("对比分析")
        print("-" * 60)

        # 找到共同的步数范围
        min_steps = min(len(steps_single), len(steps_multi))

        if min_steps > 0:
            # 计算在相同步数下的loss差异
            loss_diff = losses_single[:min_steps] - losses_multi[:min_steps]

            print(f"\n前 {min_steps} 步的差异 (单卡 - 多卡):")
            print(f"  平均差异: {np.mean(loss_diff):.6f}")
            print(f"  最大差异: {np.max(loss_diff):.6f}")
            print(f"  最小差异: {np.min(loss_diff):.6f}")
            print(f"  差异标准差: {np.std(loss_diff):.6f}")

            # 相对差异
            rel_diff = (loss_diff / losses_multi[:min_steps]) * 100
            print(f"\n相对差异 (百分比):")
            print(f"  平均相对差异: {np.mean(rel_diff):.2f}%")

    # 绘制对比图
    print()
    print("-" * 60)
    print("生成可视化图表...")
    print("-" * 60)

    plt.figure(figsize=(14, 10))

    # 子图1: Loss曲线对比
    plt.subplot(2, 2, 1)
    if len(losses_single) > 0:
        plt.plot(steps_single, losses_single, 'b-', label='单卡 (bs=8)', linewidth=2)
    if len(losses_multi) > 0:
        plt.plot(steps_multi, losses_multi, 'r-', label='多卡 (4x bs=2)', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss曲线对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: Loss差异
    plt.subplot(2, 2, 2)
    if len(losses_single) > 0 and len(losses_multi) > 0:
        min_steps = min(len(steps_single), len(steps_multi))
        loss_diff = losses_single[:min_steps] - losses_multi[:min_steps]
        plt.plot(steps_single[:min_steps], loss_diff, 'g-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Difference (单卡 - 多卡)')
        plt.title('Loss差异曲线')
        plt.grid(True, alpha=0.3)

    # 子图3: 平滑后的Loss曲线
    plt.subplot(2, 2, 3)
    window = 10  # 移动平均窗口
    if len(losses_single) >= window:
        smoothed_single = np.convolve(losses_single, np.ones(window)/window, mode='valid')
        plt.plot(steps_single[:len(smoothed_single)], smoothed_single, 'b-',
                label=f'单卡 (bs=8, 平滑窗口={window})', linewidth=2)
    if len(losses_multi) >= window:
        smoothed_multi = np.convolve(losses_multi, np.ones(window)/window, mode='valid')
        plt.plot(steps_multi[:len(smoothed_multi)], smoothed_multi, 'r-',
                label=f'多卡 (4x bs=2, 平滑窗口={window})', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Smoothed Loss')
    plt.title('平滑后的Loss曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图4: Loss分布对比（直方图）
    plt.subplot(2, 2, 4)
    if len(losses_single) > 0:
        plt.hist(losses_single, bins=30, alpha=0.5, label='单卡 (bs=8)', color='blue')
    if len(losses_multi) > 0:
        plt.hist(losses_multi, bins=30, alpha=0.5, label='多卡 (4x bs=2)', color='red')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.title('Loss分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    output_path = "/mnt/ssd_data/mingzhe/Model/robocasa365/experiments/batch_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")

    print()
    print("=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    analyze_experiments()
