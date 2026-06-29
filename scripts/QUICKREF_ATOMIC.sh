#!/bin/bash
# Quick reference for atomic_seen training scripts

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════╗
║                Atomic Seen Training - Quick Reference                  ║
╚════════════════════════════════════════════════════════════════════════╝

📁 Location: /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/scripts/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 QUICK START

  方式1: 运行单个训练
    cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
    bash scripts/train_4gpu_opencabinet_opendrawer.sh
    bash scripts/train_2gpu_opendrawer.sh
    bash scripts/train_2gpu_opencabinet.sh

  方式2: 一键启动全部（推荐）
    cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
    bash scripts/launch_all_trainings.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 TRAINING SCRIPTS

┌─────────────────────────────────────────────────────────────────────┐
│ 1. train_4gpu_opencabinet_opendrawer.sh                            │
│    任务: OpenCabinet + OpenDrawer (联合训练)                        │
│    GPU: 0,1,2,3                                                     │
│    Batch: 8/GPU (32 total)                                         │
│    输出: experiment/atomic_seen/opencabinet_opendrawer_4gpu         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2. train_2gpu_opendrawer.sh                                        │
│    任务: OpenDrawer Only                                            │
│    GPU: 4,5                                                         │
│    Batch: 16/GPU (32 total)                                        │
│    输出: experiment/atomic_seen/opendrawer_only_2gpu                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 3. train_2gpu_opencabinet.sh                                       │
│    任务: OpenCabinet Only                                           │
│    GPU: 6,7                                                         │
│    Batch: 16/GPU (32 total)                                        │
│    输出: experiment/atomic_seen/opencabinet_only_2gpu               │
└─────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 COMMON COMMANDS

  监控GPU:
    watch -n 1 nvidia-smi

  查看日志:
    tail -f logs/train_4gpu_opencabinet_opendrawer.log
    tail -f logs/train_2gpu_opendrawer.log
    tail -f logs/train_2gpu_opencabinet.log

  TensorBoard:
    tensorboard --logdir /data1/mingzhe/experiment/atomic_seen

  检查训练进程:
    ps aux | grep gr00t_finetune

  停止所有训练:
    pkill -f gr00t_finetune

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️  TRAINING CONFIG

  Base Model:    checkpoint-80000
  Max Steps:     60,000
  Save Steps:    10,000
  Learning Rate: 3e-5
  Batch Size:    8 or 16 per GPU (32 total effective)

  Trainable:     Projector + Diffusion Model
  Frozen:        LLM + Visual Backbone

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 GPU ALLOCATION

  GPUs 0,1,2,3 → 4-GPU Training (OpenCabinet + OpenDrawer)
  GPUs 4,5     → 2-GPU Training (OpenDrawer Only)
  GPUs 6,7     → 2-GPU Training (OpenCabinet Only)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 DOCUMENTATION

  详细使用指南: scripts/TRAINING_GUIDE_ATOMIC.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF
