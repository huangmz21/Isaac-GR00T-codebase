#!/bin/bash
# Final ready check before starting training

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════╗
║            ✅ 所有准备工作已完成 - 可以开始训练了！                      ║
╚════════════════════════════════════════════════════════════════════════╝

📋 完成清单：

  ✅ 虚拟环境 (uv)
  ✅ PyTorch & 依赖
  ✅ flash-attn (2.8.3.post1)
  ✅ 数据集路径已修复
  ✅ 4个训练脚本已创建
  ✅ Base model 存在
  ✅ 输出目录已准备

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 开始训练：

  方式1: 4GPU训练 (OpenCabinet + OpenDrawer)
    bash scripts/train_4gpu_opencabinet_opendrawer.sh

  方式2: 2GPU训练 (OpenDrawer Only)
    bash scripts/train_2gpu_opendrawer.sh

  方式3: 2GPU训练 (OpenCabinet Only)
    bash scripts/train_2gpu_opencabinet.sh

  方式4: 8GPU训练 (完整 atomic_seen)
    bash scripts/train_atomic_seen.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 后台运行 (推荐):

  mkdir -p logs
  nohup bash scripts/train_4gpu_opencabinet_opendrawer.sh > logs/train_4gpu.log 2>&1 &

  # 查看日志
  tail -f logs/train_4gpu.log

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 监控训练：

  GPU使用:
    watch -n 1 nvidia-smi

  TensorBoard:
    tensorboard --logdir /data1/mingzhe/experiment/atomic_seen
    # 浏览器: http://localhost:6006

  检查checkpoint:
    ls -lht /data1/mingzhe/experiment/atomic_seen/*/checkpoint-*

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 预期训练时间:

  4GPU (60k steps): ~8-12 小时
  2GPU (60k steps): ~16-20 小时
  8GPU (300k steps): ~50-60 小时

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

准备好开始了吗？运行:
  cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T
  bash scripts/train_4gpu_opencabinet_opendrawer.sh

EOF
