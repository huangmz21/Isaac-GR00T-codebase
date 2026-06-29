#!/bin/bash

# 对比实验启动脚本
# 用于启动单卡大batch和多卡小batch的对比实验
# 两个实验使用相同的数据集、相同的全局batch size (8)，以及相同的随机种子

echo "=========================================="
echo "开始对比实验"
echo "实验1: 单卡 batch_size=8 (全局batch=8)"
echo "实验2: 四卡 每卡batch_size=2 (全局batch=8)"
echo "=========================================="

# 创建输出目录
mkdir -p /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/single_gpu_bs8
mkdir -p /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/multi_gpu_bs2

echo ""
echo "实验配置:"
echo "  - 数据集: target_atomic_seen"
echo "  - Base Model: checkpoint-80000"
echo "  - 训练步数: 5000"
echo "  - 学习率: 3e-5"
echo "  - 随机种子: 42 (代码中固定)"
echo ""

# 询问用户要运行哪个实验
echo "请选择要运行的实验:"
echo "1) 单卡实验 (batch_size=8)"
echo "2) 多卡实验 (4卡, batch_size=2)"
echo "3) 依次运行两个实验"
echo "4) 并行运行两个实验 (需要8张GPU)"
read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "启动单卡实验..."
        bash /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/scripts/experiment_single_gpu_large_batch.sh
        ;;
    2)
        echo ""
        echo "启动多卡实验..."
        bash /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/scripts/experiment_multi_gpu_small_batch.sh
        ;;
    3)
        echo ""
        echo "依次运行两个实验..."
        echo ""
        echo "[1/2] 启动单卡实验..."
        bash /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/scripts/experiment_single_gpu_large_batch.sh

        echo ""
        echo "[2/2] 启动多卡实验..."
        bash /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/scripts/experiment_multi_gpu_small_batch.sh
        ;;
    4)
        echo ""
        echo "并行运行两个实验..."
        echo "注意: 这需要8张GPU (单卡实验用GPU 0, 多卡实验用GPU 1-4)"

        # 修改CUDA_VISIBLE_DEVICES以避免冲突
        echo ""
        echo "[1/2] 在后台启动单卡实验 (使用GPU 0)..."
        CUDA_VISIBLE_DEVICES=0 bash /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/scripts/experiment_single_gpu_large_batch.sh > /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/single_gpu_bs8.log 2>&1 &
        PID1=$!

        sleep 5

        echo "[2/2] 在后台启动多卡实验 (使用GPU 4-7)..."
        CUDA_VISIBLE_DEVICES=4,5,6,7 bash /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/scripts/experiment_multi_gpu_small_batch.sh > /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/multi_gpu_bs2.log 2>&1 &
        PID2=$!

        echo ""
        echo "两个实验已在后台启动:"
        echo "  - 单卡实验 PID: $PID1, 日志: /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/single_gpu_bs8.log"
        echo "  - 多卡实验 PID: $PID2, 日志: /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/multi_gpu_bs2.log"
        echo ""
        echo "使用以下命令查看实时日志:"
        echo "  tail -f /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/single_gpu_bs8.log"
        echo "  tail -f /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/multi_gpu_bs2.log"
        ;;
    *)
        echo "无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "实验完成!"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  单卡实验: /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/single_gpu_bs8"
echo "  多卡实验: /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/multi_gpu_bs2"
echo ""
echo "使用 tensorboard 查看训练曲线:"
echo "  tensorboard --logdir /mnt/ssd_data/mingzhe/Model/robocasa365/experiments --port 6006"
