#!/bin/bash
# Full fine-tuning on atomic_seen dataset with Per-Task Loss Tracking
# 8-GPU DDP training with sample-level queue for real-time per-task monitoring

# ---- paths ----
BASE_MODEL_PATH="/data1/mingzhe/models/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000"
OUTPUT_DIR="/data1/mingzhe/experiment/atomic_seen_with_per_task_tracking"

# ---- dataset ----
# 使用 atomic_seen (18个atomic任务)
DATASET_SOUP="atomic_seen"

# ---- training params ----
NUM_GPUS=8
BATCH_SIZE=16              # per-GPU；全局 batch = 16 × 8 = 128
DATALOADER_NUM_WORKERS=16
MAX_STEPS=300000
SAVE_STEPS=10000

# ---- logging ----
REPORT_TO="wandb"
WANDB_PROJECT="atomic_seen_per_task_tracking"
RUN_NAME="full_with_tracking_v2"

echo "=========================================="
echo "🚀 Starting Training with Per-Task Loss Tracking (V2)"
echo "=========================================="
echo "Base model:   $BASE_MODEL_PATH"
echo "Dataset soup: $DATASET_SOUP (18 tasks)"
echo "Output dir:   $OUTPUT_DIR"
echo "GPUs:         $NUM_GPUS  (per-GPU batch=$BATCH_SIZE, global=$((BATCH_SIZE * NUM_GPUS)))"
echo "Max steps:    $MAX_STEPS  (save every $SAVE_STEPS)"
echo "WandB:        $WANDB_PROJECT / $RUN_NAME"
echo ""
echo "📊 Per-Task Tracking Features:"
echo "  - Sample-level queue (1000 samples per task)"
echo "  - Real-time loss monitoring for each task"
echo "  - Automatic trend detection (improving/stable/degrading)"
echo "  - Summary report every 100 steps"
echo "=========================================="
echo ""

# 运行训练
python3 scripts/gr00t_finetune.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --dataset_soup "$DATASET_SOUP" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus "$NUM_GPUS" \
    --batch_size "$BATCH_SIZE" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --report_to "$REPORT_TO"

echo ""
echo "=========================================="
echo "✅ Training finished!"
echo "=========================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "📊 Check your WandB dashboard for:"
echo "  - task_*/loss_avg: Per-task loss curves"
echo "  - task_*/queue_size: Queue utilization"
echo "  - global/task_loss_mean: Average across all tasks"
echo "  - Trend indicators in console logs"
echo ""
echo "🎯 Next steps:"
echo "  1. Review per-task loss curves in WandB"
echo "  2. Identify struggling tasks (highest loss)"
echo "  3. Adjust sampling weights if needed"
echo "=========================================="
