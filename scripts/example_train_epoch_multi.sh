#!/bin/bash
# Example: Multi-task training with epoch-based control and naive concatenation
#
# This demonstrates the new epoch-based training mode for fair comparison.
# Unlike the weighted sampling in gr00t_finetune.py, this simply concatenates
# all datasets and trains for a fixed number of epochs — each sample seen
# exactly the same number of times as in single-task training.

set -e

# Activate environment
source .venv/bin/activate

# Set HuggingFace cache directories (fix permission issues)
export HF_HOME=/home/huangmingzhe/.cache/huggingface
export TRANSFORMERS_CACHE=/home/huangmingzhe/.cache/huggingface/transformers

# Configuration
DATASET_SOUP="atomic_seen"  # Multi-task (18 tasks)
DATA_CONFIG="panda_omron"
NUM_EPOCHS=10
BATCH_SIZE=16
NUM_GPUS=2
OUTPUT_DIR="./outputs/multi_task_epoch_example"
BASE_MODEL_PATH="/data1/mingzhe/models/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000"

echo "=========================================="
echo "Epoch-based Multi-task Training Example"
echo "=========================================="
echo "Dataset: ${DATASET_SOUP}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "Mode: Naive concatenation (no weighting)"
echo "=========================================="

# Run training
python scripts/gr00t_finetune_epoch.py \
    --dataset-soup "${DATASET_SOUP}" \
    --data-config "${DATA_CONFIG}" \
    --num-train-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-gpus "${NUM_GPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --tune-projector \
    --tune-diffusion-model \
    --learning-rate 3e-5 \
    --save-steps 5000 \
    --report-to tensorboard

echo "Training complete! Checkpoints saved to: ${OUTPUT_DIR}"
echo ""
echo "Fair comparison tip:"
echo "  If you trained a single task for ${NUM_EPOCHS} epochs,"
echo "  this multi-task run has seen each sample the same number"
echo "  of times (${NUM_EPOCHS}), making the comparison valid."
