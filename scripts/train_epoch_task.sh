#!/bin/bash
# Parameterized single-task (or any-soup) epoch-based training.
#
# Usage:
#   bash scripts/train_epoch_task.sh <DATASET_SOUP> [GPU_IDS] [NUM_EPOCHS]
#
# Examples:
#   bash scripts/train_epoch_task.sh opencabinet_only 0,1
#   bash scripts/train_epoch_task.sh turnonelectrickettle_only 2,3 10
#   bash scripts/train_epoch_task.sh atomic_seen 0,1,2,3,4,5,6,7 10
#
# Each soup writes to its own output dir: ./outputs/epoch/<DATASET_SOUP>

set -e

# ---- Arguments ----
DATASET_SOUP="${1:?Usage: train_epoch_task.sh <DATASET_SOUP> [GPU_IDS] [NUM_EPOCHS]}"
GPU_IDS="${2:-0,1}"
NUM_EPOCHS="${3:-10}"

# Derive number of GPUs from the comma-separated GPU_IDS
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | grep -c .)

# ---- Fixed config ----
DATA_CONFIG="panda_omron"
BATCH_SIZE=32
OUTPUT_DIR="./outputs/epoch/${DATASET_SOUP}"
BASE_MODEL_PATH="/data1/mingzhe/models/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000"

# ---- Environment ----
source .venv/bin/activate
export PYTHONPATH=/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T:$PYTHONPATH
export HF_HOME=/home/huangmingzhe/.cache/huggingface
export TRANSFORMERS_CACHE=/home/huangmingzhe/.cache/huggingface/transformers
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

echo "=========================================="
echo "Epoch-based Training"
echo "=========================================="
echo "Dataset soup:   ${DATASET_SOUP}"
echo "Epochs:         ${NUM_EPOCHS}"
echo "Batch/GPU:      ${BATCH_SIZE}"
echo "GPU IDs:        ${GPU_IDS} (${NUM_GPUS} GPUs)"
echo "Output dir:     ${OUTPUT_DIR}"
echo "=========================================="

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
