#!/bin/bash
# Example: Single-task training with epoch-based control
#
# This demonstrates the new epoch-based training mode for fair comparison.

set -e

# Activate environment
source .venv/bin/activate

# Set HuggingFace cache directories (fix permission issues)
export HF_HOME=/home/huangmingzhe/.cache/huggingface
export TRANSFORMERS_CACHE=/home/huangmingzhe/.cache/huggingface/transformers

# Configuration
DATASET_SOUP="opencabinet_opendrawer"  # Single task: OpenDrawer
DATA_CONFIG="panda_omron"
NUM_EPOCHS=10
BATCH_SIZE=32
NUM_GPUS=2
OUTPUT_DIR="./outputs/single_task_epoch_example"
BASE_MODEL_PATH="/data1/mingzhe/models/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000"

echo "=========================================="
echo "Epoch-based Single-task Training Example"
echo "=========================================="
echo "Dataset: ${DATASET_SOUP}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "GPUs: ${NUM_GPUS}"
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
