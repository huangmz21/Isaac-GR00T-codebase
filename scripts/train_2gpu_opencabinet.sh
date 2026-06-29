#!/bin/bash

# 2-GPU Training Script for OpenCabinet Only

cd /data1/mingzhe/Isaac-GR00T-codebase

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=6,7
export PYTHONPATH=/data1/mingzhe/Isaac-GR00T-codebase:$PYTHONPATH
export HF_HOME=/home/huangmingzhe/.cache/huggingface
export TRANSFORMERS_CACHE=/home/huangmingzhe/.cache/huggingface/transformers

# Training configuration
RUN_NAME="opencabinet_only_2gpu"
OUTPUT_DIR="/data1/mingzhe/experiment/atomic_seen/${RUN_NAME}"
DATA_CONFIG="panda_omron"
NUM_GPUS=2
BATCH_SIZE=16  # 16 per GPU = 32 total effective batch size
MAX_STEPS=60000
SAVE_STEPS=10000
BASE_MODEL_PATH="/data1/mingzhe/models/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000"
LEARNING_RATE=3e-5
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0.05
LORA_RANK=0
LORA_ALPHA=16
LORA_DROPOUT=0.1
DATALOADER_NUM_WORKERS=8
EMBODIMENT_TAG="new_embodiment"
VIDEO_BACKEND="opencv"
DS_WEIGHTS_ALPHA=0.4

echo "=========================================="
echo "Training: OpenCabinet Only (2 GPUs)"
echo "=========================================="
echo "GPUs: 6,7"
echo "Output: ${OUTPUT_DIR}"
echo "Dataset: OpenCabinet"
echo "=========================================="
echo ""

# Run training directly (no wrapper needed)
.venv/bin/python scripts/gr00t_finetune.py \
    --dataset-soup opencabinet_only \
    --output-dir ${OUTPUT_DIR} \
    --data-config ${DATA_CONFIG} \
    --batch-size ${BATCH_SIZE} \
    --max-steps ${MAX_STEPS} \
    --num-gpus ${NUM_GPUS} \
    --save-steps ${SAVE_STEPS} \
    --base-model-path ${BASE_MODEL_PATH} \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --no-resume \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --warmup-ratio ${WARMUP_RATIO} \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --lora-dropout ${LORA_DROPOUT} \
    --no-lora-full-model \
    --dataloader-num-workers ${DATALOADER_NUM_WORKERS} \
    --report-to tensorboard \
    --embodiment-tag ${EMBODIMENT_TAG} \
    --video-backend ${VIDEO_BACKEND} \
    --balance-dataset-weights \
    --balance-trajectory-weights \
    --ds-weights-alpha ${DS_WEIGHTS_ALPHA}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=========================================="
