#!/bin/bash

# 多卡小batch实验: 4 GPUs, batch_size=2 per GPU
# 总的全局batch size = 4 * 2 = 8 (与单卡实验相同)

cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTHONPATH=/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T:$PYTHONPATH
export HF_HOME=/home/huangmingzhe/.cache/huggingface
export TRANSFORMERS_CACHE=/home/huangmingzhe/.cache/huggingface/transformers

# Run training (use absolute path to venv python)
/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/.venv/bin/python scripts/gr00t_finetune.py \
    --dataset-soup atomic_seen \
    --base-model-path /mnt/ssd_data/mingzhe/Model/robocasa365/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000 \
    --output-dir /mnt/ssd_data/mingzhe/Model/robocasa365/experiments/multi_gpu_bs2 \
    --batch-size 2 \
    --num-gpus 4 \
    --max-steps 200 \
    --save-steps 1000 \
    --learning-rate 3e-5 \
    --weight-decay 1e-5 \
    --warmup-ratio 0.05 \
    --dataloader-num-workers 8 \
    --report-to tensorboard \
    --data-config panda_omron \
    --embodiment-tag new_embodiment \
    --video-backend opencv \
    --tune-projector \
    --tune-diffusion-model \
    --no-tune-llm \
    --no-tune-visual \
    --no-resume \
    --balance-dataset-weights \
    --balance-trajectory-weights \
    --ds-weights-alpha 0.4
