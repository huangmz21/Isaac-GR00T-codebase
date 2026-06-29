#!/bin/bash
# Launch 4 epoch-based trainings in parallel, each on its own pair of GPUs.
#
# Edit the SOUPS array below to pick which 4 dataset soups to train.
# Each entry is "soup_name:gpu_ids". Each training runs in the background
# and logs to ./outputs/epoch/<soup>/launch.log
#
# Usage:
#   bash scripts/launch_4_trainings.sh
#
# Monitor:
#   tail -f ./outputs/epoch/<soup>/launch.log
#   nvidia-smi

set -e
cd "$(dirname "$0")/.."   # repo root

NUM_EPOCHS=10

# soup_name : comma-separated GPU ids
SOUPS=(
    "opencabinet_only:0,1"
    "opendrawer_only:2,3"
    "turnonelectrickettle_only:4,5"
    "turnonsinkfaucet_only:6,7"
)

echo "=========================================="
echo "Launching ${#SOUPS[@]} parallel trainings"
echo "=========================================="

for entry in "${SOUPS[@]}"; do
    soup="${entry%%:*}"
    gpus="${entry##*:}"
    outdir="./outputs/epoch/${soup}"
    mkdir -p "${outdir}"

    echo "  ${soup}  ->  GPUs ${gpus}  (log: ${outdir}/launch.log)"
    nohup bash scripts/train_epoch_task.sh "${soup}" "${gpus}" "${NUM_EPOCHS}" \
        > "${outdir}/launch.log" 2>&1 &

    sleep 5   # stagger startup to avoid simultaneous checkpoint-shard loads
done

echo "=========================================="
echo "All launched. PIDs of background jobs:"
jobs -p
echo
echo "Monitor with:  tail -f ./outputs/epoch/<soup>/launch.log"
echo "GPU usage:     watch -n2 nvidia-smi"
echo "=========================================="

wait
echo "All trainings finished."
