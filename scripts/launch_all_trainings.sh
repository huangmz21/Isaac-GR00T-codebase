#!/bin/bash

# Main launcher for all atomic_seen training experiments
# Launches three training jobs on different GPU sets

cd /data1/mingzhe/Isaac-GR00T-codebase

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "=========================================="
echo "  Atomic Seen Training Launcher"
echo "=========================================="
echo ""

# Check if scripts exist
SCRIPT_DIR="scripts"
SCRIPTS=(
    "${SCRIPT_DIR}/train_4gpu_opencabinet_opendrawer.sh"
    "${SCRIPT_DIR}/train_2gpu_opendrawer.sh"
    "${SCRIPT_DIR}/train_2gpu_opencabinet.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo -e "${RED}✗ Script not found: $script${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✓ All training scripts found${NC}"
echo ""

# Create logs directory
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

echo "Training Configuration:"
echo "  1. 4-GPU (0,1,2,3): OpenCabinet + OpenDrawer"
echo "  2. 2-GPU (4,5):     OpenDrawer Only"
echo "  3. 2-GPU (6,7):     OpenCabinet Only"
echo ""

# Ask user for confirmation
read -p "Start all three training jobs? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "=========================================="
echo "Starting Training Jobs..."
echo "=========================================="
echo ""

# Launch training 1: 4-GPU OpenCabinet + OpenDrawer
echo -e "${BLUE}[1/3] Launching 4-GPU Training (OpenCabinet + OpenDrawer)...${NC}"
nohup bash ${SCRIPT_DIR}/train_4gpu_opencabinet_opendrawer.sh > ${LOG_DIR}/train_4gpu_opencabinet_opendrawer.log 2>&1 &
PID_4GPU=$!
echo -e "${GREEN}  ✓ Started (PID: $PID_4GPU)${NC}"
echo -e "  Log: ${LOG_DIR}/train_4gpu_opencabinet_opendrawer.log"
sleep 2

# Launch training 2: 2-GPU OpenDrawer
echo ""
echo -e "${BLUE}[2/3] Launching 2-GPU Training (OpenDrawer Only)...${NC}"
nohup bash ${SCRIPT_DIR}/train_2gpu_opendrawer.sh > ${LOG_DIR}/train_2gpu_opendrawer.log 2>&1 &
PID_2GPU_DRAWER=$!
echo -e "${GREEN}  ✓ Started (PID: $PID_2GPU_DRAWER)${NC}"
echo -e "  Log: ${LOG_DIR}/train_2gpu_opendrawer.log"
sleep 2

# Launch training 3: 2-GPU OpenCabinet
echo ""
echo -e "${BLUE}[3/3] Launching 2-GPU Training (OpenCabinet Only)...${NC}"
nohup bash ${SCRIPT_DIR}/train_2gpu_opencabinet.sh > ${LOG_DIR}/train_2gpu_opencabinet.log 2>&1 &
PID_2GPU_CABINET=$!
echo -e "${GREEN}  ✓ Started (PID: $PID_2GPU_CABINET)${NC}"
echo -e "  Log: ${LOG_DIR}/train_2gpu_opencabinet.log"

echo ""
echo "=========================================="
echo "All Training Jobs Launched!"
echo "=========================================="
echo ""

# Save PIDs to file
cat > ${LOG_DIR}/training_pids.txt << EOF
4GPU_OpenCabinet_OpenDrawer: $PID_4GPU
2GPU_OpenDrawer: $PID_2GPU_DRAWER
2GPU_OpenCabinet: $PID_2GPU_CABINET
EOF

echo "Process IDs saved to: ${LOG_DIR}/training_pids.txt"
echo ""
echo "Process IDs:"
echo "  4-GPU Training: $PID_4GPU"
echo "  2-GPU OpenDrawer: $PID_2GPU_DRAWER"
echo "  2-GPU OpenCabinet: $PID_2GPU_CABINET"
echo ""

# GPU allocation
echo "GPU Allocation:"
echo "  GPUs 0,1,2,3: 4-GPU Training (OpenCabinet + OpenDrawer)"
echo "  GPUs 4,5:     2-GPU Training (OpenDrawer Only)"
echo "  GPUs 6,7:     2-GPU Training (OpenCabinet Only)"
echo ""

echo "Output Directories:"
echo "  /data1/mingzhe/experiment/atomic_seen/opencabinet_opendrawer_4gpu"
echo "  /data1/mingzhe/experiment/atomic_seen/opendrawer_only_2gpu"
echo "  /data1/mingzhe/experiment/atomic_seen/opencabinet_only_2gpu"
echo ""

echo "=========================================="
echo "Monitoring Commands"
echo "=========================================="
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "View logs:"
echo "  tail -f ${LOG_DIR}/train_4gpu_opencabinet_opendrawer.log"
echo "  tail -f ${LOG_DIR}/train_2gpu_opendrawer.log"
echo "  tail -f ${LOG_DIR}/train_2gpu_opencabinet.log"
echo ""
echo "Monitor with TensorBoard:"
echo "  tensorboard --logdir /data1/mingzhe/experiment/atomic_seen"
echo ""
echo "Stop all trainings:"
echo "  kill $PID_4GPU $PID_2GPU_DRAWER $PID_2GPU_CABINET"
echo ""
echo "Check if still running:"
echo "  ps -p $PID_4GPU,$PID_2GPU_DRAWER,$PID_2GPU_CABINET"
echo ""
echo "=========================================="
echo ""

# Wait a bit and check if processes are running
sleep 5
echo "Verifying processes..."
for pid in $PID_4GPU $PID_2GPU_DRAWER $PID_2GPU_CABINET; do
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Process $pid is running${NC}"
    else
        echo -e "${RED}  ✗ Process $pid failed to start!${NC}"
        echo "    Check the log file for errors."
    fi
done

echo ""
echo -e "${GREEN}All trainings launched successfully!${NC}"
echo "Use the monitoring commands above to track progress."
echo ""
