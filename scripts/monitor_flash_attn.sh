#!/bin/bash
# Monitor flash-attn installation progress

echo "========================================"
echo "Flash-Attention Installation Monitor"
echo "========================================"
echo ""

# Check if installation is running
if ps aux | grep -E "flash|nvcc" | grep -v grep > /dev/null; then
    echo "✓ Installation is running"
    echo ""
    echo "Active processes:"
    ps aux | grep -E "flash|nvcc" | grep -v grep | wc -l
    echo ""
    echo "Sample process:"
    ps aux | grep nvcc | grep -v grep | head -1 | awk '{print $11, $12, $13, $14}'
else
    echo "✗ No installation process found"
fi

echo ""
echo "========================================"
echo "Checking if flash_attn is installed..."
echo "========================================"

cd /mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T

if /home/huangmingzhe/.local/bin/uv pip list | grep -i flash > /dev/null 2>&1; then
    echo "✓ flash-attn is installed!"
    /home/huangmingzhe/.local/bin/uv pip list | grep flash
else
    echo "✗ flash-attn not yet installed"
    echo ""
    echo "Estimated time remaining: ~5-10 minutes"
    echo ""
    echo "You can:"
    echo "  1. Wait for compilation to complete"
    echo "  2. Check progress: bash scripts/monitor_flash_attn.sh"
    echo "  3. View build log: tail -f /tmp/claude-1010/.../tasks/bpfhdpal0.output"
fi

echo ""
echo "========================================"
