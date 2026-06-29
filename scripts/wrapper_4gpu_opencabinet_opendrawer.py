#!/usr/bin/env python3
"""
Wrapper script for 4-GPU training: OpenCabinet + OpenDrawer
Registers custom dataset soup before running training.
"""
import sys
import os

# Make sure we're in the correct directory
os.chdir('/data1/mingzhe/Isaac-GR00T-codebase')
sys.path.insert(0, '/data1/mingzhe/Isaac-GR00T-codebase')

# Add venv bin to PATH so torchrun can be found
venv_bin = '/data1/mingzhe/Isaac-GR00T-codebase/.venv/bin'
os.environ['PATH'] = venv_bin + ':' + os.environ.get('PATH', '')

# Register custom datasets before importing training script
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
DATASET_SOUP_REGISTRY['custom_opencabinet_opendrawer'] = [
    {'path': '/data1/robocasa365/datasets_box/v1.0/target/atomic/OpenCabinet/20250813/lerobot', 'filter_key': None},
    {'path': '/data1/robocasa365/datasets_box/v1.0/target/atomic/OpenDrawer/20250816/lerobot', 'filter_key': None},
]

print("✓ Registered custom_opencabinet_opendrawer dataset soup")
print("  - OpenCabinet: /data1/robocasa365/datasets_box/v1.0/target/atomic/OpenCabinet/20250813/lerobot")
print("  - OpenDrawer: /data1/robocasa365/datasets_box/v1.0/target/atomic/OpenDrawer/20250816/lerobot")
print()

# Now run the training script
import runpy
sys.argv = ['scripts/gr00t_finetune.py'] + sys.argv[1:]
runpy.run_path('scripts/gr00t_finetune.py', run_name='__main__')
