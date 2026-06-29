#!/usr/bin/env python3
"""
Wrapper script for 2-GPU training: OpenDrawer Only
Registers custom dataset soup before running training.
"""
import sys
import os

# Make sure we're in the correct directory
os.chdir('/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T')
sys.path.insert(0, '/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T')

# Add venv bin to PATH so torchrun can be found
venv_bin = '/mnt/ssd_data/mingzhe/Code/robocasa365/Isaac-GR00T/.venv/bin'
os.environ['PATH'] = venv_bin + ':' + os.environ.get('PATH', '')

# Register custom datasets before importing
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
DATASET_SOUP_REGISTRY['custom_opendrawer_only'] = [
    {'path': '/data1/robocasa365/datasets_box/v1.0/target/atomic/OpenDrawer/20250816/lerobot', 'filter_key': None},
]

print("✓ Registered custom_opendrawer_only dataset soup")
print("  - OpenDrawer: /data1/robocasa365/datasets_box/v1.0/target/atomic/OpenDrawer/20250816/lerobot")
print()

# Now run the training script
import runpy
sys.argv = ['scripts/gr00t_finetune.py'] + sys.argv[1:]
runpy.run_path('scripts/gr00t_finetune.py', run_name='__main__')
