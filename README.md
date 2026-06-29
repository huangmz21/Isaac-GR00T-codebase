## NVIDIA Isaac GR00T
This is the NVIDIA Isaac GR00T fork repo for running RoboCasa benchmark experiments. This fork is based on the original [GR00T code](https://github.com/NVIDIA/Isaac-GR00T) from NVIDIA. Our fork supports training for **GR00T N1.5**.

### Recommended system specs
For training we recommend a GPU with at least 80 Gb of memory (H100, H200, etc).
For inference we recommend a GPU with at least 8 Gb of memory.


### Installation
```
git clone https://github.com/robocasa-benchmark/Isaac-GR00T
cd groot
pip install -e .
```

### Key files
- Training: [scripts/gr00t_finetune.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/gr00t_finetune.py)
- Evaluation: [scripts/run_eval.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/run_eval.py)
- Memory training: `scripts/gr00t_memory_finetune.py`
- Memory evaluation: `scripts/run_eval_memory.py`

### Experiment workflow

All commands below assume the repo environment is active and RoboCasa is on `PYTHONPATH`.
For multi-GPU jobs, the training scripts launch `torchrun` automatically when `--num-gpus`
is greater than 1.

#### Standard training

Train a GR00T N1.5 checkpoint with the regular RoboCasa LeRobot mixture dataset:

```
python scripts/gr00t_finetune.py \
  --output-dir /mnt/100T/users/dingxin/VLA/Isaac-GR00T/playground/Outputs/pretrain_human300_8_16_lang \
  --dataset_soup pretrain_human300 \
  --data_config panda_omron_task_name_lang \
  --batch_size 16 \
  --num_gpus 8 \
  --max_steps 300000
```

Common dataset soups come from `robocasa.utils.dataset_registry.DATASET_SOUP_REGISTRY`,
for example `pretrain_human300`, `target_atomic_seen`, `target_composite_seen`, and
`target_composite_unseen`.

#### Memory training

Train a memory-enabled GR00T N1.5 checkpoint. The memory dataset adds historical
action-effect tuples:

`(mem_pre_feat, mem_post_feat, mem_actions, mem_valid_mask)`

The default feature layout is:

```
/data2/dingxin/dino_features/<atomic|composite>/<Task>/<Date>/lerobot/dinov2_vitb14_reg/observation.images.robot0_agentview_left/chunk-000/episode_000000.npy
```

Run:

```
python scripts/gr00t_memory_finetune.py \
  --output-dir /mnt/100T/users/dingxin/VLA/Isaac-GR00T/playground/Outputs/memory_pretrain_human300 \
  --dataset_soup pretrain_human300 \
  --data_config panda_omron_task_name_lang \
  --batch_size 16 \
  --num_gpus 8 \
  --max_steps 150000 \
  --memory_feature_root /data2/dingxin/dino_features \
  --memory_num_events 4 \
  --memory_interval 4 \
  --memory_video_key video.robot0_agentview_left \
  --memory_feature_video_key observation.images.robot0_agentview_left \
  --memory_feature_backbone dinov2_vitb14_reg
```

Useful memory knobs:

- `--memory_num_events`: number of historical events retrieved per sample.
- `--memory_interval`: number of action steps in one memory event.
- `--memory_fusion`: `per_token` or `global`.
- `--memory_retrieval_layers`: number of per-token retrieval blocks.
- `--memory_feature_mmap_mode r`: can reduce RAM when features are on fast local storage.
- `--report_to tensorboard`: useful when W&B upload/network is slow.

#### Standard evaluation

Evaluate a standard checkpoint. With `--num_gpus 8`, the script launches eight
local workers, each with one inference server and one task shard:

```
python scripts/run_eval.py \
  --model_path /mnt/100T/users/dingxin/VLA/Isaac-GR00T/playground/Outputs/pretrain_human300_8_16_lang/checkpoint-120000 \
  --task_set target50 \
  --split target \
  --data_config panda_omron_task_name_lang \
  --embodiment_tag new_embodiment \
  --num_gpus 8 \
  --n_episodes 50 \
  --n_envs 10 \
  --n_action_steps 16 \
  --port 5555 \
  --no_video \
  --mujoco_gl egl \
  --pyopengl_platform egl
```

You can also evaluate only one group:

```
python scripts/run_eval.py \
  --model_path <checkpoint-path> \
  --task_set atomic_seen \
  --split target \
  --num_gpus 8
```

#### Memory evaluation

Evaluate a memory checkpoint with online action-effect memory. The script reads
`memory_cfg.action_chunk_len` from the checkpoint and uses it as `n_action_steps`
unless `--n_action_steps` is explicitly provided.

```
python scripts/run_eval_memory.py \
  --model_path /mnt/100T/users/dingxin/VLA/Isaac-GR00T/playground/Outputs/memory_pretrain_human300/checkpoint-120000 \
  --task_set target50 \
  --split target \
  --data_config panda_omron \
  --embodiment_tag new_embodiment \
  --num_gpus 8 \
  --n_episodes 50 \
  --n_envs 5 \
  --port 5555 \
  --no_video \
  --mujoco_gl egl \
  --pyopengl_platform egl
```

Memory eval defaults:

- `--memory_mode online`: use predicted action history and online DINO features.
- `--memory_mode zero`: feed zero memory tensors for ablation.
- `--memory_num_events`, `--memory_video_key`, and `--memory_dino_backbone` default to
  the checkpoint `memory_cfg` when available.
- Memory state is reset at the start of each task and after each vector-env episode ends.

For online DINO feature extraction, the script looks for local DINOv2 assets using
`DINO_REPO_DIR`, `DINO_CHECKPOINT_PATH`, or `DINO_CHECKPOINT_DIR`. In this workspace,
the default search path also covers:

```
/mnt/100T/users/dingxin/starVLA/dinov2
/mnt/100T/users/dingxin/torch_cache/checkpoints
```

#### Rendering backend

Robosuite/MuJoCo needs a working offscreen rendering backend for eval. On headless
GPU machines, use EGL:

```
--mujoco_gl egl --pyopengl_platform egl
```

If EGL fails with `eglQueryString`, the shell/container is missing a usable GLVND
`libEGL.so` (for example the `libegl1` package or an equivalent mounted system
library). The eval script can automatically unpack `/var/cache/apt/archives/libegl1_*.deb`
to `/tmp/gr00t_eval_egl` when that package is available in the apt cache. If running
through an X session instead, use GLX with a valid `DISPLAY`:

```
--mujoco_gl glx --pyopengl_platform glx
```

#### Evaluation summary

Both eval scripts write per-task stats to:

```
<video_dir_or_model_path>/evals/<split>/<task_name>/stats.json
```

They also write an aggregate summary to:

```
<video_dir_or_model_path>/evals/<split>/summary.json
```

The summary reports:

- `atomic-seen`
- `composite-seen`
- `composite-unseen`
- `total`

Existing `stats.json` files are skipped, so interrupted evaluations can be resumed by
rerunning the same command. To store videos and stats outside the checkpoint directory,
pass `--video_dir <output-dir>`.

The older reporting helper still works for completed checkpoints:

```
python gr00t/eval/get_eval_stats.py \
  --dir <checkpoint-path>
```
