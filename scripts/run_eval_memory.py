# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.run_eval import add_common_eval_args, run_eval_entrypoint


def resolve_memory_action_steps(args) -> None:
    if args.n_action_steps is not None:
        return
    if args.memory_interval is not None:
        args.n_action_steps = int(args.memory_interval)
        return

    config_path = Path(args.model_path) / "config.json"
    if config_path.exists():
        with config_path.open("r") as f:
            model_config = json.load(f)
        memory_cfg = model_config.get("memory_cfg") or {}
        action_chunk_len = memory_cfg.get("action_chunk_len")
        if action_chunk_len is not None:
            args.n_action_steps = int(action_chunk_len)
            print(f"Using n_action_steps={args.n_action_steps} from checkpoint memory_cfg.")
            return

    args.n_action_steps = 4
    print("Checkpoint memory_cfg.action_chunk_len not found; using n_action_steps=4.")


def run_memory_server_from_args(args):
    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.memory_policy import Gr00tMemoryPolicy

    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tMemoryPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        memory_mode=args.memory_mode,
        memory_num_events=args.memory_num_events,
        memory_interval=args.memory_interval,
        memory_video_key=args.memory_video_key,
        memory_feature_dim=args.memory_feature_dim,
        memory_action_dim=args.memory_action_dim,
        memory_dino_backbone=args.memory_dino_backbone,
        memory_dino_device=args.memory_dino_device,
        n_action_steps=args.n_action_steps,
    )

    server = RobotInferenceServer(policy, port=args.port)
    server.run()


def add_memory_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--memory_mode",
        choices=["online", "zero"],
        default="zero",
        help="Use online rollout memory, or feed zero memory tensors for ablations.",
    )
    parser.add_argument(
        "--memory_num_events",
        type=int,
        default=None,
        help="Defaults to checkpoint memory_cfg.num_events, or 4 when absent.",
    )
    parser.add_argument(
        "--memory_interval",
        type=int,
        default=None,
        help="Action chunk length used by memory. Defaults to checkpoint memory_cfg.",
    )
    parser.add_argument(
        "--memory_video_key",
        type=str,
        default=None,
        help="Observation video key used to extract online DINO features. Defaults to checkpoint memory_cfg.video_key.",
    )
    parser.add_argument("--memory_feature_dim", type=int, default=None)
    parser.add_argument("--memory_action_dim", type=int, default=None)
    parser.add_argument(
        "--memory_dino_backbone",
        type=str,
        default=None,
        help="Defaults to checkpoint memory_cfg.feature_backbone, or dinov2_vitb14_reg when absent.",
    )
    parser.add_argument(
        "--memory_dino_device",
        type=str,
        default=None,
        help="Device for online DINO feature extraction. Defaults to the policy device.",
    )
    return parser


if __name__ == "__main__":
    parser = add_common_eval_args(argparse.ArgumentParser())
    parser.set_defaults(n_action_steps=None)
    parser = add_memory_eval_args(parser)
    args = parser.parse_args()
    resolve_memory_action_steps(args)
    run_eval_entrypoint(
        args,
        server_fn=run_memory_server_from_args,
        reset_policy_memory=True,
    )
