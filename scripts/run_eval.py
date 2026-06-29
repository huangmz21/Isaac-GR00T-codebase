# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ctypes.util
import json
import os
import subprocess
import sys
import threading
import time
import types
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
}
REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_PATH_CANDIDATES = [
    REPO_ROOT,
    os.environ.get("ROBOCASA_ROOT"),
    "/mnt/100T/users/dingxin/starVLA/playground/Code/robocasa365",
]
for import_path in IMPORT_PATH_CANDIDATES:
    if import_path is None:
        continue
    import_path = Path(import_path).expanduser()
    if import_path.exists() and import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())


def _install_robosuite_numba_cache_override() -> None:
    if "robosuite.macros_private" in sys.modules:
        return
    for import_root in sys.path:
        macros_private_path = Path(import_root) / "robosuite" / "macros_private.py"
        if macros_private_path.exists():
            return

    macros_private = types.ModuleType("robosuite.macros_private")
    macros_private.CACHE_NUMBA = False
    sys.modules["robosuite.macros_private"] = macros_private


_install_robosuite_numba_cache_override()


def apply_thread_env_defaults(env: dict[str, str]) -> None:
    for key, value in THREAD_ENV_DEFAULTS.items():
        env.setdefault(key, value)


apply_thread_env_defaults(os.environ)

import numpy as np

_TASK_SET_REGISTRY = None
_TARGET_TASKS = None
LOCAL_EGL_RUNTIME_DIR = Path(os.environ.get("GROOT_LOCAL_EGL_DIR", "/tmp/gr00t_eval_egl"))
RENDER_ERROR_PATTERNS = (
    "eglQueryString",
    "Cannot initialize a EGL device display",
    "gladLoadGL error",
    "DISPLAY environment variable is missing",
    "The GLFW library is not initialized",
    "OSMesa",
    "glGetError",
)


def _load_task_registries():
    global _TASK_SET_REGISTRY, _TARGET_TASKS
    if _TASK_SET_REGISTRY is None or _TARGET_TASKS is None:
        from robocasa.utils.dataset_registry import TARGET_TASKS, TASK_SET_REGISTRY

        _TASK_SET_REGISTRY = TASK_SET_REGISTRY
        _TARGET_TASKS = TARGET_TASKS
    return _TASK_SET_REGISTRY, _TARGET_TASKS


def run_server(data_config, model_path, embodiment_tag, port, denoising_steps=4):
    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    data_config = DATA_CONFIG_MAP[data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
    )

    server = RobotInferenceServer(policy, port=port)
    server.run()


def run_default_server_from_args(args):
    run_server(
        data_config=args.data_config,
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        port=args.port,
        denoising_steps=args.denoising_steps,
    )


def collect_env_names(task_set_list: Iterable[str]) -> list[str]:
    task_set_registry, _ = _load_task_registries()
    all_env_names = []
    seen = set()
    for task_set in task_set_list:
        if task_set not in task_set_registry:
            available = ", ".join(sorted(task_set_registry.keys()))
            raise KeyError(f"Unknown task_set={task_set!r}. Available task sets: {available}")
        for env_name in task_set_registry[task_set]:
            if env_name in seen:
                continue
            seen.add(env_name)
            all_env_names.append(env_name)
    return all_env_names


def shard_env_names(env_names: list[str], worker_rank: int, num_workers: int) -> list[str]:
    if num_workers <= 1:
        return env_names
    return env_names[worker_rank::num_workers]


def run_client(
    host,
    port,
    task_set_list,
    video_dir,
    split,
    n_episodes,
    n_envs,
    n_action_steps,
    env_names: list[str] | None = None,
    reset_policy_memory: bool = False,
    write_summary: bool = True,
    record_video: bool = True,
):
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from gr00t.eval.simulation import (
        MultiStepConfig,
        SimulationConfig,
        SimulationInferenceClient,
        VideoConfig,
    )

    simulation_client = SimulationInferenceClient(host=host, port=port)

    print("Available modality configs:")
    modality_config = simulation_client.get_modality_config()
    print(modality_config.keys())

    all_env_names = env_names if env_names is not None else collect_env_names(task_set_list)
    print(f"Evaluating {len(all_env_names)} tasks: {all_env_names}")

    for env_name in all_env_names:
        this_video_dir = os.path.join(video_dir, "evals", split, env_name)
        os.makedirs(this_video_dir, exist_ok=True)

        stats_path = os.path.join(this_video_dir, "stats.json")
        if os.path.exists(stats_path):
            print(f"{env_name} stats already exists. skipping.")
            continue
        horizon = get_task_horizon(env_name)
        config = SimulationConfig(
            env_name=f"robocasa/{env_name}",
            split=split,
            n_episodes=n_episodes,
            n_envs=n_envs,
            video=VideoConfig(video_dir=this_video_dir if record_video else None),
            multistep=MultiStepConfig(
                n_action_steps=n_action_steps,
                max_episode_steps=horizon,
            ),
            reset_policy_memory=reset_policy_memory,
        )

        print(f"Running simulation for {env_name}...")
        try:
            _, episode_successes = simulation_client.run_simulation(config)
        except Exception as e:
            if is_render_backend_error(e):
                raise RuntimeError(render_backend_error_message(e)) from e
            print("Exception!", e)
            continue

        episode_successes = [bool(success) for success in episode_successes]
        num_successes = int(np.sum(episode_successes))
        success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0

        print(f"Results for {env_name}:")
        print(f"Success rate: {success_rate:.2f}")

        with open(stats_path, "w") as f:
            stats = {
                "num_episodes": len(episode_successes),
                "num_successes": num_successes,
                "success_rate": success_rate,
                "successes": episode_successes,
            }
            json.dump(stats, f, indent=4)
        print(f"saved stats to {stats_path}")
        print()

    if write_summary:
        write_eval_summary(video_dir=video_dir, split=split, task_set_list=task_set_list)


def _target_group_mapping() -> OrderedDict[str, list[str]]:
    _, target_tasks = _load_task_registries()
    return OrderedDict(
        [
            ("atomic-seen", list(target_tasks.get("atomic_seen", []))),
            ("composite-seen", list(target_tasks.get("composite_seen", []))),
            ("composite-unseen", list(target_tasks.get("composite_unseen", []))),
        ]
    )


def _read_eval_stats(video_dir: str, split: str) -> dict[str, dict]:
    split_dir = os.path.join(video_dir, "evals", split)
    task_stats = {}
    if not os.path.isdir(split_dir):
        return task_stats

    for task_name in sorted(os.listdir(split_dir)):
        stats_path = os.path.join(split_dir, task_name, "stats.json")
        if not os.path.exists(stats_path):
            continue
        with open(stats_path, "r") as f:
            task_stats[task_name] = json.load(f)
    return task_stats


def _success_count(stats: dict) -> float:
    if "num_successes" in stats:
        return float(stats["num_successes"])
    return float(stats.get("success_rate", 0.0)) * float(stats.get("num_episodes", 0))


def _aggregate_task_stats(task_names: Iterable[str], task_stats: dict[str, dict]) -> dict:
    task_names = list(task_names)
    present_tasks = []
    missing_tasks = []
    total_episodes = 0
    total_successes = 0.0
    per_task = OrderedDict()

    for task_name in task_names:
        stats = task_stats.get(task_name)
        if stats is None:
            missing_tasks.append(task_name)
            continue

        num_episodes = int(stats.get("num_episodes", 0))
        num_successes = _success_count(stats)
        success_rate = float(stats.get("success_rate", 0.0))
        present_tasks.append(task_name)
        total_episodes += num_episodes
        total_successes += num_successes
        per_task[task_name] = {
            "num_episodes": num_episodes,
            "num_successes": num_successes,
            "success_rate": success_rate,
            "success_rate_percent": 100.0 * success_rate,
        }

    success_rate = total_successes / total_episodes if total_episodes > 0 else None
    return {
        "num_tasks": len(present_tasks),
        "num_expected_tasks": len(task_names),
        "num_episodes": total_episodes,
        "num_successes": total_successes,
        "success_rate": success_rate,
        "success_rate_percent": None if success_rate is None else 100.0 * success_rate,
        "tasks": per_task,
        "missing_tasks": missing_tasks,
    }


def write_eval_summary(video_dir: str, split: str, task_set_list: Iterable[str]) -> dict:
    task_set_list = list(task_set_list)
    task_stats = _read_eval_stats(video_dir, split)
    expected_tasks = collect_env_names(task_set_list)
    group_mapping = _target_group_mapping()

    summary = OrderedDict()
    summary["split"] = split
    summary["task_sets"] = list(task_set_list)
    summary["groups"] = OrderedDict()

    for group_name, group_tasks in group_mapping.items():
        summary["groups"][group_name] = _aggregate_task_stats(group_tasks, task_stats)

    summary["total"] = _aggregate_task_stats(expected_tasks, task_stats)

    summary_path = os.path.join(video_dir, "evals", split, "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print_eval_summary(summary, summary_path)
    return summary


def print_eval_summary(summary: dict, summary_path: str) -> None:
    print(f"Evaluation summary for split={summary['split']}:")
    for group_name, group_stats in summary["groups"].items():
        if group_stats["num_tasks"] == 0:
            continue
        rate = group_stats["success_rate_percent"]
        print(
            f"  {group_name}: {rate:.2f}% "
            f"({group_stats['num_successes']:.0f}/{group_stats['num_episodes']} episodes, "
            f"{group_stats['num_tasks']}/{group_stats['num_expected_tasks']} tasks)"
        )

    total = summary["total"]
    if total["success_rate_percent"] is not None:
        print(
            f"  total: {total['success_rate_percent']:.2f}% "
            f"({total['num_successes']:.0f}/{total['num_episodes']} episodes, "
            f"{total['num_tasks']}/{total['num_expected_tasks']} tasks)"
        )
    print(f"saved summary to {summary_path}")


def _normalize_gpu_ids(gpu_ids: list[str] | None, num_gpus: int) -> list[str]:
    if gpu_ids:
        normalized = []
        for item in gpu_ids:
            normalized.extend(part.strip() for part in item.split(",") if part.strip())
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            normalized = [part.strip() for part in visible.split(",") if part.strip()]
        else:
            normalized = [str(index) for index in range(num_gpus)]

    if len(normalized) < num_gpus:
        raise ValueError(f"Requested {num_gpus} GPUs, but only got gpu ids: {normalized}")
    return normalized[:num_gpus]


def configure_render_env(args) -> None:
    if getattr(args, "mujoco_gl", None):
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    if getattr(args, "pyopengl_platform", None):
        os.environ["PYOPENGL_PLATFORM"] = args.pyopengl_platform
    ensure_egl_runtime(args)


def ensure_egl_runtime(args) -> None:
    wants_egl = getattr(args, "mujoco_gl", None) == "egl" or getattr(
        args, "pyopengl_platform", None
    ) == "egl"
    if not wants_egl:
        return

    env_updates = {}
    if ctypes.util.find_library("EGL") is None:
        lib_dir, vendor_json = prepare_local_egl_runtime()
        env_updates["LD_LIBRARY_PATH"] = _prepend_env_path(
            os.environ.get("LD_LIBRARY_PATH", ""),
            lib_dir.as_posix(),
        )
        env_updates.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", vendor_json.as_posix())
    elif not _has_egl_vendor_config() and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
        _, vendor_json = prepare_local_egl_runtime()
        env_updates["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor_json.as_posix()

    if not env_updates:
        return

    next_env = os.environ.copy()
    next_env.update(env_updates)
    if "LD_LIBRARY_PATH" in env_updates and os.environ.get("GROOT_EGL_RUNTIME_READY") != "1":
        next_env["GROOT_EGL_RUNTIME_READY"] = "1"
        print(
            "Local GLVND libEGL runtime prepared under "
            f"{LOCAL_EGL_RUNTIME_DIR}; re-executing with updated LD_LIBRARY_PATH."
        )
        os.execvpe(sys.executable, [sys.executable, *sys.argv], next_env)

    os.environ.update(env_updates)


def prepare_local_egl_runtime() -> tuple[Path, Path]:
    lib_dir = LOCAL_EGL_RUNTIME_DIR / "usr/lib/x86_64-linux-gnu"
    vendor_dir = LOCAL_EGL_RUNTIME_DIR / "egl_vendor.d"
    vendor_json = vendor_dir / "10_nvidia.json"
    lib_egl = lib_dir / "libEGL.so"

    if not lib_egl.exists():
        debs = sorted(Path("/var/cache/apt/archives").glob("libegl1_*_amd64.deb"))
        if not debs:
            raise RuntimeError(
                "EGL rendering was requested, but libEGL.so is not installed and no "
                "libegl1_*.deb was found in /var/cache/apt/archives."
            )
        LOCAL_EGL_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["dpkg-deb", "-x", debs[-1].as_posix(), LOCAL_EGL_RUNTIME_DIR.as_posix()], check=True)
        if not lib_egl.exists():
            lib_egl.symlink_to("libEGL.so.1")

    vendor_dir.mkdir(parents=True, exist_ok=True)
    if not vendor_json.exists():
        vendor_json.write_text(
            '{\n'
            '    "file_format_version" : "1.0.0",\n'
            '    "ICD" : {\n'
            '        "library_path" : "libEGL_nvidia.so.0"\n'
            "    }\n"
            "}\n",
            encoding="utf-8",
        )

    return lib_dir, vendor_json


def _prepend_env_path(current: str, path: str) -> str:
    parts = [part for part in current.split(":") if part]
    if path in parts:
        return current
    return ":".join([path, *parts])


def _has_egl_vendor_config() -> bool:
    vendor_paths = [
        Path("/usr/share/glvnd/egl_vendor.d"),
        Path("/etc/glvnd/egl_vendor.d"),
    ]
    return any(path.exists() and any(path.glob("*.json")) for path in vendor_paths)


def apply_render_env_to_child(env: dict, args, gpu_id: str) -> None:
    if getattr(args, "mujoco_gl", None):
        env["MUJOCO_GL"] = args.mujoco_gl
    if getattr(args, "pyopengl_platform", None):
        env["PYOPENGL_PLATFORM"] = args.pyopengl_platform
    if env.get("MUJOCO_GL") == "egl":
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)


def is_render_backend_error(error: Exception) -> bool:
    message = str(error)
    return any(pattern in message for pattern in RENDER_ERROR_PATTERNS)


def render_backend_error_message(error: Exception) -> str:
    return (
        "MuJoCo/robosuite rendering backend failed during environment creation.\n"
        f"Original error: {error}\n\n"
        "This usually means the selected rendering backend is unavailable in this shell/container.\n"
        "Observed fixes:\n"
        "  - For EGL headless rendering, install or expose GLVND libEGL.so (e.g. libegl1) and run with "
        "`--mujoco_gl egl --pyopengl_platform egl`.\n"
        "  - For GLX rendering, run inside a session with DISPLAY set and use "
        "`--mujoco_gl glx --pyopengl_platform glx`.\n"
        "  - OSMesa requires a working libOSMesa and PyOpenGL OSMesa backend.\n"
        "Current script will stop on this error so the eval does not silently skip every task."
    )


def launch_parallel_workers(script_path: str, args) -> None:
    validate_model_path(args.model_path)
    gpu_ids = _normalize_gpu_ids(args.gpu_ids, args.num_gpus)
    processes = []

    for rank, gpu_id in enumerate(gpu_ids):
        port = args.port + rank
        cmd = [
            sys.executable,
            script_path,
            *sys.argv[1:],
            "--worker_rank",
            str(rank),
            "--num_workers",
            str(len(gpu_ids)),
            "--port",
            str(port),
        ]
        env = os.environ.copy()
        apply_thread_env_defaults(env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        apply_render_env_to_child(env, args=args, gpu_id=gpu_id)
        env.setdefault("PYTHONUNBUFFERED", "1")
        print(f"[parent] launching worker {rank}/{len(gpu_ids)} on GPU {gpu_id}, port {port}")
        processes.append((rank, subprocess.Popen(cmd, env=env)))

    failed = []
    try:
        for rank, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failed.append((rank, return_code))
    except KeyboardInterrupt:
        for _, process in processes:
            process.terminate()
        raise

    if failed:
        raise RuntimeError(f"Evaluation worker(s) failed: {failed}")

    write_eval_summary(
        video_dir=args.video_dir or args.model_path,
        split=args.split,
        task_set_list=args.task_set,
    )


def validate_model_path(model_path: str) -> None:
    path = Path(model_path).expanduser()
    if path.is_absolute() and not path.exists():
        raise FileNotFoundError(
            f"Local model_path does not exist: {path}\n"
            "Use an existing checkpoint directory, or pass a HuggingFace repo id instead."
        )


def run_worker(args, server_fn: Callable, reset_policy_memory: bool = False) -> None:
    validate_model_path(args.model_path)
    env_names = args.env_names or collect_env_names(args.task_set)
    worker_rank = int(args.worker_rank)
    num_workers = int(args.num_workers)
    worker_env_names = shard_env_names(env_names, worker_rank, num_workers)

    if len(worker_env_names) == 0:
        print(f"[worker {worker_rank}] no tasks assigned; exiting.")
        return

    print(
        f"[worker {worker_rank}/{num_workers}] port={args.port}, "
        f"tasks={len(worker_env_names)}: {worker_env_names}"
    )
    server_thread = threading.Thread(target=server_fn, args=(args,), daemon=True)
    server_thread.start()
    time.sleep(args.startup_wait)
    run_client(
        host=args.host,
        port=args.port,
        task_set_list=args.task_set,
        video_dir=args.video_dir or args.model_path,
        split=args.split,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        env_names=worker_env_names,
        reset_policy_memory=reset_policy_memory,
        write_summary=False,
        record_video=not args.no_video,
    )


def run_eval_entrypoint(args, server_fn: Callable, reset_policy_memory: bool = False) -> None:
    configure_render_env(args)
    if args.server:
        validate_model_path(args.model_path)
        server_fn(args)
    elif args.client:
        validate_model_path(args.model_path)
        run_client(
            host=args.host,
            port=args.port,
            task_set_list=args.task_set,
            video_dir=args.video_dir or args.model_path,
            split=args.split,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            env_names=args.env_names,
            reset_policy_memory=reset_policy_memory,
            record_video=not args.no_video,
        )
    elif args.worker_rank is not None:
        run_worker(args, server_fn=server_fn, reset_policy_memory=reset_policy_memory)
    elif args.num_gpus > 1:
        launch_parallel_workers(Path(sys.argv[0]).resolve().as_posix(), args)
    else:
        validate_model_path(args.model_path)
        server_thread = threading.Thread(target=server_fn, args=(args,), daemon=True)
        server_thread.start()
        time.sleep(args.startup_wait)
        run_client(
            host=args.host,
            port=args.port,
            task_set_list=args.task_set,
            video_dir=args.video_dir or args.model_path,
            split=args.split,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            env_names=args.env_names,
            reset_policy_memory=reset_policy_memory,
            record_video=not args.no_video,
        )


def add_common_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="<PATH_TO_YOUR_MODEL>",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        default="panda_omron",
    )
    parser.add_argument(
        "--task_set",
        type=str,
        nargs="+",
        help="Name of the task soup(s)",
        required=True,
    )
    parser.add_argument(
        "--env_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit RoboCasa env names. Useful for smoke tests or targeted reruns.",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Split to evaluate on. Can be either pretrain or target.",
        choices=["pretrain", "target"],
        required=True,
    )
    parser.add_argument("--port", type=int, help="Base port number for the server.", default=5555)
    parser.add_argument("--host", type=str, help="Host address for the server.", default="localhost")
    parser.add_argument("--video_dir", type=str, help="Directory to save videos.", default=None)
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Disable mp4 recording while still writing stats and summary files.",
    )
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=50)
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=5)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        default=16,
    )
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument(
        "--mujoco_gl",
        type=str,
        choices=["egl", "glx", "osmesa"],
        default=None,
        help="Optional MuJoCo rendering backend. Leave unset to use the environment/default robosuite choice.",
    )
    parser.add_argument(
        "--pyopengl_platform",
        type=str,
        choices=["egl", "glx", "osmesa"],
        default=None,
        help="Optional PyOpenGL platform. Usually matches --mujoco_gl.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of local GPU workers to launch when running server+client together.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        nargs="+",
        default=None,
        help="GPU ids for worker processes, e.g. '0 1 2 3' or '0,1,2,3'.",
    )
    parser.add_argument(
        "--startup_wait",
        type=float,
        default=1.0,
        help="Seconds to wait after starting each local inference server thread.",
    )
    parser.add_argument("--server", action="store_true", help="Run the server.")
    parser.add_argument("--client", action="store_true", help="Run the client.")
    parser.add_argument("--worker_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_workers", type=int, default=1, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    parser = add_common_eval_args(argparse.ArgumentParser())
    args = parser.parse_args()
    run_eval_entrypoint(args, server_fn=run_default_server_from_args)
