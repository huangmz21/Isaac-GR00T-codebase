#!/usr/bin/env bash
# Enhanced parallel RoboCasa eval: combines flexibility of run_eval_multi.sh
# with the polish of run_eval_atomic_seen_8gpu.sh
# Usage:
#   bash run_eval_multi_enhanced.sh <model_path> [n_gpus] [n_episodes] [split] [task_set] [data_config] [seed]
# Examples:
#   bash run_eval_multi_enhanced.sh /path/to/checkpoint-30000 8 30 target atomic_seen
#   bash run_eval_multi_enhanced.sh /path/to/checkpoint-60000 4 50 target composite_seen panda_omron 42
set -euo pipefail

# Activate environment if activate_env.sh exists
[ -f activate_env.sh ] && source activate_env.sh

MODEL_PATH="${1:?need model_path}"
NGPU="${2:-8}"
NEP="${3:-30}"
SPLIT="${4:-target}"
TASK_SET="${5:-atomic_seen}"
DATA_CONFIG="${6:-panda_omron}"
SEED="${7:-42}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
N_ENVS="${N_ENVS:-5}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
BASE_PORT="${BASE_PORT:-5600}"

# Pull the task list for this task_set straight from the registry
mapfile -t TASKS < <(python - "$TASK_SET" <<'PY'
import sys
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
for t in TASK_SET_REGISTRY[sys.argv[1]]:
    print(t)
PY
)
[ ${#TASKS[@]} -eq 0 ] && { echo "no tasks found for task_set=$TASK_SET"; exit 1; }

mkdir -p logs
RUN_TAG="$(basename "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH")_${TASK_SET}"

echo "=========================================="
echo "Starting ${NGPU}-GPU Parallel Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Task set: $TASK_SET ($SPLIT split)"
echo "Total tasks: ${#TASKS[@]}"
echo "Episodes per task: $NEP"
echo "Parallel envs: $N_ENVS"
echo "Action steps: $N_ACTION_STEPS"
echo "Data config: $DATA_CONFIG"
echo "Embodiment: $EMBODIMENT_TAG"
echo "Random seed: $SEED"
echo "=========================================="

for ((g=0; g<NGPU; g++)); do
  # Round-robin: task i -> gpu (i % NGPU)
  SUBSET=()
  for ((i=g; i<${#TASKS[@]}; i+=NGPU)); do
    SUBSET+=("${TASKS[$i]}")
  done
  [ ${#SUBSET[@]} -eq 0 ] && continue

  PORT=$((BASE_PORT + g))
  LOG="logs/eval_${RUN_TAG}_gpu${g}.log"

  echo "GPU $g (port $PORT): ${SUBSET[*]} (${#SUBSET[@]} tasks)"

  CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 nohup python scripts/run_eval.py \
    --model_path "$MODEL_PATH" \
    --embodiment_tag "$EMBODIMENT_TAG" \
    --data_config "$DATA_CONFIG" \
    --task_set "$TASK_SET" \
    --split "$SPLIT" \
    --n_episodes "$NEP" \
    --n_envs "$N_ENVS" \
    --n_action_steps "$N_ACTION_STEPS" \
    --seed "$SEED" \
    --port "$PORT" \
    --env_names "${SUBSET[@]}" \
    > "$LOG" 2>&1 &

  echo "  → PID $! → $LOG"
done

echo ""
echo "All eval workers started!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  tail -f logs/eval_${RUN_TAG}_gpu0.log"
echo "  watch -n 5 'grep -h \"success\\|Running\" logs/eval_${RUN_TAG}_gpu*.log | tail -30'"
echo ""
echo "Waiting for all processes to complete..."

wait

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="

# Try to aggregate results if the script exists
if [ -f gr00t/eval/get_eval_stats.py ]; then
  echo ""
  echo "Aggregating results..."
  python gr00t/eval/get_eval_stats.py --dir "$MODEL_PATH" || true
else
  echo "Results saved to: $MODEL_PATH/evals/$SPLIT/"
  echo "To aggregate: python gr00t/eval/get_eval_stats.py --dir $MODEL_PATH"
fi

echo "Done!"
