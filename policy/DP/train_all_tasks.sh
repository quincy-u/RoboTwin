#!/usr/bin/env bash
# Run from policy/DP:
#   bash train_all_tasks.sh
#
# One GPU (default):
#   GPU_ID=0 bash train_all_tasks.sh
#
# Four GPUs in parallel (each GPU runs every 4th task, by sorted order):
#   GPUS=0,1,2,3 bash train_all_tasks.sh
#
# Other env: TASK_CONFIG=demo_clean EXPERT_DATA_NUM=50 SEED=0 ACTION_DIM=14 DATA_ROOT=../../data

cd "$(dirname "$0")"

# Ctrl-C only hits this shell; parallel "(...) &" workers keep going unless we trap.
stop_background_jobs() {
  echo "[train_all_tasks] interrupt — stopping background workers..." >&2
  local p
  for p in $(jobs -p 2>/dev/null); do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap stop_background_jobs INT TERM

TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-50}"
SEED="${SEED:-0}"
ACTION_DIM="${ACTION_DIM:-14}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-../../data}"

tasks=()
for d in "$DATA_ROOT"/*/; do
  [[ -d "$d" ]] || continue
  task_name="$(basename "$d")"
  [[ -d "${d}${TASK_CONFIG}/data" ]] || continue
  tasks+=("$task_name")
done
IFS=$'\n' tasks=($(printf '%s\n' "${tasks[@]}" | sort))
unset IFS

run_task() {
  bash process_data.sh "$1" "$TASK_CONFIG" "$EXPERT_DATA_NUM"
  bash train.sh "$1" "$TASK_CONFIG" "$EXPERT_DATA_NUM" "$SEED" "$ACTION_DIM" "$2"
}

if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -r -a gpu_list <<< "${GPUS// /}"
  n_gpus=${#gpu_list[@]}
  for ((g = 0; g < n_gpus; g++)); do
    (
      for ((i = g; i < ${#tasks[@]}; i += n_gpus)); do
        run_task "${tasks[i]}" "${gpu_list[g]}"
      done
    ) &
  done
  wait
else
  for t in "${tasks[@]}"; do
    run_task "$t" "$GPU_ID"
  done
fi
