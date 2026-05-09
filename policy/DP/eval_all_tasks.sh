#!/usr/bin/env bash
# Batch-evaluate DP on all tasks (same task discovery as train_all_tasks.sh).
# Run from policy/DP:
#   bash eval_all_tasks.sh
#
# One GPU:
#   GPU_ID=0 bash eval_all_tasks.sh
#
# Four GPUs in parallel:
#   GPUS=0,1,2,3 bash eval_all_tasks.sh
#
# Env (optional):
#   TASK_CONFIG=demo_clean       # eval environment setting
#   CKPT_SETTING=demo_clean      # which training run / checkpoint to load (often same as task_config)
#   EXPERT_DATA_NUM=50
#   SEED=0
#   DATA_ROOT=../../data
#   EVAL_TASKS=click_bell,adjust_bottle   # comma-separated; if unset, all tasks under DATA_ROOT with TASK_CONFIG/data
#   EVAL_RESULTS_DIR=eval_results          # per-task logs: <task>.log
#   RESULTS_FILE=eval_results/eval_summary.txt   # merged txt (default under EVAL_RESULTS_DIR)

cd "$(dirname "$0")"

# Ctrl-C only hits this shell by default; background "(...) &" workers keep running unless we trap.
stop_background_jobs() {
  echo "[eval_all_tasks] interrupt — stopping background workers..." >&2
  local p
  for p in $(jobs -p 2>/dev/null); do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap stop_background_jobs INT TERM

TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
CKPT_SETTING="${CKPT_SETTING:-demo_clean}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-50}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-../../data}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-eval_results}"
RESULTS_FILE="${RESULTS_FILE:-$EVAL_RESULTS_DIR/eval_summary.txt}"

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

tasks=()
if [[ -n "${EVAL_TASKS:-}" ]]; then
  IFS=',' read -ra _parts <<< "$EVAL_TASKS"
  for task_name in "${_parts[@]}"; do
    task_name="$(trim "$task_name")"
    [[ -z "$task_name" ]] && continue
    d="$DATA_ROOT/$task_name/"
    if [[ ! -d "${d}${TASK_CONFIG}/data" ]]; then
      echo "[eval_all_tasks] error: no data at ${d}${TASK_CONFIG}/data (task=${task_name})" >&2
      exit 1
    fi
    tasks+=("$task_name")
  done
else
  for d in "$DATA_ROOT"/*/; do
    [[ -d "$d" ]] || continue
    task_name="$(basename "$d")"
    [[ -d "${d}${TASK_CONFIG}/data" ]] || continue
    tasks+=("$task_name")
  done
  IFS=$'\n' tasks=($(printf '%s\n' "${tasks[@]}" | sort))
  unset IFS
fi

if ((${#tasks[@]} == 0)); then
  echo "[eval_all_tasks] error: no tasks (check EVAL_TASKS or DATA_ROOT/TASK_CONFIG)" >&2
  exit 1
fi

mkdir -p "$EVAL_RESULTS_DIR"

# Progress on stderr so you still see activity when stdout is only merged at the end.
run_eval() {
  local task="$1" gpu="$2"
  local log="${EVAL_RESULTS_DIR}/${task}.log"
  echo "[eval_all_tasks] START ${task} (GPU ${gpu})  -> ${log}" >&2
  (
    echo "=== task=${task} gpu=${gpu} start $(date -Is) ==="
    bash eval.sh "$task" "$TASK_CONFIG" "$CKPT_SETTING" "$EXPERT_DATA_NUM" "$SEED" "$gpu"
    ec=$?
    echo "=== task=${task} exit=${ec} end $(date -Is) ==="
    exit "$ec"
  ) >"$log" 2>&1
  ec=$?
  echo "[eval_all_tasks] END   ${task} (GPU ${gpu})  exit=${ec}" >&2
  return "$ec"
}

if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -r -a gpu_list <<< "${GPUS// /}"
  n_gpus=${#gpu_list[@]}
  echo "[eval_all_tasks] ${#tasks[@]} tasks, ${n_gpus} parallel workers (GPUS=${GPUS})" >&2
  echo "[eval_all_tasks] Full output per task: ${EVAL_RESULTS_DIR}/<task>.log" >&2
  echo "[eval_all_tasks] Tip: tail -f ${EVAL_RESULTS_DIR}/adjust_bottle.log   # example" >&2
  echo >&2
  for ((g = 0; g < n_gpus; g++)); do
    (
      for ((i = g; i < ${#tasks[@]}; i += n_gpus)); do
        run_eval "${tasks[i]}" "${gpu_list[g]}"
      done
    ) &
  done
  wait
else
  echo "[eval_all_tasks] ${#tasks[@]} tasks, sequential on GPU ${GPU_ID}" >&2
  for t in "${tasks[@]}"; do
    run_eval "$t" "$GPU_ID"
  done
fi

mkdir -p "$(dirname "$RESULTS_FILE")"
{
  echo "RoboTwin DP — batch eval summary"
  echo "Written: $(date -Is)"
  echo "TASK_CONFIG=$TASK_CONFIG CKPT_SETTING=$CKPT_SETTING EXPERT_DATA_NUM=$EXPERT_DATA_NUM SEED=$SEED"
  echo "Per-task logs: $EVAL_RESULTS_DIR/<task_name>.log"
  echo
  for t in "${tasks[@]}"; do
    f="${EVAL_RESULTS_DIR}/${t}.log"
    if [[ -f "$f" ]]; then
      echo "########################################################################"
      echo "# TASK: $t"
      echo "########################################################################"
      cat "$f"
      echo
    fi
  done
} >"$RESULTS_FILE"

echo "Merged results: $RESULTS_FILE"
