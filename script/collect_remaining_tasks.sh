#!/bin/bash
# Collect data (with OBB) for the 41 tasks that don't already have data.
# Skips the 9 tasks already collected under data/<task>/demo_clean/.
#
# Default: 1 episode per task into data/<task>/demo_obb_vis/ — small enough
# to inspect OBB visually for every remaining task.
#
# GPU argument accepts a single id or a comma-separated list. With multiple
# GPUs the script runs one worker per GPU; all workers pull from a single
# shared task queue, so whichever GPU finishes first picks the next task.
#
# Usage:
#   bash script/collect_remaining_tasks.sh                       # 1 ep, gpu 0
#   bash script/collect_remaining_tasks.sh demo_obb_vis 5        # gpu 5
#   bash script/collect_remaining_tasks.sh demo_obb_vis 5,6      # parallel on gpus 5 and 6
#   bash script/collect_remaining_tasks.sh demo_clean 0,1,2,3    # 50 ep, 4-way parallel
#   GPU=5,6 bash script/collect_remaining_tasks.sh
#
# Per-task policy:
#   - >= TARGET_DEMOS hdf5 episodes: skip
#   - 1..TARGET_DEMOS-1 episodes (partial): wipe <task>/<config>/ and recollect
#   - none yet: collect
# Override target with TARGET_DEMOS=N (default 50).
# collect_data.py writes proj_3d_obb per episode via Base_Task.save_obb2d.
set -e
cd "$(dirname "$0")/.."

config=${1:-demo_obb_vis}
gpu_spec=${2:-${GPU:-0}}
shift 2 2>/dev/null || true
extra=("$@")

# Optional dataset root override. SAVE_PATH=/shared/foo writes data to
# /shared/foo/<task>/<config>/... instead of ./data/<task>/<config>/...
# Used both for the resume-skip check below and exported so the python
# process picks it up via os.environ["SAVE_PATH"] in collect_data.py.
data_root="${SAVE_PATH:-data}"
export SAVE_PATH="$data_root"

# Parse comma-separated GPU list into an array.
IFS=',' read -ra gpus <<<"$gpu_spec"
n_gpu=${#gpus[@]}

target_demos=${TARGET_DEMOS:-50}

# The 9 tasks already collected. Always skipped.
done_tasks=(
    adjust_bottle
    beat_block_hammer
    click_alarmclock
    grab_roller
    handover_mic
    lift_pot
    open_laptop
    pick_dual_bottles
    turn_switch
)

is_done() {
    local t=$1
    for d in "${done_tasks[@]}"; do
        [[ "$d" == "$t" ]] && return 0
    done
    return 1
}

# Task list: TASKS=task1,task2,... env var overrides auto-discovery and
# bypasses the done_tasks filter (useful for recollecting evaluation tasks).
if [[ -n "${TASKS:-}" ]]; then
    IFS=',' read -ra remaining <<<"$TASKS"
else
    remaining=()
    for f in envs/*.py; do
        name=$(basename "$f" .py)
        [[ "$name" == _* || "$name" == __init__ ]] && continue
        is_done "$name" && continue
        remaining+=("$name")
    done
fi

echo "Config:    $config"
echo "GPUs:      ${gpus[*]} (n=$n_gpu)"
echo "Extra:     ${extra[*]}"
echo "Target:    $target_demos episodes/task"
echo "Tasks:     ${#remaining[@]}"
printf '  - %s\n' "${remaining[@]}"
echo ""

mkdir -p logs

# Shared work queue: one task per line, popped atomically under flock.
# PID-suffixed so concurrent runs don't clobber each other.
queue_file="logs/.${config}_queue.$$"
lock_file="${queue_file}.lock"
printf '%s\n' "${remaining[@]}" > "$queue_file"
touch "$lock_file"

cleanup_queue() { rm -f "$queue_file" "${queue_file}.tmp" "$lock_file"; }
trap cleanup_queue EXIT
trap 'echo "Interrupted, killing workers..."; kill 0 2>/dev/null; exit 130' INT TERM

pop_task() {
    # Atomically pop the first line of the queue file, echo it to stdout.
    # Empty output means the queue is empty.
    (
        flock 9
        local task=""
        if [[ -s "$queue_file" ]]; then
            IFS= read -r task < "$queue_file" || true
            if [[ -n "$task" ]]; then
                tail -n +2 "$queue_file" > "${queue_file}.tmp"
                mv "${queue_file}.tmp" "$queue_file"
            fi
        fi
        printf '%s' "$task"
    ) 9>"$lock_file"
}

run_one_gpu() {
    local gpu=$1
    local log_prefix="logs/collect_${config}_gpu${gpu}"
    local fail_log="${log_prefix}.failed"
    : > "$fail_log"
    while true; do
        local task
        task=$(pop_task)
        [[ -z "$task" ]] && break
        local task_root="$data_root/$task/$config"
        local out="$task_root/data"
        local n_eps=0
        if [[ -d "$out" ]]; then
            n_eps=$(find "$out" -maxdepth 1 -name 'episode*.hdf5' 2>/dev/null | wc -l)
        fi
        if (( n_eps >= target_demos )); then
            echo "[gpu $gpu] [skip] $task ($n_eps/$target_demos episodes already)"
            continue
        fi
        if [[ -d "$task_root" ]]; then
            echo "[gpu $gpu] [wipe] $task ($n_eps/$target_demos partial — removing $task_root)"
            rm -rf "$task_root"
        fi
        local log="${log_prefix}_${task}.log"
        echo "[gpu $gpu] [run]  $task  (log: $log)"
        if ! bash collect_data.sh "$task" "$config" "$gpu" "${extra[@]}" >"$log" 2>&1; then
            echo "[gpu $gpu] !! $task failed (see $log)"
            echo "$task" >> "$fail_log"
        fi
    done
    echo "[gpu $gpu] queue empty, worker exiting"
}

pids=()
for gpu in "${gpus[@]}"; do
    run_one_gpu "$gpu" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || true
done

# Aggregate failures.
all_failed=()
for gpu in "${gpus[@]}"; do
    fail_log="logs/collect_${config}_gpu${gpu}.failed"
    if [[ -s "$fail_log" ]]; then
        while IFS= read -r t; do all_failed+=("$t"); done <"$fail_log"
    fi
done

echo ""
echo "Done. Attempted ${#remaining[@]} tasks across ${n_gpu} GPU(s)."
if (( ${#all_failed[@]} )); then
    echo "Failed (${#all_failed[@]}): ${all_failed[*]}"
    exit 1
fi
