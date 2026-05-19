#!/bin/bash
# Train DP once per task (seed=0, demo_clean, 50 demos), then evaluate that
# single checkpoint at eval seeds {1,2,3} on {demo_clean, demo_randomized}.
# Aggregate mean ± std success rate per (task, eval_config) across the 3 eval seeds.
#
# deploy_policy.py:19 reads the checkpoint dir using --seed, so we symlink
# <task>-demo_clean-50-{1,2,3} -> <task>-demo_clean-50-0 to make every eval
# seed load the SAME trained model. The eval --seed still varies the rollout
# RNG (st_seed = 100000 * (1 + seed)) so the 3 evals differ as you'd expect.
#
# Skips train if the checkpoint file already exists; skips eval if a matching
# _result.txt is already present.
#
# Usage:
#   bash script/dp_train_eval_9tasks.sh 0,1,2,3,4,5,6,7         # all stages
#   bash script/dp_train_eval_9tasks.sh 0,1,2,6 fresh           # wipe ckpts+zarr first, then run all stages
#   bash script/dp_train_eval_9tasks.sh 0,1,2,3 train_only
#   bash script/dp_train_eval_9tasks.sh 0,1,2,3 eval_only
#   bash script/dp_train_eval_9tasks.sh _       aggregate_only

set -e
cd "$(dirname "$0")/.."

gpu_spec=${1:-0}
mode=${2:-all}                       # all | fresh | train_only | eval_only | aggregate_only
IFS=',' read -ra gpus <<<"$gpu_spec"

TASKS=(adjust_bottle beat_block_hammer click_alarmclock grab_roller handover_mic lift_pot open_laptop pick_dual_bottles turn_switch)
TRAIN_SEED=0
EVAL_SEEDS=(1 2 3)
EVAL_CONFIGS=(demo_clean demo_randomized)
TRAIN_CONFIG=demo_clean
EXPERT_DATA_NUM=50
ACTION_DIM=14
CKPT_SETTING=demo_clean
CHECKPOINT_NUM=${CHECKPOINT_NUM:-600}   # matches deploy_policy.yml default

# Source dataset root (the new collection lives here). Every run re-points
# data/<task>/demo_clean -> $DATA_ROOT/<task>/demo_clean so process_data.py
# reads the most recent HDF5s.
DATA_ROOT=${DATA_ROOT:-/shared/perception/datasets/robotwin}

mkdir -p logs

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
queue_t="logs/.dp_train_queue.$$"
queue_e="logs/.dp_eval_queue.$$"
lock_t="${queue_t}.lock"
lock_e="${queue_e}.lock"
cleanup() { rm -f "$queue_t" "$queue_e" "$lock_t" "$lock_e" "${queue_t}.tmp" "${queue_e}.tmp"; }
trap cleanup EXIT
trap 'echo "Interrupted, killing workers..."; kill 0 2>/dev/null; exit 130' INT TERM

pop_atomic() {
    local qfile=$1 lock=$2
    (
        flock 9
        local line=""
        if [[ -s "$qfile" ]]; then
            IFS= read -r line < "$qfile" || true
            if [[ -n "$line" ]]; then
                tail -n +2 "$qfile" > "${qfile}.tmp"
                mv "${qfile}.tmp" "$qfile"
            fi
        fi
        printf '%s' "$line"
    ) 9>"$lock"
}

ckpt_dir()  { echo "policy/DP/checkpoints/${1}-${TRAIN_CONFIG}-${EXPERT_DATA_NUM}-${2}"; }
ckpt_file() { echo "$(ckpt_dir "$1" "$2")/${CHECKPOINT_NUM}.ckpt"; }
eval_result_exists() {
    compgen -G "eval_result/${1}/DP/${2}/${CKPT_SETTING}/*_seed${3}/_result.txt" >/dev/null
}

# -----------------------------------------------------------------------------
# train — one job per task at TRAIN_SEED
# -----------------------------------------------------------------------------
build_train_queue() {
    : > "$queue_t"
    for task in "${TASKS[@]}"; do
        f=$(ckpt_file "$task" "$TRAIN_SEED")
        if [[ -f "$f" ]]; then
            echo "[skip-train] $task ($f exists)"
            continue
        fi
        echo "$task" >> "$queue_t"
    done
    touch "$lock_t"
}

run_train_worker() {
    local gpu=$1
    while :; do
        local task; task=$(pop_atomic "$queue_t" "$lock_t")
        [[ -z "$task" ]] && break
        local log="logs/dp_train_gpu${gpu}_${task}.log"
        echo "[gpu $gpu] [train] $task -> $log"
        (
            cd policy/DP
            bash train.sh "$task" "$TRAIN_CONFIG" "$EXPERT_DATA_NUM" "$TRAIN_SEED" "$ACTION_DIM" "$gpu"
        ) >"$log" 2>&1 || echo "[gpu $gpu] !! train $task FAILED (see $log)"
    done
    echo "[gpu $gpu] train worker done"
}

run_training() {
    build_train_queue
    local n; n=$(wc -l <"$queue_t")
    echo "Training jobs queued: $n  on GPUs: ${gpus[*]}"
    [[ $n -eq 0 ]] && return
    local pids=()
    for g in "${gpus[@]}"; do run_train_worker "$g" & pids+=("$!"); done
    for p in "${pids[@]}"; do wait "$p" || true; done
    echo "Training stage complete."
}

# -----------------------------------------------------------------------------
# symlink <task>-demo_clean-50-0 -> <task>-demo_clean-50-{1,2,3}
# -----------------------------------------------------------------------------
build_seed_symlinks() {
    for task in "${TASKS[@]}"; do
        local src; src=$(ckpt_dir "$task" "$TRAIN_SEED")
        [[ -d "$src" ]] || { echo "[symlink] skip $task (no $src)"; continue; }
        for s in "${EVAL_SEEDS[@]}"; do
            local dst; dst=$(ckpt_dir "$task" "$s")
            if [[ -L "$dst" ]]; then
                # Existing symlink — refresh it.
                ln -sfn "$(basename "$src")" "$dst"
            elif [[ -d "$dst" ]]; then
                if [[ -f "$dst/${CHECKPOINT_NUM}.ckpt" ]]; then
                    # Real dir with a usable checkpoint — assume it's a real
                    # per-seed training and leave it alone.
                    echo "[symlink] keep $dst (has ${CHECKPOINT_NUM}.ckpt)"
                else
                    # Real dir but no usable ckpt (partial / stale training) —
                    # remove and replace with symlink to seed=$TRAIN_SEED.
                    echo "[symlink] $dst has no ${CHECKPOINT_NUM}.ckpt — replacing with symlink to $(basename "$src")"
                    rm -rf "$dst"
                    ln -sfn "$(basename "$src")" "$dst"
                fi
            else
                ln -sfn "$(basename "$src")" "$dst"
            fi
        done
    done
}

# -----------------------------------------------------------------------------
# eval — 9 tasks × 3 eval seeds × 2 configs = 54 jobs
# -----------------------------------------------------------------------------
build_eval_queue() {
    : > "$queue_e"
    for task in "${TASKS[@]}"; do
        f=$(ckpt_file "$task" "$TRAIN_SEED")
        if [[ ! -f "$f" ]]; then
            echo "[skip-eval] $task — no checkpoint ($f). Train first."
            continue
        fi
        for seed in "${EVAL_SEEDS[@]}"; do
            for ec in "${EVAL_CONFIGS[@]}"; do
                if eval_result_exists "$task" "$ec" "$seed"; then
                    echo "[skip-eval] $task eval=$ec seed=$seed (result exists)"
                    continue
                fi
                echo "$task,$ec,$seed" >> "$queue_e"
            done
        done
    done
    touch "$lock_e"
}

run_eval_worker() {
    local gpu=$1
    while :; do
        local job; job=$(pop_atomic "$queue_e" "$lock_e")
        [[ -z "$job" ]] && break
        local task=${job%%,*} rest=${job#*,}
        local ec=${rest%,*} seed=${rest#*,}
        local log="logs/dp_eval_gpu${gpu}_${task}_${ec}_seed${seed}.log"
        echo "[gpu $gpu] [eval]  $task eval=$ec seed=$seed -> $log"
        (
            cd policy/DP
            bash eval.sh "$task" "$ec" "$CKPT_SETTING" "$EXPERT_DATA_NUM" "$seed" "$gpu"
        ) >"$log" 2>&1 || echo "[gpu $gpu] !! eval $task $ec seed=$seed FAILED (see $log)"
    done
    echo "[gpu $gpu] eval worker done"
}

run_evaluation() {
    build_seed_symlinks
    build_eval_queue
    local n; n=$(wc -l <"$queue_e")
    echo "Eval jobs queued: $n  on GPUs: ${gpus[*]}"
    [[ $n -eq 0 ]] && return
    local pids=()
    for g in "${gpus[@]}"; do run_eval_worker "$g" & pids+=("$!"); done
    for p in "${pids[@]}"; do wait "$p" || true; done
    echo "Evaluation stage complete."
}

# -----------------------------------------------------------------------------
# aggregate
# -----------------------------------------------------------------------------
aggregate() {
    local out="logs/dp_summary_$(date +%Y%m%d_%H%M).md"
    python script/aggregate_dp_results.py \
        --tasks "${TASKS[*]}" \
        --seeds "${EVAL_SEEDS[*]}" \
        --eval-configs "${EVAL_CONFIGS[*]}" \
        --ckpt-setting "$CKPT_SETTING" \
        --out "$out"
    echo ""
    echo "Summary -> $out"
    cat "$out"
}

# -----------------------------------------------------------------------------
# pre-flight setup: relink data + (optionally) wipe stale ckpts/zarr
# -----------------------------------------------------------------------------
link_data() {
    for t in "${TASKS[@]}"; do
        local src="${DATA_ROOT}/${t}/${TRAIN_CONFIG}"
        local dst="data/${t}/${TRAIN_CONFIG}"
        if [[ ! -d "$src" ]]; then
            echo "[data] !! $src missing — $t cannot train"
            continue
        fi
        # Replace any existing dir/symlink so the pointer is always fresh.
        rm -rf "$dst"
        mkdir -p "$(dirname "$dst")"
        ln -sfn "$src" "$dst"
    done
    echo "[data] linked data/<task>/${TRAIN_CONFIG} -> ${DATA_ROOT}/<task>/${TRAIN_CONFIG}"
}

fresh_wipe() {
    for t in "${TASKS[@]}"; do
        rm -rf "policy/DP/data/${t}-${TRAIN_CONFIG}-${EXPERT_DATA_NUM}.zarr"
        rm -rf policy/DP/checkpoints/${t}-${TRAIN_CONFIG}-${EXPERT_DATA_NUM}-{0,1,2,3}
    done
    echo "[fresh] wiped zarr cache + seed=0/1/2/3 ckpts for all 9 tasks"
}

case "$mode" in
    train_only)     link_data; run_training ;;
    eval_only)      run_evaluation ;;
    aggregate_only) aggregate ;;
    fresh)          link_data; fresh_wipe; run_training; run_evaluation; aggregate ;;
    all)            link_data; run_training; run_evaluation; aggregate ;;
    *) echo "Unknown mode: $mode (use all|fresh|train_only|eval_only|aggregate_only)"; exit 1 ;;
esac
