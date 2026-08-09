#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

task_name=${1}
task_config=${2}
seed=${3:-0}
gpu_id=${4:-0}
object_name=${5:-auto}
test_num=${6:-1}

if [[ "$task_config" == "heuristic_smoke" ]]; then
    smoke_target="$REPO_ROOT/task_config/heuristic_smoke.yml"
    smoke_backup=
    smoke_created=false
    restore_smoke_config() {
        if [[ -n "$smoke_backup" ]]; then
            mv -- "$smoke_backup" "$smoke_target"
        elif [[ "$smoke_created" == true ]]; then
            rm -f -- "$smoke_target"
        fi
    }
    if [[ ! -f "$smoke_target" ]] || \
        ! cmp -s "$SCRIPT_DIR/heuristic_smoke.yml" "$smoke_target"; then
        if [[ -e "$smoke_target" ]]; then
            smoke_backup=$(mktemp "$REPO_ROOT/task_config/.heuristic_smoke.yml.XXXXXX")
            cp -- "$smoke_target" "$smoke_backup"
        else
            smoke_created=true
        fi
        cp -- "$SCRIPT_DIR/heuristic_smoke.yml" "$smoke_target"
        trap restore_smoke_config EXIT
    fi
fi

export CUDA_VISIBLE_DEVICES="$gpu_id"
export SIMPLE_GRASP_ROOT=${SIMPLE_GRASP_ROOT:-"$HOME/projects/simple-grasp"}

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

python script/eval_policy.py \
    --config policy/heuristic_baseline/deploy_policy.yml \
    --overrides \
    --task_name "$task_name" \
    --task_config "$task_config" \
    --ckpt_setting heuristic_baseline \
    --seed "$seed" \
    --policy_name heuristic_baseline \
    --simple_grasp_root "$SIMPLE_GRASP_ROOT" \
    --object_name "$object_name" \
    --device cuda:0 \
    --test_num "$test_num"
