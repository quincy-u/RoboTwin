#!/bin/bash

# OBB-aware pi0.5 eval. Mirrors policy/pi05/eval.sh but:
#   - imports openpi from the vendored ../pi05/src tree (which now contains the OBB code),
#   - writes a per-episode obb-overlay MP4 next to RoboTwin's own eval video.

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi05_obb
task_name=${1}
task_config=${2}
train_config_name=${3:-pi05_aloha_full_base}
model_name=${4:-robotwin_obb}
seed=${5:-0}
gpu_id=${6:-0}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Activate the pi05 .venv (which has jax/flax/openpi deps installed). The vendored
# openpi tree at policy/pi05/src/openpi already contains the OBB code synced from
# the dev tree, so this is the one venv we need.
source "$(dirname "$0")/../pi05/.venv/bin/activate"

cd ../.. # move to RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name}
