#!/bin/bash
# Wait for a running collect_remaining_tasks.sh orchestrator (by PID) to exit,
# then run visualize_obb_all_tasks.sh against the given config.
#
# Used to chain "collect 1 ep per task" → "render OBB overlay mp4 per task"
# in a single nohup-safe pipeline that survives SSH disconnect.
#
# Usage: bash script/wait_collect_then_visualize.sh <orch_pid> <config>
set -u
cd "$(dirname "$0")/.."

orch_pid=${1:?orchestrator pid required}
config=${2:?config name required}
log=logs/visualize_${config}_chain.log

{
  echo "[chain] waiting on orchestrator pid=$orch_pid for config=$config"
  while kill -0 "$orch_pid" 2>/dev/null; do
    sleep 30
  done
  echo "[chain] orchestrator $orch_pid exited at $(date -Is)"
  echo "[chain] starting visualize_obb_all_tasks.sh $config"
  source /home/qinxiyu2/miniconda3/etc/profile.d/conda.sh
  conda activate RoboTwin
  bash script/visualize_obb_all_tasks.sh "$config"
  echo "[chain] visualize done at $(date -Is)"
} >>"$log" 2>&1
