#!/bin/bash
# Fresh single-task re-collect + convert + atomic-publish, on a DEDICATED GPU (no sharing — sharing
# a GPU between two collect_data.py runs deadlocks SAPIEN/Vulkan). Uses the fixed collect_data.py
# (Phase-2 UnStableError no longer truncates). Usage: redo_task.sh <task> <gpu>
set -o pipefail
source /home/qinxiyu2/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
set -u
cd /home/qinxiyu2/dev/RoboTwin

OPENPI=/home/qinxiyu2/dev/openpi
PARTS=/shared/perception/datasets/robotwin_eef_obb_parts
T="$1"
GPU="$2"
CFG=demo_randomized_eef

echo "[redo $T] $(date) fresh re-collect on DEDICATED GPU $GPU (fixed collect_data.py)"
rm -rf "data/$T/$CFG"
if ! bash collect_data.sh "$T" "$CFG" "$GPU" > "logs/redo_collect_${T}.log" 2>&1; then
  echo "[redo $T] COLLECT FAIL (see logs/redo_collect_${T}.log)"; exit 1
fi
n=$(find "data/$T/$CFG" -name 'episode*.hdf5' 2>/dev/null | wc -l)
echo "[redo $T] collected $n hdf5 episodes; converting..."

tmp="$PARTS/.tmp_redo_${T}_$$"
if HF_HUB_OFFLINE=1 PYTHONPATH="$OPENPI" "$OPENPI/.venv/bin/python" \
     "$OPENPI/scripts/convert_robotwin_eef_obb_to_lerobot.py" \
     --task-config-dir "data/$T/$CFG" --repo-id "eef_obb/$CFG/$T" --root "$tmp" --fps 50 \
     > "logs/redo_convert_${T}.log" 2>&1; then
  ne=$(python3 -c "import json;print(json.load(open('$tmp/meta/info.json'))['total_episodes'])" 2>/dev/null)
  mkdir -p "$PARTS/$CFG"; rm -rf "$PARTS/$CFG/$T"; mv "$tmp" "$PARTS/$CFG/$T"
  rm -rf "data/$T/$CFG"
  echo "[redo $T] DONE $(date) — published with ${ne} episodes"
else
  echo "[redo $T] CONVERT FAIL (see logs/redo_convert_${T}.log)"; rm -rf "$tmp"; exit 1
fi
