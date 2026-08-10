#!/bin/bash
# Standalone re-collect of dump_bin_bigbin/demo_randomized_eef to a full 500 episodes, using the
# FIXED collect_data.py (Phase-2 UnStableError no longer crash-truncates). Runs alongside the main
# v2 orchestrator on a shared GPU; overwrites the 475-episode part atomically only on success.
set -o pipefail
source /home/qinxiyu2/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
set -u
cd /home/qinxiyu2/dev/RoboTwin

OPENPI=/home/qinxiyu2/dev/openpi
PARTS=/shared/perception/datasets/robotwin_eef_obb_parts
GPU="${GPU:-4}"
T=dump_bin_bigbin
CFG=demo_randomized_eef

echo "[redo] $(date) fresh re-collect $T/$CFG on GPU $GPU (fixed collect_data.py)"
rm -rf "data/$T/$CFG"                                    # fresh slate (old data dir already gone; be sure)
if ! bash collect_data.sh "$T" "$CFG" "$GPU" > "logs/redo_collect_${T}.log" 2>&1; then
  echo "[redo] COLLECT FAIL $T (see logs/redo_collect_${T}.log)"; exit 1
fi
n_hdf5=$(find "data/$T/$CFG" -name 'episode*.hdf5' 2>/dev/null | wc -l)
echo "[redo] collected $n_hdf5 hdf5 episodes; converting..."

tmp="$PARTS/.tmp_redo_${T}_$$"
if HF_HUB_OFFLINE=1 PYTHONPATH="$OPENPI" "$OPENPI/.venv/bin/python" \
     "$OPENPI/scripts/convert_robotwin_eef_obb_to_lerobot.py" \
     --task-config-dir "data/$T/$CFG" --repo-id "eef_obb/$CFG/$T" --root "$tmp" --fps 50 \
     > "logs/redo_convert_${T}.log" 2>&1; then
  ne=$(python3 -c "import json;print(json.load(open('$tmp/meta/info.json'))['total_episodes'])" 2>/dev/null)
  mkdir -p "$PARTS/$CFG"; rm -rf "$PARTS/$CFG/$T"; mv "$tmp" "$PARTS/$CFG/$T"   # atomic overwrite
  rm -rf "data/$T/$CFG"
  echo "[redo] DONE $(date) — published $T with ${ne} episodes"
else
  echo "[redo] CONVERT FAIL $T (see logs/redo_convert_${T}.log)"; rm -rf "$tmp"; exit 1
fi
