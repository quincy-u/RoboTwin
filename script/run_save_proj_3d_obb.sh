#!/bin/bash
# Generic batch runner: regenerates 2D-projected 3D OBB for every collected task.
#
# Replaces the old per-task hardcoded list. Object→asset resolution comes from:
#   1. observation/object_info/{name} dataset in HDF5 (auto-written by Base_Task.save_obb2d
#      during new data collections)
#   2. scene_info.json info {A}/{B}/... matched positionally against tracked objects
#      (legacy data without object_info)
#   3. --override flag on save_obb2d_all.py for manual control.
#
# Specific tasks or demos:
#   bash script/run_save_proj_3d_obb.sh adjust_bottle pick_dual_bottles
#   bash script/run_save_proj_3d_obb.sh --demos demo_clean demo_randomized
set -e
cd "$(dirname "$0")/.."
python script/save_obb2d_all.py "$@"
