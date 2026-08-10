#!/bin/bash
# Resume data collection for put_bottles_dustbin ONLY, from where it left off
# (episode 432 -> 499) without deleting the 432 already-collected episodes.
#
# Calls collect_data.sh DIRECTLY, deliberately bypassing
# collect_remaining_tasks.sh whose "partial -> wipe" policy would rm -rf the
# whole task dir. collect_data.py natively resumes:
#   - Phase 1 skipped: seed.txt already has 500 seeds (suc_num=500).
#   - Phase 2 resumes: st_idx scans existing episode{idx}.hdf5 and starts at the
#     first missing one (432), reusing existing _traj_data pkls; scene_info.json
#     is loaded and appended to.
set -e
cd "$(dirname "$0")/.."

# conda's activate.d hooks reference unbound NVCC_PREPEND_FLAGS, so don't set -u.
source /home/qinxiyu2/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

export SAVE_PATH=/home/qinxiyu2/dev/RoboTwin/data_randomized

GPU=${1:-1}
exec bash collect_data.sh put_bottles_dustbin demo_randomized "$GPU"
