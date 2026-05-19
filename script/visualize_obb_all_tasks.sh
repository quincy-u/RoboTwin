#!/bin/bash
# Render a single OBB-overlay MP4 (4-cam grid) per task from data/<task>/<config>/.
# By default uses the demo_obb_vis 1-episode collection; pass a different config name
# as $1 to use a different demo dir. Outputs land in obb_videos/<config>/<task>/.
#
# Usage:
#   bash script/visualize_obb_all_tasks.sh                  # demo_obb_vis
#   bash script/visualize_obb_all_tasks.sh demo_clean       # use the 50-ep production runs
set +e
cd "$(dirname "$0")/.."

config=${1:-demo_obb_vis}
out_root=${2:-obb_videos/$config}
force=${FORCE:-0}      # FORCE=1 re-renders existing videos

mkdir -p "$out_root"

ok=0
skip=0
fail=0
shopt -s nullglob
for task_dir in data/*/$config; do
    task=$(basename "$(dirname "$task_dir")")
    data_dir="$task_dir/data"
    [[ -d "$data_dir" ]] || continue
    episodes=("$data_dir"/episode*.hdf5)
    (( ${#episodes[@]} )) || continue

    out_dir="$out_root/$task"
    mkdir -p "$out_dir"
    ep="${episodes[0]}"
    # Enumerate tracked objects that actually have proj_3d_obb across all
    # cameras (skip walls / synthetic actors with no model_data → no OBB).
    objs=$(python - "$ep" <<'PY'
import h5py, sys
path = sys.argv[1]
with h5py.File(path) as f:
    obj_pose = list(f["object_pose"].keys()) if "object_pose" in f else []
    cams = [c for c in f.get("observation", {}).keys() if "rgb" in f[f"observation/{c}"]]
    good = []
    for o in obj_pose:
        if all(f"observation/{c}/proj_3d_obb/{o}" in f for c in cams):
            good.append(o)
print(" ".join(sorted(good)))
PY
)
    if [[ -z "$objs" ]]; then
        echo "[skip] $task: no objects with proj_3d_obb"
        skip=$((skip + 1))
        continue
    fi
    for obj in $objs; do
        out="$out_dir/$(basename "$ep" .hdf5)_${obj}.mp4"
        if [[ -e "$out" && "$force" != "1" ]]; then
            echo "[have] $task / $obj"
            continue
        fi
        echo "==> $task / $(basename "$ep") / $obj"
        if python script/visualize_obb.py "$ep" --object "$obj" --out "$out" >/dev/null 2>&1; then
            ok=$((ok + 1))
        else
            echo "    !! failed: $task / $obj"
            fail=$((fail + 1))
        fi
    done
done

echo ""
echo "Done. ok=$ok  fail=$fail  skipped_tasks=$skip"
echo "Browse $out_root/"
