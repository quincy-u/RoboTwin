#!/bin/bash
# Polls until all (or the most we'll get) tasks have completed an episode,
# then renders OBB-overlay MP4s for everything that has data.
# Prints one progress line per minute so the parent can follow along.
set -u
cd "$(dirname "$0")/.."

config=${1:-demo_obb_vis}
expected=${2:-41}
stale_minutes=${3:-8}    # if no new episode in this many min, give up & visualize what we have
hard_cap_minutes=${4:-90}

start=$(date +%s)
last_count=0
last_change=$(date +%s)

while :; do
    count=$(ls data/*/$config/data/episode0.hdf5 2>/dev/null | wc -l)
    now=$(date +%s)
    elapsed=$(( (now - start) / 60 ))
    if (( count != last_count )); then
        last_count=$count
        last_change=$now
    fi
    stale=$(( (now - last_change) / 60 ))
    printf '[watcher] t=+%dm  done=%d/%d  stale=%dm\n' "$elapsed" "$count" "$expected" "$stale"

    if (( count >= expected )); then
        echo "[watcher] all done."
        break
    fi
    if (( stale >= stale_minutes )); then
        echo "[watcher] no progress for ${stale}m — proceeding with what we have ($count/$expected)."
        break
    fi
    if (( elapsed >= hard_cap_minutes )); then
        echo "[watcher] hard cap reached — proceeding ($count/$expected)."
        break
    fi
    sleep 60
done

echo ""
echo "[watcher] rendering OBB videos..."
source /home/qinxiyu2/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

bash script/visualize_obb_all_tasks.sh "$config"
echo "[watcher] DONE rendering. Outputs in obb_videos/$config/"
