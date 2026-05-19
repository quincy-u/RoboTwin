#!/bin/bash
# Helper to finish off the demo_obb_vis collection. Runs through every task
# that doesn't yet have an episode, with a per-task wall-clock timeout to
# prevent another motion-planning hang from blocking the queue indefinitely.
set -u
cd "$(dirname "$0")/.."

config=demo_obb_vis
gpu=${1:-6}
timeout_sec=${2:-600}   # 10 min default per task

all_tasks=(
    blocks_ranking_rgb blocks_ranking_size click_bell dump_bin_bigbin handover_block
    hanging_mug move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad
    open_microwave pick_diverse_bottles place_a2b_left place_a2b_right place_bread_basket
    place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox
    place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad
    place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe
    press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object
    shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two
    stack_bowls_three stack_bowls_two stamp_seal
)

mkdir -p logs

for task in "${all_tasks[@]}"; do
    ep="data/$task/$config/data/episode0.hdf5"
    if [[ -e "$ep" ]]; then
        continue
    fi
    log="logs/collect_${config}_gpu${gpu}_${task}.log"
    echo "[gpu $gpu] [run]  $task  (timeout=${timeout_sec}s, log: $log)"
    timeout "${timeout_sec}" bash collect_data.sh "$task" "$config" "$gpu" >"$log" 2>&1 || {
        rc=$?
        if [[ $rc -eq 124 ]]; then
            echo "[gpu $gpu] !! $task TIMED OUT after ${timeout_sec}s"
            # Pkill children so the next task starts cleanly.
            pgrep -f "collect_data.py $task " | xargs -r kill -9 2>/dev/null || true
        else
            echo "[gpu $gpu] !! $task failed (rc=$rc)"
        fi
    }
done

echo "[gpu $gpu] done."
