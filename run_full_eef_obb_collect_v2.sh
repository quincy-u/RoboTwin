#!/bin/bash
# Overlapped EEF+OBB collector: GPU collectors and CPU converters run concurrently so a
# GPU never idles during the ~5-6h CPU/AV1 video-encode step. Resumable & disk-safe.
#
#   - Collectors (one per GPU) pull (config,task) pairs, collect HDF5 (collect_data.py
#     RESUMES a partial via seed.txt, and no-ops if a task's HDF5 is already complete), then
#     hand the task to a shared convert queue and immediately pull the next collection.
#   - Converters (a CPU pool) drain the convert queue: convert -> atomic-publish to /shared
#     -> delete the HDF5. Uses the (fixed) converter that recovers from a bad episode instead
#     of cascade-dropping the rest.
#   - Disk backpressure: a collector will not START a new collection while /home free space is
#     below MIN_FREE_GB, so unconverted HDF5 can't fill the disk if converts lag.
#   - Resume: parts whose meta/info.json already exists are skipped. NO pre-wipe of data dirs
#     (that would destroy an in-progress collection), unlike v1.
set -o pipefail
source /home/qinxiyu2/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
set -u
cd /home/qinxiyu2/dev/RoboTwin

OPENPI=/home/qinxiyu2/dev/openpi
PARTS=/shared/perception/datasets/robotwin_eef_obb_parts
GPUS="${GPUS:-4,5,6,7}"
CONV_WORKERS="${CONV_WORKERS:-6}"      # concurrent CPU converters
MIN_FREE_GB="${MIN_FREE_GB:-300}"      # collectors pause below this /home free space
CONFIGS=(demo_clean_eef demo_randomized_eef)
TASKS="adjust_bottle,beat_block_hammer,blocks_ranking_rgb,blocks_ranking_size,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_block,handover_mic,hanging_mug,lift_pot,move_can_pot,move_pillbottle_pad,move_playingcard_away,move_stapler_pad,open_laptop,open_microwave,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_basket,place_bread_skillet,place_burger_fries,place_can_basket,place_cans_plasticbox,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_mouse_pad,place_object_basket,place_object_scale,place_object_stand,place_phone_stand,place_shoe,press_stapler,put_bottles_dustbin,put_object_cabinet,rotate_qrcode,scan_object,shake_bottle,shake_bottle_horizontally,stack_blocks_three,stack_blocks_two,stack_bowls_three,stack_bowls_two,stamp_seal,turn_switch"

mkdir -p "$PARTS" logs
run="logs/.eef_obb_v2.$$"
taskq="$run.taskq"; tlock="$run.taskq.lock"
convq="$run.convq"; clock="$run.convq.lock"
doneflag="$run.collectors_done"
: > "$taskq"; : > "$convq"; touch "$tlock" "$clock"

IFS=',' read -ra tarr <<< "$TASKS"
for cfg in "${CONFIGS[@]}"; do
  for t in "${tarr[@]}"; do
    [ -f "$PARTS/$cfg/$t/meta/info.json" ] && continue   # already published -> skip (resume)
    echo "$cfg $t" >> "$taskq"
  done
done
echo "=== v2 overlapped run | pending: $(wc -l < "$taskq") pairs | GPUs: $GPUS | converters: $CONV_WORKERS | min-free: ${MIN_FREE_GB}G | $(date) ==="

pop() { local q=$1 lk=$2; ( flock 9; if [ -s "$q" ]; then head -1 "$q"; tail -n +2 "$q" > "$q.tmp"; mv "$q.tmp" "$q"; fi ) 9>"$lk"; }
push() { local q=$1 lk=$2 line=$3; ( flock 9; echo "$line" >> "$q" ) 9>"$lk"; }
free_gb() { df -P --block-size=1G /home/qinxiyu2 | awk 'NR==2{print $4}'; }

collector() {
  local gpu=$1
  while :; do
    local pair cfg t; pair=$(pop "$taskq" "$tlock"); [ -z "$pair" ] && break
    cfg=${pair%% *}; t=${pair##* }
    # disk backpressure: don't start a new collection while /home is tight
    while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do
      echo "[gpu $gpu] DISK-WAIT $(free_gb)G < ${MIN_FREE_GB}G, holding $t/$cfg  $(date +%H:%M:%S)"; sleep 120
    done
    echo "[gpu $gpu] COLLECT-START $t / $cfg  $(date +%H:%M:%S)"
    rm -rf "data/$t/$cfg/.cache"                          # clear stale cache only; keep episodes (resume)
    if bash collect_data.sh "$t" "$cfg" "$gpu" > "logs/collect_${cfg}_${t}.log" 2>&1; then
      echo "[gpu $gpu] COLLECT-DONE  $t / $cfg -> convert queue  $(date +%H:%M:%S)"
      push "$convq" "$clock" "$cfg $t"
    else
      echo "[gpu $gpu] COLLECT-FAIL  $t/$cfg (see logs/collect_${cfg}_${t}.log)"
    fi
  done
  echo "[gpu $gpu] collector exit  $(date +%H:%M:%S)"
}

converter() {
  local id=$1
  while :; do
    local pair cfg t; pair=$(pop "$convq" "$clock")
    if [ -z "$pair" ]; then
      [ -f "$doneflag" ] && break        # collectors finished AND queue drained -> exit
      sleep 20; continue
    fi
    cfg=${pair%% *}; t=${pair##* }
    echo "[conv $id] CONVERT-START $t / $cfg  $(date +%H:%M:%S)"
    local tmp="$PARTS/.tmp_${cfg}_${t}_$$_$id"
    if HF_HUB_OFFLINE=1 PYTHONPATH="$OPENPI" "$OPENPI/.venv/bin/python" \
         "$OPENPI/scripts/convert_robotwin_eef_obb_to_lerobot.py" \
         --task-config-dir "data/$t/$cfg" --repo-id "eef_obb/$cfg/$t" --root "$tmp" --fps 50 \
         > "logs/convert_${cfg}_${t}.log" 2>&1; then
      mkdir -p "$PARTS/$cfg"; rm -rf "$PARTS/$cfg/$t"; mv "$tmp" "$PARTS/$cfg/$t"   # atomic publish
      rm -rf "data/$t/$cfg"                               # free local disk
      local ne; ne=$(python3 -c "import json;print(json.load(open('$PARTS/$cfg/$t/meta/info.json'))['total_episodes'])" 2>/dev/null)
      echo "[conv $id] CONVERT-DONE  $t / $cfg  (${ne} eps)  $(date +%H:%M:%S)"
    else
      echo "[conv $id] CONVERT-FAIL  $t/$cfg (see logs/convert_${cfg}_${t}.log)"; rm -rf "$tmp"
    fi
  done
  echo "[conv $id] converter exit  $(date +%H:%M:%S)"
}

collector_pids=(); conv_pids=()
IFS=',' read -ra gpus <<< "$GPUS"
for g in "${gpus[@]}"; do collector "$g" & collector_pids+=($!); done
for i in $(seq 1 "$CONV_WORKERS"); do converter "$i" & conv_pids+=($!); done

for p in "${collector_pids[@]}"; do wait "$p"; done
touch "$doneflag"                                          # no more tasks will be enqueued
echo "=== all collectors done; draining $(wc -l < "$convq") queued + in-flight converts  $(date) ==="
for p in "${conv_pids[@]}"; do wait "$p"; done

rm -f "$taskq" "$tlock" "$convq" "$clock" "$doneflag" "$taskq.tmp" "$convq.tmp"
converted=$(find "$PARTS" -name info.json 2>/dev/null | wc -l)
echo "=== ALL DONE $(date) — $converted parts present in $PARTS ==="
