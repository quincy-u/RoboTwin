#!/bin/bash
set -o pipefail
source /home/qinxiyu2/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
set -u
cd /home/qinxiyu2/dev/RoboTwin

OPENPI=/home/qinxiyu2/dev/openpi
PARTS=/shared/perception/datasets/robotwin_eef_obb_parts
GPUS="${GPUS:-6,7}"
CONFIGS=(demo_clean_eef demo_randomized_eef)   # clean first (smaller), then randomized
TASKS="adjust_bottle,beat_block_hammer,blocks_ranking_rgb,blocks_ranking_size,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_block,handover_mic,hanging_mug,lift_pot,move_can_pot,move_pillbottle_pad,move_playingcard_away,move_stapler_pad,open_laptop,open_microwave,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_basket,place_bread_skillet,place_burger_fries,place_can_basket,place_cans_plasticbox,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_mouse_pad,place_object_basket,place_object_scale,place_object_stand,place_phone_stand,place_shoe,press_stapler,put_bottles_dustbin,put_object_cabinet,rotate_qrcode,scan_object,shake_bottle,shake_bottle_horizontally,stack_blocks_three,stack_blocks_two,stack_bowls_three,stack_bowls_two,stamp_seal,turn_switch"

mkdir -p "$PARTS" logs
queue="logs/.eef_obb_queue.$$"; lock="$queue.lock"; : > "$queue"; touch "$lock"
IFS=',' read -ra tarr <<< "$TASKS"
for cfg in "${CONFIGS[@]}"; do
  for t in "${tarr[@]}"; do
    [ -f "$PARTS/$cfg/$t/meta/info.json" ] && continue    # already converted -> skip (resume)
    echo "$cfg $t" >> "$queue"
  done
done
echo "pending (config,task) pairs: $(wc -l < "$queue")  | GPUs: $GPUS | $(date)"

pop() { ( flock 9; if [ -s "$queue" ]; then head -1 "$queue"; tail -n +2 "$queue" > "$queue.tmp"; mv "$queue.tmp" "$queue"; fi ) 9>"$lock"; }

worker() {
  local gpu=$1
  while :; do
    local pair cfg t; pair=$(pop); [ -z "$pair" ] && break
    cfg=${pair%% *}; t=${pair##* }
    echo "[gpu $gpu] START $t / $cfg  $(date +%H:%M:%S)"
    rm -rf "data/$t/$cfg"                                   # clean slate
    if ! bash collect_data.sh "$t" "$cfg" "$gpu" > "logs/collect_${cfg}_${t}.log" 2>&1; then
      echo "[gpu $gpu] COLLECT FAIL $t/$cfg"; continue; fi
    local tmp="$PARTS/.tmp_${cfg}_${t}_$$"
    if HF_HUB_OFFLINE=1 PYTHONPATH="$OPENPI" "$OPENPI/.venv/bin/python" \
         "$OPENPI/scripts/convert_robotwin_eef_obb_to_lerobot.py" \
         --task-config-dir "data/$t/$cfg" --repo-id "eef_obb/$cfg/$t" --root "$tmp" --fps 50 \
         > "logs/convert_${cfg}_${t}.log" 2>&1; then
      mkdir -p "$PARTS/$cfg"; rm -rf "$PARTS/$cfg/$t"; mv "$tmp" "$PARTS/$cfg/$t"   # atomic publish
      rm -rf "data/$t/$cfg"                                # delete HDF5
      local ne; ne=$(python3 -c "import json;print(json.load(open('$PARTS/$cfg/$t/meta/info.json'))['total_episodes'])" 2>/dev/null)
      echo "[gpu $gpu] DONE  $t / $cfg  (${ne} eps)  $(date +%H:%M:%S)"
    else
      echo "[gpu $gpu] CONVERT FAIL $t/$cfg"; rm -rf "$tmp"; continue
    fi
  done
}

IFS=',' read -ra gpus <<< "$GPUS"
for g in "${gpus[@]}"; do worker "$g" & done
wait
rm -f "$queue" "$lock"
converted=$(find "$PARTS" -name info.json 2>/dev/null | wc -l)
echo "ALL PAIRS PROCESSED $(date) — $converted/100 parts present in $PARTS"
