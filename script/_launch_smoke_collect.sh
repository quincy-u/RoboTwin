#!/bin/bash
# One-shot launcher for the 50-task / 1-episode smoke run into data_randomized.
# Used by an interactive session to kick the orchestrator off as a background
# process with the right env. Not for production — use collect_remaining_tasks.sh
# directly for the 500-demo runs.
cd "$(dirname "$0")/.."

# conda's activate.d hooks reference unbound NVCC_PREPEND_FLAGS, so don't set -u here.
source /home/qinxiyu2/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

export SAVE_PATH=/home/qinxiyu2/dev/RoboTwin/data_randomized
export TARGET_DEMOS=1
export TASKS="adjust_bottle,beat_block_hammer,blocks_ranking_rgb,blocks_ranking_size,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_block,handover_mic,hanging_mug,lift_pot,move_can_pot,move_pillbottle_pad,move_playingcard_away,move_stapler_pad,open_laptop,open_microwave,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_basket,place_bread_skillet,place_burger_fries,place_can_basket,place_cans_plasticbox,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_mouse_pad,place_object_basket,place_object_scale,place_object_stand,place_phone_stand,place_shoe,press_stapler,put_bottles_dustbin,put_object_cabinet,rotate_qrcode,scan_object,shake_bottle,shake_bottle_horizontally,stack_blocks_three,stack_blocks_two,stack_bowls_three,stack_bowls_two,stamp_seal,turn_switch"

mkdir -p "$SAVE_PATH" logs
bash script/collect_remaining_tasks.sh demo_randomized_1ep 0,1,2,5
