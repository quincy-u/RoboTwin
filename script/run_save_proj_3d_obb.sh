#!/bin/bash
set -e
cd "$(dirname "$0")/.."

python script/save_obb2d.py data/adjust_bottle/demo_clean/data      assets/objects/001_bottle      data/adjust_bottle/demo_clean/scene_info.json      --object bottle

python script/save_obb2d.py data/beat_block_hammer/demo_clean/data   assets/objects/020_hammer      data/beat_block_hammer/demo_clean/scene_info.json   --object hammer

python script/save_obb2d.py data/click_alarmclock/demo_clean/data    assets/objects/046_alarm-clock data/click_alarmclock/demo_clean/scene_info.json    --object alarmclock

python script/save_obb2d.py data/grab_roller/demo_clean/data         assets/objects/102_roller      data/grab_roller/demo_clean/scene_info.json         --object roller

python script/save_obb2d.py data/handover_mic/demo_clean/data        assets/objects/018_microphone  data/handover_mic/demo_clean/scene_info.json        --object mic

python script/save_obb2d.py data/open_laptop/demo_clean/data         assets/objects/015_laptop      data/open_laptop/demo_clean/scene_info.json         --object laptop

python script/save_obb2d.py data/pick_dual_bottles/demo_clean/data   assets/objects/001_bottle      data/pick_dual_bottles/demo_clean/scene_info.json   --object bottle1
python script/save_obb2d.py data/pick_dual_bottles/demo_clean/data   assets/objects/001_bottle      data/pick_dual_bottles/demo_clean/scene_info.json   --object bottle2

python script/save_obb2d.py data/turn_switch/demo_clean/data         assets/objects/056_switch      data/turn_switch/demo_clean/scene_info.json         --object switch

python script/save_obb2d.py data/lift_pot/demo_clean/data            assets/objects/060_kitchenpot  data/lift_pot/demo_clean/scene_info.json            --object pot

echo "All done."
