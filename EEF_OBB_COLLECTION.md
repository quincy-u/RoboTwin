# RoboTwin EEF + OBB Dataset Collection (Lingbot-VA–matching)

Recollect a **27,500-episode** RoboTwin dataset that matches the Lingbot-VA dataset
`robbyant/robotwin-clean-and-aug-lerobot` **plus 2-slot per-hand OBB annotations**, for training a
pi0.5 action + OBB-head model in openpi.

**Why:** the previous `quincyu/robotwin_dynamic_27500` used **14-dim JOINT** actions and trained a
policy with ~30% lower success. Lingbot's **16-dim end-effector (endpose)** data trains a much
better policy (~77% in our eval). This recollection reproduces Lingbot's recipe (EEF, 640×480,
diverse native language, clean+randomized, aloha-agilex) and **adds OBB** for the OBB head.

---

## 0. TL;DR for the runner

```bash
# 1) create the orchestrator (script body in Section 3), then:
cd /home/qinxiyu2/dev/RoboTwin
setsid bash run_full_eef_obb_collect.sh > logs/eef_obb_collect.log 2>&1 < /dev/null &

# 2) watch progress
tail -f logs/eef_obb_collect.log
find /shared/perception/datasets/robotwin_eef_obb_parts -name info.json | wc -l   # target 100

# 3) after all 100 parts exist -> merge into the final dataset (Section 5)
```

Uses **GPUs 6,7** by default. Resumable. Peak local disk ≈ one task's HDF5 (~25 GB). ETA ≈ **4–6 days**
on 2 GPUs (measure the real rate from early log lines and refine).

---

## 1. What is ALREADY prepared (do not recreate)

| Item | Path | Notes |
|---|---|---|
| Clean config (50 eps/task) | `task_config/demo_clean_eef.yml` | = demo_clean + **Large_D435 (640×480)** |
| Randomized config (500 eps/task) | `task_config/demo_randomized_eef.yml` | = demo_randomized + Large_D435 |
| HDF5→LeRobot converter | `/home/qinxiyu2/dev/openpi/scripts/convert_robotwin_eef_obb_to_lerobot.py` | validated on pilot |
| Merge script | `/home/qinxiyu2/dev/openpi/scripts/robotwin_pi05/merge_eef_lerobot.py` | carries `obb_head` through |
| 2-slot OBB visualizer | `script/visualize_obb_2slot.py` | optional QA |
| Pilot QA videos | `/home/qinxiyu2/dev/openpi/obb_pilot_videos/` | 50 tasks, verified |

Both configs already have: `endpose: true`, `qpos: true`, `rgb: true`, `embodiment: [aloha-agilex]`,
`language_num: 100`, and OBB is written unconditionally by `Base_Task.save_obb2d`
(`observation/<cam>/proj_3d_obb/<obj>` + `object_target/{left,right}`).

---

## 2. Format the output MUST have (Lingbot-matching + OBB)

Per-frame LeRobot v2.1 features (what the converter writes):

- `observation.state` : float32**[16]** = `[left x,y,z,q1..q4,gripper, right x,y,z,q1..q4,gripper]` (endpose)
- `action` : float32**[16]** = **next-frame endpose** (`action[t] = state[t+1]`, last frame repeated)
- `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` : video, **480×640, TRUE RGB**
- `observation.obb_head` : float32**[32]** = `[left_16 | right_16]` **2-slot per-hand OBB** (head-cam pixels;
  slot 0 = left-hand target, slot 1 = right-hand target, from `object_target`; **empty hand → zeros**)
- `task` (per frame) : one native RoboTwin **`seen`** instruction for the episode (Lingbot-style)
- fps label : **50**

**Critical gotchas (already handled by the converter — keep them):**
1. **RGB channel flip.** RoboTwin's `pkl2hdf5` JPEG-encodes SAPIEN RGB via `cv2.imencode` → HDF5 stores
   **BGR-in-RGB**. The converter's `_decode_native` flips `[..., ::-1]` to true RGB (the AGILE**X** logo
   must render **red**, not blue). Skipping this reproduces the old BGR bug.
2. **Ignore `front_camera`.** The HDF5 has 4 cameras; Lingbot uses only head + 2 wrists.
3. **OBB is head-camera + object_target-gated** (2 slots), not one-box-per-object.

---

## 3. The orchestrator — save as `run_full_eef_obb_collect.sh` (RoboTwin root)

Resumable, disk-safe: per `(task, config)` it does clean-slate → collect HDF5 → convert to a LeRobot
"part" on `/shared` → **delete the HDF5**. One worker per GPU pulls from a shared queue. A part whose
`meta/info.json` already exists is **skipped**, so re-running resumes; add GPUs via `GPUS=4,5,6,7`.

```bash
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
```

Launch (detached, survives disconnect):

```bash
cd /home/qinxiyu2/dev/RoboTwin
setsid bash run_full_eef_obb_collect.sh > logs/eef_obb_collect.log 2>&1 < /dev/null &
```

---

## 4. Monitor / resume / add GPUs

```bash
# live log
tail -f logs/eef_obb_collect.log

# parts completed (target 100 = 50 clean + 50 randomized)
find /shared/perception/datasets/robotwin_eef_obb_parts -name info.json | wc -l

# per-part episode counts
for f in $(find /shared/perception/datasets/robotwin_eef_obb_parts -name info.json); do \
  echo "$(python3 -c "import json;print(json.load(open('$f'))['total_episodes'])") $f"; done
```

**Resume / add GPUs:** just relaunch the same script (kill the old one first if still up). Completed
parts are skipped. To use more GPUs when they free up:

```bash
GPUS=4,5,6,7 setsid bash run_full_eef_obb_collect.sh > logs/eef_obb_collect.log 2>&1 < /dev/null &
```

**Measure real ETA:** count `DONE` lines over a known wall-clock window → episodes/hour, then
`remaining_episodes / rate`. Expected ~2,500 clean + ~25,000 randomized = 27,500 total.

---

## 5. Merge parts → final dataset (after all 100 parts)

```bash
cd /home/qinxiyu2/dev/openpi
.venv/bin/python scripts/robotwin_pi05/merge_eef_lerobot.py \
  --src-root /shared/perception/datasets/robotwin_eef_obb_parts \
  --splits demo_clean_eef,demo_randomized_eef \
  --out-root /shared/perception/datasets/robotwin_eef_obb_27500 \
  --repo-id quincyu/robotwin_eef_obb_27500
```

The merge renumbers `episode_index`/`index`/`task_index`, copies mp4 (no re-encode), and **carries the
`observation.obb_head` column verbatim**. Result: one LeRobot dataset at
`/shared/perception/datasets/robotwin_eef_obb_27500`, symlinked into `$HF_LEROBOT_HOME`.

> Note: `/shared` is NFS (slow for training). For the actual training run, stage a local copy (or
> re-point the `~/.cache/huggingface/lerobot/...` symlink to fast local disk), as was done for the
> Lingbot dataset.

---

## 6. Verify a part (sanity)

```bash
cd /home/qinxiyu2/dev/openpi
HF_HUB_OFFLINE=1 .venv/bin/python - <<'PY'
import glob, json, numpy as np, pyarrow.parquet as pq
root = sorted(glob.glob("/shared/perception/datasets/robotwin_eef_obb_parts/*/*"))[0]
info = json.load(open(f"{root}/meta/info.json"))
print("features:", {k: v["shape"] for k, v in info["features"].items()})
pqf = sorted(glob.glob(f"{root}/data/chunk-000/episode_*.parquet"))[0]
t = pq.read_table(pqf)
st = np.array(t["observation.state"].to_pylist()); ac = np.array(t["action"].to_pylist())
ob = np.array(t["observation.obb_head"].to_pylist())
print("state", st.shape, "action", ac.shape, "obb_head", ob.shape)
print("action[t]==state[t+1]:", np.allclose(ac[:-1], st[1:], atol=1e-5))
print("task:", [json.loads(l)["task"] for l in open(f"{root}/meta/tasks.jsonl")][:1])
PY
```

Expect: state[16], action[16], obb_head[32], 3 cams [3,480,640], `action[t]==state[t+1]`, a natural-language task.
OBB overlay QA: `python script/visualize_obb_2slot.py data/<task>/<config>/data/episode0.hdf5 --out /tmp/x.mp4`.

---

## 7. Troubleshooting

- **A task fails to collect** → check `logs/collect_<config>_<task>.log`; the worker logs `COLLECT FAIL`
  and moves on. Re-run later to retry (it's still in the pending queue since no part was published).
- **Convert fails** → `logs/convert_<config>_<task>.log`; no part is published (atomic temp dir), so a
  re-run retries it.
- **Disk** → HDF5 is transient on `/home` (one task at a time); parts land on `/shared` (~52 GB total
  across all 100). If `/home` fills, a stuck HDF5 dir under `data/<task>/<config>` wasn't cleaned —
  safe to `rm -rf` any `data/*/demo_*_eef` whose part already exists on `/shared`.
- **RGB looks swapped (blue X)** → the converter's channel flip was bypassed; do NOT edit
  `_decode_native` out.
```
